# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness + perf for the gfx1250 asm DSA sparse-prefill MLA kernel.

Exercises ``aiter.ops.mla_sparse_prefill.mla_sparse_prefill_fp8_asm`` (kernel
``mla_a8w8_qh128_1tg_32mx4_32nx1_sparse_pfl``) against a torch reference, with
the portable Triton sparse-prefill kernel as a second candidate.

Note on the perf comparison: the asm kernel is fp8 and the Triton kernel is
bf16.
"""

import argparse
import itertools

import pandas as pd
import torch
import triton
from test_pa_sparse_prefill_opus import (
    _FP8_D_HEAD,
    _FP8_D_NOPE,
    _FP8_D_NOPE_PADDED,
    _FP8_D_ROPE,
    _dense_csr,
    _empty_csr,
    _quantize_nope,
    _random_csr,
    _ref_pa_sparse_prefill_fp8,
)

import aiter
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.mla_sparse_prefill import mla_sparse_prefill_fp8_asm
from aiter.ops.triton._triton_kernels.attention.sparse_attention_dsv4 import (
    _sparse_attn_prefill_kernel,
)
from aiter.test_common import benchmark, checkAllclose, run_perftest

# Deliberately NOT torch.set_default_device("cuda"): the CSR generators reused
# from test_pa_sparse_prefill_opus build their index lists on the CPU with a CPU
# torch.Generator, and a cuda default device makes torch.randint try to pair
# that generator with a cuda allocation ("Expected a 'cuda' device type for
# generator"). Every tensor in this file names its device explicitly.

SUPPORTED_GFX = ["gfx1250"]
_ASM_SUB_KV = 32
_ASM_HEADS = 128

# How the CSR index lists are built:
#   "sparse" -- random nnz per row, with the leading rows pinned to tile
#               boundaries (0, 1, SUB_KV+/-1, 2*SUB_KV, ..., pool) so every
#               sweep hits the partial-tile branch
#   "dense"  -- every row references the whole pool (nnz == total_pages)
#   "empty"  -- no entries at all; exercises the sink-only path
#   "fixed"  -- exactly `nnz_prefix` / `nnz_extend` entries per row. Not in this
#               tuple: it is selected implicitly whenever -p/-e give an explicit
#               nnz, so the `mode` column is a constant in that sweep's table.
_MODES = ("sparse", "dense", "empty")

# checkAllclose mismatch ratio above which a configuration counts as failed.
_ERR_TOL = 0.05


def _rows_fp8(rows: int, device, gen: torch.Generator):
    """One pool of ``rows`` KV/Q rows in the kernel's split fp8/bf16 layout.

    Returns ``(nope_fp8[rows, 512], rope_bf16[rows, 64], fp32[rows, 512])``
    where the fp32 rows are ``concat(dequant_nope, rope)`` -- exactly what the
    kernel sees after dequantisation, and what the reference consumes.
    """
    real = torch.randn(rows, _FP8_D_NOPE, device=device, generator=gen) * 0.5
    nope, deq = _quantize_nope(real)
    rope = (torch.randn(rows, _FP8_D_ROPE, device=device, generator=gen) * 0.5).to(
        torch.bfloat16
    )
    return nope, rope, torch.cat([deq, rope.to(torch.float32)], dim=1)


def _fixed_csr(n: int, nnz: int, total_rows: int, *, device):
    """CSR with exactly ``nnz`` entries on every row (poc_kl-style)."""
    if nnz == 0:
        return _empty_csr(n, device=device)
    indptr = torch.arange(n + 1, dtype=torch.int32, device=device) * nnz
    indices = (
        torch.arange(nnz, dtype=torch.int32, device=device) % max(total_rows, 1)
    ).repeat(n)
    return indptr, indices


def _merge_two_sources(ukv, kv, ix_p, ip_p, ix_e, ip_e):
    """Fold the two KV regions into the single pool the Triton kernel takes: one
    concatenated row pool and one CSR whose per-token row is
    ``prefix_indices ++ (extend_indices + total_pages)``.

    Sound because the op is region-order-invariant -- both regions feed one
    shared online-softmax accumulator -- so concatenating them per token
    computes the identical result.
    """
    total_pages = ukv.shape[0]
    pool = torch.cat([ukv, kv], dim=0)
    lens_p = (ip_p[1:] - ip_p[:-1]).to(torch.int64)
    lens_e = (ip_e[1:] - ip_e[:-1]).to(torch.int64)
    indptr = torch.zeros(ip_p.numel(), dtype=torch.int32, device=ukv.device)
    indptr[1:] = torch.cumsum(lens_p + lens_e, 0).to(torch.int32)
    parts = []
    pp = ip_p.to(torch.int64).tolist()
    pe = ip_e.to(torch.int64).tolist()
    for i in range(len(pp) - 1):
        parts.append(ix_p[pp[i] : pp[i + 1]])
        parts.append(ix_e[pe[i] : pe[i + 1]] + total_pages)
    indices = (
        torch.cat(parts).to(torch.int32)
        if parts
        else torch.zeros(0, dtype=torch.int32, device=ukv.device)
    )
    return pool, indices, indptr


def run_triton(q, pool, indices, indptr, attn_sink, softmax_scale):
    """Portable Triton sparse prefill.

    Driven directly rather than through ``aiter.ops.triton.attention.
    pa_prefill_sparse``: that wrapper hard-routes gfx1250 to a gluon kernel
    which does not compile against the Triton version installed here.
    """
    out = torch.empty_like(q)
    num_queries, num_heads, head_dim = q.shape

    def grid(META):
        return (num_queries, triton.cdiv(num_heads, META["BLOCK_H"]))

    _sparse_attn_prefill_kernel[grid](
        q,
        pool,
        indices,
        indptr,
        attn_sink,
        out,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        pool.stride(0),
        pool.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        num_heads,
        head_dim,
        pool.shape[0],
        softmax_scale,
        HAS_ATTN_SINK=True,
        BLOCK_D=triton.next_power_of_2(head_dim),
    )
    return out


def make_inputs(
    t: int,
    h: int,
    total_pages: int,
    total_tokens: int,
    *,
    mode: str,
    nnz_prefix: int | None = None,
    nnz_extend: int | None = None,
    device="cuda",
    seed: int = 0,
) -> dict:
    """Build both the fp8 (asm) and bf16 (triton) views of one problem.

    Both views are derived from the *same* fp32 rows, so the two candidates
    solve the same numerical problem and each can be checked against a reference
    at its own precision.
    """
    device = torch.device(device)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    qn, qr, q_fp32 = _rows_fp8(t * h, device, gen)
    qn = qn.reshape(t, h, _FP8_D_NOPE_PADDED)
    qr = qr.reshape(t, h, _FP8_D_ROPE)
    q_fp32 = q_fp32.reshape(t, h, _FP8_D_HEAD)
    ukn, ukr, ukv_fp32 = _rows_fp8(total_pages, device, gen)
    kn, kr, kv_fp32 = _rows_fp8(total_tokens, device, gen)

    attn_sink = torch.randn(h, device=device, generator=gen, dtype=torch.float32) * 0.25

    def csr(total_rows: int, nnz: int | None, seed_offset: int):
        if nnz is not None:
            return _fixed_csr(t, nnz, total_rows, device=device)
        if mode == "sparse":
            # kv_tile_size=_ASM_SUB_KV: seed the leading rows with *this*
            # kernel's tile boundaries.
            return _random_csr(
                t,
                total_rows,
                device=device,
                kv_tile_size=_ASM_SUB_KV,
                seed=seed * 2 + seed_offset,
            )
        if mode == "dense":
            return _dense_csr(t, total_rows, device=device)
        return _empty_csr(t, device=device)

    ip_p, ix_p = csr(total_pages, nnz_prefix, 1)
    ip_e, ix_e = csr(total_tokens, nnz_extend, 2)

    return {
        "asm": {
            "q_nope": qn,
            "q_rope": qr,
            "unified_kv_nope": ukn,
            "unified_kv_rope": ukr,
            "kv_indices_prefix": ix_p,
            "kv_indptr_prefix": ip_p,
            "kv_nope": kn,
            "kv_rope": kr,
            "kv_indices_extend": ix_e,
            "kv_indptr_extend": ip_e,
            "attn_sink": attn_sink,
        },
        "q_bf16": q_fp32.to(torch.bfloat16),
        "merged": _merge_two_sources(
            ukv_fp32.to(torch.bfloat16),
            kv_fp32.to(torch.bfloat16),
            ix_p,
            ip_p,
            ix_e,
            ip_e,
        ),
        "ref": {
            "q_fp32": q_fp32,
            "ukv_fp32": ukv_fp32,
            "kv_fp32": kv_fp32,
            "kv_indices_prefix": ix_p,
            "kv_indptr_prefix": ip_p,
            "kv_indices_extend": ix_e,
            "kv_indptr_extend": ip_e,
            "attn_sink": attn_sink,
        },
        "nnz": (int(ip_p[-1].item()), int(ip_e[-1].item())),
    }


@benchmark()
def test_mla_sparse_prefill(
    t: int,
    h: int,
    total_pages: int,
    total_tokens: int,
    mode: str,
    nnz_prefix: int | None,
    nnz_extend: int | None,
):
    data = make_inputs(
        t,
        h,
        total_pages,
        total_tokens,
        mode=mode,
        nnz_prefix=nnz_prefix,
        nnz_extend=nnz_extend,
    )
    nnz_p, nnz_e = data["nnz"]
    total_nnz = nnz_p + nnz_e
    softmax_scale = 1.0 / (_FP8_D_HEAD**0.5)

    # One reference per input precision. The asm kernel consumes fp8-quantized
    # rows; Triton consumes the bf16 rounding of the same values. Checking a
    # bf16 candidate against the fp8 golden would report the quantization gap as
    # kernel error (~0.45 mismatch ratio), which says nothing about either.
    ref_fp8 = _ref_pa_sparse_prefill_fp8(**data["ref"], softmax_scale=softmax_scale)
    ref_args_bf16 = dict(data["ref"])
    for k in ("q_fp32", "ukv_fp32", "kv_fp32"):
        ref_args_bf16[k] = ref_args_bf16[k].to(torch.bfloat16).to(torch.float32)
    ref_bf16 = _ref_pa_sparse_prefill_fp8(**ref_args_bf16, softmax_scale=softmax_scale)
    refs = {"fp8": ref_fp8, "bf16": ref_bf16}

    candidates = {}
    if h == _ASM_HEADS:  # the kernel is built for exactly this head count
        candidates["asm"] = (
            lambda: mla_sparse_prefill_fp8_asm(
                **data["asm"], softmax_scale=softmax_scale
            ),
            "fp8",
            1,  # NoPE element size
        )
    q_bf16 = data["q_bf16"]
    pool, m_ix, m_ip = data["merged"]
    candidates["triton"] = (
        lambda: run_triton(
            q_bf16, pool, m_ix, m_ip, data["asm"]["attn_sink"], softmax_scale
        ),
        "bf16",
        2,
    )

    # 2 GEMMs (QK^T and PV) x 2 flops/MAC x H x nnz x D.
    flops = 4.0 * h * total_nnz * _FP8_D_HEAD

    ret = {}
    for name, (fn, dtype, kv_esz) in candidates.items():
        out, us = run_perftest(fn)
        # Traffic: Q read + the gathered KV rows read + O written (bf16).
        nbytes = (
            t * h * _FP8_D_HEAD * kv_esz  # Q
            + total_nnz * _FP8_D_HEAD * kv_esz  # gathered KV
            + t * h * _FP8_D_HEAD * 2  # O, always bf16
        )
        err = checkAllclose(
            refs[dtype].to(torch.float32),
            out.to(torch.float32),
            rtol=3e-2,
            atol=3e-2,
            tol_err_ratio=_ERR_TOL,
            msg=f"{name}: mla_sparse_prefill t={t} mode={mode}",
        )
        ret[f"{name} dtype"] = dtype
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


def main():
    # Whole-op arch gate goes HERE, not inside the @benchmark fn: that wrapper
    # always returns the call-args dict, so an in-fn return still emits an
    # args-only NaN row.
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "mla_sparse_prefill asm is gfx1250-only; found %s -- skipping", get_gfx()
        )
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="config input of test",
    )
    parser.add_argument(
        "-t",
        "--tokens",
        type=int,
        nargs="*",
        default=[1, 64, 256, 512, 1024],
        help="query token counts (one workgroup each)",
    )
    parser.add_argument(
        "-n",
        "--nhead",
        type=int,
        nargs="*",
        default=[_ASM_HEADS],
        help=f"head counts; the asm kernel only supports {_ASM_HEADS}",
    )
    parser.add_argument(
        "-m",
        "--modes",
        type=str,
        nargs="*",
        default=list(_MODES),
        help="CSR shape: sparse / dense / empty",
    )
    parser.add_argument(
        "--nnz",
        type=int,
        nargs="*",
        default=[0, 1, 31, 32, 33, 64, 96],
        help="per-row nnz values for the tile-boundary sweep "
        f"(asm SUB_KV={_ASM_SUB_KV})",
    )
    parser.add_argument(
        "-p",
        "--nnz-prefix",
        type=int,
        nargs="*",
        default=[256, 1024],
        help="per-row prefix nnz for the explicit shape sweep",
    )
    parser.add_argument(
        "-e",
        "--nnz-extend",
        type=int,
        nargs="*",
        default=[128],
        help="per-row extend nnz for the explicit shape sweep",
    )
    parser.add_argument(
        "--pool",
        type=int,
        default=4096,
        help="prefix/extend pool rows for the explicit shape sweep. Must be >= the\n"
        "largest per-row nnz, otherwise the index list wraps and every KV row\n"
        "gets gathered several times per token.",
    )
    args = parser.parse_args()

    total_pages, total_tokens = 128, 128
    failures = []

    def summarize(title, rows):
        df = pd.DataFrame(rows)
        # `mode` is the axis of the CSR-shape sweep, but a constant ("fixed") in
        # the explicit -p/-e sweep, where it is pure noise. Drop it only when it
        # has a single value in this table, so the sweep it actually indexes
        # keeps it.
        if "mode" in df.columns and df["mode"].nunique() <= 1:
            df = df.drop(columns=["mode"])
        aiter.logger.info("%s:\n%s", title, df.to_markdown(index=False))
        for col in [c for c in df.columns if c.endswith(" err")]:
            for _, r in df[df[col] > _ERR_TOL].iterrows():
                failures.append(f"{title}: {col}={r[col]:.4f} at {dict(r)}")

    # 1) CSR-shape sweep: sparse / dense / empty.
    summarize(
        "mla_sparse_prefill -- CSR shape sweep",
        [
            test_mla_sparse_prefill(t, h, total_pages, total_tokens, mode, None, None)
            for t, h, mode in itertools.product(args.tokens, args.nhead, args.modes)
        ],
    )

    # 2) Tile-boundary sweep: exact per-row nnz around multiples of SUB_KV,
    #    including prefix-only (extend empty -> the kernel parks its
    #    prefix->extend switch index) and extend-only (prefix empty -> the
    #    head-of-stream select jumps straight to the extend source).
    summarize(
        f"mla_sparse_prefill -- tile-boundary (SUB_KV={_ASM_SUB_KV}) sweep",
        [
            test_mla_sparse_prefill(8, h, total_pages, total_tokens, "fixed", npx, nex)
            for h, nnz in itertools.product(args.nhead, args.nnz)
            for npx, nex in ((nnz, 0), (0, nnz), (nnz, nnz))
        ],
    )

    # 3) Explicit (prefix, extend) shapes at realistic per-row nnz. The two sweeps
    #    above only ever reach nnz=128/row (dense mode over a 128-row pool), so
    #    this is the only coverage of a long prefix with a short extend. Its pool
    #    is sized so the index lists do not wrap.
    if args.nnz_prefix and args.nnz_extend:
        pool = max(args.pool, max(args.nnz_prefix), max(args.nnz_extend))
        summarize(
            "mla_sparse_prefill -- explicit (prefix, extend) sweep",
            [
                test_mla_sparse_prefill(t, h, pool, pool, "fixed", npx, nex)
                for t, h, npx, nex in itertools.product(
                    args.tokens,
                    args.nhead,
                    args.nnz_prefix,
                    args.nnz_extend,
                )
            ],
        )

    if failures:
        for f in failures:
            aiter.logger.error("FAIL %s", f)
        raise AssertionError(
            f"{len(failures)} configuration(s) exceeded the mismatch tolerance "
            f"of {_ERR_TOL}"
        )


if __name__ == "__main__":
    main()
