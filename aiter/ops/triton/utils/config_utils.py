# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Config-file infrastructure for the AITER Triton ops.

Every tuned Triton config ships as JSON under ``aiter/ops/triton/configs`` in
the nested layout ``<arch>/<backend>/<op>/<d_type>/``. This module owns both
halves of reading it: the shared path/probe/parse core, and the per-op loaders
built on top of it.

Sections
--------
Shared core
    ``AITER_TRITON_CONFIGS_PATH``, ``load_config_json()``,
    ``resolve_config_dir()`` -- the path constants, the cached JSON parse and
    the directory probe every loader below goes through.
GEMM
    ``get_gemm_config()`` plus the splitk / num-stages helpers.
Conv
    ``get_conv_config()`` and its canonical shape-key formatter.
MHC
    ``get_mhc_config()`` / ``get_mhc_post_config()``, with the gfx942 arch
    fallback.
MOE
    ``get_moe_configs()`` and the optimal-config wrappers.
Tuned kernel entries
    ``get_tuned_kernel_config()``, for kernels whose autotune search space
    lives in Python and only need one pinned tile per device.
"""

import copy
import functools
import glob
import itertools
import json
import os
import re
import warnings
from typing import Any

import torch
import triton

from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.utils.logger import AiterTritonLogger

logger = AiterTritonLogger()

# =============================================================================
# Shared core: paths, JSON loading, directory resolution
# =============================================================================

this_dir = os.path.dirname(os.path.abspath(__file__))
AITER_TRITON_OPS_PATH = os.path.abspath(f"{this_dir}/../")
AITER_TRITON_CONFIGS_PATH = os.path.abspath(f"{this_dir}/../configs")

# This flag should be set to True, unless it is being used for debugging.
# When False, config JSON files are re-read on every call, so live edits to
# the JSON are picked up.
USE_LRU_CACHE = True


@functools.lru_cache(maxsize=None if USE_LRU_CACHE else 0)
def load_config_json(fpath: str, required: bool = True) -> dict | None:
    """Load a config JSON file, cached per path (including negative results —
    add config files before process start, or call
    ``load_config_json.cache_clear()``). Raises FileNotFoundError if the file
    doesn't exist, consistently on every call (exceptions are never cached);
    pass required=False for probe/fallback lookups to get None instead.

    The returned dict is the shared cached object — copy before mutating:
    a shallow ``.copy()`` suffices for flat bucket dicts (scalar values),
    ``copy.deepcopy`` when nested sub-dicts will be mutated."""
    try:
        with open(fpath, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        if required:
            raise FileNotFoundError(
                f"Required config file doesn't exist: {fpath}"
            ) from None
        return None


def _dtype_dir(config_name: str) -> str:
    """Nested-layout directory for a config family:
    ``GEMM-AFP4WFP4`` -> ``gemm_afp4wfp4``."""
    return config_name.lower().replace("-", "_")


def resolve_config_dir(
    op: str,
    config_name: str,
    backend: str | None = None,
    arch: str | None = None,
) -> str:
    """Return the config directory of the first candidate that has a
    ``DEFAULT.json`` (the nested layout ``<arch>/<backend>/<op>/<d_type>``,
    the leaf from _dtype_dir()). With ``backend=None`` the probe tries
    ``triton`` then ``gluon``. Falls back to the last candidate so the
    missing-file assertion names the nested path. ``arch`` overrides the
    running architecture, for loaders that retry under another arch when the
    running one has no tuned configs."""
    dtype_dir = _dtype_dir(config_name)
    dev = arch if arch is not None else arch_info.get_arch()
    if backend is None:
        candidates = [
            f"{AITER_TRITON_CONFIGS_PATH}/{dev}/triton/{op}/{dtype_dir}",
            f"{AITER_TRITON_CONFIGS_PATH}/{dev}/gluon/{op}/{dtype_dir}",
        ]
    else:
        candidates = [
            f"{AITER_TRITON_CONFIGS_PATH}/{dev}/{backend}/{op}/{dtype_dir}",
        ]
    for cfg_dir in candidates:
        # Nested dirs name their default DEFAULT.json.
        if os.path.exists(f"{cfg_dir}/DEFAULT.json"):
            return cfg_dir
    return candidates[-1]


# =============================================================================
# GEMM
# =============================================================================

# Standard bounds for M_LEQ_x keys (tuple for hashability with LRU cache)
STANDARD_M_BOUNDS = (1, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)


@functools.lru_cache(maxsize=1024 if USE_LRU_CACHE else 0)
def _get_gemm_config_cached(
    config_name: str,
    M: int,
    N: int | None = None,
    K: int | None = None,
    bounds: tuple[int, ...] | None = None,
    specialized_filename: str | None = None,
    backend: str | None = None,
    B: int | None = None,
) -> tuple[dict, bool]:
    """
    Internal cached implementation. Do NOT use this directly — use
    ``get_gemm_config()`` instead, which returns a defensive deep-copy so
    callers can freely mutate the returned dict without polluting the cache.

    Resolves from ``<arch>/<backend>/gemm/<d_type>/`` (prefix-less filenames,
    default named ``DEFAULT.json``); ``backend=None`` tries triton then gluon.
    """
    # Input validation
    assert M >= 0, "M must be positive."
    assert N is None or N > 0, "N must be positive when provided."
    assert K is None or K > 0, "K must be positive when provided."
    assert bounds is None or (
        len(bounds) > 0
        and all(x > 0 for x in bounds)
        and all(x < y for x, y in itertools.pairwise(bounds))
    ), "When provided, bounds must be a non-empty tuple of strictly increasing positive numbers."

    # Every GEMM family lives in the nested layout <arch>/<backend>/gemm/
    # <d_type>/ (no arch prefix, default named DEFAULT.json); the shared probe
    # lives in resolve_config_dir().
    cfg_dir = resolve_config_dir("gemm", config_name, backend=backend)

    # Load default config (must exist)
    default_fpath = f"{cfg_dir}/DEFAULT.json"
    config_dict = load_config_json(default_fpath, required=False)
    if config_dict is None:
        raise AssertionError(f"Required config file doesn't exist: {default_fpath}")

    # Specialized configs override the default; the first existing file wins.
    # A custom specialized_filename (fused kernels with multiple N dims)
    # bypasses the B/N/K candidates.
    specialized_suffixes = []
    if specialized_filename is not None:
        specialized_suffixes = [specialized_filename]
    elif N is not None and K is not None:
        if B is not None:
            specialized_suffixes.append(f"B={B}-N={N}-K={K}")
        specialized_suffixes.append(f"N={N}-K={K}")

    is_tuned = False
    for suffix in specialized_suffixes:
        specialized_config = load_config_json(
            f"{cfg_dir}/{config_name}-{suffix}.json", required=False
        )
        if specialized_config is not None:
            config_dict, is_tuned = specialized_config, True
            break

    # use standard bounds unless custom bounds are passed
    search_bounds = bounds if bounds is not None else STANDARD_M_BOUNDS

    # Search for M_LEQ_x keys
    for bound in search_bounds:
        key = f"M_LEQ_{bound}"
        if M <= bound and key in config_dict:
            return dict(config_dict[key]), is_tuned

    # Search for M_GEQ_x keys
    for bound in reversed(search_bounds):
        key = f"M_GEQ_{bound}"
        if M >= bound and key in config_dict:
            return dict(config_dict[key]), is_tuned

    if "any" in config_dict:
        return dict(config_dict["any"]), False

    raise KeyError(
        f"No matching configuration found for M={M}, N={N}, K={K}, B={B}, "
        f"specialized_filename={specialized_filename!r} in config '{config_name}'."
    )


def get_gemm_config(
    config_name: str,
    M: int,
    N: int | None = None,
    K: int | None = None,
    bounds: tuple[int, ...] | None = None,
    specialized_filename: str | None = None,
    backend: str | None = None,
    B: int | None = None,
) -> tuple[dict, bool]:
    """
    Load a GEMM configuration using the standardized M_LEQ_x/M_GEQ_y/any format.

    This function provides a unified way to load GEMM configs across all kernels.
    It uses the following logic:
    1. Load default config file: <d_type>/DEFAULT.json
    2. If B, N and K are provided, try B-specialized config: {config_name}-B={B}-N={N}-K={K}.json
    3. If N and K are provided, try to load specialized config: {config_name}-N={N}-K={K}.json
       Or if specialized_filename is provided, use: {config_name}-{specialized_filename}.json
    4. Search for M_LEQ_x keys in order of bounds (default: STANDARD_M_BOUNDS)
    5. If no M_LEQ_x matches, search for M_GEQ_x keys in reverse order
    6. Fall back to "any" if no bounds match

    Args:
        config_name: Name of the config (example - "GEMM-A16W16")
        M: M dimension of the GEMM
        N: N dimension of the GEMM (optional)
        K: K dimension of the GEMM (optional)
        bounds: Custom bounds to use instead of STANDARD_M_BOUNDS (optional)
        specialized_filename: Custom specialized filename suffix (optional)
        backend: Backend name for per-backend config subdirectory (optional)
        B: Batch dimension for batched GEMM (optional)

    Returns:
        Dictionary with the config params (a fresh deep-copy safe to mutate),
        bool indicating if the config is tuned.(True if tuned, False otherwise)
    """
    config, is_tuned = _get_gemm_config_cached(
        config_name, M, N, K, bounds, specialized_filename, backend, B
    )
    return copy.deepcopy(config), is_tuned


def add_default_gemm_config_params(config: dict) -> dict:
    """
    this fn ensures that all configs have required default values.

    Args:
        config: Dictionary containing GEMM configuration parameters.

    Returns:
        same object as input
    """
    if "NUM_KSPLIT" not in config:
        config["NUM_KSPLIT"] = 1

    # adding default cache_modifier if not present as some kernels need this
    if "cache_modifier" not in config and "BLOCK_SIZE_K" in config:
        config["cache_modifier"] = None

    return config


def compute_splitk_params(config: dict, K: int) -> dict:
    """
    this fn calculates the SPLITK_BLOCK_SIZE and adjusts BLOCK_SIZE_K
    if necessary based on the NUM_KSPLIT value in the config.

    Args:
        config: Dictionary containing GEMM configuration parameters.
        K: K dimension of the GEMM operation (must be positive)

    Returns:
        same object as input
    """
    assert K > 0, "K must be positive"

    add_default_gemm_config_params(config)

    config["SPLITK_BLOCK_SIZE"] = triton.cdiv(K, config["NUM_KSPLIT"])

    if "BLOCK_SIZE_K" in config:
        # If NUM_KSPLIT makes K too small, then BLOCK_K will decrease to be smaller than
        # GROUP_K.
        while (
            config["NUM_KSPLIT"] > 1
            and config["BLOCK_SIZE_K"] > config["SPLITK_BLOCK_SIZE"]
        ):
            config["NUM_KSPLIT"] = max(config["NUM_KSPLIT"] // 2, 1)
            config["SPLITK_BLOCK_SIZE"] = triton.cdiv(K, config["NUM_KSPLIT"])

        # If BLOCK_SIZE_K is still too large with NUM_KSPLIT=1, fix it to equal K dim.
        if config["BLOCK_SIZE_K"] > config["SPLITK_BLOCK_SIZE"]:
            config["BLOCK_SIZE_K"] = triton.next_power_of_2(config["SPLITK_BLOCK_SIZE"])

            if config["BLOCK_SIZE_K"] > config["SPLITK_BLOCK_SIZE"]:
                config["BLOCK_SIZE_K"] = config["BLOCK_SIZE_K"] // 2

        config["BLOCK_SIZE_K"] = max(config["BLOCK_SIZE_K"], 16)

        # Round the SPLITK_BLOCK_SIZE to multiple of BLOCK_SIZE_K and update NUM_KSPLIT to again.
        if config["NUM_KSPLIT"] > 1 and (
            config["SPLITK_BLOCK_SIZE"] % config["BLOCK_SIZE_K"] != 0
        ):
            config["SPLITK_BLOCK_SIZE"] = (
                triton.cdiv(config["SPLITK_BLOCK_SIZE"], config["BLOCK_SIZE_K"])
                * config["BLOCK_SIZE_K"]
            )
            config["NUM_KSPLIT"] = triton.cdiv(K, config["SPLITK_BLOCK_SIZE"])

    return config


def _padded_size_32_4(n):
    pad = (n >> 5) << 2
    if (n & 31) == 0 and pad >= 4:
        pad -= 4
    return n + pad


def _padded_size_pow2(n, interval, padding):
    log2_i = (interval - 1).bit_length()
    log2_p = (padding - 1).bit_length() if padding else 0
    pad = (n >> log2_i) << log2_p
    if n % interval == 0 and pad >= padding:
        pad -= padding
    return n + pad


def _gemm_lds_bytes(
    block_m, block_n, block_k, bits_a, bits_b, num_stages, use_async_padding
):
    elem_a = block_m * block_k
    elem_b = block_k * block_n
    if use_async_padding:
        # Padded shared encoding + N buffers (matches TensorAtlas
        # _estimate_triton_lds_async_copy / tritonBLAS origami).
        pa = _padded_size_32_4(elem_a)
        pb = _padded_size_32_4(elem_b)
        if block_k & (block_k - 1) == 0:
            pa = max(pa, _padded_size_pow2(elem_a, block_k, 8))
        if block_n & (block_n - 1) == 0:
            pb = max(pb, _padded_size_pow2(elem_b, block_n, 8))
        return num_stages * (pa * bits_a + pb * bits_b) // 8
    # Non-async: (N-1) extra buffer pairs beyond the active stage.
    LDSA = elem_a * bits_a
    LDSB = elem_b * bits_b
    if num_stages <= 1:
        return max(LDSA, LDSB) // 8
    return (LDSA + LDSB) * (num_stages - 1) // 8


def pick_gemm_num_stages(
    arch, block_m, block_n, block_k, bits_a, bits_b, use_async_padding=False
):
    assert min(block_m, block_n, block_k, bits_a, bits_b) > 0
    # bits_a / bits_b: element bit-widths (8 for fp8, 4 for mxfp4).
    # use_async_padding: True when the kernel lowers to async direct-to-LDS
    # with padded shared encoding (e.g. a4w4 on gfx950).
    cap = arch_info._LDS_CAP_BYTES.get(arch)
    if cap is None:
        return 2
    lds = _gemm_lds_bytes(
        block_m, block_n, block_k, bits_a, bits_b, 2, use_async_padding
    )
    return 2 if lds <= cap else 1


# =============================================================================
# Conv
# =============================================================================

CONV_STANDARD_M_BOUNDS: tuple[int, ...] = (
    4,
    8,
    16,
    32,
    64,
    128,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
    65536,
    131072,
    262144,
)


def format_shape_key(
    N: int,
    C: int,
    H: int,
    W: int,
    K: int,
    R: int,
    S: int,
    sh: int,
    sw: int,
    ph: int,
    pw: int,
    dh: int,
    dw: int,
) -> str:
    """Canonical string key for a user-visible conv2d call. Same format used by
    the loader and the kernel-side _get_config helpers.
    """
    return (
        f"N={N},C={C},H={H},W={W},K={K},R={R},S={S},"
        f"sh={sh},sw={sw},ph={ph},pw={pw},dh={dh},dw={dw}"
    )


@functools.lru_cache(maxsize=512 if USE_LRU_CACHE else 0)
def _get_conv_config_cached(
    config_name: str,
    shape_key: str | None,
    M: int | None,
) -> dict:
    """Three-tier walk: literal shape entry -> M_LEQ bucket -> 'any'."""
    dev = arch_info.get_arch()
    # Nested layout <arch>/triton/conv/<d_type>/DEFAULT.json; the shared probe
    # lives in resolve_config_dir().
    cfg_dir = resolve_config_dir("conv", config_name, backend="triton")
    config_dict = load_config_json(f"{cfg_dir}/DEFAULT.json")

    # Tier 1: literal shape key.
    shapes = config_dict.get("shapes", {})
    if shape_key is not None and shape_key in shapes:
        return shapes[shape_key]

    # Tier 2: M-bucket walk.
    if M is not None and M >= 0:
        for bound in CONV_STANDARD_M_BOUNDS:
            key = f"M_LEQ_{bound}"
            if M <= bound and key in config_dict:
                return config_dict[key]

    # Tier 3: any fallback.
    if "any" in config_dict:
        return config_dict["any"]

    raise KeyError(
        f"No matching config in '{config_name}' for shape_key={shape_key!r}, "
        f"M={M} on arch {dev} (no literal shape, no bucket, no 'any' fallback)."
    )


def get_conv_config(
    config_name: str,
    shape_key: str | None = None,
    M: int | None = None,
) -> dict:
    """Load a conv kernel config for the running GPU arch.

    Walk order (first hit wins):
        1. ``shapes[shape_key]`` — exact-shape pin from the offline sweep.
        2. ``M_LEQ_<n>`` — row-count bucket walk (M_total for GEMM-like
           kernels, T for Winograd).
        3. ``"any"`` — global fallback.

    Returns a fresh deep-copy of the config dict; safe to mutate.

    Modeled on :func:`get_gemm_config` but with conv-native (shape-key first)
    dispatch and no splitk / N=K= specialization.
    """
    config = _get_conv_config_cached(config_name, shape_key, M)
    return copy.deepcopy(config)


# =============================================================================
# MHC
# =============================================================================

_FALLBACK_DEV = "gfx942"


def _mhc_config_dir(dev: str, config_name: str) -> str:
    """``dev``'s nested config directory for the ``config_name`` family."""
    return resolve_config_dir("mhc", config_name, backend="triton", arch=dev)


def _load_with_fallback(
    dev: str, config_name: str, fname: str, required: bool = False
) -> dict | None:
    """Load ``fname`` from ``dev``'s ``config_name`` directory, falling back to
    the gfx942 copy for arches without tuned MHC configs (may be suboptimal)."""
    config = load_config_json(
        f"{_mhc_config_dir(dev, config_name)}/{fname}", required=False
    )
    if config is None:
        config = load_config_json(
            f"{_mhc_config_dir(_FALLBACK_DEV, config_name)}/{fname}", required=required
        )
    return config


@functools.lru_cache(maxsize=None if USE_LRU_CACHE else 0)
def _c_thresholds(dev: str, actual_config_name: str) -> tuple[int, ...]:
    """C values that have a specialized config file (arch-specific plus the
    gfx942 fallback), sorted ascending."""
    thresholds = set()
    for d in {dev, _FALLBACK_DEV}:
        cfg_dir = _mhc_config_dir(d, actual_config_name)
        pattern = f"{cfg_dir}/{actual_config_name}-C=*.json"
        for fpath in glob.glob(pattern):
            match = re.search(r"-C=(\d+)\.json$", os.path.basename(fpath))
            if match:
                thresholds.add(int(match.group(1)))
    return tuple(sorted(thresholds))


@functools.lru_cache(maxsize=1024 if USE_LRU_CACHE else 0)
def get_mhc_config(
    config_name: str,
    M: int,
    C: int,
    mode: str | None = None,
) -> tuple[dict, bool]:
    """
    Load MHC configuration with threshold matching of M_LEQ_x keys, C, and mode.

    Selection logic:
    - C: Finds the largest C-specific config file threshold <= input C value.
      Available C configs are discovered from the files named
      {config}-C={value}.json in the arch's config directory.
    - M: Within the selected config, finds the largest M_LEQ_x threshold <= input M value.

    Architecture fallback:
    - If configs for the current GPU architecture don't exist, falls back to gfx942 configs.
    - This allows MHC operations to work on GPUs without tuned configs (may be suboptimal).

    Config file naming convention:
    - For MHC_FUSED: mode is required ("sinkhorn")
      - e.g., gfx942/triton/mhc/mhc_fused_sinkhorn/MHC_FUSED_SINKHORN-C=128.json

    Args:
        config_name: Base name of the config (e.g., "MHC_FUSED")
        M: M dimension (batch/sequence size)
        C: C dimension (hidden dim per stream). Uses threshold matching
            to find the largest available C config <= input C.
        mode: H_res mode for MHC_FUSED - "sinkhorn" (required for MHC_FUSED)

    Returns:
        Tuple of (config dict, bool indicating if C-specialized config was used)

    Raises:
        ValueError: If mode is invalid or missing when required
        KeyError: If no matching config found
    """
    dev = arch_info.get_arch()

    if mode is None or mode != "sinkhorn":
        raise ValueError(f"mode must be 'sinkhorn', got '{mode}'")
    actual_config_name = f"{config_name}_{mode.upper()}"

    # Default config (must exist for the arch or the gfx942 fallback)
    config_dict = _load_with_fallback(
        dev, actual_config_name, "DEFAULT.json", required=True
    )
    used_specialized = False

    # C-specific config: largest discovered threshold <= input C wins
    for c_threshold in reversed(_c_thresholds(dev, actual_config_name)):
        if C >= c_threshold:
            specialized = _load_with_fallback(
                dev, actual_config_name, f"{actual_config_name}-C={c_threshold}.json"
            )
            if specialized is not None:
                config_dict = specialized
                used_specialized = True
                break

    # Extract M_LEQ_x keys and their thresholds, sorted ascending
    m_leq_keys = []
    for key in config_dict:
        if key.startswith("M_LEQ_"):
            try:
                threshold = int(key[6:])  # Extract number after "M_LEQ_"
                m_leq_keys.append((threshold, key))
            except ValueError:
                continue
    m_leq_keys.sort()  # Sort by threshold value

    # Find largest threshold <= M
    matched_key = None
    for threshold, key in m_leq_keys:
        if M >= threshold:
            matched_key = key
        else:
            break

    if matched_key is not None:
        return dict(config_dict[matched_key]), used_specialized

    # Fallback to "any" if no matching key found
    if "any" in config_dict:
        return dict(config_dict["any"]), used_specialized

    raise KeyError(
        f"No matching config for M={M}, C={C}, mode={mode} in '{config_name}'"
    )


@functools.lru_cache(maxsize=1024 if USE_LRU_CACHE else 0)
def get_mhc_post_config(M: int, C: int) -> dict:
    """Pick the mhc_post config for ``(M, C)`` from the arch's ``mhc_post``
    ``DEFAULT.json``.

    Picks the largest ``C_<value> <= C``, else ``"default"``.
    """
    dev = arch_info.get_arch()
    cfg = load_config_json(f"{_mhc_config_dir(dev, 'MHC_POST')}/DEFAULT.json")

    c_thresholds = sorted(
        int(k[2:]) for k in cfg if k.startswith("C_") and k[2:].isdigit()
    )
    for c_threshold in reversed(c_thresholds):
        if C >= c_threshold:
            return dict(cfg[f"C_{c_threshold}"])

    if "default" in cfg:
        return dict(cfg["default"])

    raise KeyError(f"No matching config for M={M}, C={C} in 'MHC_POST'")


def hip_post_dispatch_block(C: int, arch_id: str) -> int | None:
    """Return the ``residual_block`` ``aiter.mhc_post`` selects for this C.

    Mirrors ``MHC_POST_KERNEL_DISPATCH`` in
    ``csrc/kernels/mhc_kernels.cu``:

        non-gfx942 + C % 1024 == 0 -> 1024
        C % 512 == 0               -> 512
        C % 256 == 0               -> 256
        else                       -> None  (unsupported, caller should skip)

    The HIP kernel additionally enforces ``C >= 2 * residual_block`` via
    ``TORCH_CHECK``, so callers should reject shapes where
    ``C < 2 * hip_post_dispatch_block(C, arch_id)``.
    """
    if arch_id != "gfx942" and C % 1024 == 0:
        return 1024
    if C % 512 == 0:
        return 512
    if C % 256 == 0:
        return 256
    return None


# =============================================================================
# MOE
# =============================================================================

M_THRESHOLD_SMALL = 256
M_THRESHOLD_MEDIUM = 1024


def get_config_dtype_str(
    dtype: torch.dtype,
    use_int8_w8a16: bool | None = False,
    use_int8_w8a8: bool | None = False,
    use_fp8_w8a8: bool | None = False,
    use_int4_w4a16: bool | None = False,
    use_mxfp4: bool | None = False,
):
    if use_fp8_w8a8:
        return "FP8_W8A8"
    elif use_int8_w8a16:
        return "INT8_W8A16"
    elif use_int8_w8a8:
        return "INT8_W8A8"
    elif use_int4_w4a16:
        return "INT4_W4A16"
    elif use_mxfp4:
        return "MX_FP4"
    elif dtype == torch.float:
        # avoiding cases where kernel fails when float32 MoE
        # use fp16/bfloat16 configs
        return "float32"
    return None


@functools.lru_cache(maxsize=1024 if USE_LRU_CACHE else 0)
def get_moe_configs(dtype: str | None) -> dict[int, Any] | None:
    """
    Return optimized configurations for the fused MoE kernel.

    The return value will be a dictionary that maps an irregular grid of
    batch sizes to configurations of the fused_moe kernel. To evaluate the
    kernel on a given batch size bs, the closest batch size in the grid should
    be picked and the associated configuration chosen to invoke the kernel.
    """
    # First look up if an optimized configuration is available in the configs
    # directory
    dtype_str = "DEFAULT" if dtype is None else dtype
    dev = arch_info.get_arch()
    # Nested layout only: the default file is DEFAULT.json under
    # <arch>/triton/moe/<d_type>/.
    cfg_dir = resolve_config_dir("moe", f"MOE-{dtype_str}", backend="triton")
    configs = load_config_json(f"{cfg_dir}/DEFAULT.json", required=False)
    if configs is not None:
        return configs

    # If no optimized configuration is available, we will use the default
    # configuration
    warnings.warn(
        f"No MoE configuration found for device '{dev}' with dtype '{dtype_str}'. Using default configuration."
    )
    return None


def get_optimal_moe_config(
    dtype: torch.dtype,
    use_int8_w8a16: bool | None = False,
    use_int8_w8a8: bool | None = False,
    use_fp8_w8a8: bool | None = False,
    use_int4_w4a16: bool | None = False,
    use_mxfp4: bool | None = False,
    M: int = 1,
):
    dtype_str = get_config_dtype_str(
        dtype, use_int8_w8a16, use_int8_w8a8, use_fp8_w8a8, use_int4_w4a16, use_mxfp4
    )
    # print(f"dtype_str={dtype_str}")
    configs = get_moe_configs(dtype_str)
    if configs:
        if M < M_THRESHOLD_SMALL:
            config = configs["small_M"]
        elif M < M_THRESHOLD_MEDIUM:
            config = configs["medium_M"]
        else:
            config = configs["large_M"]
    else:
        # default config
        config = {
            "BLOCK_SIZE_M": 256,
            "BLOCK_SIZE_N": 256,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 8,
            "num_warps": 8,
            "num_stages": 2,
            "waves_per_eu": 0,
            "matrix_instr_nonkdim": 16,
            "kpack": 1,
        }

    # print(f"config={config}")
    return config


def get_optimal_moe_config_func(
    dtype: torch.dtype,
    use_int8_w8a16: bool | None = False,
    use_int8_w8a8: bool | None = False,
    use_fp8_w8a8: bool | None = False,
    use_int4_w4a16: bool | None = False,
    use_mxfp4: bool | None = False,
):
    return functools.partial(
        get_optimal_moe_config,
        dtype,
        use_int8_w8a16,
        use_int8_w8a8,
        use_fp8_w8a8,
        use_int4_w4a16,
        use_mxfp4,
    )


# =============================================================================
# Tuned kernel entries
# =============================================================================
#
# Per-arch tuned tiles for kernels that carry a Python autotune search space.
#
# The GEMM and MOE families resolve a tuned entry per shape at launch time.
# These kernels need less: their search is opt-in, so all that has to be
# decided is the one config registered when it is off. Keeping that in a
# config file rather than in Python means pinning a tile for a new device is
# a file and not a branch.


@functools.lru_cache(maxsize=1024 if USE_LRU_CACHE else 0)
def _get_tuned_kernel_entry(
    op: str, config_name: str, kernel_name: str, backend: str
) -> tuple[str, dict | None]:
    """Internal cached lookup returning ``(config path, entry or None)``.

    Do NOT use this directly — the entry is the shared cached object, so
    ``get_tuned_kernel_config()`` copies it before handing it out.
    """
    arch = arch_info.get_arch()
    # Nested layout of configs/CLAUDE.md: <arch>/<backend>/<op>/<d_type>/DEFAULT.json,
    # <d_type> being the config name lowercased with dashes folded to underscores.
    dtype_dir = _dtype_dir(config_name)
    config_path = (
        f"{AITER_TRITON_CONFIGS_PATH}/{arch}/{backend}/{op}/{dtype_dir}/DEFAULT.json"
    )
    published = load_config_json(config_path, required=False) or {}
    return config_path, published.get(kernel_name)


def get_tuned_kernel_config(
    op: str,
    config_name: str,
    kernel_name: str,
    fallback: triton.Config,
    backend: str = "triton",
) -> triton.Config:
    """The tile pinned for this device, or ``fallback`` where none is published.

    What fits is not portable: the same tile can compile to 16KB of LDS on one
    arch and to more than the 64KB another one has. A device nobody has measured
    therefore gets the fallback, which has to be launchable anywhere rather than
    fastest somewhere, and stays on it until a measured entry is published.

    Args:
        op: Op family directory, e.g. ``"attention"``.
        config_name: Config family, e.g. ``"CHUNK_DELTA_ATTN"``.
        kernel_name: Key of the kernel's entry within the config file.
        fallback: Config to register when this device has no published entry.
        backend: ``"triton"`` or ``"gluon"``.
    """
    try:
        config_path, entry = _get_tuned_kernel_entry(
            op, config_name, kernel_name, backend
        )
    except BaseException as error:  # noqa: BLE001 -- no accelerator/unreadable file
        logger.warning(
            f"Unable to load tuned Triton config '{config_name}' for "
            f"kernel '{kernel_name}'; using fallback {fallback}: {error}"
        )
        return fallback
    if not entry:
        logger.warning(
            f"No tuned Triton config for kernel '{kernel_name}' in "
            f"'{config_path}'; using fallback {fallback}"
        )
        return fallback
    entry = dict(entry)
    num_warps = entry.pop("num_warps", fallback.num_warps)
    num_stages = entry.pop("num_stages", fallback.num_stages)
    return triton.Config(entry, num_warps=num_warps, num_stages=num_stages)
