# Triton kernel configs — rules for automated edits

Scope: **every tuned config JSON under `aiter/ops/triton/configs/`.** Read this
before adding, moving, renaming, or tuning any of them, and before touching
`utils/config_utils.py` — the single module that owns the path constants, the
JSON loader, the directory probe, and every per-op loader built on them.

The tree is in one layout, `<arch>/<backend>/<op>/<d_type>/`. There is no
legacy flat layout and no fallback to one: a config file that is not in the
nested layout does not resolve.

Two non-negotiables:

1. **Tuning values live in JSON, never in Python.** No `setdefault`, no inline
   dict literals, no arch-conditional constants, no hardcoded fallback configs.
   If a value is missing, fix the JSON.
2. **A config family is one directory.** Everything for one `<arch>` ×
   `<backend>` × `<CONFIG_NAME>` — default and specialized alike — lives in
   that family's `<d_type>/` directory, and the directory is only visible if it
   contains a `DEFAULT.json` (see §2).

`GEMM-AFP4WFP4` (gfx950 triton, gfx950/gfx1250 gluon) is the worked reference —
copy its shape when in doubt.

---

## 1. Layout

```
configs/<arch>/<backend>/<op>/<d_type>/DEFAULT.json
configs/<arch>/<backend>/<op>/<d_type>/<CONFIG_NAME>-<suffix>.json
```

| Segment     | Values                                                    |
| ----------- | --------------------------------------------------------- |
| `<arch>`    | `gfx942`, `gfx950`, `gfx1250`, `gfx1100`, `gfx1151`, `gfx1200`, `gfx1201` |
| `<backend>` | `triton` or `gluon`                                        |
| `<op>`      | `gemm`, `conv`, `attention`, `gmm`, `mhc`, `moe`, `fusions` |
| `<d_type>`  | `config_name.lower().replace("-", "_")` — `GEMM-AFP4WFP4` → `gemm_afp4wfp4`. The transform lives in `config_utils._dtype_dir()` |
| filename    | **no arch prefix** — the arch is the directory. The default is literally `DEFAULT.json`; specialized files keep the `<CONFIG_NAME>-` stem |

```
configs/gfx950/triton/gemm/gemm_afp4wfp4/DEFAULT.json
configs/gfx950/triton/gemm/gemm_afp4wfp4/GEMM-AFP4WFP4-N=8192-K=8192.json
configs/gfx950/gluon/gemm/gemm_afp4wfp4/DEFAULT.json
configs/gfx1250/gluon/gemm/gemm_afp4wfp4/DEFAULT.json
configs/gfx1201/triton/conv/conv_3x3_nhwc/DEFAULT.json
configs/gfx942/triton/mhc/mhc_fused_sinkhorn/MHC_FUSED_SINKHORN-C=4096.json
configs/gfx950/triton/moe/moe_fp8_w8a8/DEFAULT.json
```

Not every arch carries every op — the RDNA parts `gfx1100` and `gfx1200` ship
conv configs only, and a `gluon/` directory exists only where a Gluon kernel is
tuned. Regenerate rather than trusting any listing:
`git ls-tree -r --name-only HEAD aiter/ops/triton/configs/`

A few `<arch>/<backend>/<op>/` directories are held open with `.gitkeep` while
no family has been tuned there yet. Keep them.

---

## 2. Resolution — `resolve_config_dir()`

Every loader picks its directory through one probe in `utils/config_utils.py`:

```python
def resolve_config_dir(op: str, config_name: str, backend: str | None = None,
                       arch: str | None = None) -> str:
    """Return the config directory of the first candidate that has a
    ``DEFAULT.json`` (the nested layout ``<arch>/<backend>/<op>/<d_type>``,
    the leaf from _dtype_dir()). With ``backend=None`` the probe tries
    ``triton`` then ``gluon``. Falls back to the last candidate so the
    missing-file assertion names the nested path. ``arch`` overrides the
    running architecture, for loaders that retry under another arch when the
    running one has no tuned configs."""
```

`<arch>` below is the running arch unless the caller passes `arch=` — MHC is
the one loader that does, retrying under gfx942 for arches with no tuned MHC
files.

**`backend=None`**:

1. `configs/<arch>/triton/<op>/<d_type>/DEFAULT.json`
2. `configs/<arch>/gluon/<op>/<d_type>/DEFAULT.json`

**`backend="triton"|"gluon"`**:

1. `configs/<arch>/<backend>/<op>/<d_type>/DEFAULT.json`

Specialized files are then read from whichever directory won. If no candidate
has a `DEFAULT.json` the last one is returned anyway, so the missing-default
assertion names the nested path the file should have been at.

Consequences to keep in mind:

- **A directory is chosen as a unit.** The unit is the family's `<d_type>/`
  directory. A file for that family kept anywhere else is dead weight the probe
  will never see; move a family wholesale or not at all.
- **A `<d_type>/` directory with specialized files but no `DEFAULT.json` is
  invisible.** The probe keys only on `DEFAULT.json`, so the whole directory is
  ignored — and with `backend=None` the probe simply moves on to the other
  backend.
- **`backend=None` prefers `triton` over `gluon`.** On an arch that has only a
  gluon default for the family, lookup falls through to gluon. Adding a triton
  `DEFAULT.json` for that family later would change which file that arch
  resolves to — verify that is intended.
- Results are cached twice: `functools.lru_cache` on the full argument
  tuple, plus a per-path cache of parsed JSON
  (`utils/config_utils.py::load_config_json`) that also caches negative results
  (missing files). Adding a config file at runtime therefore has no effect;
  restart the process (tooling may call `load_config_json.cache_clear()`
  instead).

Direct-path loaders bypass the probe. Grep for
`f"{AITER_TRITON_CONFIGS_PATH}/..."` before moving anything —
`fusions/fused_clamp_act_mul.py` builds its
`<arch>/<backend>/fusions/fused_clamp_act_mul/` paths by hand (via
`load_config_json`) so it can fall back to the gfx950 file on an arch with no
tuned copy, and must be edited when its configs move. Everything else, Gluon
GEMM included, goes through `resolve_config_dir()` and needs no change.

---

## 3. GEMM config file contents

Required top-level shape:

```json
{
  "M_LEQ_64":   { "...": "..." },
  "M_GEQ_4096": { "...": "..." },
  "any":        { "...": "..." }
}
```

- `M_LEQ_x` is searched ascending over `STANDARD_M_BOUNDS =
  (1, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)`, then
  `M_GEQ_x` descending, then `any`. A caller may override with
  `bounds=(...)`, which must be strictly increasing positive ints.
- `any` must exist unless every reachable `M` is covered by an explicit bound.
- The deprecated `{"large": ..., "small": ...}` shape must not be introduced.
- A `KeyError` at lookup time means no bound matched — usually a missing `any`.

Each `M_*` entry carries at minimum:

```
BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M,
num_warps, num_stages, waves_per_eu, matrix_instr_nonkdim,
cache_modifier, NUM_KSPLIT
```

`add_default_gemm_config_params()` backfills `NUM_KSPLIT=1` and
`cache_modifier=None` as a last resort, and `compute_splitk_params()` derives
`SPLITK_BLOCK_SIZE` and may clamp `BLOCK_SIZE_K` / `NUM_KSPLIT`. Neither is a
license to omit keys.

`get_gemm_config()` returns `(config, is_tuned)`. `is_tuned` is `True` only when
a specialized (N/K-, B-, or `specialized_filename`-keyed) file was hit, `False`
for the default file or `any`. Do not discard it.

### The JSON is the only place tuning values live

A `_get_config()` should do nothing but call `get_gemm_config()` and return:

```python
def _get_config(M: int, N: int, K: int):
    return get_gemm_config("GEMM-AFP4WFP4", M, N, K)
```

`_triton_kernels/gemm/basic/gemm_afp4wfp4.py` carried a block of `setdefault`
calls and it was deleted — it masked incomplete config files with values nobody
had tuned, and made the effective config un-inspectable from the JSON.

---

## 4. GEMM naming

| Kind              | File inside the family's `<d_type>/` directory  |
| ----------------- | ----------------------------------------------- |
| Default           | `gemm_a16w16/DEFAULT.json`                       |
| N/K specialized   | `gemm_a16w16/GEMM-A16W16-N=256-K=7168.json`      |
| Batched (B, N, K) | `batched_gemm_a16w16/BATCHED_GEMM-A16W16-B=4-N=1024-K=4096.json` |
| Custom suffix     | `fused_gemm_afp4wfp4_a16w16/FUSED-GEMM-AFP4WFP4-A16W16-N4=512-N16=256-K=7168.json` |

The `<d_type>` directory name is `config_name.lower().replace("-", "_")`.
Dashes, underscores, and case all fold together, so new config names must stay
distinct under that transform — `GEMM-FOO-BAR` and `GEMM-FOO_BAR` would collide.

Config-name patterns: `GEMM-A{x}W{y}`, `BATCHED_GEMM-A{x}W{y}`,
`FUSED-GEMM-{op}`, `FF-A{x}W{y}-fused`; variant suffixes
`_PRESHUFFLED`, `_BLOCKSCALE`.

**`K` in AFP4WFP4 filenames is the logical K, i.e. `2 * K_bytes`.** The kernel
does `K = 2 * K` before calling `get_gemm_config`. Tuning output that names
files by the packed byte width will never be found.

---

## 5. MOE configs

MOE does **not** go through `get_gemm_config()`. There is no `is_tuned` signal
and no shared schema: four independent loaders each call
`resolve_config_dir("moe", ...)` for their own family and then interpret the
file their own way.

| Loader | Config family → directory | Schema |
| ------ | ------------------------- | ------ |
| `utils/config_utils.py::get_moe_configs` | `MOE-<dtype_str>` → `moe/moe_<dtype_str>/` | `small_M` / `medium_M` / `large_M` |
| `moe/moe_op_gemm_a8w4.py::_get_a8w4_dispatch` | `A8W4` → `moe/a8w4/` | `bm<block_m>_n<N>_k<K>`; the gluon path keys `bm<block_m>_n<N>_k<K>_<bucket>` then `bm<block_m>_any` |
| `moe/moe_op_gemm_a4w4.py::_get_a4w4_dispatch` | `A4W4` → `moe/a4w4/` | `bm<block_m>_n<N>_k<K>_<bucket>`, then `bm<block_m>_any` |
| `_triton_kernels/moe/moe_routing_sigmoid_top1_fused.py::_get_config` | `MOE_ROUTING_SIGMOID_TOPK1` → `moe/moe_routing_sigmoid_topk1/` | `N16` / `N128` → `small` / `medium` / `large` / `xlarge` |

All four read `DEFAULT.json` out of the resolved directory. Only the routing
loader requires it to exist; the other three treat a missing file as "this arch
is untuned" and fall back in Python — see the debt below.

`<dtype_str>` comes from `get_config_dtype_str()`: `DEFAULT`, `FP8_W8A8`,
`INT8_W8A16`, `INT8_W8A8`, `INT4_W4A16`, `MX_FP4`, so the directories are
`moe_default`, `moe_fp8_w8a8`, … under `<arch>/triton/moe/`.

`small_M` / `medium_M` / `large_M` split on `M_THRESHOLD_SMALL = 256` and
`M_THRESHOLD_MEDIUM = 1024`, both module constants in `utils/config_utils.py`.
This is **not** the GEMM `M_LEQ_x` / `M_GEQ_y` scheme — do not mix them.

`<bucket>` is a *third* M scheme — `m2bucket()`, defined identically in
`moe_op_gemm_a4w4.py` and `moe_op_gemm_a8w4.py`, splitting on 8 / 32 / 128 /
256 / 512 into `tiny` / `small` / `medium` / `medium2` / `large` / `xlarge`. It
keys the **gluon paths** of both families (a4w4's triton path still computes its
config in Python). Lookup is two tiers: `bm<block_m>_n<N>_k<K>_<bucket>`, then
`bm<block_m>_any`. Since a missing bucket falls all the way through to `_any`
and loses the shape's tuning, a tuned shape must supply **all six** buckets,
even where the values repeat.

Because a8w4's two paths key the same file differently, an `a8w4/DEFAULT.json`
is written for whichever path that arch actually runs — gfx950's is `bm_n_k`
keyed, gfx1250's is bucket-keyed with the `bm<block_m>_any` tier. Match the
arch's existing file rather than mixing both key shapes into one.

### MOE is the main offender for tuning values in Python

Known debt, not a pattern to copy. Fix these as you touch them; do not add more:

- `get_optimal_moe_config()` returns a hardcoded dict (`BLOCK_SIZE_M: 256`,
  `BLOCK_SIZE_N: 256`, …) when no config file exists, behind a
  `warnings.warn`. A missing config silently runs untuned values.
- `moe_op_gemm_a8w4.py` has a three-tier Python fallback: exact
  `bm_n_k` hit → any-`block_m` proxy with matching `(N, K)` → a gfx942-gated
  shape heuristic → a conservative default. Only the first tier reads tuned
  numbers from JSON.

Retiring them is blocked on coverage, not on the loader: only gfx942 and gfx950
ship the `MOE-<dtype_str>` families today, so every other arch lands on the
hardcoded dict. Once each supported arch has a `moe_<dtype_str>/DEFAULT.json`,
that dict and its `warnings.warn` come out.

---

## 6. Adding and moving configs

### Adding a tuned config to an existing family

Put it in that family's `<d_type>/` directory: `DEFAULT.json` for the default,
the config-name stem for a specialized file
(`GEMM-A16W16-N=256-K=7168.json`), never an arch prefix. Nothing else is
needed — the probe finds the directory by its `DEFAULT.json` and reads
specialized files from it.

If the family has no directory on this arch yet, create it *with* its
`DEFAULT.json` in the same commit. A `<d_type>/` directory holding only
specialized files is invisible (§2). The directory needs no `.gitkeep` — it is
created populated.

### Adding a new config family

1. **Pick the name.** `<d_type>` is `config_name.lower().replace("-", "_")`, so
   dashes, underscores, and case all fold together: the new name must stay
   distinct from every existing one under that transform.
2. **Pick `<op>`.** Reuse an existing op directory (`gemm`, `conv`,
   `attention`, `gmm`, `mhc`, `moe`, `fusions`) if the family belongs to one. A
   genuinely new op means a new directory under each arch/backend that ships
   the family, and a loader that passes that `op` string through.
3. **Load it through `utils/config_utils.py`.** Reuse a loader if the schema
   fits — `get_gemm_config()` for M-bound GEMM files,
   `get_tuned_kernel_config()` for kernels that only need one pinned tile per
   device. Otherwise add the loader to that module and have it call
   `resolve_config_dir()`. Do not hand-build a config path, and do not put a
   config loader anywhere else.
4. **Ship a `DEFAULT.json` for every arch you support**, and pull any tuning
   values the kernel still holds in Python into it. A family must be fully
   described by its config files.
5. **Update the docs.** `aiter/ops/triton/README.md` ("Tuned configs", "Config
   naming") and `aiter/ops/triton/utils/_triton/tunning/README.md` (the copy
   step under "Verify performance") describe the layout — keep them current.
6. **Verify on the target arch:** the config resolves, `is_tuned` is `True` for
   a shape that has a specialized file (GEMM), and numerics are unchanged.

### Renaming or moving config files

**Move with `git mv`** so the change reviews as a rename, and **do not edit
contents in the same commit** — keep renames at 100% similarity; content
changes go in a follow-up commit. Before moving anything, grep for the config
name and for `AITER_TRITON_CONFIGS_PATH` in `aiter/ops/triton/`: hand-built
paths (§2) must be rewritten in the same commit, while `resolve_config_dir()`
callers need no change. Renaming a family means renaming its `<d_type>/`
directory, and every file in it moves along.

### Do not

- Rename or delete `.gitkeep` placeholder directories.
- Put an arch prefix on a file inside `<arch>/...`, or name a default anything
  other than `DEFAULT.json`.
- Leave a `<d_type>/` directory without a `DEFAULT.json`.
- Put tuning values in `.py` files — no `setdefault`, no inline dicts, no
  arch-conditional constants, no hardcoded fallback configs.
- Mix the GEMM `M_LEQ_x`/`M_GEQ_y` scheme with the MOE
  `small_M`/`medium_M`/`large_M` scheme.

### Not tuning configs

Two AOT code paths build directories under this tree at runtime that are **not
checked in and out of scope**:

- `configs/gemm/aot/<kernel>_M=…-N=…-K=…` — `gemm/fused/fused_gemm_afp4wfp4_a16w16.py`,
  `gemm/fused/fused_gemm_afp4wfp4_mul_add.py`
- `configs/paged_mqa_logits/aot/<kernel>` — `attention/pa_mqa_logits.py`

Both are guarded by `use_aot and os.path.exists(...)` and hold compiled-kernel
metadata, not tuning parameters. Their `configs/<name>/` parents are not
`<arch>` directories and are not part of the layout in §1. Do not create,
migrate, or document them as config directories.
