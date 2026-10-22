# K3-2x collection runbook — known failures and how to avoid them

Written after the 2026-10 B300 collection and the simulation work that consumed
it. Every item here is a mistake that was actually hit, diagnosed and fixed in
this repo — not hypothetical advice. Read this before collecting more K3-2x
data or running a K3-2x sweep.

The model under study ("K3-2x") is a hypothetical scale-up of Kimi-K3:
hidden 14336, 192 MLA/KDA heads, 1792 routed experts top-32, latent 7168,
MoE inter 6144, dense inter 67584. Head dims, layer count and the 69-KDA /
24-MLA split are unchanged from real Kimi-K3.

---

## 1. Environment / build

**`uv` is not on PATH and the network only reaches Git + Artifactory.**
```bash
export PATH="$HOME/.local/bin:$PATH"
export UV_DEFAULT_INDEX="https://artifactory.marvell.com/artifactory/api/pypi/odsp-sw-pypi-remote/simple"
export UV_INDEX="https://artifactory.marvell.com/artifactory/api/pypi/odsp-sw-pypi-local/simple"
export SSL_CERT_FILE="/home/jding/.certs/ca-bundle-marvell.pem"
uv sync --project python/aisimulate --extra dev --reinstall-package aisimulate
```
`uv` does **not** read `pip.conf`; the index vars must be set explicitly.

**The compiled extension goes stale silently and breaks in a confusing way.**
`cargo build` does NOT update
`python/aisimulate/src/aisimulate/_runtime.abi3.so`. After ANY merge or any
change under `crates/`, rebuild with the `uv sync` line above.

Observed symptom when stale (real example, after an upstream merge removed a
kwarg from `compile_engine`):
```
ValueError: unsupported model for Rust core estimator: compile_engine:
TypeError: compile_engine() got an unexpected keyword argument 'decode_workload_distribution'
```
That error is NOT a config problem — it means the `.so` predates the Python
source. Verify with `ls -la` on the `.so` vs `git log -1 --format=%ci`, then
rebuild. A stale `.so` can also fail *silently* (wrong numbers, no error), so
rebuild on principle rather than waiting for a crash.

**`AIC_ALLOW_UNLISTED_VERSIONS=1`** must be set to query raw version
coordinates outside the `current`/`previous`/`next` slots.

---

## 2. SDK API traps (all of these raised in practice)

| Trap | Symptom | Fix |
|---|---|---|
| `backend_version="next"` on a **custom override system** | `ValueError: no queryable versions defined for sglang on b300_sxm_nvl72; cannot resolve alias 'next'` | Pass the **literal** version string (`"0.5.17"`). The slot-alias resolution inside the Rust→Python engine-spec callback does not see the override's `systems_paths`, even though `get_version_slots()` resolves fine when called directly. Backward-fill still reaches the 0.5.14 tables. |
| `CommQuantMode.bfloat16` | `InvalidEngineConfigurationError: 'bfloat16'` | `CommQuantMode` has only `half`/`int8`/`fp8`. Use `"half"`. |
| Passing enum objects to `ForwardPassPerfModelConfig` | type error / silent reject | Its quant fields are typed `str \| None` — pass `"bfloat16"`, `"w4a16_mxfp4"`. `Task(...)` is the opposite: it wants real `common.*QuantMode` enum members. |
| Topology mismatch | `topology requires tp * attention_dp * cp_size == moe_tp_size * moe_ep_size` | Hard assert in `KimiK3Model.__init__`. Also `kda_num_heads % tp_size == 0` and `num_experts >= moe_ep`. |
| `get_version_slots(..., systems_paths=[dir, "default"])` returns `None` | silently no slots | That entry point does not normalize the literal string `"default"` to the packaged path the way `set_systems_paths()` does. Call `set_systems_paths([OVERRIDE_DIR, "default"])` first, then call `get_version_slots(system, backend)` with no `systems_paths` argument. |
| `Task(...)` defaults every quant mode to **bfloat16** | `InsufficientMemoryError` everywhere | At bf16 the K3-2x MoE experts alone are ~44 TB. Always set `moe_quant_mode=common.MoEQuantMode.w4a16_mxfp4` explicitly (~11.3 TB) to match what was actually collected. |
| `pareto_sweep=False` vs `True` | one raises, one doesn't | `False` raises `InsufficientMemoryError` / `KVCacheCapacityError` / `PerfDataNotAvailableError` on a bad point. `True` **silently drops** infeasible points. Use `False` when probing one config, `True` for sweeps. |

---

## 3. Silent-wrong-number traps (the dangerous ones)

These do **not** raise. They return a confident latency tagged
`source="silicon"` even under `fallback_policy="deny"`.

**KDA — unscaled nearest-shard borrow.**
`StateSpaceTable::query_kda` keys on exact `d_model`; on a miss its ONLY
fallback searches *within the same `d_model`*, picking the nearest
`num_v_heads` and returning that row **unscaled**. Collected K3-2x head
counts at `d_model=14336` are exactly `{24, 48, 96, 192}` (tp 8/4/2/1).
A query at 12 or 6 heads silently returns the 24-head latency verbatim —
verified: tp=16 and tp=32 produced byte-identical KDA numbers.
If `d_model` itself has no rows at all, the op degrades to analytic SOL and
IS tagged `source="sol"` (that part is at least visible).

**MLA — `num_heads` is a continuous interpolation axis.**
`GENERATION_AXES` in `perf_database/mla.rs` treats `num_heads` as an ordinary
grid axis, so an uncollected head count is **interpolated** between
neighbours and reported as `silicon`. That is correct for most of the range
but wrong across a kernel-dispatch cliff (see §4). Nothing in the engine
models kernel-support boundaries.

**Consequence:** a wide `tp` sweep will happily return results for tp values
whose kernels cannot run. `fallback_policy="deny"` does NOT catch this —
confirmed. Filtering must be done at the candidate-list stage, informed by
§4.

---

## 4. flashinfer trtllm-gen MLA decode: the head-count rule

From `fmhaKernels.cuh:442-447,684-710` (reproduced in `k3x_b300_audit/AUDIT.md`
with bare flashinfer AND a live `sglang.launch_server` crash):

```
numHeadsPerCta = min(numHeadsQPerKv, tileSizeQ)   # must DIVIDE the head count
heads <= 32  ->  SwapsMmaAb kernel, tileSizeQ = 16
heads >  32  ->  tileSizeQ = 64 (tile-16 only for small per-CTA KV length)
```

Applying it (MLA ⇒ `num_kv_heads = 1`, so `numHeadsQPerKv = num_heads`):

| local heads | tp | tile | perCta | divides? | audit |
|---|---|---|---|---|---|
| 192 | 1 | 64 | 64 | yes | 71/71 pass |
| 96 | 2 | 64 | 64 | no | 25/71 (tile-16 only, small workloads) |
| 48 | 4 | 64 | 48 | yes | 71/71 pass |
| 32 | — | 16 | 16 | yes | 71/71 pass |
| **24** | **8** | 16 | 16 | **no** | **0/71 FAIL** |
| 16 | — | 16 | 16 | yes | 71/71 pass |
| **12** | **16** | 16 | **12** | **yes** | **never tested** |
| **6** | **32** | 16 | **6** | **yes** | **never tested** |
| **3** | **64** | 16 | **3** | **yes** | **never tested** |

**Key non-obvious consequence:** head counts *below* the tile size are
predicted to WORK, because `perCta` collapses to the head count itself, which
divides trivially. `tp=8` (24 heads) fails precisely because 24 is the one
value above 16 that is not a multiple of it. So **tp=16/32/64 are predicted
to be better supported than tp=8** — the opposite of the usual "finer sharding
is less tested" intuition.

This is a prediction from the documented mechanism, validated against every
audit-tested point. It is **not** measured below 16 heads. Preflight it
(§5) before committing GPU hours.

---

## 5. Always preflight before a long collection

The audit's probe pattern: call the kernel directly, outside sglang and
outside the collector, over the head counts and a batch×seq grid, and record
pass/fail. One short script answers "will this shape run at all" before hours
of collection. Do this for any new head count.

---

## 6. What cannot be collected on one GPU

NCCL / custom all-reduce / DeepEP all-to-all at group sizes ≥2 nodes.
Collectives need ≥2 communicating ranks. Shipped `b300_sxm` comm tables cover
`num_gpus ∈ {2,4,8}` only, and no sglang `moe_a2a_perf.parquet` exists at all —
which is why cross-node EP (`moe_ep > gpus_per_node`, `moe_comm_resolver.py:141`)
raises `PerfDataNotAvailableError` and gets dropped from sweeps. Any NVL72-vs-
NVL288 conclusion rests on this uncollected data; state that caveat explicitly
in any result.

---

## 7. MoE rows are all-gather style, no comm

`num_tokens` in `moe_perf.parquet` is the **global** token count of the EP
group; the kernel routes over all 1792 experts and only rank-0's share does
GEMM work. Dispatch / all-gather / combine cost is **not** included. If a
future experiment models all-to-all (DeepEP-style), these rows overstate
input-side work at high EP.
