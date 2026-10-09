# Kimi-K3-2x (hypothetical) kernel measurements on one NVIDIA B300

Single-GPU kernel latency tables for a **hypothetical, scaled-up Kimi-K3 variant**, collected with the
`aisimulate` collector code on one B300 (`device = "NVIDIA B300 SXM6 AC"`, SM 10.3), 2026-10-08 → 2026-10-09.
Rows follow the repo's own perf-table schema (same columns, same order, same dtypes as the existing
`b300_sxm` parquet files) so they can be loaded next to repo data. **They are not a published checkpoint.**

> Read sections 4 (what is missing) and 5 (measurement conditions) before building on these numbers.

## 1. Layout

```
data/b300_sxm/<family>/sglang/<version>/<table>.parquet   <- the data, in the repo's systems/data layout
data/b300_sxm/<family>/sglang/<version>/collection_meta.yaml
csv/<table>.csv            same rows as the parquet, plain CSV
gemm_nk_pairs.csv          the 40 GEMM (n,k) pairs collected and which model op each belongs to
failures/                  every failed / missing case, classified (see section 4)
verification/              verify_report.txt (exact-key asserts + coverage), overlap_vs_shipped_b300.txt
env/                       package freeze, clock samples + lock attempt, raw per-case ledgers, gzipped raw logs, plans
scripts/                   k3x_collect.py (driver), finalize.py, verify_bundle.py, kda_supp.py, overlap_check.py, chain.sh
MANIFEST.json              machine-readable summary (versions, row counts, per-op planned/ok/failed, clock summary, sha256)
```

Re-check anywhere with `python scripts/verify_bundle.py <bundle_dir>` (needs only pandas + pyarrow).

| table (parquet) | runtime | rows | what |
|---|---|---|---|
| `moe/.../moe_perf` | sglang 0.5.14 | 3712 | routed-expert MoE, 2 quant modes x 2 routing distributions |
| `kda/.../kda_perf` | **sglang 0.5.17** | 940 | KDA linear attention (context / generation / verify) |
| `mla/.../context_mla_perf` | sglang 0.5.14 | 880 | MLA prefill, 192 heads sharded by TP |
| `mla/.../generation_mla_perf` | sglang 0.5.14 | 844 | MLA decode (**TP8 absent, TP2 partial**, see 4) |
| `mla_bmm/.../mla_bmm_perf` | sglang 0.5.14 | 424 | MLA helper BMMs (`mla_gen_pre`, `mla_gen_post`) |
| `attention/.../context_attention_perf` | sglang 0.5.14 | 816 | GQA fallback attention profile, prefill |
| `attention/.../generation_attention_perf` | sglang 0.5.14 | 1072 | GQA fallback attention profile, decode |
| `gemm/.../gemm_perf` | sglang 0.5.14 | 11691 | dense GEMMs at the variant's exact (n,k) pairs |

## 2. The variant (only these differ from real Kimi-K3; layer counts, 69 KDA + 24 MLA split, head dims unchanged)

| | K3 | K3-2x |
|---|---|---|
| hidden_size | 7168 | **14336** |
| MLA heads / KDA heads | 96 / 96 | **192 / 192** |
| routed experts / top-k | 896 / 16 | **1792 / 32** |
| MoE latent width (`hidden_size` in `moe_perf`) | 3584 | **7168** |
| moe_intermediate_size | 3072 | **6144** |
| dense (layer 0) intermediate | 33792 | **67584** |

Unchanged: qk_nope 128, qk_rope 64, v 128, KDA head_dim 128, short_conv 4, q_lora 1536, kv_lora 512, vocab 163840.

## 3. How to read each table (semantics that matter for new experiments)

All `latency` columns are **milliseconds per kernel call**, B300, 1 GPU. TP/EP are **emulated on one GPU** (the kernel sees one rank's share).

**moe_perf** (`sglang_flashinfer_trtllm_moe`)
- `hidden_size = 7168` is the **latent** width the MoE kernel sees (not the model hidden 14336); `inter_size = 6144`, `topk = 32`, `num_experts = 1792`.
- Grid: `moe_tp_size` 1,2,4,8 x `moe_ep_size` 16,32,64,128,256,448,896,1792 x `moe_dtype` {`w4a16_mxfp4`, `w4a8_mxfp4_mxfp8`} x `distribution` {`balanced`, `power_law_1.01`} x 29 token counts (1 … 28672). Complete: 3712/3712, 0 failures.
- **`num_tokens` is the GLOBAL token count of the EP group (all-gather style).** The kernel receives all `num_tokens` rows, routes over all 1792 global experts, and only the pairs landing on rank 0's `1792/moe_ep` experts do expert GEMM work. It is **not** an all-to-all per-rank view (~topk*T/EP rows). Dispatch / all-gather / combine communication is **not** included. If your new experiment models all-to-all (e.g. DeepEP-style), these rows overstate input-side work at high EP; look up with the global T for all-gather deployments.
- Routing knobs copied from the real K3 plan: sigmoid scoring, DeepSeekV3 routing, renormalize, correction bias, 1 expert group, topk_group 1, routed_scaling_factor 1.0. Backend `flashinfer_mxfp4`.
- Each case averages the kernel over 5 independent routing samples. **Provenance note:** the first 819 cases (collection order) drew fresh random routing per case. After a driver change (approved) the remaining 2893 cases **shared pools of 5 draws per (tokens, experts, topk, EP, alpha)** across `moe_tp_size` and `moe_dtype` (the power-law generator is CPU-bound Python and dominated run time). Equivalent distribution, but rows sharing a key are correlated. Balanced routing is unaffected.

**kda_perf** (`phase` = context | generation | verify; `d_model=14336, d_conv=4, head_k/v_dim=128`; `num_k_heads == num_v_heads` in {192,96,48,24} = attention TP 1/2/4/8)
- context: `batch_size` x `seq_len` (tokens/request); two kernels per cell (`causal_conv1d_fn_qkv3`, `chunk_kda`) -> sum both for a layer.
- generation: `batch_size` only (`seq_len=1`); two kernels per cell (`causal_conv1d_update`, `fused_recurrent_kda_packed_decode`).
- verify (speculative target-verify): `seq_len` = **draft tokens per request** (2,4,8), `num_tokens = batch*draft`; one kernel `fused_kda_decode_mtp_dspark`.
- **`version` is 0.5.17, outside the collector's declared `__compat__` (`sglang==0.5.16`)**: stock 0.5.16 lacks the decode/verify kernels (`kda_fused_decode`, `sglang.kernels.ops.kimi_k3`); they exist from 0.5.17. The original `kimi-k3` branch is no longer on the sglang remote.

**context_mla_perf / generation_mla_perf** (`trtllm_mla`): `num_heads` is the **local (per-TP-rank) head count** = 192/`tp_size`; keys include `mla_dtype`/`kv_cache_dtype` (bfloat16, fp8), `batch_size`, `isl`, `step`. Shipped repo MLA tables only have power-of-two head counts up to 128, bf16 only.

**mla_bmm_perf**: `op_name` `mla_gen_pre` / `mla_gen_post`, `num_heads` local {192,96,48,24}, `bmm_dtype` bfloat16 (`sglang_torch_bmm`) / fp8 (`sglang_sgl_kernel_bmm_fp8`). Complete.

**context/generation_attention_perf** (`trtllm_mha`): K3's GQA-fallback profile doubled: local q-heads = kv-heads = 192/tp in {192,96,48,24}, head_dim 128, window 0, precision combos bf16/bf16 and fp8-KV/fp8-context-FMHA.

**gemm_perf**: `m` = tokens, (`n`,`k`) = output/input features, `gemm_dtype` in bfloat16 / fp8 / fp8_block / nvfp4. **Exact (n,k) pairs** derived from `aisimulate_core/sdk/models/kimi_k3.py` at the variant's dims for TP 1/2/4/8 (list in `gemm_nk_pairs.csv`: 40 pairs incl. replicated router/latent/down-scale ops), each with the repo's standard 74 `m` values. It is **not** the cross-product of new axis values: models that interpolate on an (n,k) grid will have to interpolate between these points. Dense `gate_up/down`, shared-expert and logits GEMMs are included; the DSPARK draft model is not.

## 4. What is missing or failed (nothing was hidden, skipped, retried, or synthesized)

**Largest outstanding gap - collectives.** NCCL and custom all-reduce at **group sizes 8 through 288 were not and cannot be collected on one GPU** (collectives need >= 2 ranks). The shipped tables cover NCCL for {2,4} and custom all-reduce for {2,4,8,16}. The GB300 72-vs-288 comparison differs only in `num_gpus_per_node`, so the whole comparison rides on collective cost vs group size - **this bundle alone does not make that comparison sound.** MoE rows also exclude all-to-all/all-gather cost (see 3).

Failures by cause (every failed case: `failures/failed_cases.csv`; classes: `failure_summary_by_error_class.csv`; raw ledgers in `env/raw_ledgers/`):

| op | failed / planned | cause | confirmed? |
|---|---|---|---|
| MLA generation | 604 / 1448 | flashinfer trtllm FMHA rejects the head ratio (`computeCtaAndClusterConfig`): **local heads 24 (TP8) fail 100%, 96 (TP2) ~67%**; 192 and 48 work | single error class; kernel limit, **serving-parity audit NOT done** |
| attention context | 408 / 1224 | one invalid precision combo (bf16 context-FMHA on fp8 KV cache); sglang `trtllm_mha` quantizes Q to fp8 when KV is fp8; collector raises this itself | yes - not a real serving config, **no data gap** (both real combos complete, 102 rows per head count) |
| KDA context | 74 cells in 4 cases | sglang `causal_conv1d` Triton int32 token-offset overflow (`total_tokens * conv_channels >= 2**31`); guard raised by the collector with source citation; cells missing: 28 @192 heads, 21 @96, 15 @48, 10 @24 (`kda_missing_cells.csv`) | yes (kernel guard) |
| KDA verify | 4 cells | CUDA illegal memory access at `batch*draft*heads >= 196608` (192h: (128,8),(256,4),(256,8); 96h: (256,8)). Re-run one cell per fresh process: deterministic; the one cell below the threshold ((192h,256,2)=98304) succeeded | reproducible shape-dependent crash; consistent with an int32 offset overflow (tokens*heads*128*128 > 2^31) but **cause unconfirmed** (kernel source not read) |
| GEMM nvfp4 | 148 / 11840 | `n % 128 == 0` rule of the NVFP4 block-scale layout: n=320 (`kda_bfa`) and n=2112 (`mla_downscale`) x 74 `m` values | yes (framework's own check) |
| GEMM fp8_block | 1 / 11840 | CUDA illegal memory access at m=32768, n=14336, k=67584 (m*k ~ 2.2e9 > 2^31) | **unconfirmed** |

Not collected at all: vLLM; NCCL/custom-AR collectives (above); the DSPARK draft model.

## 5. Measurement conditions (what to assume when comparing)

- Device string stamped by the collector: **`NVIDIA B300 SXM6 AC`** (what `torch.cuda.get_device_name` returns). Data is meant for the `b300_sxm` system tree; it will **not** be found by a `gb300` lookup (decision: keep the honest label).
- **GPU clocks were NOT locked.** `nvidia-smi -ac/-lgc` were refused (container user lacks permission; `env/clock_lock_attempt.log`). Application clocks were already at max (SM 2032 / mem 3996 MHz). `env/clock_samples.csv` (5 s samples): under load (util >= 50%, 219 samples) SM clock median 1897 MHz, **min 960 MHz, max 2032 MHz, peak 66 C, ~940 W median**; throttle-reason bits seen include `0x4` (SW power cap). Expect run-to-run noise, largest for sub-0.1 ms kernels. The sampler started ~2 min after the MoE run began (a first sampler attempt was lost to a file-handling mistake) and ran to the end, so it covers essentially the whole collection.
- Sanity vs the repo's shipped **real B300** data on identical keys (`verification/overlap_vs_shipped_b300.txt`): attention median ratio 1.001 / 1.006 (p10-p90 0.98-1.03), GEMM median 1.011 (p10-p90 0.93-1.13). MoE, KDA, MLA, MLA-BMM shapes are new (no overlapping keys).
- Software: sglang 0.5.14 (moe, gemm, mla, mla_bmm, attention; torch 2.11.0+cu130, flashinfer 0.6.12) and sglang 0.5.17 (kda); CUDA 13.0 nvcc via pip, driver 580.173.02. Full freeze in `env/pip_freeze.txt`. Not run in the repo's container images (no docker on the pod).
- Collector repo commit `5b718eda`. **The collector's case generator was bypassed** (approved): `scripts/k3x_collect.py` builds each op's cases from the repo's own K3 case lists with only the variant's dimensions rewritten, then calls the stock `run_*` functions unmodified (plus the MoE routing-logit cache described in 3). The rows are the same kind the standard pipeline writes, but they did **not** go through its case-ID/failure/parquet-finalization machinery; `collection_meta.yaml` here is hand-written (schema_version 2, `provenance: local`) and has **not** been validated against the repo's loader.
- No synthetic, interpolated, or relabeled rows. Duplicate keys: none.

## 6. Verification performed

`verification/verify_report.txt`: ALL ASSERTS PASSED. Exact rows exist for
`(topk=32, num_experts=1792, hidden=7168, inter=6144, moe_tp=1, moe_ep=64)` in all 4 (quant x distribution) combos (29 token rows each) and for
`(d_model=14336, num_k_heads=96, head_k_dim=128, d_conv=4, phase=generation)` (22 rows); all 3712 requested MoE keys present; nothing outside the requested shapes; all 40 GEMM pairs present.
Not done: lookup through the repo's Rust/Python perf-database loader (these tables were checked as raw parquet, exact-key only).

## 7. Reproducing / extending

`scripts/k3x_collect.py --op <moe|kda|mla_context|mla_generation|mla_bmm_gen_pre|mla_bmm_gen_post|attention_context|attention_generation|gemm> [--plan|--smoke]` (resumable per-op ledger; env needs CUDA_HOME/PATH/CPATH for the venv's nvcc + CCCL headers, see `env/pip_freeze.txt`; the pip CUDA-13.0 environment additionally needs `nvidia-cuda-cccl==13.0.*`, and `lib64`/`libcudart.so` symlinks inside `nvidia/cu13`). Edit the `VARIANT` constants and grids at the top to measure new shapes; `finalize.py` rebuilds this bundle.
