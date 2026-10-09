# B300 collection for a scaled-up Kimi-K3 ("K3-2x") and 256-head MLA (2026-10-09)

Kernel measurements on one `NVIDIA B300 SXM6 AC` (SM 10.3) for a hypothetical Kimi-K3 variant ("K3-2x": hidden 14336,
192 MLA/KDA heads, 1792 experts, top-32, latent 7168, MoE inter 6144, dense inter 67584; head dims and layer counts as K3),
plus a 256-head MLA profile meant to make the shipped GB300 MLA tables interpolate to 192 heads. The same rows were imported
into `python/aisimulate/src/aisimulate_core/systems/data/` (see "Data import"); this directory keeps the full evidence.

## Contents
| Directory | What |
|---|---|
| `k3x_b300_results/` | The K3-2x collection: parquet/CSV tables, failures, verification reports, environment capture, raw ledgers and logs, plans. Start with its `README.md`. |
| `k3x_mla256_results/` | MLA at 256 heads, TP 1/2/4/8 (local heads 256/128/64/32), the merge candidate for the GB300 tables, overlap/interpolation checks. |
| `k3x_b300_audit/` | Serving-parity audit of the three crash groups, probes' raw output, and a real sglang server reproduction (`AUDIT.md`). |
| `sdk_lookup_verification.md` | Before/after queries through the SDK engine (exact-key equality with the parquet rows, provenance tags, KDA version routing). |

The driver, finalize/verify and probe scripts (`.py`/`.sh`) are **not** in this directory: the repository's header check
requires an NVIDIA copyright header on source files, which should not be applied to them without a decision. They exist in the
original bundles; the references to `scripts/` and `probes/` inside the sub-READMEs point to files that are not included here.

## Results (rows collected / planned)
| Table | Rows | Notes |
|---|---|---|
| MoE | 3712 / 3712 | sglang 0.5.14, `sglang_flashinfer_trtllm_moe`, w4a16_mxfp4 and w4a8_mxfp4_mxfp8 |
| KDA | 940 | sglang **0.5.17** (see caveats); 78 cells failed |
| MLA context / generation | 880 / 844 (of 1448) | 192 heads, TP 1/2/4/8 |
| mla_bmm | 424 / 424 | |
| Attention context / generation | 816 (of 1224) / 1072 | K3 "MLA as GQA" fallback profile at 192 heads |
| GEMM | 11691 (of 11840) | 40 (n,k) pairs from the K3-2x model definition |
| MLA 256 heads: generation / context | 1448 / 880 | all passed |

## Failures (all classified; none is a collector bug)
- **MLA generation, 604 cases:** flashinfer's trtllm-gen MLA decode kernel needs the local head count to fit its tile (16 or 64); 24 local heads
  never works and 96 only for small workloads (`fmhaKernels.cuh:444`). Reproduced with bare flashinfer (0.6.12 and 0.6.15) and with a real
  `sglang.launch_server` on a dummy 24-head model.
- **Attention context, 408 cases:** a bf16-compute prefill on an fp8 KV cache does not exist (trtllm_mha quantizes Q to FP8).
- **GEMM, 149 cases:** 148 nvfp4 (n=2112, n=320 violate the n%128 / k%64 layout); 1 fp8_block (m=32768, n=14336, k=67584: activation above 2^31
  elements, the quantizer mis-indexes it).
- **KDA, 78 cells:** 74 context cells hit the collector's int32 conv guard; 4 verify cells hit an illegal memory access from a 32-bit offset in
  `fused_kda_decode_mtp_dspark`.

## Caveats
- **GPU clocks were not locked** (permission refused); SM clocks varied 960-2032 MHz with power-cap throttling. Points under ~0.1 ms are noisy.
- **KDA ran on stock sglang 0.5.17**, not the `kimi-k3` 0.5.16 branch build the collector declares in `__compat__` (that tag does not exist upstream).
- **MoE router logits were memoized** across cases that share a routing key (TP and quant variants), so those rows reuse the same 5 draws instead of 5 new ones.
- **A custom driver bypassed the collector's case generator** (stock `run_*` measurement functions, cases derived from the repo's own K3 case list with
  the K3-2x dimensions); no case YAML was authored. No container was used (pip venvs).
- **MoE rows have no communication cost** (all-gather style emulation on one GPU). NCCL / custom all-reduce for group sizes 8-288 cannot be
  collected on one GPU and remain the largest gap for the 72-vs-288 GPU comparison.
- **Label conventions followed on import:** in the generation MLA tables of both `b300_sxm` and `gb300` the new fp8-KV rows carry
  `mla_dtype=bfloat16` like the shipped fp8-KV rows (generation lookups ignore `mla_dtype`); `b300_sxm` context rows keep `fp8` (that table's own
  convention), `gb300` context fp8-KV rows carry `bfloat16`. The as-collected files keep the true `fp8` label throughout.
- **Registry edit:** `python/aisimulate/collector/op_backend_facts.yaml` got three entries for `b300_sxm` KDA at sglang 0.5.17 (required by
  `backend_facts.py --check`; values are the ones derived from the new rows' `kernel_source` labels).
- **KDA version routing:** on `b300_sxm` the queryable sglang versions are `current = 0.5.14` and `next = 0.5.17`; K3-2x KDA lookups resolve as measured
  only on 0.5.17 (see `sdk_lookup_verification.md`).
- fp8-KV MLA context rows ran FP8 compute (sglang 0.5.14 casts prefill Q/K/V to FP8 for an FP8 KV cache). In `gb300/mla` they carry
  `mla_dtype=bfloat16` to follow that table's existing convention; the as-collected files here keep `fp8`.

## Data import
Rows were appended after the shipped rows (shipped rows unchanged); rows whose key already existed were not added.
| Table (system / sglang version) | Shipped | Appended | Dropped (key already present) |
|---|---|---|---|
| b300_sxm moe 0.5.14 | 260756 | 3712 | 0 |
| b300_sxm gemm 0.5.14 | 130906 | 10211 | 1480 |
| b300_sxm mla_bmm 0.5.14 | 1696 | 424 | 0 |
| b300_sxm mla context / generation 0.5.14 | 3080 / 5792 | 880 / 844 | 0 / 0 |
| b300_sxm attention context / generation 0.5.14 | 44167 / 49680 | 204 / 304 | 612 / 768 |
| b300_sxm kda 0.5.17 (new directory) | - | 940 | - |
| gb300 mla context / generation 0.5.14 | 3520 / 5792 | 220 / 362 | 0 / 0 |

On the overlapping keys (attention, GEMM) the new B300 latencies agree with the shipped ones (median ratio 1.00 to 1.01), which is an independent
sanity check of the collection. The new gb300 MLA rows were measured on a B300 and keep the device string `NVIDIA B300 SXM6 AC`
(mixed device strings in one table were approved). Interpolating 192 heads from the 128 and 256 rows: context MAPE 4.2%, generation 9.1%
(13.8% above 0.1 ms) when tested against measured 192-head rows.
