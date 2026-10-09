# MLA, 256 heads, TP 1/2/4/8 - B300 (to extend the shipped GB300 MLA tables and interpolate 192 heads)

Measured 2026-10-09 on one B300 (`NVIDIA B300 SXM6 AC`, SM 10.3), sglang 0.5.14, flashinfer 0.6.12, kernel `trtllm_mla`,
stock collector `run_mla` (cases built by my driver, not the case generator). **GPU clocks were not locked** (no permission).
Case grid = the repo's own MLA grid (same as the shipped tables): generation 1448 cases, context 880 cases. **All 2328 passed, 0 failures.**

## What is where
| Path | Content |
|---|---|
| `data_as_collected/{generation,context}_mla_perf.parquet` (+ `.csv`) | Every measured row, repo schema/dtypes, exactly as collected. `num_heads` = LOCAL heads = 256 / tp. |
| `merge_candidate_gb300/{generation,context}_mla_perf.parquet` | Shipped GB300 table + only the genuinely new rows (local heads 256). Ready to import. See "Merge notes". |
| `verification_report.txt`, `interpolation_check_192.txt` | Overlap vs shipped tables, scaling sanity, 192-head interpolation test. |
| `raw/`, `scripts/`, `build_and_verify.py` | Raw perf text, ledgers, logs, driver, probe, builder. |

Rows: generation 1448 (local heads 256/128/64/32 for TP 1/2/4/8, 362 each), context 880 (220 each).
Only local **256** is new relative to the shipped tables (they already hold 1..128). The 128/64/32 rows are cross-checks.

## Merge notes (read before importing)
1. **Overlap is ignored by the SDK.** The loader keys generation on `[kv_dtype][num_heads][batch][isl+step]` and context on
   `[mla_dtype][kv_dtype][num_heads][isl][batch]`, `insert_first_wins`; `tp_size` is not a key. So the merge candidate adds only
   local-256 rows (generation +362, context +220). Rows for local 128/64/32 would be shadowed by the shipped ones.
2. **mla_dtype label mismatch on context fp8-KV rows.** Shipped GB300 labels fp8-KV rows `mla_dtype=bfloat16`; the current collector writes `fp8`.
   `data_as_collected` keeps `fp8`. In `merge_candidate_gb300/context_mla_perf.parquet` the 110 fp8-KV rows were relabelled to `bfloat16` so they
   land in the same lookup branch as the shipped rows. Generation ignores `mla_dtype`, so nothing changed there. **Decision (user, 2026-10-09): keep the shipped GB300 convention** (fp8-KV context rows labelled `bfloat16`) in the merge candidate. Physically these rows ran FP8 compute (sglang 0.5.14 trtllm_mla casts prefill Q/K/V to FP8 when the KV cache is FP8); `data_as_collected` keeps the true `fp8` label.
3. **Device label.** New rows say `NVIDIA B300 SXM6 AC`, shipped ones `NVIDIA GB300`. The schema has no other provenance column, so the merged
   tables mix two device strings. **Decision (user, 2026-10-09): mixed device names are fine**; the new rows keep `NVIDIA B300 SXM6 AC`. B300 vs shipped GB300 on the overlapping rows: generation median ratio 1.001, context 1.045 (vs shipped b300_sxm: 1.017 / 0.999).
4. Nothing was written into the repo; the repo is unmodified.

## Interpolating 192 heads - what the data says (see `interpolation_check_192.txt`)
I already had 192 local heads measured at TP=1 (earlier bundle), so I tested interpolation between 128 and 256 against it.
- **Context:** linear interpolation in heads is good: MAPE 4.2% (3.8% on rows > 0.1 ms), 97.7% of 192-head rows lie between the 128 and 256 rows.
- **Generation:** weaker. MAPE 9.1% overall, 13.8% on rows > 0.1 ms, worst 35-51%; linear underestimates (median -6%). Reason: 128 local heads runs
  on a faster kernel path (2-CTA), so 128 -> 256 grows 2.0-3.3x at large workloads, while 192 behaves like 256 (192/128 = 2.06x, 256/192 = 1.27x).
  Linear-in-heads is better than geometric. If generation accuracy at 192 matters, adding the measured 192 TP=1 rows (in `../k3x_b300_results/`) would remove the gap.
- Small workloads (< 0.1 ms) are noisy with unlocked clocks; ratios there can go below 1 (min 0.65).
