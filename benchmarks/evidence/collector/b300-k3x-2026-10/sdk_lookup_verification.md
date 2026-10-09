# SDK-loader lookup verification (2026-10-09)

Queries went through the SDK's compiled engine (`EngineHandle.evaluate_ops_json` via the internal single-op plumbing), `database_mode=SILICON`,
on two trees: **BEFORE** = the fork's unmodified `main` (`0dedbb5e`), **AFTER** = the same tree with the data import.
`source` is the engine's provenance tag. Note that `silicon` means "served from the measured tables" and also covers interpolation or
extrapolation inside them; exact-key hits are therefore checked separately below against the parquet rows.

## Exact keys (AFTER; engine value equals the parquet row, ratio 1.000000)
| Query | Parquet rows | Parquet latency (ms) | Engine (ms) |
|---|---|---|---|
| MoE E=1792 topk=32 hidden=7168 inter=6144 tp=1 ep=64 w4a16_mxfp4 balanced, 1024 tokens | 1 | 0.525256 | 0.525256 |
| MoE same shape w4a8_mxfp4_mxfp8 power_law_1.01, 1024 tokens | 1 | 0.428419 | 0.428419 |
| gb300 GenerationMLA 256 local heads, bf16 KV, batch 8, s=1024 | 1 | 0.023562 | 0.023562 |

## Before / after
| Query | BEFORE | AFTER |
|---|---|---|
| MoE exact K3-2x keys (2 quants x 2 distributions), 1000 tokens (between grid points), tp8/ep1792 28672 tokens | `PerfDataNotAvailableError` (MoeKey missing) | `silicon` |
| MLA BMM 192 heads bf16 pre, 64 tokens (b300_sxm) | `PerfDataNotAvailableError` | `silicon` |
| KDA K3-2x (d_model 14336, 96 heads) generation / context (b300_sxm, sglang 0.5.17) | `sol` (analytical estimate) | `silicon` |
| KDA real K3 (d_model 7168, 12 heads), same version | `silicon`, 0.006656 / 0.129011 | identical values (shipped 0.5.16 rows still inherited) |
| b300_sxm GenerationMLA 192 heads (bf16 / fp8 KV), ContextMLA 192 fp8/fp8 | `silicon` (extrapolated) 22.57 / 19.30 / 433.6 us | `silicon` (measured rows) 23.04 / 17.30 / 429.1 us |
| gb300 GenerationMLA 256 heads bf16 / fp8 KV (b=8, s=1024) | `silicon` (extrapolated) 26.18 / 24.49 us | `silicon` (measured) 23.56 / 19.02 us |
| gb300 ContextMLA 256 heads fp8-KV / bf16-KV (b=4, s=1024) | `silicon` (extrapolated) 500.8 / 438.7 us | `silicon` (measured) 537.7 / 445.3 us |
| gb300 GenerationMLA 192 heads bf16 (interpolation target) | 22.90 us | 22.23 us (b300_sxm measured 192-head row: 23.04 us) |
| gb300 ContextMLA 192 heads bf16 | 330.2 us | 333.4 us |
| gb300 GenerationMLA 128 heads (pre-existing control) | 20.9008 us | 20.9008 us (unchanged) |
| ContextAttention 192/192 heads bf16 (b300_sxm) | `mixed`, 385.4 us | `mixed`, 379.8 us |

The extrapolated 256-head values before the import were 11% (bf16 KV) and 29% (fp8 KV) too high for generation, which is the gap the
256-head measurements close for interpolating 192 heads.

## Version routing (b300_sxm, sglang)
Queryable versions are `current = 0.5.14` and `next = 0.5.17`; `0.5.16` is "not a queryable version". The K3-2x KDA rows live in the new
`kda/sglang/0.5.17` directory. Querying KDA through the 0.5.14 database still returns `sol` (KDA data is not under 0.5.14), so KDA lookups
need the 0.5.17 ("next") version.
