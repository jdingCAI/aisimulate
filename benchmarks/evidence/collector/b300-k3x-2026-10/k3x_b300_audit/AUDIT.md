# Serving-parity audit of the K3-2x B300 failures

Date: 2026-10-09. Device: NVIDIA B300 SXM6 AC (SM 10.3), clocks unlocked.
Method follows `failure_handling.md` / `layer_permissions.md` ("Metadata/input parity"): for each crash, check
whether the collector hands the kernel anything serving would not, then reproduce with the smallest possible
program. No repo file was changed. Probes are in `probes/`, raw output in `results/`.

## Verdict

| Failure | Rows lost | Collector-side divergence found? | Root cause (reproduced) |
|---|---|---|---|
| MLA generation, 24 and 96 local heads | 604 | No | flashinfer 0.6.12 trtllm-gen MLA decode has no kernel for that head count / workload |
| KDA verify illegal memory access | 4 verify cells (192h b128 d8, 192h b256 d4, 192h b256 d8, 96h b256 d8) | No | `fused_kda_decode_mtp_dspark` overflows a 32-bit element offset into `intermediate_ssm` |
| GEMM fp8_block m=32768 n=14336 k=67584 | 1 | No | `sglang_per_token_group_quant_fp8` mis-indexes activations with >= 2^31 elements |

In all three the failing input is the shape itself, which serving would also construct. These are kernel/framework
limits, not collector bugs. Nothing here changes the collected data that passed.

## 1. MLA generation (604 cases: tp=8 -> 24 local heads 362, tp=2 -> 96 local heads 242)

- Path audited: collector -> `RadixAttention` -> `TRTLLMMLABackend.forward_decode` -> `flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla`
  (trtllm_mla_backend.py:711-805, 640-670). Kernel arguments come from serving code. Collector-built inputs:
  q_nope/q_rope (bs, local_heads, 512/64), the MLA KV pool (page 64), `req_to_token`, `seq_lens=history+1`, scaling.
  All match what `forward_decode` consumes. One collector/serving difference: the mocked `model_config.num_attention_heads` is the
  global head count with attn_tp=1, so `backend.num_q_heads` is 192 instead of local (trtllm_mla_backend.py:152). It is only read for
  CUDA-graph buffers, not on this path.
- Reproduction without sglang or the collector: calling flashinfer directly with random inputs, heads {16,24,32,48,64,96,128,192},
  batch 1..256, seq 1..32767, bf16 and fp8 KV.
  24 heads: 0/71 pass. 96 heads: 25/71 pass (batch*seq small), identical to the collector's pass/fail map. All others: 71/71.
  Error: `fmhaKernels.cuh:444 The numHeadsQ/numHeadsKv is not supported`.
- Mechanism (flashinfer fmhaKernels.cuh:442-447, 684-710): `numHeadsPerCta = min(numHeadsQPerKv, tileSizeQ)` must divide the head
  count. Heads <= 32 force the SwapsMmaAb kernel with tile 16, so 24 is never divisible. 96 only works with the
  tile-16 kernel, which is chosen only when the per-CTA KV length is small; otherwise tile 64 is chosen and 96 % 64 != 0.
- Consequence: K3-2x at TP=8 (24 heads) cannot use trtllm_mla decode at all, and TP=2 (96 heads) only for small workloads.
  Serving would hit the same wall, so these rows should stay missing, not be synthesised.

## 2. KDA verify illegal memory access

- Collector fields vs serving (`kda_backend.py:953-1045`): matching shapes/dtypes for x_q/k/v, conv weights (fp32), conv states,
  A_log, dt_bias, recurrent_state, cu_seqlens, state indices. One layout difference: collector `intermediate_conv_*` are fresh
  contiguous `[B, draft, proj, 3]`; serving passes transposed views of `[slots, draft, 3, conv_dim]`. Not causal here, since
  the experiments below change nothing but the state slots.
- Boundary: (heads=192, draft=2) batch 340 (tokens*heads 130560) ok, batch 342 (131328) fails, batch 256 ok. The threshold is
  tokens*heads*128*128 = 2^31 elements of `intermediate_ssm`.
- Direct kernel call, all arguments identical except slot placement:
  - B=128, D=8, scratch=128 slots (3.2e9 elements): IMA.
  - B=64, D=8, scratch=256 slots (6.4e9 elements), active slots 0..63: ok. So a >2^31-element tensor alone is fine.
  - Same, active slots 192..255 (element offset > 2^31): IMA.
  So the kernel computes `slot_index * (draft*H*V*K)` in 32 bits. The wrapper's `_fits_32bit_stride` (kda_decode_mtp.py:897) does not cover this.
- Serving relevance: this would also hit serving when slot_index * draft * heads * 16384 >= 2^31, i.e. large max-running-request pools.
  Example: 24 heads, draft 8 -> slot 683 and up. Worth reporting upstream. The kernel source was not read line-by-line;
  the 32-bit offset conclusion rests on the experiments above.
- The 74 KDA context cells (Triton causal_conv int32 token offset) are a classified raise in the collector citing source lines. Not re-audited here.

## 3. GEMM fp8_block (m=32768, n=14336, k=67584)

- The activation has 2,214,592,512 elements, above 2^31. Single-shot probe of quant -> DeepGEMM passed (silently corrupt), but the exact
  collector `run_gemm` crashed in a fresh process; 10 eager iterations and a CUDA-graph capture also crash.
- Quantizer correctness check: quantize the full tensor vs the last 997 rows quantized alone. Rows from 31775 on (the 2^31 mark)
  mismatch: 993 of 997 rows differ for `sglang_per_token_group_quant_fp8`. fp8 `sgl_per_token_quant_fp8` and nvfp4
  `flashinfer_fp4_quantize` on the same tensor: 0 mismatches, so the bf16/fp8/nvfp4 rows at this (m,k) are unaffected.
- Block m=31744 (A = 2.145e9 elements, just under 2^31) passes; m=32768,k=32768 passes.
- Serving relevance: only for a single >2^31-element activation (e.g. a 32k-token prefill into a 67584-wide down-projection). Rare but possible.
  Kernel source not read.

## What this does not do
- No collector change and no `FIXME(kernel-limit)` was added (a code change you have not approved). `collect_kda.py:737` already has one for the
  KDA verify case, but its stated hypothesis (per-SM resource growth) is not what was found.
- Upstream bug reports not filed.

## Addendum: does real serving handle 24 / 96 local heads? (2026-10-09, follow-up question)
- Code: sglang 0.5.14 and 0.5.17 `TRTLLMMLABackend` pass `layer.tp_q_head_num` (unpadded) to
  `flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla`; no head padding or fallback in the backend or in
  `DeepseekV2AttentionMLA`/`KimiK3MLAAttention`. 0.5.17 defaults Kimi-K3 to trtllm_mla on SM100/103 when no backend is set
  (arg_groups/overrides.py:449-455); DeepSeek-V3 arch likewise (overrides.py:633-641).
- Direct probe with flashinfer 0.6.15.post1 (the 0.5.17 env): 24 heads 0/71, 48 heads 71/71, 96 heads 25/71, 192 heads 71/71 (same as 0.6.12).
- Real server (`serving_repro/`): `sglang.launch_server` 0.5.14, tiny dummy-weight DeepseekV3ForCausalLM, 24 heads, TP=1,
  `--attention-backend trtllm_mla`, CUDA graphs on (default). The scheduler crashed during startup warmup
  (`_flashinfer_autotune` -> `_dummy_run` -> `forward_decode` -> `trtllm_mla_backend._run_decode_kernel`) with the same
  `fmhaKernels.cuh:444 numHeadsQ/numHeadsKv is not supported`. Not tested: 96 heads, the K3 architecture on 0.5.17, other MLA backends.
