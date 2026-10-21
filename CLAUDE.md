<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

@AGENTS.md

## Session notes: Kimi-K3 / GB300 Pareto investigation

Learnings from an extended investigation into Kimi-K3 EP/TP sensitivity,
NVL72-vs-NVL288 Pareto curves, and an interactive dashboard. Kept here
because they weren't discoverable from the code alone.

### Architecture / conventions

- Real (non-deprecated) Pareto path: `Task(pareto_sweep=True)` +
  `aisimulate.sdk.picking.pick_default`. **Not** `aisimulate.sweeper.Sweeper`
  — it's hard-capped at 16 GPUs/worker, has no cross-node-EP gate, and
  doesn't propagate systems paths into its worker processes. Unusable for
  Kimi-K3's EP range.
- `pick_default` computes `tokens/s/gpu_cluster = tokens/s/gpu *
  (total_gpus // num_total_gpus) * num_total_gpus / total_gpus` and fronts
  on `(tokens/s/user, tokens/s/gpu_cluster)`.
- `aisimulate predict --stack engine -c <cfg.yaml>` is a separate
  trace-replay/agentic path from the scalar `Task` SDK. `engine.systems_paths`
  in the YAML is the config-level equivalent of `set_systems_paths()`.
- `workload_distribution` isn't exposed via `build_model_config`/CLI for
  Kimi-K3 (hardcoded `"power_law_1.01"`) — only reachable by monkeypatching
  `aisimulate_core.sdk.config.ModelConfig.__init__`.
- For a per-op latency breakdown of one exact config (not available from
  `Task.run()` or predict's `power_diagnostics`, which is energy-only):
  build `aisimulate_core.sdk.rust_engine_step.ForwardPassPerfModelConfig(
  model=, system=, backend=, worker_type='aggregated', tp=, attention_dp=,
  moe_tp_size=, moe_ep_size=, gemm_quant_mode=, moe_quant_mode=,
  kvcache_quant_mode=, fmha_quant_mode=, comm_quant_mode=,
  estimation_mode='op_level', fallback_policy='deny',
  database_mode='SILICON', systems_paths=(...))`, then
  `RustForwardPassPerfModel.best_available(cfg).static_phase_diagnostics(
  batch_size=, context_length=, prefill=False)` → list of
  `{name, latency_ms, source, details}`.

### Gotchas

- **A single business rule can be hand-duplicated across multiple Rust call
  sites with no shared source of truth.** The "route TP all-reduce through
  NCCL when `num_gpus_per_node == 72` and `tp>4`" rule existed independently
  in `readiness.rs` (data-requirements check only — editing it alone changes
  *nothing* observable), `operators/communication.rs`
  (`query_custom_allreduce_table`, the real dispatch), and
  `perf_database/communication.rs` (`query_custom_allreduce_scaled`, the
  low-level lookup). All three must change together. Always grep for every
  occurrence of a condition before assuming one edit changed behavior, and
  verify empirically (re-run a known case), not just by reading the diff.
- Changing that `==72` to `>=72` also pulls in NVL288 (and anything else
  `>=72`) — it stops being an NVL72-only rule. Confirm that's intended.
- **The compiled Python extension goes stale silently.** `cargo build -p
  aisimulate-core` does not update `python/aisimulate/src/aisimulate/
  _runtime.abi3.so`. Must rebuild via `uv sync --project python/aisimulate
  --extra dev --reinstall-package aisimulate` (needs the Artifactory mirror
  env vars below). Verify by checking the `.so` mtime *and* re-running a
  known test case — a stale `.so` fails silently, not loudly.
- The full-candidates sweep output has duplicate-looking rows per
  `(config, bs)`: the engine samples multiple phase-offsets of the periodic
  chunked-prefill cycle for the same steady-state point.
  `num_ctx_reqs + num_gen_reqs` is constant per group; `ttft` genuinely
  varies (different queueing position) but `tpot`/throughput agree to
  within ~1% (float noise). Dedup key is `(system, tp, dp, moe_tp, moe_ep,
  bs)` — do **not** include `tpot`/`tokens/s/*` in the key, float noise
  splits what should be one group.
- `tokens/s/gpu = bs / (tpot_ms * tp)` — `dp` cancels out entirely. When
  comparing configs at *different* `tp`, this metric is not a clean
  "did more GPUs help" signal; the `tp` change rescales the denominator by
  itself. Check whether `tp` differs before interpreting the result.
- MoE `moe_ep` scaling is non-monotonic in the real measured kernel data
  (`moe_perf_*.csv`): growing `moe_ep` while the batch size fed to the local
  kernel stays fixed can *overshoot* the efficient EP width and get worse
  (confirmed: EP128 slower than EP64 at fixed `num_tokens=64`). Scaling
  `moe_ep` and batch size together (constant tokens-per-expert-group) stays
  on the efficient side and wins on both TPOT and tokens/s/gpu.
- This network only reaches Git + Artifactory (no direct PyPI/HF/crates.io).
  `uv` doesn't read `pip.conf` — must set `UV_DEFAULT_INDEX`/`UV_INDEX`
  explicitly. `uv` itself lives at `~/.local/bin/uv`, not on default PATH.

### Commands that work

```bash
# Rust-only build check (does NOT update the Python-visible .so)
cargo build -p aisimulate-core --release

# Rebuild the Python extension after any crates/core change (required)
export PATH="$HOME/.local/bin:$PATH"
export UV_DEFAULT_INDEX="https://artifactory.marvell.com/artifactory/api/pypi/odsp-sw-pypi-remote/simple"
export UV_INDEX="https://artifactory.marvell.com/artifactory/api/pypi/odsp-sw-pypi-local/simple"
export SSL_CERT_FILE="/home/jding/.certs/ca-bundle-marvell.pem"
uv sync --project python/aisimulate --extra dev --reinstall-package aisimulate
```

HuggingFace dataset downloads: use the Artifactory HF mirror
(`lab-art-huggingface-remote`) with HTTP `Range` requests rather than
pulling a large file in full.

### Unresolved / flagged follow-ups

- Disagg sweep used `agentic_mooncake` format with no warmup, for
  protocol-consistency with the agg sweep. The official full disagg
  protocol (`format: weka` + `agentic_lanes`/`agentic_snapshot`/
  `agentic_warmup`, a three-stage primer/warmup/profile run) was never
  exercised — see `docs/agentic-warmup.md`.
- The disagg sweep gives prefill and decode the **identical** parallelism
  shape (not independently tuned), which misses the actual point of
  disaggregation. A true independent search is ~29h brute-force or ~2h via
  a two-stage solo-optimize-then-cross-shortlists approach — not yet run.
- FPM model-onboarding workflow (see AGENTS.md) was never exercised here.
