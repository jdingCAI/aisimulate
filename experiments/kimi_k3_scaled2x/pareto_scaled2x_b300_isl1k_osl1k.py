"""
Kimi-K3-2x (scaled variant) / B300: NVL72 vs NVL288 Pareto curve at
ISL=1000 / OSL=1000, total_gpus=1024, replica sizes 64/128/256 GPUs.

Uses the real (non-deprecated) Pareto path -- Task(pareto_sweep=True) +
aisimulate.sdk.picking.pick_default -- NOT aisimulate.sweeper.Sweeper (hard-
capped at 16 GPUs/worker, no cross-node-EP gate, doesn't propagate
set_systems_paths() into its worker processes). Same scaffolding as
experiments/kimi_k3_ep_sensitivity/pareto_nvl72_vs_nvl288_osl1m.py, retargeted
at the B300 collection on origin/data/b300-k3x-and-mla256.

See /home/jding/.claude/plans/go-into-aisimulate-repo-encapsulated-sutton.md
for the full rationale, the search-space constraints below, and what was
verified empirically before this script was trusted:

- backend_version is the LITERAL "0.5.17", not the "next" alias -- the alias
  resolution path fails on a custom override system (confirmed bug in the
  Rust->Python engine-spec-compilation callback's systems_paths handling).
  The literal version was confirmed to still reach the MoE/GEMM/MLA/mla_bmm
  rows (native at 0.5.14) via the same backward-fill the alias would have
  used, with zero source="sol" ops on a representative config.
- agg_tp_candidates=[4, 8, 16, 32]. tp=1/2 are memory-infeasible everywhere
  in the search space (the full unsharded attention+dense+shared-expert
  stack, ~438GB bf16 with no tp division, exceeds the B300's ~288GB per GPU)
  and are not included -- they would simply never appear in the output.
  tp=8/16/32 ARE included, on request, despite a known data-fidelity problem:
  at tp=8 (24 local heads), the generation_attention op resolves by
  INTERPOLATING num_heads as a continuous grid axis across the measured
  16-head/32-head points, but the audit (k3x_b300_audit/AUDIT.md) found the
  real kernel has a 0% success rate at exactly 24 local heads -- a hard
  kernel-dispatch wall (tile-16 decode requires num_heads % 16 == 0), not a
  point on a smooth curve -- reproduced with a live `sglang.launch_server`
  crash. tp=16/32 (12/6 local heads) compound this: KDA has no measured row
  below 24 local heads for this d_model, so its nearest-same-d_model fallback
  returns the 24-head row UNSCALED for both. None of this is enforced by the
  repo -- `fallback_policy="deny"` does not fire for any of these shapes, so
  nothing stops a sweep from including them; the finding lives only in a
  human-authored audit document the engine never reads. This script reports
  BOTH a `tp4_only` frontier (the trusted one, real exact-key or legitimate
  in-range interpolation throughout) and an `all_tp` frontier (everything the
  engine will compute without complaint) so the gap between "what the tool
  reports" and "what's actually backed by sound data" is visible, not assumed.
- moe_quant_mode is explicitly w4a16_mxfp4 (Task defaults every quant mode
  to bfloat16; at bf16 the MoE experts alone are ~44TB, infeasible at any
  replica size in this ladder).
- moe_tp=1 is memory-infeasible at every moe_ep; left in the candidate list
  anyway because pareto_sweep=True drops infeasible points silently rather
  than raising (verified on both systems) -- it will simply never appear.
- moe_ep=128 is expected to survive on b300_sxm_nvl288 but be dropped
  entirely on b300_sxm_nvl72 (cross-node EP there needs DeepEP A2A data this
  collection doesn't have) -- confirmed directly before this run. That drop
  IS the NVL72-vs-NVL288 story, not a bug.
"""

import os
import sys

os.environ.setdefault("AIC_ALLOW_UNLISTED_VERSIONS", "1")

REPO = "/home/jding/Documents/aisimulate"
OVERRIDE_DIR = os.path.join(REPO, "systems_override")
MODEL_DIR = os.path.join(REPO, "experiments/kimi_k3_scaled2x/model")
OUT_DIR = os.path.join(REPO, "experiments", "kimi_k3_scaled2x")

from aisimulate_core.sdk.perf_database import set_systems_paths  # noqa: E402

set_systems_paths([OVERRIDE_DIR, "default"])

from aisimulate.sdk.task_v2 import Task  # noqa: E402
from aisimulate.sdk.picking import pick_default  # noqa: E402
from aisimulate_core.sdk import common  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TOTAL_GPUS = 1024
COMMON = dict(
    serving_mode="agg",
    model_path=MODEL_DIR,
    backend_name="sglang",
    backend_version="0.5.17",
    isl=1000,
    osl=1000,
    total_gpus=TOTAL_GPUS,
    ttft=1e9,  # unconstrained -- full envelope, not a TTFT-filtered one
    tpot=1e9,  # irrelevant: pareto_sweep=True replaces this with the internal TPOT grid
    pareto_sweep=True,
    agg_num_gpu_candidates=[64, 128, 256],
    agg_tp_candidates=[4, 8, 16, 32],
    agg_pp_candidates=[1],
    agg_dp_candidates=[1, 2, 4, 8, 16, 32, 64, 128, 256],
    agg_moe_tp_candidates=[1, 2, 4],
    agg_moe_ep_candidates=[16, 32, 64, 128, 256],
    moe_quant_mode=common.MoEQuantMode.w4a16_mxfp4,
    gemm_quant_mode=common.GEMMQuantMode.bfloat16,
    kvcache_quant_mode=common.KVCacheQuantMode.bfloat16,
    fmha_quant_mode=common.FMHAQuantMode.bfloat16,
    comm_quant_mode=common.CommQuantMode.half,
)

SYSTEMS = ["b300_sxm_nvl72", "b300_sxm_nvl288"]
COLORS = {"b300_sxm_nvl72": "#2a78d6", "b300_sxm_nvl288": "#eb6834"}
TRUST_STYLE = {"tp4_only": dict(linestyle="-", marker="o"), "all_tp": dict(linestyle="--", marker="^")}


def _frontier(df: pd.DataFrame, system_name: str, trust: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    res = pick_default(pareto_df=df, total_gpus=TOTAL_GPUS, serving_mode="agg", strict_sla=False)
    frontier = res["pareto_frontier_df"].copy()
    if frontier.empty:
        return frontier
    frontier["system"] = system_name
    frontier["trust"] = trust
    cols = [c for c in ["num_total_gpus", "moe_ep", "moe_tp", "tp", "dp", "bs", "tokens/s/user", "tokens/s/gpu_cluster"] if c in frontier.columns]
    print(f"  [{trust}] pareto frontier: {len(frontier)} points", file=sys.stderr)
    print(frontier[cols].sort_values("tokens/s/user").to_string(index=False), file=sys.stderr)
    return frontier


def run_system(system_name: str):
    print(f"\n=== {system_name} ===", file=sys.stderr)
    t = Task(**COMMON, system_name=system_name)
    t.validate()
    df = t.run()
    print(f"  candidate set: {len(df)} rows", file=sys.stderr)
    if df.empty:
        print("  WARNING: empty candidate set -- nothing survived the search space.", file=sys.stderr)
        return df, pd.DataFrame(), pd.DataFrame()

    tp_seen = sorted(df.tp.unique().tolist()) if "tp" in df.columns else []
    moe_ep_seen = sorted(df.moe_ep.unique().tolist()) if "moe_ep" in df.columns else []
    print(f"  tp values present: {tp_seen}", file=sys.stderr)
    print(f"  moe_ep values present: {moe_ep_seen}", file=sys.stderr)

    frontier_all = _frontier(df, system_name, "all_tp")
    frontier_tp4 = _frontier(df[df.tp == 4] if "tp" in df.columns else df, system_name, "tp4_only")
    return df, frontier_tp4, frontier_all


def main():
    frontiers_tp4, frontiers_all = [], []
    raw_sets = []
    for system_name in SYSTEMS:
        df, frontier_tp4, frontier_all = run_system(system_name)
        if not df.empty:
            tagged = df.copy()
            tagged["system"] = system_name
            raw_sets.append(tagged)
        if not frontier_tp4.empty:
            frontiers_tp4.append(frontier_tp4)
        if not frontier_all.empty:
            frontiers_all.append(frontier_all)

    if raw_sets:
        raw_combined = pd.concat(raw_sets, ignore_index=True)
        raw_combined = raw_combined.drop(columns=[c for c in raw_combined.columns if c.startswith("_")], errors="ignore")
        raw_out_csv = os.path.join(OUT_DIR, "pareto_scaled2x_full_candidates.csv")
        raw_combined.to_csv(raw_out_csv, index=False)
        print(f"\nWrote {len(raw_combined)} full candidate rows to {raw_out_csv}", file=sys.stderr)

    if not frontiers_tp4 and not frontiers_all:
        print("\nNo frontier points for either system -- nothing to plot.", file=sys.stderr)
        return

    combined_tp4 = pd.concat(frontiers_tp4, ignore_index=True) if frontiers_tp4 else pd.DataFrame()
    combined_all = pd.concat(frontiers_all, ignore_index=True) if frontiers_all else pd.DataFrame()

    tp4_csv = os.path.join(OUT_DIR, "pareto_scaled2x_frontier_tp4_only.csv")
    all_csv = os.path.join(OUT_DIR, "pareto_scaled2x_frontier_all_tp.csv")
    combined_tp4.to_csv(tp4_csv, index=False)
    combined_all.to_csv(all_csv, index=False)
    print(f"\nWrote {len(combined_tp4)} rows to {tp4_csv}", file=sys.stderr)
    print(f"Wrote {len(combined_all)} rows to {all_csv}", file=sys.stderr)

    # Backward-compatible name, pointing at the trusted (tp=4-only) frontier --
    # this is the curve with no known kernel-support or unscaled-fallback issue.
    legacy_csv = os.path.join(OUT_DIR, "pareto_scaled2x_frontier.csv")
    combined_tp4.to_csv(legacy_csv, index=False)

    combined = pd.concat([combined_tp4, combined_all], ignore_index=True)

    fig, ax = plt.subplots(figsize=(9, 7))
    for system_name in SYSTEMS:
        for trust, label_suffix in [("tp4_only", "tp=4 only, trusted"), ("all_tp", "tp∈{4,8,16,32}, unfiltered")]:
            sub = combined[(combined.system == system_name) & (combined.trust == trust)].sort_values("tokens/s/user")
            if sub.empty:
                continue
            style = TRUST_STYLE[trust]
            ax.plot(
                sub["tokens/s/user"], sub["tokens/s/gpu_cluster"],
                ms=6, label=f"{system_name} ({label_suffix})", color=COLORS[system_name], **style,
            )
    ax.set_xlabel("tokens/s/user")
    ax.set_ylabel("tokens/s/gpu (cluster-normalized)")
    ax.set_title(
        "Kimi-K3-2x / B300: NVL72 vs NVL288 Pareto frontier\n"
        "(ISL=1000, OSL=1000, total_gpus=1024)\n"
        "dashed = includes tp=8/16/32, known to rest on an unscaled KDA fallback\n"
        "and (at tp=8) a kernel-dispatch cliff reproduced as a live server crash"
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_png = os.path.join(OUT_DIR, "pareto_scaled2x_nvl72_vs_nvl288.png")
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_png}", file=sys.stderr)


if __name__ == "__main__":
    main()
