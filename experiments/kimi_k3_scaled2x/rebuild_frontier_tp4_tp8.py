"""
Rebuild the final Pareto deliverable restricted to tp in {4, 8} -- dropping
tp=16/32, which have zero measured KDA data at this model's d_model (confirmed:
they return the 24-head row unscaled) on top of the already-known tp=8 MLA
generation-decode kernel-cliff issue. tp=8 is kept: its KDA data is real and
exact-key (192/8=24 local heads is one of the four measured shapes), and the
separate MLA-decode problem at that shape is a distinct, already-flagged risk,
not grounds for removal here -- that's the user's call, made explicitly.

Operates on the already-computed full candidate set from
pareto_scaled2x_b300_isl1k_osl1k.py (pick_default is a pure function over the
dataframe -- no need to re-run the engine sweep).
"""

import os
import sys

REPO = "/home/jding/Documents/aisimulate"
OUT_DIR = os.path.join(REPO, "experiments", "kimi_k3_scaled2x")

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from aisimulate.sdk.picking import pick_default

TOTAL_GPUS = 1024
SYSTEMS = ["b300_sxm_nvl72", "b300_sxm_nvl288"]
COLORS = {"b300_sxm_nvl72": "#2a78d6", "b300_sxm_nvl288": "#eb6834"}
KEEP_TP = [4, 8]

full = pd.read_csv(os.path.join(OUT_DIR, "pareto_scaled2x_full_candidates.csv"))
print(f"Loaded {len(full)} rows, tp values present: {sorted(full.tp.unique().tolist())}", file=sys.stderr)

filtered = full[full.tp.isin(KEEP_TP)].copy()
print(f"Kept {len(filtered)} rows at tp in {KEEP_TP}", file=sys.stderr)
filtered.to_csv(os.path.join(OUT_DIR, "pareto_scaled2x_full_candidates.csv"), index=False)

frontiers = []
for system_name in SYSTEMS:
    df = filtered[filtered.system == system_name]
    print(f"\n=== {system_name} ===", file=sys.stderr)
    print(f"  candidates: {len(df)} rows, tp present: {sorted(df.tp.unique().tolist())}", file=sys.stderr)
    if df.empty:
        continue
    res = pick_default(pareto_df=df, total_gpus=TOTAL_GPUS, serving_mode="agg", strict_sla=False)
    frontier = res["pareto_frontier_df"].copy()
    frontier["system"] = system_name
    print(f"  pareto frontier: {len(frontier)} points", file=sys.stderr)
    cols = [c for c in ["num_total_gpus", "moe_ep", "moe_tp", "tp", "dp", "bs", "tokens/s/user", "tokens/s/gpu_cluster"] if c in frontier.columns]
    print(frontier[cols].sort_values("tokens/s/user").to_string(index=False), file=sys.stderr)
    print(f"  tp value counts on this frontier: {frontier.tp.value_counts().to_dict()}", file=sys.stderr)
    frontiers.append(frontier)

combined = pd.concat(frontiers, ignore_index=True)
out_csv = os.path.join(OUT_DIR, "pareto_scaled2x_frontier.csv")
combined.to_csv(out_csv, index=False)
print(f"\nWrote {len(combined)} rows to {out_csv}", file=sys.stderr)

# the tp4_only / all_tp split files from the exploratory step no longer apply --
# this run replaces the single canonical frontier/full-candidates files.
for stale in ("pareto_scaled2x_frontier_tp4_only.csv", "pareto_scaled2x_frontier_all_tp.csv"):
    path = os.path.join(OUT_DIR, stale)
    if os.path.exists(path):
        os.remove(path)
        print(f"Removed stale {stale}", file=sys.stderr)

fig, ax = plt.subplots(figsize=(8, 6))
for system_name in SYSTEMS:
    sub = combined[combined.system == system_name].sort_values("tokens/s/user")
    if sub.empty:
        continue
    ax.plot(
        sub["tokens/s/user"], sub["tokens/s/gpu_cluster"],
        marker="o", ms=6, label=system_name, color=COLORS[system_name],
    )
ax.set_xlabel("tokens/s/user")
ax.set_ylabel("tokens/s/gpu (cluster-normalized)")
ax.set_title(
    "Kimi-K3-2x / B300: NVL72 vs NVL288 Pareto frontier\n"
    "(ISL=1000, OSL=1000, total_gpus=1024, tp in {4, 8})"
)
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
out_png = os.path.join(OUT_DIR, "pareto_scaled2x_nvl72_vs_nvl288.png")
fig.savefig(out_png, dpi=150)
plt.close(fig)
print(f"Wrote {out_png}", file=sys.stderr)
