"""
Verification step 3 (plan: go-into-aisimulate-repo-encapsulated-sutton.md):
one single-point Task run (pareto_sweep=False) before committing to the full
sweep. Confirms Task()/pick_default's own plumbing (distinct from the
ForwardPassPerfModelConfig diagnostic already verified) resolves cleanly
against the K3-2x model + b300_sxm_nvl72 override + literal backend_version.
"""

import os
import sys

os.environ.setdefault("AIC_ALLOW_UNLISTED_VERSIONS", "1")

REPO = "/home/jding/Documents/aisimulate"
OVERRIDE_DIR = os.path.join(REPO, "systems_override")
MODEL_DIR = os.path.join(REPO, "experiments/kimi_k3_scaled2x/model")

from aisimulate_core.sdk.perf_database import set_systems_paths  # noqa: E402

set_systems_paths([OVERRIDE_DIR, "default"])

from aisimulate.sdk.task_v2 import Task  # noqa: E402
from aisimulate_core.sdk import common  # noqa: E402

t = Task(
    serving_mode="agg",
    model_path=MODEL_DIR,
    backend_name="sglang",
    backend_version="0.5.17",  # literal -- see verify_version_routing.py for why not "next"
    system_name="b300_sxm_nvl72",
    isl=1000,
    osl=1000,
    total_gpus=1024,
    ttft=1e9,
    tpot=1e9,
    pareto_sweep=False,
    agg_num_gpu_candidates=[256],
    agg_tp_candidates=[4],
    agg_pp_candidates=[1],
    agg_dp_candidates=[64],
    agg_moe_tp_candidates=[4],   # moe_ep stays <=64 (within-node, b300_sxm_nvl72's 72-GPU domain)
    agg_moe_ep_candidates=[64],  # and moe_tp=4 shrinks the per-GPU expert shard to fit memory
    # Task defaults EVERY quant mode to bfloat16 -- at bf16 the MoE experts
    # alone are ~44TB (unquantized), which doesn't fit any replica size in
    # this ladder. Match what the B300 collection actually measured: mxfp4
    # routed experts, bf16 everywhere else (~11.3TB weights, fits from 64
    # GPUs up per the earlier memory-floor estimate).
    moe_quant_mode=common.MoEQuantMode.w4a16_mxfp4,
    gemm_quant_mode=common.GEMMQuantMode.bfloat16,
    kvcache_quant_mode=common.KVCacheQuantMode.bfloat16,
    fmha_quant_mode=common.FMHAQuantMode.bfloat16,
    comm_quant_mode=common.CommQuantMode.half,
)
t.validate()
df = t.run()
print(f"rows: {len(df)}", file=sys.stderr)
if df.empty:
    print("EMPTY -- infeasible config or a routing problem, not necessarily a crash.", file=sys.stderr)
    sys.exit(1)
cols = [c for c in ["tp", "dp", "moe_tp", "moe_ep", "bs", "tpot", "tokens/s/user", "tokens/s/gpu", "num_total_gpus"] if c in df.columns]
print(df[cols].head(10).to_string(index=False), file=sys.stderr)
print("\nPASS: non-empty result.", file=sys.stderr)
