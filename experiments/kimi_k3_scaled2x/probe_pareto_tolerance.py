"""
Probe: does Task(pareto_sweep=True) tolerate a MIX of feasible and infeasible
points in one candidate set (memory-infeasible moe_tp=1; cross-node-EP
moe_ep=128 on the 72-GPU-per-node b300_sxm_nvl72 override, which needs DeepEP
A2A data we don't have), or does a single bad combo abort the whole sweep the
way it did in pareto_sweep=False single-point mode?
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

for system in ("b300_sxm_nvl72", "b300_sxm_nvl288"):
    print(f"\n=== {system}: mixed candidate set (moe_tp 1&4, moe_ep 64&128) ===", file=sys.stderr)
    t = Task(
        serving_mode="agg",
        model_path=MODEL_DIR,
        backend_name="sglang",
        backend_version="0.5.17",
        system_name=system,
        isl=1000,
        osl=1000,
        total_gpus=1024,
        ttft=1e9,
        tpot=1e9,
        pareto_sweep=True,
        agg_num_gpu_candidates=[256],
        agg_tp_candidates=[4],
        agg_pp_candidates=[1],
        agg_dp_candidates=[64],
        agg_moe_tp_candidates=[1, 2, 4],   # 1 is memory-infeasible per the smoke test
        agg_moe_ep_candidates=[64, 128],  # 128 crosses 72-node boundary on nvl72 only; needs moe_tp=2 to reach product 256
        moe_quant_mode=common.MoEQuantMode.w4a16_mxfp4,
        gemm_quant_mode=common.GEMMQuantMode.bfloat16,
        kvcache_quant_mode=common.KVCacheQuantMode.bfloat16,
        fmha_quant_mode=common.FMHAQuantMode.bfloat16,
        comm_quant_mode=common.CommQuantMode.half,
    )
    try:
        t.validate()
        df = t.run()
        print(f"  rows: {len(df)}", file=sys.stderr)
        if not df.empty:
            cols = [c for c in ["tp", "dp", "moe_tp", "moe_ep", "bs"] if c in df.columns]
            print(df[cols].drop_duplicates().to_string(index=False), file=sys.stderr)
    except Exception as e:
        print(f"  RAISED: {type(e).__name__}: {e}", file=sys.stderr)
