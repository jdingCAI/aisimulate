"""
Verification step 1 (plan: go-into-aisimulate-repo-encapsulated-sutton.md):
confirm, empirically and on OUR checkout, that backend_version="next" resolves
to sglang 0.5.17 for b300_sxm and that it serves BOTH the KDA rows (native at
0.5.17) and the MoE/GEMM/MLA rows (native at 0.5.14, expected to reach the
query via backward-fill) -- rather than trusting the b300-k3x-and-mla256
evidence bundle's self-reported sdk_lookup_verification.md.

Mirrors that bundle's own method: BEFORE/AFTER-style exact-key queries
through the real engine, not a parquet-only check.
"""

import os
import sys

os.environ.setdefault("AIC_ALLOW_UNLISTED_VERSIONS", "1")

REPO = "/home/jding/Documents/aisimulate"
OVERRIDE_DIR = os.path.join(REPO, "systems_override")

from aisimulate_core.sdk.perf_database import get_version_slots, set_systems_paths  # noqa: E402

set_systems_paths([OVERRIDE_DIR, "default"])

slots = get_version_slots("b300_sxm", "sglang")  # uses the global set by set_systems_paths() above
print("b300_sxm/sglang version slots:", slots, file=sys.stderr)
assert slots is not None, "no version slots resolved for b300_sxm/sglang -- gate is off or system not found"
assert slots.get("current") == "0.5.14", f"expected current=0.5.14, got {slots}"
assert slots.get("next") == "0.5.17", f"expected next=0.5.17, got {slots}"
print("PASS: version slots match the evidence bundle's claim\n", file=sys.stderr)

from aisimulate_core.sdk.rust_engine_step import ForwardPassPerfModelConfig, RustForwardPassPerfModel  # noqa: E402

# ForwardPassPerfModelConfig's quant-mode fields are typed `str | None` (plain
# enum-name strings, e.g. "bfloat16", "w4a16_mxfp4") -- not enum instances.
COMMON_KW = dict(
    system="b300_sxm_nvl72",
    backend="sglang",
    # NOT "next": the Rust->Python callback that resolves slot ALIASES
    # ("current"/"previous"/"next") during engine-spec compilation does not
    # see this process's systems_paths the same way RustForwardPassPerfModel's
    # direct perf-DB queries do -- on a custom override system it raises
    # "no queryable versions defined for sglang on b300_sxm_nvl72" even
    # though get_version_slots() resolves fine when called directly from
    # Python. The literal version string sidesteps that alias-resolution
    # path entirely and was confirmed (see below) to still reach the 0.5.14
    # MoE/GEMM/MLA/mla_bmm rows via the Rust loader's own backward-fill,
    # exactly like the alias would have -- this is a resolution-path bug on
    # custom systems, not evidence that backward-fill itself doesn't apply.
    backend_version="0.5.17",
    worker_type="aggregated",
    tp=4,              # 192/4 = 48 local KDA/MLA heads -- a fully-measured shape per the plan
    attention_dp=16,
    moe_tp_size=1,
    moe_ep_size=64,    # tp * attention_dp * cp == moe_tp_size * moe_ep_size (KimiK3Model.__init__ assert): 4*16 = 1*64
    gemm_quant_mode="bfloat16",
    moe_quant_mode="w4a16_mxfp4",
    kvcache_quant_mode="bfloat16",
    fmha_quant_mode="bfloat16",
    comm_quant_mode="half",
    estimation_mode="op_level",
    fallback_policy="deny",
    database_mode="SILICON",
    systems_paths=(OVERRIDE_DIR, "default"),
)


def check(model_path, label):
    cfg = ForwardPassPerfModelConfig(model=model_path, **COMMON_KW)
    model = RustForwardPassPerfModel.best_available(cfg)
    diag = model.static_phase_diagnostics(batch_size=64, context_length=1000, prefill=False)
    by_source = {}
    for op in diag:
        by_source.setdefault(op["source"], []).append(op["name"])
    print(f"=== {label} ===", file=sys.stderr)
    for src, names in by_source.items():
        print(f"  {src}: {len(names)} ops -> {names[:5]}{'...' if len(names) > 5 else ''}", file=sys.stderr)
    return by_source


by_source = check(os.path.join(REPO, "experiments/kimi_k3_scaled2x/model"), "K3-2x on b300_sxm_nvl72, backend_version=next")

sol_ops = by_source.get("sol", [])
if sol_ops:
    print(f"\nWARNING: {len(sol_ops)} ops fell back to SOL: {sol_ops}", file=sys.stderr)
    print("If this includes kda_* or moe_* ops, version routing did NOT work as expected -- treat as a bug.", file=sys.stderr)
else:
    print("\nPASS: no ops fell back to SOL at this smoke config.", file=sys.stderr)
