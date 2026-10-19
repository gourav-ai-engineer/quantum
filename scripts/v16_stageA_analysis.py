"""V16 Stage A decision table (docs/PROJECT_STATE.md, "Pre-registration V16"), applied mechanically.

beats(X): mean held-out-train KL of X (5 seeds) <= 0.9 x official K-prune's at BOTH c0.5 and c0.4 on MRPC, and also
          on SST-2 at both constraints (the SST-2 win must survive).
A-1: beats(d2ii) and D1 upper-layer underpricing on MRPC for both D1 seeds
     (median rho over the layer 9-11 modules >= 2 x median rho over the layer 0-8 modules).
A-2: beats(d2i) and not beats(d2ii).
A-3: otherwise.

    python scripts/v16_stageA_analysis.py <stageA_dir> results/v15_h10/c43dd3e/h10/v15_h10_verdict.json
"""
from __future__ import annotations

import json
import os
import statistics
import sys

TASKS, CONS, SEEDS = ("mrpc", "sst2"), (0.5, 0.4), (7, 42, 77, 123, 2024)


def mean_kl(d, part, task, c):
    vals = []
    for s in SEEDS:
        p = os.path.join(d, f"v16_{part}_{task}_seed{s}_c{c}.json")
        if not os.path.exists(p):
            return None
        vals.append(json.load(open(p))["heldout_kl"])
    return sum(vals) / len(vals)


def underpricing(d, task, seed):
    p = os.path.join(d, f"v16_d1_{task}_seed{seed}_c0.5.json")
    if not os.path.exists(p):
        return None
    rows = [r for r in json.load(open(p))["modules"] if r.get("rho") is not None]
    top = [r["rho"] for r in rows if r["layer"] >= 9]
    low = [r["rho"] for r in rows if r["layer"] <= 8]
    mt, ml = statistics.median(top), statistics.median(low)
    return {"median_rho_top": mt, "median_rho_low": ml, "ratio": mt / ml if ml > 0 else float("inf"),
            "underpriced": ml > 0 and mt >= 2 * ml}


def decide(d, verdict_path):
    kp = {k: v["kprune"]["mean_kl"] for k, v in json.load(open(verdict_path))["table"].items()}
    table, beats = {}, {}
    for part in ("d2i", "d2ii", "no_sens"):
        ok = True
        for task in TASKS:
            for c in CONS:
                m = mean_kl(d, part, task, c)
                ref = kp[f"{task}_c{c}"]
                table[f"{part}_{task}_c{c}"] = {"mean_kl": m, "kprune_kl": ref,
                                                "ratio": None if m is None else m / ref}
                ok = ok and m is not None and m <= 0.9 * ref
        beats[part] = ok
    d1 = {f"{t}_seed{s}": underpricing(d, t, s) for t in TASKS for s in (7, 42)}
    under = all(d1[f"mrpc_seed{s}"] and d1[f"mrpc_seed{s}"]["underpriced"] for s in (7, 42))
    if beats["d2ii"] and under:
        branch = "A-1"
    elif beats["d2i"] and not beats["d2ii"]:
        branch = "A-2"
    else:
        branch = "A-3"
    return {"branch": branch, "beats": beats, "mrpc_upper_layer_underpricing": under, "table": table, "d1": d1}


def main():
    out = decide(sys.argv[1], sys.argv[2])
    for k, v in out["table"].items():
        print(k, v)
    print("d1", out["d1"])
    print("beats", out["beats"], "underpricing(MRPC)", out["mrpc_upper_layer_underpricing"])
    print("branch:", out["branch"])
    json.dump(out, open(os.path.join(sys.argv[1], "v16_stageA_verdict.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
