"""V17 Stage R-dev gate (docs/PROJECT_STATE.md, "Pre-registration V17"), applied mechanically.

gate (proceed to R-conf): GRP-R mean KL <= 1.0 x official K-prune at c0.5 and c0.4 on MRPC, and <= 0.9 x at both on SST-2.
mechanism (descriptive): d2i_r within 1.1 x K-prune at both budgets on both tasks.

    python scripts/v17_rdev_analysis.py <dir> results/v15_h10/c43dd3e/h10/v15_h10_verdict.json
"""
from __future__ import annotations

import json
import os
import sys

SEEDS = (7, 42, 77, 123, 2024)


def mean_kl(d, part, task, c):
    paths = [os.path.join(d, f"v17_{part}_{task}_seed{s}_c{c}.json") for s in SEEDS]
    if not all(os.path.exists(p) for p in paths):
        return None
    return sum(json.load(open(p))["heldout_kl"] for p in paths) / len(paths)


def decide(d, verdict_path):
    kp = {k: v["kprune"]["mean_kl"] for k, v in json.load(open(verdict_path))["table"].items()}
    table = {f"{part}_{t}_c{c}": {"mean_kl": mean_kl(d, part, t, c), "kprune_kl": kp[f"{t}_c{c}"]}
             for part in ("grp_r", "d2i_r") for t in ("mrpc", "sst2") for c in (0.5, 0.4)}
    for v in table.values():
        v["ratio"] = None if v["mean_kl"] is None else v["mean_kl"] / v["kprune_kl"]

    def ok(part, task, bound):
        return all(table[f"{part}_{task}_c{c}"]["ratio"] is not None and table[f"{part}_{task}_c{c}"]["ratio"] <= bound
                   for c in (0.5, 0.4))

    gate = ok("grp_r", "mrpc", 1.0) and ok("grp_r", "sst2", 0.9)
    mechanism = ok("d2i_r", "mrpc", 1.1) and ok("d2i_r", "sst2", 1.1)
    return {"gate_pass": gate, "mechanism_confirmed": mechanism, "table": table}


if __name__ == "__main__":
    out = decide(sys.argv[1], sys.argv[2])
    for k, v in out["table"].items():
        print(k, v)
    print("gate_pass:", out["gate_pass"], "mechanism_confirmed:", out["mechanism_confirmed"])
    json.dump(out, open(os.path.join(sys.argv[1], "v17_rdev_verdict.json"), "w"), indent=2)
