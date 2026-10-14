"""V15 H10 gate (docs/PROJECT_STATE.md): GRP's mean held-out-TRAIN logit KL(teacher || pruned) must be at least 10%
lower than BOTH Kwon et al. and K-prune at >= 2 of the 4 constraints, on BOTH MRPC and SST-2 (means over 5 seeds).

Inputs (from scripts/v15_kaggle.sh h10): <root>/kwon_<task>_seed<s>_c<c>/v15_{teacher,student}_heldout.pt,
<root>/kprune_out/kprune_<task>_seed<s>_c<c>/v15_{teacher,student}_heldout.pt, <root>/grp/v15_grp_<task>_seed<s>_c<c>.json.
Writes <root>/v15_h10_verdict.json.

    python scripts/v15_h10_analysis.py /kaggle/working/v15/h10
"""
from __future__ import annotations

import argparse
import json
import os

import torch

TASKS, SEEDS, CONS = ("mrpc", "sst2"), (7, 42, 77, 123, 2024), (0.7, 0.6, 0.5, 0.4)
MARGIN, MIN_CONS = 0.10, 2


def kl_from_dumps(d):
    t = torch.load(os.path.join(d, "v15_teacher_heldout.pt"), weights_only=True)
    s = torch.load(os.path.join(d, "v15_student_heldout.pt"), weights_only=True)
    assert t["idx"] == s["idx"]
    pt = t["logits"].double().log_softmax(-1)
    return float((pt.exp() * (pt - s["logits"].double().log_softmax(-1))).sum(-1).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    a = ap.parse_args()
    paths = {"kwon": lambda t, s, c: os.path.join(a.root, f"kwon_{t}_seed{s}_c{c}"),
             "kprune": lambda t, s, c: os.path.join(a.root, "kprune_out", f"kprune_{t}_seed{s}_c{c}")}
    table, missing = {}, []
    for t in TASKS:
        for c in CONS:
            row = {}
            for m in ("kwon", "kprune", "grp", "no_sens"):
                vals = []
                for s in SEEDS:
                    try:
                        if m in paths:
                            vals.append(kl_from_dumps(paths[m](t, s, c)))
                        else:
                            f = os.path.join(a.root, "grp", f"v15_{m}_{t}_seed{s}_c{c}.json")
                            vals.append(json.load(open(f))["heldout_kl"])
                    except FileNotFoundError:
                        missing.append((m, t, s, c))
                row[m] = {"mean_kl": sum(vals) / len(vals) if vals else None, "n_seeds": len(vals)}
            table[f"{t}_c{c}"] = row
    complete = not [x for x in missing if x[0] != "no_sens"]
    out = {"table": table, "missing": missing, "complete": complete}
    if complete:
        per_task = {t: [c for c in CONS if all(table[f"{t}_c{c}"]["grp"]["mean_kl"] <=
                                               (1 - MARGIN) * table[f"{t}_c{c}"][b]["mean_kl"]
                                               for b in ("kwon", "kprune"))] for t in TASKS}
        out["constraints_won"] = per_task
        out["verdict"] = "supported" if all(len(w) >= MIN_CONS for w in per_task.values()) else "not_supported"
    else:
        out["verdict"] = "insufficient_data"
    json.dump(out, open(os.path.join(a.root, "v15_h10_verdict.json"), "w"), indent=2)
    for k, row in table.items():
        print(k, {m: (None if r["mean_kl"] is None else round(r["mean_kl"], 4)) for m, r in row.items()})
    print("verdict:", out["verdict"], out.get("constraints_won"))


if __name__ == "__main__":
    main()
