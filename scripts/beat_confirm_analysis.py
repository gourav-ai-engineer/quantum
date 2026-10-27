"""V18 confirmatory analysis (docs/PROJECT_STATE.md, "Pre-registration V18"): d2i-RI minus the better of {Kwon et al.,
K-prune} at MAC budget 0.5 on QNLI, MNLI-m, QQP. "Better" is fixed from the exploration baselines file
(results/beat_explore/baselines_v15.json, MRPC + SST-2 mean accuracy at 0.5), never from confirmatory data. Paired
cluster bootstrap over (seed x example), 10,000 resamples, Holm over the 3 tasks. Supported if Holm p < 0.05 and the
95% CI lower bound > 0 on >= 2 of 3 tasks and no Holm-significant loss against EITHER baseline on any task.

Inputs under <root>: d2i_ri/d2i_ri_<task>_seed<s>_c0.5.json (beat_run.py, key valid_logits), and the official-code dumps
kwon_<task>_seed<s>_c0.5/v15_student_validation.pt, kprune_out/kprune_<task>_seed<s>_c0.5/v15_student_validation.pt.

    python scripts/beat_confirm_analysis.py <root>
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from qfc.metrics import bootstrap_p_ge, bootstrap_p_le, cluster_bootstrap_delta, holm_adjust

SEEDS, C, CONF, ALPHA = (7, 42, 77, 123, 2024), 0.5, ("qnli", "mnli", "qqp"), 0.05
BASELINES = "results/beat_explore/baselines_v15.json"


def better_baseline(path=BASELINES):
    runs = json.load(open(path))["runs"]
    m = {b: float(np.mean([runs[f"{b}|{t}|{C}"]["mean"] for t in ("mrpc", "sst2")])) for b in ("kwon", "kprune")}
    return max(m, key=m.get), m


def load(root, method, task):
    """([S, N] predicted classes, [N] labels or None for d2i_ri)."""
    rows, labels = [], None
    for s in SEEDS:
        if method == "d2i_ri":
            r = json.load(open(os.path.join(root, "d2i_ri", f"d2i_ri_{task}_seed{s}_c{C}.json")))
            rows.append(np.asarray(r["valid_logits"]).argmax(-1))
        else:
            sub = f"kwon_{task}_seed{s}_c{C}" if method == "kwon" else os.path.join("kprune_out", f"kprune_{task}_seed{s}_c{C}")
            d = torch.load(os.path.join(root, sub, "v15_student_validation.pt"), weights_only=True)
            rows.append(d["logits"].argmax(-1).numpy())
            labels = d["labels"].numpy() if labels is None else labels
    return np.stack(rows), labels


def compare(root, a, b, task, n_boot, seed):
    pa, _ = load(root, a, task)
    pb, lab = load(root, b, task)
    r = cluster_bootstrap_delta(pa, pb, lab, "accuracy", n_boot, seed)
    return {"observed_delta": r["observed_delta"], "ci_low": r["ci_low"], "ci_high": r["ci_high"],
            "p_le_zero": bootstrap_p_le(r["boot"], 0.0), "p_ge_zero": bootstrap_p_ge(r["boot"], 0.0)}


def decide(tests):
    """tests = {'vs_best': {task: stats}, 'vs_other': {task: stats}} with raw p-values. Adds Holm p, returns the verdict."""
    tasks = list(tests["vs_best"])
    for key in tests:
        for p in ("p_le_zero", "p_ge_zero"):
            for t, adj in zip(tasks, holm_adjust([tests[key][t][p] for t in tasks])):
                tests[key][t]["holm_" + p] = adj
    wins = [t for t in tasks if tests["vs_best"][t]["ci_low"] > 0 and tests["vs_best"][t]["holm_p_le_zero"] < ALPHA]
    losses = [(k, t) for k in tests for t in tasks if tests[k][t]["holm_p_ge_zero"] < ALPHA]
    return wins, losses, "supported" if len(wins) >= 2 and not losses else "not_supported"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()
    best, dev_mean = better_baseline()
    other = "kprune" if best == "kwon" else "kwon"
    tests = {"vs_best": {}, "vs_other": {}}
    for i, t in enumerate(CONF):
        tests["vs_best"][t] = compare(a.root, "d2i_ri", best, t, a.n_boot, a.seed + i)
        tests["vs_other"][t] = compare(a.root, "d2i_ri", other, t, a.n_boot, a.seed + 10 + i)
    wins, losses, verdict = decide(tests)
    out = {"dev_mean_accuracy": dev_mean, "best_baseline": best, "tests": tests, "wins": wins, "losses": losses,
           "verdict": verdict}
    json.dump(out, open(os.path.join(a.root, "beat_confirm_verdict.json"), "w"), indent=2)
    print(json.dumps({k: out[k] for k in ("dev_mean_accuracy", "best_baseline", "wins", "losses", "verdict")}, indent=2))


if __name__ == "__main__":
    main()
