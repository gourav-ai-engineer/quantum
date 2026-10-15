"""V15 H11 (docs/PROJECT_STATE.md, primary test, run only if H10 is supported).

At FLOPs constraint 0.5: GRP minus the better of {Kwon et al., K-prune}, where "better" is chosen ONCE from the
development tasks (MRPC, SST-2; mean validation accuracy at 0.5) and then fixed. Supported if Holm-adjusted p < 0.05
and 95% CI lower bound > 0 on >= 2 of 3 confirmatory tasks (QNLI, MNLI-m, SQuAD v1.1) and no Holm-significant loss
against EITHER baseline on any confirmatory task. Paired cluster bootstrap over (seed x example), 10,000 resamples.

Inputs under <root> (from the H11 runs): grp/v15_grp_<task>_seed<s>_c0.5.json (step22 --mode accuracy; key
valid_pred, or valid_f1 for SQuAD), kwon_<task>_seed<s>_c0.5/v15_student_validation.pt and
kprune_out/kprune_<task>_seed<s>_c0.5/v15_student_validation.pt (official codes + v15 add-on; key logits, or f1).

    python scripts/v15_h11_analysis.py /kaggle/working/v15/h11
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from qfc.metrics import bootstrap_p_ge, bootstrap_p_le, cluster_bootstrap_delta, holm_adjust

SEEDS, C = (7, 42, 77, 123, 2024), 0.5
DEV, CONF = ("mrpc", "sst2"), ("qnli", "mnli", "squad")
ALPHA = 0.05


def load(root, method, task):
    """[S, N] per-example values (predicted class, or F1 for SQuAD) and the labels [N] (None for SQuAD)."""
    rows, labels = [], None
    for s in SEEDS:
        if method == "grp":
            r = json.load(open(os.path.join(root, "grp", f"v15_grp_{task}_seed{s}_c{C}.json")))
            rows.append(np.asarray(r["valid_f1"] if task == "squad" else r["valid_pred"]))
        else:
            sub = (f"kwon_{task}_seed{s}_c{C}" if method == "kwon"
                   else os.path.join("kprune_out", f"kprune_{task}_seed{s}_c{C}"))
            d = torch.load(os.path.join(root, sub, "v15_student_validation.pt"), weights_only=True)
            if task == "squad":
                rows.append(np.asarray(d["f1"]))
            else:
                rows.append(d["logits"].argmax(-1).numpy())
                labels = d["labels"].numpy() if labels is None else labels
    return np.stack(rows), labels


def compare(root, a, b, task, n_boot, seed):
    pa, lab_a = load(root, a, task)
    pb, lab_b = load(root, b, task)
    lab = lab_a if lab_a is not None else lab_b
    metric = "score" if task == "squad" else "accuracy"
    r = cluster_bootstrap_delta(pa, pb, lab if lab is not None else np.zeros(pa.shape[1]), metric, n_boot, seed)
    return {"observed_delta": r["observed_delta"], "ci_low": r["ci_low"], "ci_high": r["ci_high"],
            "p_le_zero": bootstrap_p_le(r["boot"], 0.0), "p_ge_zero": bootstrap_p_ge(r["boot"], 0.0)}


def dev_accuracy(root, method):
    accs = []
    for t in DEV:
        preds, labels = load(root, method, t)
        accs.append(float((preds == labels[None]).mean()))
    return float(np.mean(accs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()
    dev_mean = {m: dev_accuracy(a.root, m) for m in ("kwon", "kprune")}
    best = max(dev_mean, key=dev_mean.get)
    other = "kprune" if best == "kwon" else "kwon"
    tests = {"vs_best": {}, "vs_other": {}}
    for i, t in enumerate(CONF):
        tests["vs_best"][t] = compare(a.root, "grp", best, t, a.n_boot, a.seed + i)
        tests["vs_other"][t] = compare(a.root, "grp", other, t, a.n_boot, a.seed + 10 + i)
    for key in tests:
        for p in ("p_le_zero", "p_ge_zero"):
            for t, adj in zip(CONF, holm_adjust([tests[key][t][p] for t in CONF])):
                tests[key][t]["holm_" + p] = adj
    wins = [t for t in CONF if tests["vs_best"][t]["ci_low"] > 0 and tests["vs_best"][t]["holm_p_le_zero"] < ALPHA]
    losses = [(k, t) for k in tests for t in CONF if tests[k][t]["holm_p_ge_zero"] < ALPHA]
    out = {"dev_mean_accuracy": dev_mean, "best_baseline": best, "tests": tests, "wins": wins, "losses": losses,
           "verdict": "supported" if len(wins) >= 2 and not losses else "not_supported"}
    json.dump(out, open(os.path.join(a.root, "v15_h11_verdict.json"), "w"), indent=2)
    print(json.dumps({k: out[k] for k in ("dev_mean_accuracy", "best_baseline", "wins", "losses", "verdict")}, indent=2))


if __name__ == "__main__":
    main()
