"""V19 analysis (docs/PROJECT_STATE.md, "Pre-registration V19" and "V19 step 0/1/2"): frozen d2i-RI against official
Kwon et al. and K-prune on the 6-layer BERT-format DistilBERT-weights checkpoints, MAC budget 0.5, seven tasks
(MRPC, SST-2, STS-B, QNLI, MNLI-m, QQP, SQuAD v1.1), five calibration seeds. Copy of scripts/beat_confirm_analysis.py:
the task list, the metrics named in the pre-registration (accuracy; Pearson for STS-B; per-example F1 for SQuAD) and the
H19a/H19b rules below are the only differences.

Comparator for H19a: K-prune (fixed in the pre-registration, no development data). Paired cluster bootstrap over
(seed x example), 10,000 resamples, seed 1234 (+i per task, +10+i for the Kwon comparison), Holm over the 7 tasks.
  H19a (superiority): wins = tasks with 95% CI lower bound of (d2i-RI - K-prune) > 0 and Holm one-sided p < 0.05;
       supported iff wins >= 5 of 7 AND no Holm-significant loss (d2i-RI worse) against EITHER baseline on any task.
  H19b (non-inferiority, margin 1.5 points = 0.015 in accuracy / Pearson / F1 units, all on a 0-1 scale): supported iff on
       EVERY task the 95% CI lower bound of (d2i-RI - K-prune) > -0.015 AND the Holm-adjusted one-sided p for
       H0: delta <= -0.015 is < 0.05.
Before any statistic is computed, check_integrity() must pass for every (task, seed); a failure is a pipeline mismatch
(engineering failure, no verdict), never a result. Inputs per task: one kernel directory conf_<task>/ with
  beat/d2i_ri_<task>_seed<s>_c0.5.json      (scripts/beat_run.py)
  bl/kwon_<task>_seed<s>_c0.5/              (official code + v15 add-on dumps)
  bl/kprune_out/kprune_<task>_seed<s>_c0.5/ (same)
  bl/kprune_masks.npz                       (K-prune masks d2i-RI started from)

    python scripts/v19_analysis.py conf_mrpc conf_sst2 conf_stsb conf_qnli conf_mnli conf_qqp conf_squad
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from qfc.metrics import bootstrap_p_ge, bootstrap_p_le, cluster_bootstrap_delta, holm_adjust

SEEDS, C, ALPHA, MARGIN = (7, 42, 77, 123, 2024), 0.5, 0.05, 0.015
TASKS = ("mrpc", "sst2", "stsb", "qnli", "mnli", "qqp", "squad")
METRIC = {"stsb": "pearson", "squad": "score"}  # everything else: accuracy
WINS_NEEDED = 5  # >= 2/3 of the 7 tasks
TEACHER_ROWS_TOL = 2  # rows; teacher quality through beat_run vs the official pipeline


class PipelineMismatch(RuntimeError):
    pass


def _metric(task):
    return METRIC.get(task, "accuracy")


def _paths(root, task, s):
    return {"d2i_ri": os.path.join(root, "beat", f"d2i_ri_{task}_seed{s}_c{C}.json"),
            "kwon": os.path.join(root, "bl", f"kwon_{task}_seed{s}_c{C}"),
            "kprune": os.path.join(root, "bl", "kprune_out", f"kprune_{task}_seed{s}_c{C}")}


def _dump(d, which="student"):
    return torch.load(os.path.join(d, f"v15_{which}_validation.pt"), weights_only=True)


def find_roots(roots, tasks=TASKS):
    out = {}
    for t in tasks:
        hit = [r for r in roots if os.path.exists(_paths(r, t, SEEDS[0])["d2i_ri"])]
        if len(hit) != 1:
            raise PipelineMismatch(f"{t}: expected exactly one kernel directory with outputs, found {hit}")
        out[t] = hit[0]
    return out


def _pearson(x, y):
    return float(np.corrcoef(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64))[0, 1])


def check_integrity(root, task, s):
    """Row alignment and teacher/mask parity between our d2i-RI run and the official dumps; raises PipelineMismatch."""
    p = _paths(root, task, s)
    r = json.load(open(p["d2i_ri"]))
    ds, dk, dt = _dump(p["kprune"]), _dump(p["kwon"]), _dump(p["kprune"], "teacher")
    if task == "squad":
        n = len(ds["f1"])
        if not (len(dk["f1"]) == n and r["evaluation_size"] == n and len(r["valid_f1"]) == n):
            raise PipelineMismatch(f"{task} seed {s}: evaluation size differs between pipelines")
        if abs(float(np.mean(r["valid_f1"])) - r["valid"]["f1"]) > 1e-9:
            raise PipelineMismatch(f"{task} seed {s}: per-example F1 does not average to the recorded F1")
        t_off = float(dt["f1"].float().mean())
        if abs(t_off - r["unpruned_valid"]["f1"]) * n > TEACHER_ROWS_TOL:
            raise PipelineMismatch(f"{task} seed {s}: teacher F1 {r['unpruned_valid']['f1']} vs official {t_off}")
    else:
        labels = ds["labels"]
        if not (torch.equal(labels, dk["labels"]) and r["evaluation_size"] == len(labels)):
            raise PipelineMismatch(f"{task} seed {s}: evaluation size / labels differ between pipelines")
        lg = torch.tensor(r["valid_logits"])
        if task == "stsb":
            own = _pearson(lg[:, 0], labels.numpy())
            if abs(own - r["valid"]["pearson"]) > 1e-6:
                raise PipelineMismatch(f"{task} seed {s}: Pearson against dump labels {own} != own {r['valid']['pearson']} (row order)")
            t_off = _pearson(dt["logits"][:, 0], labels.numpy())
            if abs(t_off - r["unpruned_valid"]["pearson"]) > 1e-3:
                raise PipelineMismatch(f"{task} seed {s}: teacher Pearson {r['unpruned_valid']['pearson']} vs official {t_off}")
        else:
            acc = float((lg.argmax(-1) == labels).float().mean())
            if abs(acc - r["valid"]["accuracy"]) * len(labels) > 0.5:
                raise PipelineMismatch(f"{task} seed {s}: accuracy against dump labels {acc} != own {r['valid']['accuracy']} (row order)")
            t_off = float((dt["logits"].argmax(-1) == labels).float().mean())
            if abs(t_off - r["unpruned_valid"]["accuracy"]) * len(labels) > TEACHER_ROWS_TOL:
                raise PipelineMismatch(f"{task} seed {s}: teacher accuracy {r['unpruned_valid']['accuracy']} vs official {t_off}")
    z = np.load(os.path.join(root, "bl", "kprune_masks.npz"))
    for key, ours in (("head", r["heads_kept"]), ("neuron", r["neurons_kept"])):
        m = z[f"kprune_{task}_seed{s}_c{C}__{key}"]
        if any(sorted(ours[str(l)]) != np.flatnonzero(m[l]).tolist() for l in range(m.shape[0])):
            raise PipelineMismatch(f"{task} seed {s}: d2i-RI {key}s differ from the K-prune masks")


def load(root, method, task):
    """([S, N] per-example predictions (class / STS-B score / SQuAD F1), [N] labels from the official dump or None)."""
    rows, labels = [], None
    for s in SEEDS:
        p = _paths(root, task, s)
        if method == "d2i_ri":
            r = json.load(open(p["d2i_ri"]))
            rows.append(np.asarray(r["valid_f1"]) if task == "squad" else
                        np.asarray(r["valid_logits"])[:, 0] if task == "stsb" else np.asarray(r["valid_logits"]).argmax(-1))
        else:
            d = _dump(p[method])
            if task == "squad":
                rows.append(d["f1"].numpy().astype(np.float64))
            else:
                rows.append(d["logits"][:, 0].numpy() if task == "stsb" else d["logits"].argmax(-1).numpy())
                labels = d["labels"].numpy() if labels is None else labels
    return np.stack(rows), labels


def compare(root, a, b, task, n_boot, seed):
    pa, la = load(root, a, task)
    pb, lb = load(root, b, task)
    lab = la if la is not None else lb
    r = cluster_bootstrap_delta(pa, pb, lab if lab is not None else np.zeros(pa.shape[1]), _metric(task), n_boot, seed)
    return {"observed_delta": r["observed_delta"], "ci_low": r["ci_low"], "ci_high": r["ci_high"],
            "p_le_zero": bootstrap_p_le(r["boot"], 0.0), "p_ge_zero": bootstrap_p_ge(r["boot"], 0.0),
            "p_le_margin": bootstrap_p_le(r["boot"], -MARGIN)}


def decide(tests, tasks=TASKS):
    """tests = {'vs_kprune': {task: stats}, 'vs_kwon': {task: stats}}. Adds Holm-adjusted p-values; returns the
    H19a / H19b verdicts exactly as pre-registered."""
    tasks = list(tasks)
    for key in tests:
        for p in ("p_le_zero", "p_ge_zero", "p_le_margin"):
            if p in tests[key][tasks[0]]:
                for t, adj in zip(tasks, holm_adjust([tests[key][t][p] for t in tasks])):
                    tests[key][t]["holm_" + p] = adj
    kp = tests["vs_kprune"]
    wins = [t for t in tasks if kp[t]["ci_low"] > 0 and kp[t]["holm_p_le_zero"] < ALPHA]
    losses = [(k, t) for k in tests for t in tasks if tests[k][t]["holm_p_ge_zero"] < ALPHA]
    h19a = "supported" if len(wins) >= WINS_NEEDED and not losses else "not_supported"
    noninf = [t for t in tasks if kp[t]["ci_low"] > -MARGIN and kp[t]["holm_p_le_margin"] < ALPHA]
    h19b = "supported" if len(noninf) == len(tasks) else "not_supported"
    return {"wins": wins, "losses": losses, "H19a": h19a, "noninferior_tasks": noninf, "H19b": h19b}


def means(root_of):
    """Descriptive per-task means over the five seeds (the table in the paper; not a statistic)."""
    out = {}
    for t, root in root_of.items():
        row = {}
        lab = load(root, "kprune", t)[1]  # labels come from the official dump (d2i-RI's JSON stores none)
        for m in ("d2i_ri", "kprune", "kwon"):
            preds = load(root, m, t)[0]
            if t == "squad":
                row[m] = float(preds.mean())
            elif t == "stsb":
                row[m] = float(np.mean([_pearson(p, lab) for p in preds]))
            else:
                row[m] = float((preds == lab[None]).mean())
        r0 = json.load(open(_paths(root, t, SEEDS[0])["d2i_ri"]))
        row["unpruned"] = r0["unpruned_valid"][{"squad": "f1", "stsb": "pearson"}.get(t, "accuracy")]
        row["evaluation_size"] = r0["evaluation_size"]
        out[t] = row
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default="v19_verdict.json")
    ap.add_argument("--integrity-only", action="store_true", help="run the gate for the tasks present; no statistics")
    a = ap.parse_args()
    if a.integrity_only:
        for r in a.roots:
            for t in TASKS:
                if os.path.exists(_paths(r, t, SEEDS[0])["d2i_ri"]):
                    for s in SEEDS:
                        check_integrity(r, t, s)
                    print(t, "integrity OK for", len(SEEDS), "seeds")
        return
    root_of = find_roots(a.roots)
    for t in TASKS:
        for s in SEEDS:
            check_integrity(root_of[t], t, s)
    tests = {"vs_kprune": {}, "vs_kwon": {}}
    for i, t in enumerate(TASKS):
        tests["vs_kprune"][t] = compare(root_of[t], "d2i_ri", "kprune", t, a.n_boot, a.seed + i)
        tests["vs_kwon"][t] = compare(root_of[t], "d2i_ri", "kwon", t, a.n_boot, a.seed + 10 + i)
    verdict = decide(tests)
    out = {"comparator": "kprune", "margin": MARGIN, "tests": tests, "means": means(root_of), **verdict}
    json.dump(out, open(a.out, "w"), indent=2)
    print(json.dumps({k: out[k] for k in ("wins", "losses", "H19a", "noninferior_tasks", "H19b")}, indent=2))


if __name__ == "__main__":
    main()
