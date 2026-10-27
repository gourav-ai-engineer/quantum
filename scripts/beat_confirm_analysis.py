"""V18 confirmatory analysis (docs/PROJECT_STATE.md, "Pre-registration V18"): d2i-RI minus the better of {Kwon et al.,
K-prune} at MAC budget 0.5 on QNLI, MNLI-m, QQP. "Better" is fixed from the exploration baselines file
(results/beat_explore/baselines_v15.json, MRPC + SST-2 mean accuracy at 0.5), never from confirmatory data. Paired
cluster bootstrap over (seed x example), 10,000 resamples, Holm over the 3 tasks. Supported if Holm p < 0.05 and the
95% CI lower bound > 0 on >= 2 of 3 tasks and no Holm-significant loss against EITHER baseline on any task.

Inputs: one directory per kernel as the kernels write it (conf_<task>/), each holding
  beat/d2i_ri_<task>_seed<s>_c0.5.json         (scripts/beat_run.py; keys valid_logits, valid, unpruned_valid, ...)
  bl/kwon_<task>_seed<s>_c0.5/                 (official code + v15 add-on: v15_{student,teacher}_validation.pt)
  bl/kprune_out/kprune_<task>_seed<s>_c0.5/    (same)
  bl/kprune_masks.npz                          (the K-prune masks d2i-RI started from)
Before any statistic is computed, check_integrity() must pass for every (task, seed); a failure is a pipeline mismatch
(engineering failure, no verdict), never a result.

    python scripts/beat_confirm_analysis.py conf_qnli conf_mnli conf_qqp
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
TEACHER_ROWS_TOL = 2  # rows; teacher accuracy through beat_run vs the official pipeline (fp16/batching noise)


class PipelineMismatch(RuntimeError):
    pass


def better_baseline(path=BASELINES):
    runs = json.load(open(path))["runs"]
    m = {b: float(np.mean([runs[f"{b}|{t}|{C}"]["mean"] for t in ("mrpc", "sst2")])) for b in ("kwon", "kprune")}
    return max(m, key=m.get), m


def _paths(root, task, s):
    return {"d2i_ri": os.path.join(root, "beat", f"d2i_ri_{task}_seed{s}_c{C}.json"),
            "kwon": os.path.join(root, "bl", f"kwon_{task}_seed{s}_c{C}"),
            "kprune": os.path.join(root, "bl", "kprune_out", f"kprune_{task}_seed{s}_c{C}")}


def _dump(d, which="student"):
    return torch.load(os.path.join(d, f"v15_{which}_validation.pt"), weights_only=True)


def find_roots(roots):
    """task -> kernel directory containing that task's d2i-RI outputs."""
    out = {}
    for t in CONF:
        hit = [r for r in roots if os.path.exists(_paths(r, t, SEEDS[0])["d2i_ri"])]
        if len(hit) != 1:
            raise PipelineMismatch(f"{t}: expected exactly one kernel directory with outputs, found {hit}")
        out[t] = hit[0]
    return out


def check_integrity(root, task, s):
    """Row alignment and teacher/mask parity between our d2i-RI run and the official dumps; raises PipelineMismatch."""
    p = _paths(root, task, s)
    r = json.load(open(p["d2i_ri"]))
    ds, dk, dt = _dump(p["kprune"]), _dump(p["kwon"]), _dump(p["kprune"], "teacher")
    labels = ds["labels"]
    if not (torch.equal(labels, dk["labels"]) and r["evaluation_size"] == len(labels)):
        raise PipelineMismatch(f"{task} seed {s}: evaluation size / labels differ between pipelines")
    acc = float((torch.tensor(r["valid_logits"]).argmax(-1) == labels).float().mean())
    if abs(acc - r["valid"]["accuracy"]) * len(labels) > 0.5:  # same number of correct rows (float32 mean noise ok)
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
    """([S, N] predicted classes, [N] labels from the official dump)."""
    rows, labels = [], None
    for s in SEEDS:
        p = _paths(root, task, s)
        if method == "d2i_ri":
            rows.append(np.asarray(json.load(open(p["d2i_ri"]))["valid_logits"]).argmax(-1))
        else:
            d = _dump(p[method])
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
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default="beat_confirm_verdict.json")
    ap.add_argument("--integrity-only", action="store_true", help="run the gate for the tasks present; no statistics")
    a = ap.parse_args()
    if a.integrity_only:
        for r in a.roots:
            for t in CONF:
                if os.path.exists(_paths(r, t, SEEDS[0])["d2i_ri"]):
                    for s in SEEDS:
                        check_integrity(r, t, s)
                    print(t, "integrity OK for", len(SEEDS), "seeds")
        return
    root_of = find_roots(a.roots)
    for t in CONF:
        for s in SEEDS:
            check_integrity(root_of[t], t, s)
    best, dev_mean = better_baseline()
    other = "kprune" if best == "kwon" else "kwon"
    tests = {"vs_best": {}, "vs_other": {}}
    for i, t in enumerate(CONF):
        tests["vs_best"][t] = compare(root_of[t], "d2i_ri", best, t, a.n_boot, a.seed + i)
        tests["vs_other"][t] = compare(root_of[t], "d2i_ri", other, t, a.n_boot, a.seed + 10 + i)
    wins, losses, verdict = decide(tests)
    out = {"dev_mean_accuracy": dev_mean, "best_baseline": best, "tests": tests, "wins": wins, "losses": losses,
           "verdict": verdict}
    json.dump(out, open(a.out, "w"), indent=2)
    print(json.dumps({k: out[k] for k in ("dev_mean_accuracy", "best_baseline", "wins", "losses", "verdict")}, indent=2))


if __name__ == "__main__":
    main()
