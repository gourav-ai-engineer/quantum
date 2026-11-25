"""E7 high-compression development kill test (docs/EXPLORATION_LOG.md E7; design = docs/STRATEGY_AUDIT_2026-11-19.md section 3.3,
fixed 2026-11-19 before any data). Committed before the kernels run.

Question: once headroom exists (MAC budgets 0.3, 0.2), does an allocation different from K-prune's (GRP-RI) beat official K-prune?
Per budget c (all must hold; metric = accuracy, Pearson for STS-B; seeds 7,42,77,123,2024):
  H  teacher - mean(K-prune) >= 0.03 on all three tasks (MRPC, SST-2, STS-B)
  W  mean(GRP-RI) - mean(K-prune) >= 0.010 on all three tasks
  S  GRP-RI > K-prune on >= 4 of 5 seeds on every task
  L  no task with mean(GRP-RI) < mean(K-prune)   (implied by W; checked anyway)
PASS if some budget passes all four, else FAIL = stop the allocation path (this family, these budgets/tasks/thresholds only).
Descriptive only: Kwon, d2i-RI (K-prune masks + our repair: repair headroom, never a compression win), MAC ratios.
Integrity gate (raises before any statistic): labels / evaluation size equal across pipelines; teacher metric of our pipeline vs
the official dump within 2 rows (accuracy) or 0.002 (Pearson); d2i-RI masks equal K-prune's; measured MAC ratios <= c + 0.005.

    python scripts/e7_highcomp_analysis.py <kernel_dir_mrpc> <kernel_dir_sst2> <kernel_dir_stsb> --out e7_verdict.json
"""
import argparse
import json
import os

import numpy as np
import torch
from transformers import BertConfig

from beat_run import mac_ratio, valid_metrics
from qfc.grp import AVG_SEQ_LEN

TASKS, BUDGETS, SEEDS = ("mrpc", "sst2", "stsb"), (0.3, 0.2), (7, 42, 77, 123, 2024)
HEADROOM, WIN, SEED_WINS = 0.03, 0.010, 4


class PipelineMismatch(RuntimeError):
    pass


def metric(task, logits, labels):
    return valid_metrics(task, logits, labels)["pearson" if task == "stsb" else "accuracy"]


def paths(root, task, s, c):
    return {"grp_ri": os.path.join(root, "beat", f"grp_ri_{task}_seed{s}_c{c}.json"),
            "d2i_ri": os.path.join(root, "beat", f"d2i_ri_{task}_seed{s}_c{c}.json"),
            "kwon": os.path.join(root, "bl", f"kwon_{task}_seed{s}_c{c}"),
            "kprune": os.path.join(root, "bl", "kprune_out", f"kprune_{task}_seed{s}_c{c}")}


def dump(d, which="student"):
    return torch.load(os.path.join(d, f"v15_{which}_validation.pt"), weights_only=True)


def unit(root, task, s, c):
    """All metrics for one (task, seed, budget), after the integrity checks."""
    p = paths(root, task, s, c)
    ours = {m: json.load(open(p[m])) for m in ("grp_ri", "d2i_ri")}
    dk, dw, dt = dump(p["kprune"]), dump(p["kwon"]), dump(p["kprune"], "teacher")
    lab = dk["labels"]
    if not torch.equal(lab, dw["labels"]) or any(r["evaluation_size"] != len(lab) for r in ours.values()):
        raise PipelineMismatch(f"{task} s{s} c{c}: labels / evaluation size differ")
    key = "pearson" if task == "stsb" else "accuracy"
    t_ours, t_off = ours["grp_ri"]["unpruned_valid"][key], metric(task, dt["logits"], lab)
    if abs(t_ours - t_off) > (0.002 if task == "stsb" else 2 / len(lab)):
        raise PipelineMismatch(f"{task} s{s} c{c}: teacher {t_ours} (ours) vs {t_off} (official)")
    z = np.load(os.path.join(root, "bl", "kprune_masks.npz"))
    hm, nm = z[f"kprune_{task}_seed{s}_c{c}__head"], z[f"kprune_{task}_seed{s}_c{c}__neuron"]
    kh = {l: np.flatnonzero(hm[l]).tolist() for l in range(hm.shape[0])}
    kn = {l: np.flatnonzero(nm[l]).tolist() for l in range(nm.shape[0])}
    if any(sorted(ours["d2i_ri"]["heads_kept"][str(l)]) != kh[l] or sorted(ours["d2i_ri"]["neurons_kept"][str(l)]) != kn[l]
           for l in kh):
        raise PipelineMismatch(f"{task} s{s} c{c}: d2i-RI masks differ from K-prune's")
    macs = {"kprune": mac_ratio(kh, kn, BertConfig(), AVG_SEQ_LEN[task]), "grp_ri": ours["grp_ri"]["mac_ratio"]}
    if max(macs.values()) > c + 0.005:
        raise PipelineMismatch(f"{task} s{s} c{c}: MAC ratio over budget {macs}")
    for r in ours.values():  # our score must equal a recomputation from the saved logits (row order)
        if abs(metric(task, torch.tensor(r["valid_logits"]), lab) - r["valid"][key]) > 1e-6:
            raise PipelineMismatch(f"{task} s{s} c{c}: {r['method']} metric vs dump labels")
    return {"teacher": t_ours, "kprune": metric(task, dk["logits"], lab), "kwon": metric(task, dw["logits"], lab),
            "grp_ri": ours["grp_ri"]["valid"][key], "d2i_ri": ours["d2i_ri"]["valid"][key], "mac": macs}


def analyse(roots):
    root_of = {}
    for t in TASKS:
        hit = [r for r in roots if os.path.exists(paths(r, t, SEEDS[0], BUDGETS[0])["grp_ri"])]
        if len(hit) != 1:
            raise PipelineMismatch(f"{t}: expected one kernel directory, found {hit}")
        root_of[t] = hit[0]
    units = {(t, c, s): unit(root_of[t], t, s, c) for t in TASKS for c in BUDGETS for s in SEEDS}  # gate first
    out = {"budgets": {}}
    for c in BUDGETS:
        rows = {}
        for t in TASKS:
            u = [units[(t, c, s)] for s in SEEDS]
            m = {k: float(np.mean([x[k] for x in u])) for k in ("teacher", "kprune", "kwon", "grp_ri", "d2i_ri")}
            rows[t] = {"mean": m, "headroom": m["teacher"] - m["kprune"], "delta_grp_ri": m["grp_ri"] - m["kprune"],
                       "delta_d2i_ri": m["d2i_ri"] - m["kprune"],
                       "seed_wins_grp_ri": sum(x["grp_ri"] > x["kprune"] for x in u),
                       "per_seed": {str(s): x for s, x in zip(SEEDS, u)}}
        checks = {"H": all(r["headroom"] >= HEADROOM for r in rows.values()),
                  "W": all(r["delta_grp_ri"] >= WIN for r in rows.values()),
                  "S": all(r["seed_wins_grp_ri"] >= SEED_WINS for r in rows.values()),
                  "L": all(r["delta_grp_ri"] >= 0 for r in rows.values())}
        out["budgets"][str(c)] = {"tasks": rows, "checks": checks, "pass": all(checks.values())}
    out["verdict"] = "PASS" if any(b["pass"] for b in out["budgets"].values()) else "FAIL"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    r = analyse(a.roots)
    json.dump(r, open(a.out, "w"), indent=1)
    for c, b in r["budgets"].items():
        print(f"c={c} checks={b['checks']}")
        for t, x in b["tasks"].items():
            print(f"  {t:5s} teacher {x['mean']['teacher']:.4f} kprune {x['mean']['kprune']:.4f} kwon {x['mean']['kwon']:.4f} "
                  f"grp_ri {x['mean']['grp_ri']:.4f} (d {x['delta_grp_ri']:+.4f}, wins {x['seed_wins_grp_ri']}/5) "
                  f"d2i_ri d {x['delta_d2i_ri']:+.4f}")
    print("VERDICT", r["verdict"])


if __name__ == "__main__":
    main()
