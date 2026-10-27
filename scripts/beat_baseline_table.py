"""Accuracy bar for the beat-the-baselines mission: official Kwon / K-prune validation accuracy per (task, budget, seed)
from the V15 H10 logit dumps (results/v15_h10/c43dd3e/h10_logits, archive sha256 in ../run_meta.json).

    python scripts/beat_baseline_table.py
"""
import glob
import json
import os
import re

import torch

ROOTS = ["results/v15_h10/c43dd3e/h10_logits", "results/beat_explore/C_stsb_baselines/036e44c/logits"]
OUT = "results/beat_explore/baselines_v15.json"
acc = {}  # primary dev metric: accuracy, or Pearson for STS-B
kl = {}  # held-out distance to the teacher: KL, or MSE for STS-B
teacher = {}


def pearson(x, y):
    return float(torch.corrcoef(torch.stack([x[:, 0].double(), y.double()]))[0, 1])


files = [f for r in ROOTS for f in sorted(glob.glob(r + "/**/v15_*_validation.pt", recursive=True))]
for f in files:
    p = f.replace("\\", "/")
    m = re.search(r"(kwon|kprune)_(mrpc|sst2|stsb)_seed(\d+)_c([\d.]+)/v15_(student|teacher)_validation", p)
    d = torch.load(f, weights_only=True)
    reg = m.group(2) == "stsb"
    a = pearson(d["logits"], d["labels"]) if reg else float((d["logits"].argmax(-1) == d["labels"]).float().mean())
    if m.group(5) == "teacher":
        teacher[m.group(2)] = a  # same pinned checkpoint for every run; identical across files
    else:
        key, seed = f"{m.group(1)}|{m.group(2)}|{m.group(4)}", int(m.group(3))
        acc.setdefault(key, {})[seed] = a
        lt = torch.load(f.replace("student_validation", "teacher_heldout"), weights_only=True)["logits"]
        ls = torch.load(f.replace("student_validation", "student_heldout"), weights_only=True)["logits"]
        if reg:
            kl.setdefault(key, {})[seed] = float(((lt - ls) ** 2).mean())
        else:
            lt, ls = lt.log_softmax(-1), ls.log_softmax(-1)
            kl.setdefault(key, {})[seed] = float((lt.exp() * (lt - ls)).sum(-1).mean())
res = {"source": ROOTS, "teacher_accuracy": teacher,
       "note": "acc = accuracy (Pearson for stsb); heldout_kl = KL (MSE for stsb)", "runs": {}}
for k, v in sorted(acc.items()):
    seeds = sorted(v)
    res["runs"][k] = {"seeds": seeds, "acc": [v[s] for s in seeds], "mean": sum(v.values()) / len(v),
                      "heldout_kl": [kl[k][s] for s in seeds], "heldout_kl_mean": sum(kl[k].values()) / len(seeds)}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(res, open(OUT, "w"), indent=1)
for k, r in res["runs"].items():
    print(k, len(r["acc"]), round(r["mean"], 4))
print("teacher", teacher)
