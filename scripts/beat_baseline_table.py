"""Accuracy bar for the beat-the-baselines mission: official Kwon / K-prune validation accuracy per (task, budget, seed)
from the V15 H10 logit dumps (results/v15_h10/c43dd3e/h10_logits, archive sha256 in ../run_meta.json).

    python scripts/beat_baseline_table.py
"""
import glob
import json
import os
import re

import torch

ROOT = "results/v15_h10/c43dd3e/h10_logits"
OUT = "results/beat_explore/baselines_v15.json"
acc = {}
kl = {}
teacher = {}
for f in sorted(glob.glob(ROOT + "/**/v15_*_validation.pt", recursive=True)):
    p = f.replace("\\", "/")
    m = re.search(r"(kwon|kprune)_(mrpc|sst2)_seed(\d+)_c([\d.]+)/v15_(student|teacher)_validation", p)
    d = torch.load(f, weights_only=True)
    a = float((d["logits"].argmax(-1) == d["labels"]).float().mean())
    if m.group(5) == "teacher":
        teacher[m.group(2)] = a  # same pinned checkpoint for every run; identical across files
    else:
        key, seed = f"{m.group(1)}|{m.group(2)}|{m.group(4)}", int(m.group(3))
        acc.setdefault(key, {})[seed] = a
        lt = torch.load(f.replace("student_validation", "teacher_heldout"), weights_only=True)["logits"].log_softmax(-1)
        ls = torch.load(f.replace("student_validation", "student_heldout"), weights_only=True)["logits"].log_softmax(-1)
        kl.setdefault(key, {})[seed] = float((lt.exp() * (lt - ls)).sum(-1).mean())
res = {"source": ROOT, "teacher_accuracy": teacher, "runs": {}}
for k, v in sorted(acc.items()):
    seeds = sorted(v)
    res["runs"][k] = {"seeds": seeds, "acc": [v[s] for s in seeds], "mean": sum(v.values()) / len(v),
                      "heldout_kl": [kl[k][s] for s in seeds], "heldout_kl_mean": sum(kl[k].values()) / len(seeds)}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(res, open(OUT, "w"), indent=1)
for k, r in res["runs"].items():
    print(k, len(r["acc"]), round(r["mean"], 4))
print("teacher", teacher)
