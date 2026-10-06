"""One-off diagnostic (no selector changed): why is VonNeumann keep-high identical across calibration
seeds on MRPC (Jaccard 1.0) but not on SST-2 (0.92)?

For each task and calibration seed it computes the VNE scores of the mean density states, reports the
per-layer score gap at the top-k boundary, and the selection overlap (Jaccard) between conditions:
real text, word-shuffled text, and text from the other task. Engineering diagnostic, not a result.

    python scripts/diag_vne_seed_stability.py --out results/diag_vne/vne_seed_stability.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
from itertools import combinations

import torch
from datasets import Dataset, load_dataset

from qfc.hf_experiments import collect_mean_density_states, load_sequence_classifier, make_text_loader
from step13_mrpc_stability import select_topk, vn_scores
from step18_v10_confirmatory import SPECS


def spearmanr(x, y):
    rx, ry = x.argsort().argsort().double(), y.argsort().argsort().double()
    rx, ry = rx - rx.mean(), ry - ry.mean()
    return [(rx * ry).sum() / (rx.norm() * ry.norm())]


def jaccard(a, b):
    A = {(l, h) for l, hs in a.items() for h in hs}
    B = {(l, h) for l, hs in b.items() for h in hs}
    return len(A & B) / len(A | B)


def shuffled(rows, fields, seed):
    rng = random.Random(seed)
    out = {f: [] for f in fields}
    for r in rows:
        for f in fields:
            w = r[f].split()
            rng.shuffle(w)
            out[f].append(" ".join(w))
    out["label"] = [0] * len(rows)  # loader needs a label column; VNE scores never use it
    return Dataset.from_dict(out)


def other_task_rows(rows_src, fields, n, seed):
    """Pair sentences of another task into the same [CLS] s1 [SEP] s2 [SEP] layout."""
    src = rows_src.shuffle(seed=seed).select(range(2 * n))
    key = "sentence" if "sentence" in src.column_names else "sentence1"
    texts = src[key]
    if len(fields) == 2:
        return Dataset.from_dict({fields[0]: texts[:n], fields[1]: texts[n:], "label": [0] * n})
    return Dataset.from_dict({fields[0]: texts[:n], "label": [0] * n})


def scores_for(model, tok, rows, fields, device, bs, max_len):
    loader = make_text_loader(rows, tok, text_fields=fields, batch_size=bs, max_length=max_len)
    return vn_scores(collect_mean_density_states(model, loader, device))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", default="mrpc,sst2")
    p.add_argument("--seeds", default="7,42,77")
    p.add_argument("--calibration-size", type=int, default=128)
    p.add_argument("--k", type=int, default=6)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default="results/diag_vne/vne_seed_stability.json")
    a = p.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    data = {"mrpc": load_dataset("glue", "mrpc")["train"], "sst2": load_dataset("stanfordnlp/sst2")["train"]}
    report = {}
    for task in a.tasks.split(","):
        spec = SPECS[task]
        fields = spec["text_fields"]
        model, tok = load_sequence_classifier(spec["model_id"], a.device, revision=spec["revision"])
        other = "sst2" if task == "mrpc" else "mrpc"
        sel = {c: {} for c in ("real", "shuffled", "other_task")}
        scores = {c: {} for c in sel}
        for s in seeds:
            cal = data[task].shuffle(seed=s).select(range(a.calibration_size))
            conds = {
                "real": cal,
                "shuffled": shuffled(cal, fields, s),
                "other_task": other_task_rows(data[other], fields, a.calibration_size, s),
            }
            for c, rows in conds.items():
                sc = scores_for(model, tok, rows, fields, a.device, 16, 128)
                scores[c][s] = sc
                sel[c][s] = select_topk(sc, a.k)
        s0 = seeds[0]
        gaps, rho = [], []
        for layer in range(len(scores["real"][s0])):
            v = torch.sort(scores["real"][s0][layer], descending=True).values
            gaps.append({"layer": layer, "gap_k_to_k1": (v[a.k - 1] - v[a.k]).item(), "spread": (v[0] - v[-1]).item()})
            rho.append(float(spearmanr(scores["real"][s0][layer], scores["other_task"][s0][layer])[0]))
        report[task] = {
            "seed_to_seed_jaccard_real": [jaccard(sel["real"][x], sel["real"][y]) for x, y in combinations(seeds, 2)],
            "seed_to_seed_jaccard_shuffled": [jaccard(sel["shuffled"][x], sel["shuffled"][y]) for x, y in combinations(seeds, 2)],
            "real_vs_shuffled_jaccard": [jaccard(sel["real"][s], sel["shuffled"][s]) for s in seeds],
            f"real_vs_{other}_text_jaccard": [jaccard(sel["real"][s], sel["other_task"][s]) for s in seeds],
            f"per_layer_spearman_real_vs_{other}_text_seed{s0}": rho,
            f"boundary_gaps_real_seed{s0}": gaps,
        }
        print(task, {k: v for k, v in report[task].items() if "gaps" not in k and "spearman" not in k})
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(report, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
