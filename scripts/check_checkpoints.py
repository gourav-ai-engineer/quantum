"""Engineering check (not a result): do the pinned textattack checkpoints load, and is their label order
aligned with the GLUE dataset labels? Reports unpruned validation accuracy (and MCC for CoLA), the predicted
and true class counts, and the accuracy if the two labels were swapped. Uses no pruning.

    python scripts/check_checkpoints.py --tasks rte,qnli,cola --out results/diag_checkpoints/check.json
"""
from __future__ import annotations

import argparse
import json
import os

import torch
from datasets import load_dataset

from qfc.hf_experiments import load_sequence_classifier, make_text_loader
from step18_v10_confirmatory import evaluate

CHECKS = {
    "rte": {"model_id": "textattack/bert-base-uncased-RTE", "revision": "44f1d994cbd4a349cb7867681940bdb1f0472f53", "dataset": ("glue", "rte"), "text_fields": ("sentence1", "sentence2")},
    "qnli": {"model_id": "textattack/bert-base-uncased-QNLI", "revision": "a63ef5bad18761ededbc04fb8e0f0a2729b1508d", "dataset": ("glue", "qnli"), "text_fields": ("question", "sentence")},
    "cola": {"model_id": "textattack/bert-base-uncased-CoLA", "revision": "5fed03dd6bc5f0b40e86cb04cd1a16eb404ba391", "dataset": ("glue", "cola"), "text_fields": ("sentence",)},
}


def mcc(y, p):
    tp = ((p == 1) & (y == 1)).sum().item(); tn = ((p == 0) & (y == 0)).sum().item()
    fp = ((p == 1) & (y == 0)).sum().item(); fn = ((p == 0) & (y == 1)).sum().item()
    d = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    return 0.0 if d == 0 else (tp * tn - fp * fn) / d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="rte,qnli,cola")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="results/diag_checkpoints/check.json")
    a = ap.parse_args()
    out = {}
    for t in a.tasks.split(","):
        s = CHECKS[t]
        ds = load_dataset(*s["dataset"])["validation"]
        model, tok = load_sequence_classifier(s["model_id"], a.device, revision=s["revision"])
        loader = make_text_loader(ds, tok, text_fields=s["text_fields"], batch_size=16, max_length=128)
        _, correct, y, p = evaluate(model, loader, a.device)
        acc = float(correct.mean())
        out[t] = {"n": len(ds), "revision": s["revision"], "accuracy": acc, "accuracy_if_labels_swapped": 1.0 - acc,
                  "mcc": mcc(y, p), "mcc_if_labels_swapped": mcc(y, 1 - p),
                  "true_counts": [int((y == c).sum()) for c in (0, 1)], "pred_counts": [int((p == c).sum()) for c in (0, 1)]}
        print(t, out[t])
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
