"""Eval parity (docs/EXPLORATION_LOG.md, "Eval parity"): our pipeline (step22 make_batches/logits_of, max_length 128)
vs the official-code dumps of the same pinned checkpoint. CPU is fine (inference on <= 872 rows).

    PYTHONPATH=scripts python scripts/beat_eval_parity.py <dir with v15_teacher_validation.pt> mrpc
"""
import sys

import torch
from datasets import load_dataset

from step22_grp import SPECS, load_model, logits_of, make_batches

path, task = sys.argv[1], sys.argv[2]
d = torch.load(path, weights_only=True)
spec = SPECS[task]
model, tok = load_model(spec, "cpu")
valid = load_dataset(*[x for x in spec["dataset"] if x])["validation"]
ours = logits_of(model, make_batches(valid, tok, spec["text_fields"], 32, 128), "cpu")
idx = torch.tensor(d["idx"])
print("rows ours/dump:", len(ours), len(d["logits"]), "idx is identity:", bool((idx == torch.arange(len(idx))).all()))
ref = d["logits"].float()
lab = torch.tensor(valid["label"])
print("label match:", bool((torch.as_tensor(d["labels"]) == lab[idx]).all()))
print("max|logit diff|: %.2e" % float((ours[idx] - ref).abs().max()))
print("argmax agreement: %.4f" % float((ours[idx].argmax(-1) == ref.argmax(-1)).float().mean()))
print("acc ours %.4f dump %.4f" % (float((ours.argmax(-1) == lab).float().mean()),
                                   float((ref.argmax(-1) == lab[idx]).float().mean())))
