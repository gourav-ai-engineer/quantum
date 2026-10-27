"""CPU parity of the HELD-OUT rows: our held-out split train.shuffle(seed)[2048:2560] vs the official runs' dumped
v15_teacher_heldout.pt (logits, labels), for one seed per task. The KL/MSE half of the "beats both" rule depends on it.

    PYTHONPATH=scripts python scripts/beat_heldout_parity.py
"""
import glob

import torch
from datasets import load_dataset

from step22_grp import SPECS, load_model, logits_of, make_batches

DUMPS = {"mrpc": "results/v15_h10/c43dd3e/h10_logits", "sst2": "results/v15_h10/c43dd3e/h10_logits",
         "stsb": "results/beat_explore/C_stsb_baselines/036e44c/logits"}
SEED, N0, N1 = 7, 2048, 2560
for task in ("mrpc", "sst2", "stsb"):
    f = sorted(glob.glob(f"{DUMPS[task]}/**/kwon_{task}_seed{SEED}_c0.5/v15_teacher_heldout.pt", recursive=True))[0]
    d = torch.load(f, weights_only=True)
    spec = SPECS[task]
    ds = load_dataset(*spec["dataset"])
    teacher, tok = load_model(spec, "cpu")
    rows = ds["train"].shuffle(seed=SEED).select(range(N0, N1))
    ours = logits_of(teacher, make_batches(rows, tok, spec["text_fields"], 32, 128), "cpu")
    print(f"{task}: dump {tuple(d['logits'].shape)} ours {tuple(ours.shape)} max|dlogit| "
          f"{(d['logits'] - ours).abs().max():.2e} labels_equal "
          f"{torch.equal(d['labels'].float(), torch.tensor(rows['label']).float())}", flush=True)
