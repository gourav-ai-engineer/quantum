"""V15: inputs for running the official Kwon / K-prune codes on OUR checkpoints and calibration samples.

For each (task, seed) writes <out>/indices/<task>_seed<seed>.json with the 2048 calibration and 512 held-out TRAIN
row indices (train.shuffle(seed) in this pinned environment; GLUE 'idx' == row index), and saves each pinned
checkpoint as a plain pytorch_model.bin directory that transformers 4.12 / 4.25 can load.

    PYTHONPATH=scripts python scripts/v15_prepare_baselines.py --tasks mrpc,sst2 --out /kaggle/working/v15_inputs
"""
from __future__ import annotations

import argparse
import json
import os

from datasets import load_dataset

from qfc.hf_experiments import load_sequence_classifier
from step20_round2 import SPECS


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", default="mrpc,sst2")
    p.add_argument("--seeds", default="7,42,77,123,2024")
    p.add_argument("--calibration-size", type=int, default=2048)
    p.add_argument("--heldout-size", type=int, default=512)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    os.makedirs(os.path.join(a.out, "indices"), exist_ok=True)
    for task in a.tasks.split(","):
        spec = SPECS[task]
        dn, cfg = spec["dataset"]
        train = (load_dataset(dn, cfg) if cfg else load_dataset(dn))["train"]
        assert train["idx"] == list(range(len(train))), "GLUE idx must equal row index"
        n, h = a.calibration_size, a.heldout_size
        for seed in [int(s) for s in a.seeds.split(",")]:
            order = train.shuffle(seed=seed)["idx"]
            json.dump({"task": task, "seed": seed, "calibration": order[:n], "heldout": order[n:n + h]},
                      open(os.path.join(a.out, "indices", f"{task}_seed{seed}.json"), "w"))
        model, tok = load_sequence_classifier(spec["model_id"], "cpu", revision=spec["revision"])
        ck = os.path.join(a.out, "ckpt", task)
        model.save_pretrained(ck, safe_serialization=False)
        tok.save_pretrained(ck)
        print(task, "indices + checkpoint ->", a.out)


if __name__ == "__main__":
    main()
