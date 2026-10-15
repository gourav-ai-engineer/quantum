"""Pair-level attention-pattern fidelity vs output fidelity for every within-layer head pair (paper figure; V14
diagnostic D1 at pair resolution). Calibration data only (train.shuffle(seed)[:512]); CPU allowed.

    PYTHONPATH=scripts python scripts/v14_pairs.py --tasks sst2,mrpc --seed 7 --out results/v14_pairs/<commit>
"""
from __future__ import annotations

import argparse
import json
import os

import torch
from datasets import load_dataset

from qfc.fidelity import pairwise_fidelity
from qfc.hf_experiments import collect_mean_density_states, load_sequence_classifier, make_text_loader
from qfc.output_coverage import collect_context_grams, output_fidelity
from step20_round2 import SPECS


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", default="sst2,mrpc")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--calibration-size", type=int, default=512)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for task in a.tasks.split(","):
        spec = SPECS[task]
        dn, cfg = spec["dataset"]
        train = (load_dataset(dn, cfg) if cfg else load_dataset(dn))["train"]
        model, tok = load_sequence_classifier(spec["model_id"], "cpu", revision=spec["revision"])
        H = model.config.num_attention_heads
        cal = train.shuffle(seed=a.seed).select(range(a.calibration_size))
        cl = make_text_loader(cal, tok, text_fields=spec["text_fields"], batch_size=16, max_length=128)
        Gs = collect_context_grams(model, cl, "cpu")
        states = collect_mean_density_states(model, cl, "cpu")
        iu = torch.triu_indices(H, H, 1)
        pairs = []
        for l, G in enumerate(Gs):
            fo = output_fidelity(G, H)[iu[0], iu[1]]
            fa = pairwise_fidelity(states[l]).double()[iu[0], iu[1]]
            pairs += [{"layer": l, "i": int(i), "j": int(j), "attn_F": float(x), "out_F": float(y)}
                      for i, j, x, y in zip(iu[0], iu[1], fa, fo)]
        json.dump({"task": task, "seed": a.seed, "calibration_size": a.calibration_size, "pairs": pairs},
                  open(os.path.join(a.out, f"pairs_{task}_seed{a.seed}.json"), "w"))
        print(task, len(pairs), "pairs", flush=True)


if __name__ == "__main__":
    main()
