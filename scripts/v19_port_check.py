"""V19 step 1 acceptance (a), CPU, engineering check only (not a result, rule 9): re-run the frozen d2i-RI through the
adapter-refactored code on BERT-base and compare the validation logits with an existing reference JSON written by
scripts/beat_run.py before the refactor (same method code at tag v18-freeze-fix1).

    PYTHONPATH=scripts python scripts/v19_port_check.py --task mrpc --seed 7 \
        --reference results/beat_explore/E1_align/e0063df/d2i_ri_mrpc_seed7_c0.5.json --out results/v19_port/a_mrpc.json
"""
from __future__ import annotations

import argparse
import json
import os
import time
from types import SimpleNamespace

import torch
from datasets import load_dataset

from beat_run import build, heldout_dist, mac_ratio, valid_metrics
from qfc.grp import AVG_SEQ_LEN, permute_classifier
from step22_grp import SPECS, load_model, logits_of, make_batches


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="mrpc")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--constraint", type=float, default=0.5)
    p.add_argument("--reference", required=True)
    p.add_argument("--kprune-masks", default="results/v15_h10/c43dd3e/h10/kprune_masks.npz")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    spec, task, seed, c = SPECS[a.task], a.task, a.seed, a.constraint
    ns = SimpleNamespace(kprune_masks=a.kprune_masks, device="cpu", ridge_rel=1e-4)
    ds = load_dataset(*spec["dataset"])
    teacher, tok = load_model(spec, "cpu")
    if "label_order" in spec:
        permute_classifier(teacher, spec["label_order"])
    valid = ds[spec.get("valid_split", "validation")]
    vb = make_batches(valid, tok, spec["text_fields"], 32, 128)
    labels = torch.tensor(valid["label"])
    shuf = ds["train"].shuffle(seed=seed)
    cb = make_batches(shuf.select(range(2048)), tok, spec["text_fields"], 32, 128)
    hb = make_batches(shuf.select(range(2048, 2048 + 512)), tok, spec["text_fields"], 32, 128)
    t0 = time.time()
    lt_held = logits_of(teacher, hb, "cpu")
    student, heads, neurons = build("d2i_ri", teacher, cb, AVG_SEQ_LEN[task], c, ns, task, seed, None)
    ls = logits_of(student, vb, "cpu")
    dist, _ = heldout_dist(task, lt_held, logits_of(student, hb, "cpu"))
    ref = json.load(open(a.reference))
    rl = torch.tensor(ref["valid_logits"])
    diff = (ls - rl).abs()
    out = {"task": task, "seed": seed, "constraint": c, "device": "cpu", "torch": torch.__version__,
           "reference": a.reference, "reference_valid": ref["valid"], "ours_valid": valid_metrics(task, ls, labels),
           "max_abs_logit_diff": float(diff.max()), "mean_abs_logit_diff": float(diff.mean()),
           "argmax_agreement": float((ls.argmax(-1) == rl.argmax(-1)).float().mean()),
           "heldout_dist_ours": dist, "heldout_dist_reference": ref["heldout_dist"],
           "mac_ratio_ours": mac_ratio(heads, neurons, teacher.config, AVG_SEQ_LEN[task]),
           "mac_ratio_reference": ref["mac_ratio"],
           "heads_kept_equal": {str(l): h for l, h in heads.items()} == ref["heads_kept"],
           "neurons_kept_equal": {str(l): n for l, n in neurons.items()} == ref["neurons_kept"],
           "seconds": time.time() - t0}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
