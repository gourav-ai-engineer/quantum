"""V15 (docs/PROJECT_STATE.md, "Pre-registration V15"): GRP on BERT-base.

--mode kl       : H10 calibration-side gate. Teacher vs GRP-pruned logit KL and label agreement on a held-out slice
                  of TRAIN (never validation). Allowed on CPU (no accuracy number is produced).
--mode accuracy : full validation accuracy of the pruned model (GPU required unless --smoke --allow-cpu).

Calibration = train.shuffle(seed)[:n] (n = 2048 by default, as for every V15 method); held-out = the next 512.
Writes one JSON per (task, seed, constraint, variant) with the kept heads/neurons, MAC ratio, parameter count.

    PYTHONPATH=scripts python scripts/step22_grp.py --tasks mrpc --seeds 7 --constraints 0.5 --mode kl \
        --output-dir results/v15_grp
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch
from datasets import load_dataset

from qfc.grp import AVG_SEQ_LEN, grp_prune, teacher_stats
from qfc.hf_experiments import load_sequence_classifier
from step20_round2 import SPECS


def make_batches(rows, tok, fields, batch_size, max_length):
    """Dynamically padded batches (padding only to the longest example in each batch)."""
    out = []
    for i in range(0, len(rows), batch_size):
        chunk = rows[i:i + batch_size]
        enc = tok(*[chunk[f] for f in fields], truncation=True, max_length=max_length, padding=True,
                  return_tensors="pt")
        out.append({k: v for k, v in enc.items() if k in ("input_ids", "attention_mask", "token_type_ids")})
    return out


@torch.no_grad()
def logits_of(model, batches, device):
    model.eval()
    return torch.cat([model(**{k: v.to(device) for k, v in b.items()}).logits.float().cpu() for b in batches])


def kl_and_agreement(lt, ls):
    pt = lt.log_softmax(-1)
    kl = (pt.exp() * (pt - ls.log_softmax(-1))).sum(-1)
    return float(kl.mean()), float((lt.argmax(-1) == ls.argmax(-1)).float().mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", default="mrpc,sst2")
    p.add_argument("--seeds", default="7,42,77,123,2024")
    p.add_argument("--constraints", default="0.7,0.6,0.5,0.4")
    p.add_argument("--variants", default="grp", help="comma list of grp,no_sens,no_refit")
    p.add_argument("--calibration-size", type=int, default=2048)
    p.add_argument("--heldout-size", type=int, default=512)
    p.add_argument("--ridge-rel", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--mode", choices=("kl", "accuracy"), default="kl")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--allow-cpu", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output-dir", default="results/v15_grp")
    a = p.parse_args()
    if a.mode == "accuracy" and a.device == "cpu" and not (a.smoke and a.allow_cpu):
        raise SystemExit("accuracy runs need a GPU (CLAUDE.md); use --mode kl, or --smoke --allow-cpu")
    os.makedirs(a.output_dir, exist_ok=True)
    variants = {"grp": dict(use_sensitivity=True, refit=True), "no_sens": dict(use_sensitivity=False, refit=True),
                "no_refit": dict(use_sensitivity=True, refit=False)}
    for task in a.tasks.split(","):
        spec = SPECS[task]
        dn, cfg = spec["dataset"]
        ds = load_dataset(dn, cfg) if cfg else load_dataset(dn)
        teacher, tok = load_sequence_classifier(spec["model_id"], a.device, revision=spec["revision"])
        n_seq = AVG_SEQ_LEN[task]
        if a.mode == "accuracy":
            valid = ds["validation"]
            vb = make_batches(valid, tok, spec["text_fields"], a.batch_size, a.max_length)
            lt_valid, labels = logits_of(teacher, vb, a.device), torch.tensor(valid["label"])
        for seed in [int(s) for s in a.seeds.split(",")]:
            shuf, n = ds["train"].shuffle(seed=seed), a.calibration_size
            cb = make_batches(shuf.select(range(n)), tok, spec["text_fields"], a.batch_size, a.max_length)
            hb = make_batches(shuf.select(range(n, n + a.heldout_size)), tok, spec["text_fields"], a.batch_size,
                              a.max_length)
            lt_held = logits_of(teacher, hb, a.device)
            stats = None  # budget-independent; computed on first use, shared by all constraints and variants
            for c in [float(x) for x in a.constraints.split(",")]:
                for v in a.variants.split(","):
                    path = os.path.join(a.output_dir, f"v15_{v}_{task}_seed{seed}_c{c}.json")
                    if os.path.exists(path):
                        continue
                    t0 = time.time()
                    stats = stats or teacher_stats(teacher, cb, a.device, seed)
                    student, heads, neurons, info = grp_prune(teacher, cb, a.device, n_seq, c, a.ridge_rel,
                                                              seed=seed, stats=stats, **variants[v])
                    rec = {"task": task, "seed": seed, "constraint": c, "variant": v, "mode": a.mode,
                           "calibration_size": n, "heldout": f"train.shuffle({seed})[{n}:{n + a.heldout_size}]",
                           "seq_len_for_macs": n_seq, "ridge_rel": a.ridge_rel, "seconds": time.time() - t0,
                           "heads_kept": {str(l): h for l, h in heads.items()},
                           "neurons_kept": {str(l): x for l, x in neurons.items()},
                           "params_teacher": sum(q.numel() for q in teacher.parameters()),
                           "params_student": sum(q.numel() for q in student.parameters()), **info,
                           "model_id": spec["model_id"], "model_revision": spec["revision"]}
                    rec["heldout_kl"], rec["heldout_teacher_agreement"] = kl_and_agreement(
                        lt_held, logits_of(student, hb, a.device))
                    if a.mode == "accuracy":
                        ls = logits_of(student, vb, a.device)
                        rec.update(evaluation_size=len(labels), valid_pred=ls.argmax(-1).tolist(),
                                   accuracy=float((ls.argmax(-1) == labels).float().mean()),
                                   unpruned_accuracy=float((lt_valid.argmax(-1) == labels).float().mean()))
                    print(f"{task} seed={seed} c={c} {v}: mac={info['mac_ratio']:.3f} heads/layer="
                          f"{[len(h) for h in heads.values()]} neurons/layer={[len(x) for x in neurons.values()]} "
                          f"kl={rec['heldout_kl']:.4f} agree={rec['heldout_teacher_agreement']:.3f} "
                          f"({rec['seconds']:.0f}s)", flush=True)
                    json.dump(rec, open(path, "w"))
                    del student


if __name__ == "__main__":
    main()
