"""Exploration runner for docs/CLI_MISSION_BEAT.md: build a pruned student with a named method, score held-out-train
KL (MSE for STS-B) AND validation accuracy / MCC / Pearson, dump the validation logits (for paired tests), one JSON
per (method, task, seed, constraint). Resumable. Eval path = step22 make_batches/logits_of (parity with the official
codes verified in scripts/beat_eval_parity.py). Calibration = train.shuffle(seed)[:2048], held-out = next 512.

    PYTHONPATH=scripts python scripts/beat_run.py --methods grp_r,d2i_r --tasks mrpc --seeds 7 --constraints 0.5 \
        --output-dir results/beat_explore/smoke --smoke --allow-cpu
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import time

import numpy as np
import torch
from datasets import load_dataset

from qfc.align import sequential_refit_io
from qfc.grp import AVG_SEQ_LEN, grp_prune, prune_model, sequential_refit, teacher_stats
from step22_grp import SPECS, kl_and_agreement, load_model, logits_of, make_batches


def kprune_masks(npz, task, seed, c):
    z = np.load(npz)
    hm, nm = z[f"kprune_{task}_seed{seed}_c{c}__head"], z[f"kprune_{task}_seed{seed}_c{c}__neuron"]
    return ({l: np.nonzero(hm[l])[0].tolist() for l in range(hm.shape[0])},
            {l: np.nonzero(nm[l])[0].tolist() for l in range(nm.shape[0])})


def mac_ratio(heads, neurons, cfg, n_seq):
    from qfc.grp import model_mac
    hs = cfg.hidden_size // cfg.num_attention_heads
    full = model_mac([cfg.num_attention_heads] * cfg.num_hidden_layers, [cfg.intermediate_size] * cfg.num_hidden_layers,
                     n_seq, cfg.hidden_size, hs)
    return model_mac([len(heads[l]) for l in heads], [len(neurons[l]) for l in neurons], n_seq, cfg.hidden_size, hs) / full


def valid_metrics(task, logits, labels):
    if task == "stsb":
        x, y = logits[:, 0].double(), labels.double()
        return {"pearson": float(torch.corrcoef(torch.stack([x, y]))[0, 1])}
    pred = logits.argmax(-1)
    out = {"accuracy": float((pred == labels).float().mean())}
    if task == "cola":
        tp, tn = float(((pred == 1) & (labels == 1)).sum()), float(((pred == 0) & (labels == 0)).sum())
        fp, fn = float(((pred == 1) & (labels == 0)).sum()), float(((pred == 0) & (labels == 1)).sum())
        den = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
        out["mcc"] = (tp * tn - fp * fn) / den if den else 0.0
    return out


def heldout_dist(task, lt, ls):
    if task == "stsb":
        return float(((lt - ls) ** 2).mean()), None
    return kl_and_agreement(lt, ls)


def build(method, teacher, cb, n_seq, c, a, task, seed, stats):
    """Returns (student, heads, neurons). New methods are added here, one entry per EXPLORATION_LOG idea."""
    if method == "grp_r":
        s, h, n, _ = grp_prune(teacher, cb, a.device, n_seq, c, a.ridge_rel, seed=seed, stats=stats, drift_correct=True)
        return s, h, n
    if method == "d2i_r":  # K-prune's own masks + the drift-corrected refit (reproduces K-prune compensation, V17)
        h, n = kprune_masks(a.kprune_masks, task, seed, c)
        s = prune_model(copy.deepcopy(teacher), h, n)
        sequential_refit(s, teacher, cb, a.device, a.ridge_rel, True)
        return s, h, n
    if method in ("grp_ri", "d2i_ri"):  # idea 1 (EXPLORATION_LOG): + input-side alignment (qkv rows, FFN up rows)
        if method == "grp_ri":
            s, h, n, _ = grp_prune(teacher, cb, a.device, n_seq, c, a.ridge_rel, seed=seed, stats=stats, refit=False)
        else:
            h, n = kprune_masks(a.kprune_masks, task, seed, c)
            s = prune_model(copy.deepcopy(teacher), h, n)
        sequential_refit_io(s, teacher, cb, a.device, h, n, a.ridge_rel)
        return s, h, n
    raise SystemExit(f"unknown method {method}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--methods", default="grp_r,d2i_r")
    p.add_argument("--tasks", default="mrpc,sst2")
    p.add_argument("--seeds", default="7,42,77,123,2024")
    p.add_argument("--constraints", default="0.5,0.4")
    p.add_argument("--kprune-masks", default="results/v15_h10/c43dd3e/h10/kprune_masks.npz")
    p.add_argument("--calibration-size", type=int, default=2048)
    p.add_argument("--heldout-size", type=int, default=512)
    p.add_argument("--ridge-rel", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--allow-cpu", action="store_true")
    a = p.parse_args()
    if a.device == "cpu" and not (a.smoke and a.allow_cpu):
        raise SystemExit("exploration runs need a GPU (CLAUDE.md); CPU only with --smoke --allow-cpu")
    os.makedirs(a.output_dir, exist_ok=True)
    for task in a.tasks.split(","):
        spec = SPECS[task]
        ds = load_dataset(*spec["dataset"])
        teacher, tok = load_model(spec, a.device)
        cfg, n_seq = teacher.config, AVG_SEQ_LEN[task]
        valid = ds["validation"]
        if a.smoke:
            valid = valid.select(range(64))
        vb = make_batches(valid, tok, spec["text_fields"], a.batch_size, a.max_length)
        labels = torch.tensor(valid["label"])
        lt_valid = logits_of(teacher, vb, a.device)
        for seed in [int(s) for s in a.seeds.split(",")]:
            shuf, n = ds["train"].shuffle(seed=seed), a.calibration_size
            if a.smoke:
                n = 64
            cb = make_batches(shuf.select(range(n)), tok, spec["text_fields"], a.batch_size, a.max_length)
            hb = make_batches(shuf.select(range(n, n + (64 if a.smoke else a.heldout_size))), tok, spec["text_fields"],
                              a.batch_size, a.max_length)
            lt_held = logits_of(teacher, hb, a.device)
            stats = teacher_stats(teacher, cb, a.device, seed) if any(
                m.startswith("grp") for m in a.methods.split(",")) else None
            for c in [float(x) for x in a.constraints.split(",")]:
                for method in a.methods.split(","):
                    path = os.path.join(a.output_dir, f"{method}_{task}_seed{seed}_c{c}.json")
                    if os.path.exists(path):
                        continue
                    t0 = time.time()
                    student, heads, neurons = build(method, teacher, cb, n_seq, c, a, task, seed, stats)
                    dist, agree = heldout_dist(task, lt_held, logits_of(student, hb, a.device))
                    ls = logits_of(student, vb, a.device)
                    rec = {"method": method, "task": task, "seed": seed, "constraint": c, "calibration_size": n,
                           "mode": "smoke" if a.smoke else "full", "evaluation_size": len(labels),
                           "heldout": f"train.shuffle({seed})[{n}:{n + (64 if a.smoke else a.heldout_size)}]",
                           "heldout_dist": dist, "heldout_dist_kind": "mse" if task == "stsb" else "kl",
                           "heldout_teacher_agreement": agree, "valid": valid_metrics(task, ls, labels),
                           "unpruned_valid": valid_metrics(task, lt_valid, labels), "valid_logits": ls.tolist(),
                           "seq_len_for_macs": n_seq, "mac_ratio": mac_ratio(heads, neurons, cfg, n_seq),
                           "heads_kept": {str(l): h for l, h in heads.items()},
                           "neurons_kept": {str(l): x for l, x in neurons.items()}, "seconds": time.time() - t0,
                           "model_id": spec["model_id"], "model_revision": spec["revision"]}
                    print(f"{task} seed={seed} c={c} {method}: mac={rec['mac_ratio']:.3f} dist={dist:.4f} "
                          f"valid={rec['valid']} unpruned={rec['unpruned_valid']} ({rec['seconds']:.0f}s)", flush=True)
                    json.dump(rec, open(path, "w"))
                    del student


if __name__ == "__main__":
    main()
