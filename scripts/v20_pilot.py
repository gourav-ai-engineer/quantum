"""V20 PILOT runner (docs/PROJECT_STATE.md, "V20 PILOT pre-registration"). Exploration on VALIDATION only.

Fixed official K-prune masks (English selection) on the fine-tuned multilingual teacher; repair factor
{none, d2i_r, d2i_ri} x calibration language {en, as, bn, hi, mixed}. Calibration rows are CONTENT-MATCHED: the same
2048 MNLI train row indices (the official runs' index files) taken from GLUE MNLI (en) or from IndicXNLI train (a
row-aligned translation of MNLI train); 'mixed' assigns them round-robin to en/as/bn/hi. Labels are never read by the
repair. Evaluation: XNLI-en + IndicXNLI validation (n-way parallel), per-example predictions stored.
One JSON per (seed, repair, calibration language); resumable.

    PYTHONPATH=scripts python scripts/v20_pilot.py --ckpt /tmp/v20/ft/mnli --kprune-masks <npz> --index-dir <dir> \
        --output-dir /kaggle/working/v20/pilot
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import time

import torch
from datasets import Dataset, load_dataset

from qfc.align import sequential_refit_io
from qfc.grp import AVG_SEQ_LEN, prune_model, sequential_refit
from qfc.hf_experiments import load_sequence_classifier
from beat_run import kprune_masks, mac_ratio
from step22_grp import logits_of, make_batches
from v20_finetune import INDIC, XNLI, load_indicxnli

MIX = ("en", "as", "bn", "hi")
FIELDS = ("premise", "hypothesis")


def mixed_assignment(n, langs=MIX):
    """Position i of the calibration index list -> language langs[i % len(langs)] (content stays matched)."""
    return [langs[i % len(langs)] for i in range(n)]


def calibration_rows(train_of, idx, lang, row_of=lambda l, i: i):
    """Unlabeled premise/hypothesis rows at MNLI train positions idx, in `lang` (or round-robin for 'mixed');
    row_of(lang, mnli_pos) -> that language's row (identity for en)."""
    langs = mixed_assignment(len(idx)) if lang == "mixed" else [lang] * len(idx)
    rows = {f: [] for f in FIELDS}
    for i, l in zip(idx, langs):
        r = train_of(l)[int(row_of(l, int(i)))]
        for f in FIELDS:
            rows[f].append(r[f])
    return Dataset.from_dict(rows)


def align_rows(labels_en, labels_other, k=32, max_shift=2000):
    """IndicXNLI train = MNLI train with rows dropped (the shift grows: +1 at row 6073, +82 at row 300000; checked
    2026-11-17) and the same length, so other rows exist that MNLI does not explain. Monotone map other row i -> MNLI row
    i + o: row i is mapped only if the k labels starting at i match MNLI at the current shift o (chance 3^-k); otherwise
    the nearest shift within +-max_shift that matches is taken (drops move it up, insertions down). Unmatched rows are
    left out. Returns {other_row: mnli_row}."""
    import numpy as np
    a, b = np.asarray(labels_other), np.asarray(labels_en)
    out, o, i = {}, 0, 0
    while i + k <= len(a):
        if 0 <= i + o and i + o + k <= len(b) and (a[i:i + k] == b[i + o:i + o + k]).all():
            out[i] = i + o
            i += 1
            continue
        for o2 in (o + d * s for d in range(1, max_shift + 1) for s in (1, -1)):
            if 0 <= i + o2 and i + o2 + k <= len(b) and (a[i:i + k] == b[i + o2:i + o2 + k]).all():
                o = o2
                break
        else:
            i += 1  # row i of `other` is not in MNLI near here: skip it
            continue
    return out


def pick_calibration(order, ok, n, heldout=512):
    """First n rows of the seeded shuffle `order` whose position passes ok(); rows that fail ok() among the first n are
    replaced by the next matching rows from index n + heldout onward (disjoint from the held-out slice, amendment 3).
    Returns (rows, number_replaced)."""
    keep = [i for i in order[:n] if ok(i)]
    fill = (i for i in order[n + heldout:] if ok(i))
    for i in fill:
        if len(keep) >= n:
            break
        keep.append(i)
    if len(keep) < n:
        raise SystemExit(f"only {len(keep)} content-matched calibration rows (need {n})")
    return keep, n - sum(1 for i in order[:n] if ok(i))


def parallel_evals(evals, ref="en"):
    """Truncate every eval set to the first len(evals[ref]) rows (IndicXNLI-ta dev has 3,238 rows; its first 2,490 are the
    parallel ones, amendment 2) and require identical label vectors, so predictions can be paired by example id."""
    n = len(evals[ref])
    short = [k for k, v in evals.items() if len(v) < n]
    if short:
        raise SystemExit(f"eval sets shorter than {ref} ({n} rows): {short}")
    out = {k: v.select(range(n)) for k, v in evals.items()}
    bad = [k for k in out if out[k]["label"] != out[ref]["label"]]
    if bad:
        raise SystemExit(f"eval sets not label-parallel with {ref}: {bad}")
    return out


def repair(method, teacher, heads, neurons, cb, device, ridge):
    s = prune_model(copy.deepcopy(teacher), heads, neurons)
    if method == "d2i_r":
        sequential_refit(s, teacher, cb, device, ridge, True)
    elif method == "d2i_ri":
        sequential_refit_io(s, teacher, cb, device, heads, neurons, ridge)
    elif method != "none":
        raise SystemExit(f"unknown repair {method}")
    return s


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--kprune-masks", required=True)
    p.add_argument("--index-dir", default=None, help="official runs' <task>_seed<s>.json; default: recompute shuffle")
    p.add_argument("--seeds", default="7,42,77")
    p.add_argument("--constraint", type=float, default=0.5)
    p.add_argument("--repairs", default="none,d2i_r,d2i_ri")
    p.add_argument("--cal-langs", default="en,as,bn,hi,mixed")
    p.add_argument("--calibration-size", type=int, default=2048)
    p.add_argument("--ridge-rel", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--allow-cpu", action="store_true")
    a = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and not (a.smoke and a.allow_cpu):
        raise SystemExit("pilot runs need a GPU (CLAUDE.md); CPU only with --smoke --allow-cpu")
    os.makedirs(a.output_dir, exist_ok=True)
    teacher, tok = load_sequence_classifier(a.ckpt, device)
    teacher.eval()
    cfg, n_seq = teacher.config, AVG_SEQ_LEN["mnli"]
    cache = {}

    inv = {}  # lang -> {mnli_pos: lang_row}

    def train_of(lang):
        if lang not in cache:
            cache[lang] = load_dataset("glue", "mnli")["train"] if lang == "en" else load_indicxnli(lang, "train")
            if lang != "en":  # IndicXNLI train = MNLI train with rows dropped: rebuild the row map (align_rows)
                m = align_rows(train_of("en")["label"], cache[lang]["label"])
                inv[lang] = {j: i for i, j in m.items()}
                print(f"{lang} train rows={len(cache[lang])} mapped to MNLI={len(m)} ({len(m) / len(cache[lang]):.4f})",
                      flush=True)
                if len(m) < 0.98 * len(cache[lang]):
                    raise SystemExit(f"IndicXNLI-{lang} train: row map covers only {len(m)} rows")
        return cache[lang]

    need = sorted({l for c in a.cal_langs.split(",") for l in (MIX if c == "mixed" else (c,)) if l != "en"})
    for l in need:
        train_of(l)

    evals = {"en": load_dataset(XNLI[0], "en", revision=XNLI[1])["validation"]}
    for lang in INDIC:
        evals[lang] = load_indicxnli(lang, "validation")
    evals = parallel_evals(evals)
    if a.smoke:
        evals = {k: v.select(range(64)) for k, v in evals.items()}
    labels = {k: v["label"] for k, v in evals.items()}
    vb = {k: make_batches(v, tok, FIELDS, 128, a.max_length) for k, v in evals.items()}
    lab = torch.tensor(labels["en"])
    t_pred = {k: logits_of(teacher, b, device).argmax(-1) for k, b in vb.items()}
    unpruned = {k: float((t_pred[k] == lab).float().mean()) for k in vb}
    print("unpruned", {k: round(v, 4) for k, v in unpruned.items()}, flush=True)
    for seed in [int(s) for s in a.seeds.split(",")]:
        heads, neurons = kprune_masks(a.kprune_masks, "mnli", seed, a.constraint)
        n = 64 if a.smoke else a.calibration_size
        order = train_of("en").add_column("_row", list(range(len(train_of("en"))))).shuffle(seed=seed)["_row"]
        if a.index_dir:  # the official runs' calibration indices must be the first positions of this same shuffle
            official = json.load(open(os.path.join(a.index_dir, f"mnli_seed{seed}.json")))["calibration"]
            assert order[:len(official)] == official, "shuffle differs from the official index file"
        idx, replaced = pick_calibration(order, lambda i: all(i in inv[l] for l in need), n)
        print(f"seed={seed}: {replaced} of {n} official calibration rows replaced (no content match)", flush=True)
        for method in a.repairs.split(","):
            for lang in (["none"] if method == "none" else a.cal_langs.split(",")):
                path = os.path.join(a.output_dir, f"{method}_{lang}_seed{seed}_c{a.constraint}.json")
                if os.path.exists(path):
                    continue
                t0 = time.time()
                rows = None if method == "none" else calibration_rows(
                    train_of, idx, lang, lambda l, i: i if l == "en" else inv[l][i])
                cb = None if rows is None else make_batches(rows, tok, FIELDS, a.batch_size, a.max_length)
                s = repair(method, teacher, heads, neurons, cb, device, a.ridge_rel)
                preds = {k: logits_of(s, b, device).argmax(-1) for k, b in vb.items()}
                acc = {k: float((preds[k] == lab).float().mean()) for k in preds}
                rec = {"repair": method, "cal_lang": lang, "seed": seed, "constraint": a.constraint,
                       "calibration_size": n, "calibration_rows": "MNLI train positions from the official index file"
                       if a.index_dir else f"glue mnli train.shuffle({seed})[:{n}]", "mode": "smoke" if a.smoke else "full",
                       "calibration_mnli_positions": idx, "calibration_rows_replaced": replaced, "evaluation_size": len(lab), "accuracy": acc, "unpruned_accuracy": unpruned,
                       "preds": {k: v.tolist() for k, v in preds.items()}, "labels": lab.tolist(),
                       "mac_ratio": mac_ratio(heads, neurons, cfg, n_seq),
                       "heads_kept_per_layer": [len(heads[l]) for l in sorted(heads)],
                       "neurons_kept_per_layer": [len(neurons[l]) for l in sorted(neurons)],
                       "seconds": time.time() - t0}
                print(f"seed={seed} {method}/{lang}: mac={rec['mac_ratio']:.3f} heads={rec['heads_kept_per_layer']} "
                      f"acc={ {k: round(v, 4) for k, v in acc.items()} } ({rec['seconds']:.0f}s)", flush=True)
                json.dump(rec, open(path, "w"))
                del s


if __name__ == "__main__":
    main()
