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

import copy

from qfc.grp import (AVG_SEQ_LEN, grp_prune, permute_classifier, prune_model, random_select, sequential_refit,
                     teacher_stats)
from qfc.hf_experiments import load_sequence_classifier
from step20_round2 import SPECS as _SPECS

SPECS = dict(_SPECS)
# V15 Phase 0 (docs/PROJECT_STATE.md): textattack MNLI predicts in a permuted label order; fixed from TRAIN data as
# pred m -> GLUE (0->2, 1->0, 2->1), i.e. new class g takes old classifier row label_order[g].
SPECS["mnli"] = {"model_id": "textattack/bert-base-uncased-MNLI", "revision": "3a97f689528cbd91bcc71ab29ea6c20c089d8f28",
                 "dataset": ("glue", "mnli"), "text_fields": ("premise", "hypothesis"), "metric": "accuracy",
                 "valid_split": "validation_matched", "label_order": [1, 2, 0]}
# Pinned 2026-10-19 for V17 R-conf (docs/PROJECT_STATE.md): label order checked on 1000 TRAIN rows (shuffle(0)):
# identity 0.970, swapped 0.030, so no permutation.
SPECS["qqp"] = {"model_id": "textattack/bert-base-uncased-QQP", "revision": "226962dced5a914fa7d486b5f91307f2aee5f37e",
                "dataset": ("glue", "qqp"), "text_fields": ("question1", "question2"), "metric": "accuracy"}
# Pinned 2026-10-27 for the beat-baselines mission (docs/EXPLORATION_LOG.md); regression head (1 output), Pearson.
SPECS["stsb"] = {"model_id": "textattack/bert-base-uncased-STS-B", "revision": "8a1d8f1cc0523b7af746daf104ddd0c1ce5911c1",
                 "dataset": ("glue", "stsb"), "text_fields": ("sentence1", "sentence2"), "metric": "pearson"}
# Pinned in V15 Phase 0. SQuAD v1.1, extractive QA: features of length 384 with doc stride 128 (as both official codes).
SPECS["squad"] = {"model_id": "csarron/bert-base-uncased-squad-v1", "revision": "c0773ca28abf86a669cd1cccc9104c7f829f2be0",
                  "dataset": ("squad", None), "metric": "f1", "qa": True}


def load_model(spec, device):
    if not spec.get("qa"):
        return load_sequence_classifier(spec["model_id"], device, revision=spec["revision"])
    from transformers import AutoModelForQuestionAnswering, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(spec["model_id"], revision=spec["revision"])
    model = AutoModelForQuestionAnswering.from_pretrained(spec["model_id"], revision=spec["revision"],
                                                          attn_implementation="eager")
    return model.to(device).eval(), tok


def task_batches(spec, rows, tok, batch_size, max_length):
    if not spec.get("qa"):
        return make_batches(rows, tok, spec["text_fields"], batch_size, max_length)
    from qfc.squad import batches_from_features, make_features
    return batches_from_features(make_features(rows[:], tok), tok, max(1, batch_size // 2))


@torch.no_grad()
def outputs_of(spec, model, batches, device):
    """Classification: logits [N, C]. QA: list of (start, end) log-probabilities over real tokens per feature."""
    if not spec.get("qa"):
        return logits_of(model, batches, device)
    model.eval()
    out = []
    for b in batches:
        o = model(**{k: v.to(device) for k, v in b.items()})
        for s, e, m in zip(o.start_logits.float().cpu(), o.end_logits.float().cpu(), b["attention_mask"]):
            n = int(m.sum())
            out.append((s[:n].log_softmax(-1), e[:n].log_softmax(-1)))
    return out


def compare_outputs(spec, ref, new):
    """Mean KL(teacher || pruned) and teacher agreement (QA: sum of start and end KL; agreement of the argmax span
    endpoints)."""
    if not spec.get("qa"):
        return kl_and_agreement(ref, new)
    kls, agree = [], []
    for (ts, te), (ss, se) in zip(ref, new):
        kls.append(float((ts.exp() * (ts - ss)).sum() + (te.exp() * (te - se)).sum()))
        agree.append(float(int(ts.argmax()) == int(ss.argmax()) and int(te.argmax()) == int(se.argmax())))
    return sum(kls) / len(kls), sum(agree) / len(agree)


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
    p.add_argument("--variants", default="grp", help="comma list of grp,no_sens,no_refit,random")
    p.add_argument("--random-per-seed", type=int, default=6, help="random unit sets per calibration seed (H13)")
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
        teacher, tok = load_model(spec, a.device)
        if "label_order" in spec:
            permute_classifier(teacher, spec["label_order"])
        n_seq = AVG_SEQ_LEN[task]
        cfg_m = teacher.config
        names = [x for v in a.variants.split(",")
                 for x in ([f"random{i}" for i in range(a.random_per_seed)] if v == "random" else [v])]
        if a.mode == "accuracy":
            valid = ds[spec.get("valid_split", "validation")]
            if spec.get("qa"):
                from qfc.squad import make_features, per_example_scores, postprocess, predict_logits
                v_feats = make_features(valid[:], tok)
                from qfc.squad import batches_from_features
                vb = batches_from_features(v_feats, tok, max(1, a.batch_size // 2))
                t_st, t_en = predict_logits(teacher, vb, a.device)
                valid_d = valid[:]  # plain dict of lists: O(1) column access in postprocess
                unpruned_f1 = per_example_scores(valid_d, postprocess(valid_d, v_feats, t_st, t_en))[0]
            else:
                vb = make_batches(valid, tok, spec["text_fields"], a.batch_size, a.max_length)
                lt_valid, labels = logits_of(teacher, vb, a.device), torch.tensor(valid["label"])
        for seed in [int(s) for s in a.seeds.split(",")]:
            shuf, n = ds["train"].shuffle(seed=seed), a.calibration_size
            cb = task_batches(spec, shuf.select(range(n)), tok, a.batch_size, a.max_length)
            hb = task_batches(spec, shuf.select(range(n, n + a.heldout_size)), tok, a.batch_size, a.max_length)
            lt_held = outputs_of(spec, teacher, hb, a.device)
            stats = None  # budget-independent; computed on first use, shared by all constraints and variants
            for c in [float(x) for x in a.constraints.split(",")]:
                for v in names:
                    path = os.path.join(a.output_dir, f"v15_{v}_{task}_seed{seed}_c{c}.json")
                    if os.path.exists(path) and json.load(open(path)).get("mode") == a.mode:
                        continue  # resume only a finished run of the same mode (a KL result is not an accuracy one)
                    t0 = time.time()
                    if v.startswith("random"):  # H13: random units (same budget), same joint refit
                        heads, neurons, info = random_select(cfg_m.num_hidden_layers, cfg_m.num_attention_heads,
                                                             cfg_m.intermediate_size, n_seq, cfg_m.hidden_size, c,
                                                             seed=100 * seed + int(v[6:]))
                        student = prune_model(copy.deepcopy(teacher), heads, neurons)
                        sequential_refit(student, teacher, cb, a.device, a.ridge_rel)
                    else:
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
                    rec["heldout_kl"], rec["heldout_teacher_agreement"] = compare_outputs(
                        spec, lt_held, outputs_of(spec, student, hb, a.device))
                    if a.mode == "accuracy" and spec.get("qa"):
                        st_, en_ = predict_logits(student, vb, a.device)
                        f1, em = per_example_scores(valid_d, postprocess(valid_d, v_feats, st_, en_))
                        rec.update(evaluation_size=len(f1), valid_f1=f1, f1=sum(f1) / len(f1), em=sum(em) / len(em),
                                   unpruned_f1=sum(unpruned_f1) / len(unpruned_f1))
                    elif a.mode == "accuracy":
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
