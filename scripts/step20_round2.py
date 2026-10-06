"""Round 2 (docs/PROJECT_STATE.md, "Pre-registration Round 2"): H1 / H2 / H6.

Per task and calibration seed it calibrates on TRAIN, evaluates on the full VALIDATION split, and saves
per-example predictions, losses and class-1 probabilities for every method (npz), so
scripts/round2_analysis.py can recompute every test from files. Methods:

  VonNeumann_keep_high/low, Shannon_keep_high/low   (uniform k per layer)
  MichelGate                                        (uniform k per layer = "uniform-k Michel")
  DelimiterMass_keep_low                            (uniform k per layer; keep-low ONLY, pre-registered)
  MichelGlobal, MichelGlobal_min1                   (72 heads in total; per-layer counts recorded)
  Random                                            (>= 100 masks, one distribution per task)

Uniform methods are asserted to keep exactly k distinct heads per layer; global methods are asserted to keep
exactly L*k distinct heads in total (per-layer counts vary by design). Heads kept per layer are printed.

    python scripts/step20_round2.py --tasks sst2,mrpc,rte,qnli,cola --output-dir results/round2
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from datasets import load_dataset

from qfc.baselines import (
    delimiter_mass_from_attention,
    entropy_baseline_selections,
    heads_kept_per_layer,
    michel_global_selection,
    random_selection,
    topk_selection,
    validate_global_selection,
    validate_selection,
)
from qfc.hf_experiments import (
    _move_batch,
    collect_mean_density_states,
    head_mask_from_selection,
    load_sequence_classifier,
    make_text_loader,
    michel_head_importance,
)
from qfc.metrics import auroc, mcc_binary
from step13_mrpc_stability import shannon_scores, vn_scores
from step18_v10_confirmatory import SPECS as SPECS_V10

SPECS = dict(SPECS_V10)
SPECS["sst2"] = {**SPECS["sst2"], "metric": "accuracy"}
SPECS["mrpc"] = {**SPECS["mrpc"], "metric": "accuracy"}
# Revisions checked 2026-10-06 (docs/PROJECT_STATE.md, Round 2; label order verified by scripts/check_checkpoints.py).
SPECS["rte"] = {"model_id": "textattack/bert-base-uncased-RTE", "revision": "44f1d994cbd4a349cb7867681940bdb1f0472f53",
                "dataset": ("glue", "rte"), "text_fields": ("sentence1", "sentence2"), "metric": "accuracy"}
SPECS["qnli"] = {"model_id": "textattack/bert-base-uncased-QNLI", "revision": "a63ef5bad18761ededbc04fb8e0f0a2729b1508d",
                 "dataset": ("glue", "qnli"), "text_fields": ("question", "sentence"), "metric": "accuracy"}
SPECS["cola"] = {"model_id": "textattack/bert-base-uncased-CoLA", "revision": "5fed03dd6bc5f0b40e86cb04cd1a16eb404ba391",
                 "dataset": ("glue", "cola"), "text_fields": ("sentence",), "metric": "mcc"}


@torch.no_grad()
def evaluate(model, loader, device, head_mask=None):
    """Per-example predictions, losses and class-1 probabilities (numpy) plus labels."""
    model.eval()
    preds, losses, p1, labels = [], [], [], []
    for batch in loader:
        batch = _move_batch(batch, device)
        y = batch.pop("labels")
        z = model(**batch, head_mask=head_mask, return_dict=True).logits.float()
        losses.append(torch.nn.functional.cross_entropy(z, y, reduction="none").cpu())
        preds.append(z.argmax(-1).cpu())
        p1.append(z.softmax(-1)[:, 1].cpu())
        labels.append(y.cpu())
    cat = lambda xs: torch.cat(xs).numpy()  # noqa: E731
    return cat(preds).astype(np.int8), cat(losses).astype(np.float32), cat(p1).astype(np.float32), cat(labels).astype(np.int8)


def summarize(pred, loss, p1, labels):
    return {"accuracy": float((pred == labels).mean()), "mcc": mcc_binary(labels, pred),
            "loss": float(loss.mean()), "auroc": auroc(labels, p1)}


@torch.no_grad()
def delimiter_mass_scores(model, loader, device, delimiter_ids):
    """Per layer, [H] mean attention mass on [CLS]/[SEP] keys (real query positions only)."""
    model.eval()
    sums, count = None, 0
    for batch in loader:
        batch = _move_batch(batch, device)
        batch.pop("labels", None)
        out = model(**batch, output_attentions=True, return_dict=True)
        ids = batch["input_ids"]
        delim = torch.zeros_like(ids)
        for token_id in delimiter_ids:
            delim = delim | (ids == token_id).long()
        if sums is None:
            sums = [torch.zeros(a.shape[1], dtype=torch.float64) for a in out.attentions]
        for li, attn in enumerate(out.attentions):
            sums[li] += delimiter_mass_from_attention(attn.float(), batch["attention_mask"], delim).sum(0).double().cpu()
        count += int(ids.shape[0])
    if sums is None or count == 0:
        raise ValueError("empty loader")
    return [s / count for s in sums]


def build_selections(model, tok, loader, device, k):
    mean_states = collect_mean_density_states(model, loader, device)
    layers, heads = len(mean_states), int(mean_states[0].shape[0])
    vn, sh = vn_scores(mean_states), shannon_scores(model, loader, device)
    michel = michel_head_importance(model, loader, device).detach().cpu()
    dm = delimiter_mass_scores(model, loader, device, [tok.cls_token_id, tok.sep_token_id])
    uniform = entropy_baseline_selections(vn, sh, k)
    uniform["MichelGate"] = topk_selection([michel[i] for i in range(layers)], k, largest=True)
    uniform["DelimiterMass_keep_low"] = topk_selection(dm, k, largest=False)
    glob = {"MichelGlobal": (michel_global_selection(michel, layers * k), 0),
            "MichelGlobal_min1": (michel_global_selection(michel, layers * k, min_per_layer=1), 1)}
    for name, sel in uniform.items():
        validate_selection(sel, layers, heads, k, name=name)
    for name, (sel, floor) in glob.items():
        validate_global_selection(sel, layers, heads, layers * k, min_per_layer=floor, name=name)
    return layers, heads, uniform, {n: s for n, (s, _) in glob.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", default="sst2,mrpc,rte,qnli,cola")
    p.add_argument("--seeds", default="7,42,77,123,2024")
    p.add_argument("--calibration-size", type=int, default=128)
    p.add_argument("--evaluation-size", type=int, default=-1, help="-1 = full validation split")
    p.add_argument("--heads-to-keep", type=int, default=6)
    p.add_argument("--random-masks", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output-dir", default="results/round2")
    a = p.parse_args()
    k = a.heads_to_keep
    os.makedirs(a.output_dir, exist_ok=True)

    for task in a.tasks.split(","):
        spec = SPECS[task]
        dn, cfg = spec["dataset"]
        ds = load_dataset(dn, cfg) if cfg else load_dataset(dn)
        train, valid = ds["train"], ds["validation"]
        if a.evaluation_size >= 0:
            if a.evaluation_size > len(valid):
                raise ValueError("evaluation-size exceeds validation split")
            valid = valid.select(range(a.evaluation_size))
        model, tok = load_sequence_classifier(spec["model_id"], a.device, revision=spec["revision"])
        vl = make_text_loader(valid, tok, text_fields=spec["text_fields"], batch_size=a.batch_size, max_length=a.max_length)
        base = evaluate(model, vl, a.device)
        base_summary = summarize(*base)
        print(f"{task} eval_n={len(valid)} unpruned: {base_summary}")

        for seed in [int(s) for s in a.seeds.split(",")]:
            cal = train.shuffle(seed=seed).select(range(a.calibration_size))
            cl = make_text_loader(cal, tok, text_fields=spec["text_fields"], batch_size=a.batch_size, max_length=a.max_length)
            layers, heads, uniform, glob = build_selections(model, tok, cl, a.device, k)
            arrays = {"labels": base[3], "Unpruned__pred": base[0], "Unpruned__loss": base[1], "Unpruned__p1": base[2]}
            record = {"task": task, "seed": seed, "metric": spec["metric"], "model_id": spec["model_id"],
                      "model_revision": spec["revision"], "calibration_size": a.calibration_size,
                      "evaluation_size": len(valid), "heads_to_keep_per_layer": k, "unpruned": base_summary, "methods": {}}
            for name, sel in {**uniform, **glob}.items():
                pred, loss, p1, labels = evaluate(model, vl, a.device, head_mask_from_selection(layers, heads, sel, a.device))
                arrays[f"{name}__pred"], arrays[f"{name}__loss"], arrays[f"{name}__p1"] = pred, loss, p1
                kept = heads_kept_per_layer(sel)
                record["methods"][name] = {**summarize(pred, loss, p1, labels), "heads_kept_per_layer": kept,
                                           "total_heads_kept": sum(kept), "selected_heads_zero_based": sel}
                print(f"  {task} seed={seed} {name:24s} {spec['metric']}={record['methods'][name][spec['metric']]:.4f} "
                      f"auroc={record['methods'][name]['auroc']:.4f} heads_kept_per_layer={kept}")
            np.savez_compressed(os.path.join(a.output_dir, f"round2_{task}_seed{seed}.npz"), **arrays)
            json.dump(record, open(os.path.join(a.output_dir, f"round2_{task}_seed{seed}.json"), "w", encoding="utf-8"), indent=2)

        # Random: one distribution per task, independent of the calibration seed.
        masks = []
        for i in range(a.random_masks):
            sel = random_selection(layers, heads, k, 2027 + 1000 * k + i)
            validate_selection(sel, layers, heads, k, name=f"Random[{i}]")
            masks.append(summarize(*evaluate(model, vl, a.device, head_mask_from_selection(layers, heads, sel, a.device))))
        quant = {m: {f"q{q}": float(np.nanquantile([r[m] for r in masks], q / 100)) for q in (2.5, 25, 50, 75, 97.5)}
                 for m in ("accuracy", "mcc", "loss", "auroc")}
        json.dump({"task": task, "metric": spec["metric"], "random_masks": a.random_masks, "heads_kept_per_layer": [k] * layers,
                   "per_mask": masks, "quantiles": quant}, open(os.path.join(a.output_dir, f"round2_{task}_random.json"), "w"), indent=2)
        print(f"  {task} Random (n={a.random_masks}) {spec['metric']} quantiles: {quant[spec['metric']]}")


if __name__ == "__main__":
    main()
