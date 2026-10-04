from __future__ import annotations

import argparse
import copy
import json
import os

import torch
from datasets import load_dataset

from qfc.coverage import greedy_select, weighted_greedy_select
from qfc.fidelity import pairwise_fidelity
from qfc.hf_experiments import (
    _move_batch,
    collect_mean_density_states,
    load_sequence_classifier,
    make_text_loader,
    michel_head_importance,
    parameter_count,
    structured_prune,
)
from qfc.metrics import evaluate_per_example, paired_bootstrap_delta


def f1_binary(correct_predictions: torch.Tensor, labels: torch.Tensor):
    # This helper expects binary predictions/labels, returned on CPU.
    pred = correct_predictions.to(torch.int64)
    lab = labels.to(torch.int64)
    tp = int(((pred == 1) & (lab == 1)).sum())
    fp = int(((pred == 1) & (lab == 0)).sum())
    fn = int(((pred == 0) & (lab == 1)).sum())
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)


@torch.no_grad()
def evaluate_mrpc(model, loader, device, head_mask=None):
    model.eval()
    losses = []
    predictions = []
    labels_all = []

    for batch in loader:
        batch = _move_batch(batch, device)
        labels = batch.pop("labels")
        out = model(
            **batch,
            labels=labels,
            head_mask=head_mask,
            return_dict=True,
        )
        logits = out.logits.float()
        losses.append(
            torch.nn.functional.cross_entropy(
                logits,
                labels,
                reduction="none",
            ).cpu()
        )
        predictions.append(logits.argmax(-1).cpu())
        labels_all.append(labels.cpu())

    losses = torch.cat(losses)
    predictions = torch.cat(predictions)
    labels_all = torch.cat(labels_all)
    correct = (predictions == labels_all).float()

    return {
        "accuracy": float(correct.mean()),
        "f1": f1_binary(predictions, labels_all),
        "loss": float(losses.mean()),
        "losses": losses,
        "correct": correct,
        "predictions": predictions,
        "labels": labels_all,
    }


def mask_from(selected, layers, heads, device):
    mask = torch.zeros((layers, heads), dtype=torch.float32, device=device)
    for layer, hs in selected.items():
        mask[layer, hs] = 1.0
    return mask


def select_topk(scores, k):
    return {
        layer: torch.topk(score, k=k, largest=True).indices.tolist()
        for layer, score in enumerate(scores)
    }


def random_select(layers, heads, k, seed):
    generator = torch.Generator().manual_seed(seed)
    return {
        layer: torch.randperm(heads, generator=generator)[:k].tolist()
        for layer in range(layers)
    }


def mean_shannon(model, loader, device):
    model.eval()
    sums = None
    count = 0

    with torch.no_grad():
        for batch in loader:
            batch = _move_batch(batch, device)
            batch.pop("labels", None)
            mask = batch["attention_mask"].float()
            out = model(**batch, output_attentions=True, return_dict=True)
            if out.attentions is None:
                raise RuntimeError("no attention tensors returned")

            if sums is None:
                sums = [torch.zeros(a.shape[1], dtype=torch.float64) for a in out.attentions]

            valid = mask.sum(-1).clamp_min(1.0)
            for li, attn in enumerate(out.attentions):
                p = attn.float().clamp_min(1e-12)
                p = p * mask[:, None, None, :]
                entropy = -(p * p.log()).sum(-1)
                entropy = entropy * mask[:, None, :]
                per_example = entropy.sum(-1) / valid[:, None]
                sums[li] += per_example.sum(0).cpu().double()

            count += int(mask.shape[0])

    if sums is None or count == 0:
        raise ValueError("empty calibration loader")
    return [s / count for s in sums]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="textattack/bert-base-uncased-MRPC")
    parser.add_argument("--model-revision", default="ddeddf4a04cd7b9415b00e40b00e78f0c61a7921")
    parser.add_argument("--calibration-size", type=int, default=32)
    parser.add_argument("--evaluation-size", type=int, default=64)
    parser.add_argument("--heads-to-keep", type=int, default=9)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--output-dir", default="results/step7_mrpc_smoke")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    train = load_dataset("nyu-mll/glue", "mrpc", split="train")
    validation = load_dataset("nyu-mll/glue", "mrpc", split="validation")

    if args.calibration_size + args.evaluation_size > len(validation):
        raise ValueError("smoke sizes exceed MRPC validation size")

    # For a smoke run we use a held-out slice of validation to keep selection/evaluation
    # disjoint. The final paper run will select calibration examples from training.
    calibration = train.select(range(min(args.calibration_size, len(train))))
    evaluation = validation.select(range(args.evaluation_size))

    model, tokenizer = load_sequence_classifier(args.model_id, args.device, revision=args.model_revision)

    cal_loader = make_text_loader(
        calibration,
        tokenizer,
        text_fields=("sentence1", "sentence2"),
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    eval_loader = make_text_loader(
        evaluation,
        tokenizer,
        text_fields=("sentence1", "sentence2"),
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    base = evaluate_mrpc(model, eval_loader, args.device)
    states = collect_mean_density_states(model, cal_loader, args.device)
    layers = len(states)
    heads = int(states[0].shape[0])

    shannon = mean_shannon(model, cal_loader, args.device)
    vn = [
        torch.stack([
            -(torch.linalg.eigvalsh(rho).clamp_min(1e-12)
              * torch.linalg.eigvalsh(rho).clamp_min(1e-12).log()).sum()
            for rho in layer_states
        ])
        for layer_states in states
    ]
    michel = michel_head_importance(model, cal_loader, args.device).detach().cpu()

    selections = {}
    qfc = {}
    for layer, layer_states in enumerate(states):
        selected, _ = greedy_select(pairwise_fidelity(layer_states), args.heads_to_keep)
        qfc[layer] = selected
    selections["QFC"] = qfc
    iwqfc = {}
    for layer, layer_states in enumerate(states):
        sim = pairwise_fidelity(layer_states)
        raw = michel[layer].clamp_min(0.0)
        quality = raw / raw.sum() if float(raw.sum()) > 0 else torch.full_like(raw, 1.0 / len(raw))
        iwqfc[layer], _ = weighted_greedy_select(sim, quality, args.heads_to_keep)
    selections["IWQFC"] = iwqfc
    selections["VonNeumann"] = select_topk(vn, args.heads_to_keep)
    selections["Shannon"] = select_topk(shannon, args.heads_to_keep)
    selections["MichelGate"] = select_topk(
        [michel[i] for i in range(layers)],
        args.heads_to_keep,
    )
    selections["Random"] = random_select(
        layers, heads, args.heads_to_keep, seed=2026
    )

    results = {}
    for name, selected in selections.items():
        mask = mask_from(selected, layers, heads, args.device)
        metric = evaluate_mrpc(model, eval_loader, args.device, mask)
        acc_ci = paired_bootstrap_delta(
            metric["correct"], base["correct"], n_boot=args.bootstrap, seed=10
        )
        loss_ci = paired_bootstrap_delta(
            metric["losses"], base["losses"], n_boot=args.bootstrap, seed=11
        )
        results[name] = {
            "accuracy": metric["accuracy"],
            "f1": metric["f1"],
            "loss": metric["loss"],
            "accuracy_delta_vs_full": acc_ci,
            "loss_delta_vs_full": loss_ci,
            "selected_heads_zero_based": selected,
        }

    physical = copy.deepcopy(model).to(args.device)
    physical, pruned = structured_prune(physical, qfc)
    phys = evaluate_mrpc(physical, eval_loader, args.device)

    results["QFC_structured"] = {
        "accuracy": phys["accuracy"],
        "f1": phys["f1"],
        "loss": phys["loss"],
        "parameters": parameter_count(physical),
        "parameter_fraction_remaining": parameter_count(physical)
        / parameter_count(model),
        "pruned_heads_zero_based": pruned,
    }

    payload = {
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "dataset": "nyu-mll/glue/mrpc",
        "calibration_split": "train",
        "evaluation_split": "validation_smoke_slice",
        "calibration_size": args.calibration_size,
        "evaluation_size": args.evaluation_size,
        "heads_to_keep_per_layer": args.heads_to_keep,
        "max_length": args.max_length,
        "baseline": {
            "accuracy": base["accuracy"],
            "f1": base["f1"],
            "loss": base["loss"],
        },
        "methods": results,
        "research_note": (
            "Engineering smoke test. Final MRPC paper evaluation must use the complete "
            "validation split with calibration separated from evaluation and repeated runs."
        ),
    }

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
