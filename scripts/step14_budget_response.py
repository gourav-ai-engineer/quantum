from __future__ import annotations

import argparse
import json
import os
from statistics import mean, pstdev

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
)


MODELS = {
    "sst2": {
        "model_id": "textattack/bert-base-uncased-SST-2",
        "revision": "205ffbd1bc5c5b89802266f4948a601f53556b00",
        "dataset": ("stanfordnlp/sst2", None),
        "text_fields": ("sentence",),
    },
    "mrpc": {
        "model_id": "textattack/bert-base-uncased-MRPC",
        "revision": "ddeddf4a04cd7b9415b00e40b00e78f0c61a7921",
        "dataset": ("glue", "mrpc"),
        "text_fields": ("sentence1", "sentence2"),
    },
}


def shannon_scores(model, loader, device):
    model.eval()
    sums = None
    count = 0
    with torch.no_grad():
        for batch in loader:
            batch = _move_batch(batch, device)
            labels = batch.pop("labels", None)
            del labels
            mask = batch["attention_mask"].float()
            out = model(**batch, output_attentions=True, return_dict=True)
            if out.attentions is None:
                raise RuntimeError("attention outputs unavailable")
            if sums is None:
                sums = [
                    torch.zeros(a.shape[1], dtype=torch.float64)
                    for a in out.attentions
                ]

            valid_queries = mask.sum(-1).clamp_min(1.0)
            for li, attn in enumerate(out.attentions):
                p = attn.float().clamp_min(1e-12)
                p = p * mask[:, None, None, :]
                entropy = -(p * p.log()).sum(-1)
                entropy = entropy * mask[:, None, :]
                per_example = entropy.sum(-1) / valid_queries[:, None]
                sums[li] += per_example.sum(0).cpu().double()
            count += int(mask.shape[0])

    if sums is None or count == 0:
        raise ValueError("empty loader")
    return [s / count for s in sums]


def vn_scores(mean_states):
    result = []
    for states in mean_states:
        vals = []
        for rho in states:
            eig = torch.linalg.eigvalsh(0.5 * (rho + rho.T)).clamp_min(1e-12)
            vals.append(-(eig * eig.log()).sum())
        result.append(torch.stack(vals))
    return result


def select_topk(scores, k):
    return {
        layer: torch.topk(score, k=k, largest=True).indices.tolist()
        for layer, score in enumerate(scores)
    }


def random_select(layers, heads, k, seed):
    g = torch.Generator().manual_seed(seed)
    return {
        layer: torch.randperm(heads, generator=g)[:k].tolist()
        for layer in range(layers)
    }


def make_mask(selection, layers, heads, device):
    mask = torch.zeros((layers, heads), dtype=torch.float32, device=device)
    for layer, hs in selection.items():
        mask[layer, hs] = 1.0
    return mask


def f1_score(labels, predictions):
    labels = labels.long()
    predictions = predictions.long()
    tp = ((predictions == 1) & (labels == 1)).sum().item()
    fp = ((predictions == 1) & (labels == 0)).sum().item()
    fn = ((predictions == 0) & (labels == 1)).sum().item()
    denom = 2 * tp + fp + fn
    return 0.0 if denom == 0 else float(2 * tp / denom)


@torch.no_grad()
def evaluate(model, loader, device, head_mask=None):
    model.eval()
    losses = []
    labels_all = []
    predictions_all = []

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
                logits, labels, reduction="none"
            ).cpu()
        )
        labels_all.append(labels.detach().cpu())
        predictions_all.append(logits.argmax(-1).detach().cpu())

    losses = torch.cat(losses)
    labels_all = torch.cat(labels_all)
    predictions_all = torch.cat(predictions_all)
    correct = (predictions_all == labels_all).float()
    return losses, correct, labels_all, predictions_all


def mean_std(values):
    return {
        "mean": mean(values),
        "std": pstdev(values) if len(values) > 1 else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Full validation budget-response experiment comparing QFC and "
            "gradient-weighted QFC across SST-2 and MRPC."
        )
    )
    parser.add_argument("--calibration-size", type=int, default=256)
    parser.add_argument("--calibration-seed", type=int, default=42)
    parser.add_argument("--budgets", default="3,6,9")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--output-dir", default="results/step14_budget_response")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    budgets = [int(x) for x in args.budgets.split(",") if x.strip()]
    if not budgets:
        raise ValueError("at least one budget is required")

    output = {
        "calibration_size": args.calibration_size,
        "calibration_seed": args.calibration_seed,
        "budgets_heads_kept_per_layer": budgets,
        "tasks": {},
        "research_note": (
            "Budget-response study at one fixed calibration seed. "
            "Validation sets are full and fixed; calibration is sampled from train. "
            "This experiment is for response curves, not seed-level statistical claims."
        ),
    }

    for task_name, spec in MODELS.items():
        dataset_name, config = spec["dataset"]
        train = (
            load_dataset(dataset_name, config, split="train")
            if config is not None
            else load_dataset(dataset_name, split="train")
        )
        validation = (
            load_dataset(dataset_name, config, split="validation")
            if config is not None
            else load_dataset(dataset_name, split="validation")
        )

        if args.calibration_size > len(train):
            raise ValueError(
                f"{task_name}: calibration-size exceeds training split"
            )

        calibration = train.shuffle(seed=args.calibration_seed).select(
            range(args.calibration_size)
        )

        model, tokenizer = load_sequence_classifier(
            spec["model_id"],
            args.device,
            revision=spec["revision"],
        )

        cal_loader = make_text_loader(
            calibration,
            tokenizer,
            text_fields=spec["text_fields"],
            batch_size=args.batch_size,
            max_length=args.max_length,
        )
        eval_loader = make_text_loader(
            validation,
            tokenizer,
            text_fields=spec["text_fields"],
            batch_size=args.batch_size,
            max_length=args.max_length,
        )

        baseline_losses, baseline_correct, baseline_labels, baseline_predictions = (
            evaluate(model, eval_loader, args.device)
        )

        states = collect_mean_density_states(
            model, cal_loader, args.device
        )
        shannon = shannon_scores(model, cal_loader, args.device)
        vn = vn_scores(states)
        michel = michel_head_importance(
            model, cal_loader, args.device
        ).detach().cpu()

        layers = len(states)
        heads = int(states[0].shape[0])
        methods = ("QFC", "IWQFC", "VonNeumann", "Shannon", "MichelGate", "Random")

        task_result = {
            "model_id": spec["model_id"],
            "model_revision": spec["revision"],
            "dataset": dataset_name if config is None else f"{dataset_name}/{config}",
            "train_size": len(train),
            "validation_size": len(validation),
            "baseline": {
                "accuracy": float(baseline_correct.mean()),
                "loss": float(baseline_losses.mean()),
            },
            "budgets": {},
        }

        if task_name == "mrpc":
            task_result["baseline"]["f1"] = f1_score(
                baseline_labels, baseline_predictions
            )

        for k in budgets:
            if k <= 0 or k > heads:
                raise ValueError(
                    f"{task_name}: heads-to-keep must be in [1, {heads}], got {k}"
                )

            qfc = {}
            iwqfc = {}
            for layer, layer_states in enumerate(states):
                sim = pairwise_fidelity(layer_states)
                qfc[layer], _ = greedy_select(sim, k)

                raw = michel[layer].clamp_min(0.0)
                if float(raw.sum()) > 0:
                    weights = raw / raw.sum()
                else:
                    weights = torch.full_like(raw, 1.0 / len(raw))
                iwqfc[layer], _ = weighted_greedy_select(
                    sim, weights, k
                )

            selections = {
                "QFC": qfc,
                "IWQFC": iwqfc,
                "VonNeumann": select_topk(vn, k),
                "Shannon": select_topk(shannon, k),
                "MichelGate": select_topk(
                    [michel[i] for i in range(layers)], k
                ),
                "Random": random_select(
                    layers, heads, k, seed=args.calibration_seed + 1000 + k
                ),
            }

            budget_result = {}
            for name in methods:
                mask = make_mask(
                    selections[name], layers, heads, args.device
                )
                losses, correct, labels, predictions = evaluate(
                    model, eval_loader, args.device, mask
                )
                entry = {
                    "accuracy": float(correct.mean()),
                    "accuracy_delta": float(
                        correct.mean() - baseline_correct.mean()
                    ),
                    "loss": float(losses.mean()),
                    "loss_delta": float(
                        losses.mean() - baseline_losses.mean()
                    ),
                    "selected_heads_zero_based": selections[name],
                }
                if task_name == "mrpc":
                    entry["f1"] = f1_score(labels, predictions)
                    entry["f1_delta"] = entry["f1"] - task_result["baseline"]["f1"]

                budget_result[name] = entry

            budget_result["retention_fraction"] = k / heads
            budget_result["heads_pruned_fraction"] = 1.0 - (k / heads)
            task_result["budgets"][str(k)] = budget_result

            print(
                f"{task_name} k={k} "
                + " | ".join(
                    f"{m}={budget_result[m]['accuracy']:.4f}"
                    for m in methods
                )
            )

        output["tasks"][task_name] = task_result

    os.makedirs(args.output_dir, exist_ok=True)
    path = os.path.join(args.output_dir, "summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(json.dumps(
        {
            task: {
                k: {
                    method: {
                        key: value
                        for key, value in budget_data[method].items()
                        if key in {"accuracy", "f1", "accuracy_delta", "f1_delta"}
                    }
                    for method in ("QFC", "IWQFC", "VonNeumann", "Shannon", "MichelGate", "Random")
                }
                for k, budget_data in task_data["budgets"].items()
            }
            for task, task_data in output["tasks"].items()
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
