from __future__ import annotations

import argparse
import json
import os

import torch
from datasets import load_dataset

from qfc.adaptive import layer_adaptive_greedy
from qfc.coverage import greedy_select, weighted_greedy_select
from qfc.fidelity import pairwise_fidelity
from qfc.hf_experiments import (
    _move_batch,
    collect_mean_density_states,
    load_sequence_classifier,
    make_text_loader,
    michel_head_importance,
)
from qfc.metrics import paired_bootstrap_delta


SPECS = {
    "sst2": {
        "model_id": "textattack/bert-base-uncased-SST-2",
        "revision": "205ffbd1bc5c5b89802266f4948a601f53556b00",
        "loader": lambda: load_dataset("stanfordnlp/sst2"),
        "text_fields": ("sentence",),
    },
    "mrpc": {
        "model_id": "textattack/bert-base-uncased-MRPC",
        "revision": "ddeddf4a04cd7b9415b00e40b00e78f0c61a7921",
        "loader": lambda: load_dataset("glue", "mrpc"),
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
            batch.pop("labels", None)
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


def random_select(layers, heads, total_budget, seed, min_per_layer=1):
    if total_budget < layers * min_per_layer:
        raise ValueError("random budget is too small for the layer minimum")
    g = torch.Generator().manual_seed(seed)
    selected = {layer: [] for layer in range(layers)}
    for layer in range(layers):
        selected[layer].append(int(torch.randint(heads, (1,), generator=g)))
    remaining = total_budget - layers * min_per_layer
    available = [
        (layer, head)
        for layer in range(layers)
        for head in range(heads)
        if head not in selected[layer]
    ]
    perm = torch.randperm(len(available), generator=g).tolist()
    for idx in perm[:remaining]:
        layer, head = available[idx]
        selected[layer].append(head)
    return selected


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


def allocation_counts(selection):
    return {str(layer): len(heads) for layer, heads in selection.items()}


def main():
    parser = argparse.ArgumentParser(
        description="Layer-adaptive QFC allocation diagnostic."
    )
    parser.add_argument("--calibration-size", type=int, default=32)
    parser.add_argument("--calibration-seed", type=int, default=42)
    parser.add_argument("--evaluation-size", type=int, default=64)
    parser.add_argument("--heads-to-keep", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--bootstrap", type=int, default=300)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default="results/step15_layer_adaptive_smoke")
    args = parser.parse_args()

    output = {
        "calibration_size": args.calibration_size,
        "calibration_seed": args.calibration_seed,
        "evaluation_size": args.evaluation_size,
        "equal_heads_to_keep": args.heads_to_keep,
        "tasks": {},
        "research_note": (
            "Diagnostic ablation: compare equal-per-layer selection against "
            "global-budget layer-adaptive allocation at the same total head budget. "
            "The adaptive seeded procedure is a heuristic and carries no new "
            "1-1/e claim beyond the base cardinality result."
        ),
    }

    for task_name, spec in SPECS.items():
        ds = spec["loader"]()
        train = ds["train"]
        validation = ds["validation"]
        if args.calibration_size > len(train):
            raise ValueError(f"{task_name}: calibration-size exceeds train")
        if args.evaluation_size > len(validation):
            raise ValueError(f"{task_name}: evaluation-size exceeds validation")

        calibration = train.shuffle(seed=args.calibration_seed).select(
            range(args.calibration_size)
        )
        evaluation = validation.select(range(args.evaluation_size))

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
            evaluation,
            tokenizer,
            text_fields=spec["text_fields"],
            batch_size=args.batch_size,
            max_length=args.max_length,
        )

        (
            baseline_losses,
            baseline_correct,
            baseline_labels,
            baseline_predictions,
        ) = evaluate(model, eval_loader, args.device)

        states = collect_mean_density_states(model, cal_loader, args.device)
        shannon = shannon_scores(model, cal_loader, args.device)
        vn = vn_scores(states)
        michel = michel_head_importance(model, cal_loader, args.device).detach().cpu()
        similarities = [pairwise_fidelity(layer_states) for layer_states in states]

        layers = len(states)
        heads = int(states[0].shape[0])
        total_budget = layers * args.heads_to_keep

        equal_qfc = {}
        equal_iwqfc = {}
        adaptive_qfc, _ = layer_adaptive_greedy(
            similarities,
            total_budget=total_budget,
            min_per_layer=1,
        )
        adaptive_iwqfc_weights = []
        for layer in range(layers):
            raw = michel[layer].clamp_min(0.0)
            if float(raw.sum()) > 0:
                adaptive_iwqfc_weights.append(raw)
            else:
                adaptive_iwqfc_weights.append(
                    torch.ones_like(raw)
                )
            equal_qfc[layer], _ = greedy_select(
                similarities[layer], args.heads_to_keep
            )
            equal_iwqfc[layer], _ = weighted_greedy_select(
                similarities[layer],
                adaptive_iwqfc_weights[layer],
                args.heads_to_keep,
            )
        adaptive_iwqfc, _ = layer_adaptive_greedy(
            similarities,
            total_budget=total_budget,
            weights=adaptive_iwqfc_weights,
            min_per_layer=1,
        )

        selections = {
            "EqualQFC": equal_qfc,
            "AdaptiveQFC": adaptive_qfc,
            "EqualIWQFC": equal_iwqfc,
            "AdaptiveIWQFC": adaptive_iwqfc,
            "VonNeumann": select_topk(vn, args.heads_to_keep),
            "MichelGate": select_topk(
                [michel[i] for i in range(layers)],
                args.heads_to_keep,
            ),
            "Random": random_select(
                layers,
                heads,
                total_budget,
                seed=args.calibration_seed + 1000,
            ),
        }

        task = {
            "model_id": spec["model_id"],
            "model_revision": spec["revision"],
            "baseline": {
                "accuracy": float(baseline_correct.mean()),
                "loss": float(baseline_losses.mean()),
            },
            "total_budget": total_budget,
            "methods": {},
        }
        if task_name == "mrpc":
            task["baseline"]["f1"] = f1_score(
                baseline_labels, baseline_predictions
            )

        for name, selection in selections.items():
            mask = make_mask(selection, layers, heads, args.device)
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
                "selection_counts": allocation_counts(selection),
                "selected_heads_zero_based": selection,
                "accuracy_bootstrap_delta": paired_bootstrap_delta(
                    correct,
                    baseline_correct,
                    n_boot=args.bootstrap,
                    seed=args.calibration_seed + 20000,
                ),
            }
            if task_name == "mrpc":
                entry["f1"] = f1_score(labels, predictions)
                entry["f1_delta"] = entry["f1"] - task["baseline"]["f1"]
            task["methods"][name] = entry

        output["tasks"][task_name] = task
        print(
            f"{task_name}: "
            + " | ".join(
                f"{name}={task['methods'][name]['accuracy']:.4f}"
                for name in (
                    "EqualQFC",
                    "AdaptiveQFC",
                    "EqualIWQFC",
                    "AdaptiveIWQFC",
                    "VonNeumann",
                    "MichelGate",
                    "Random",
                )
            )
        )

    os.makedirs(args.output_dir, exist_ok=True)
    with open(
        os.path.join(args.output_dir, "summary.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(output, f, indent=2)


if __name__ == "__main__":
    main()
