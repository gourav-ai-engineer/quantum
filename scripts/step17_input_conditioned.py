from __future__ import annotations

import argparse
import json
import os

import torch
from datasets import load_dataset

from qfc.conditional import (
    collect_per_sample_density_states,
    conditional_fidelity_kernels,
    conditional_greedy_select,
    conditional_weighted_greedy_select,
)
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


@torch.no_grad()
def evaluate(model, loader, device, head_mask=None):
    model.eval()
    losses = []
    correct = []
    labels_all = []
    preds_all = []

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
        preds = logits.argmax(-1).detach().cpu()
        labels_cpu = labels.detach().cpu()
        correct.append((preds == labels_cpu).float())
        labels_all.append(labels_cpu)
        preds_all.append(preds)

    return (
        torch.cat(losses),
        torch.cat(correct),
        torch.cat(labels_all),
        torch.cat(preds_all),
    )


def f1_score(labels, predictions):
    labels = labels.long()
    predictions = predictions.long()
    tp = ((predictions == 1) & (labels == 1)).sum().item()
    fp = ((predictions == 1) & (labels == 0)).sum().item()
    fn = ((predictions == 0) & (labels == 1)).sum().item()
    denom = 2 * tp + fp + fn
    return 0.0 if denom == 0 else float(2 * tp / denom)


def make_mask(selection, layers, heads, device):
    mask = torch.zeros((layers, heads), dtype=torch.float32, device=device)
    for layer, chosen in selection.items():
        mask[layer, chosen] = 1.0
    return mask


def select_average_fidelity(per_sample_kernel, k):
    mean_kernel = per_sample_kernel.mean(dim=0)
    return greedy_select(mean_kernel, k)[0]


def select_average_weighted_fidelity(
    per_sample_kernel,
    weights,
    k,
):
    mean_kernel = per_sample_kernel.mean(dim=0)
    return weighted_greedy_select(mean_kernel, weights, k)[0]


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Input-conditioned QFC ablation: compare fidelity of mean states, "
            "mean per-example fidelity, and direct per-example coverage."
        )
    )
    parser.add_argument("--calibration-size", type=int, default=128)
    parser.add_argument("--calibration-seed", type=int, default=42)
    parser.add_argument("--evaluation-size", type=int, default=64)
    parser.add_argument("--heads-to-keep", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default="results/step17_input_conditioned")
    args = parser.parse_args()

    output = {
        "calibration_size": args.calibration_size,
        "calibration_seed": args.calibration_seed,
        "evaluation_size": args.evaluation_size,
        "heads_to_keep_per_layer": args.heads_to_keep,
        "tasks": {},
        "research_question": (
            "Does QFC underperform because fidelity is computed after averaging "
            "density operators, rather than optimizing input-conditioned fidelity coverage?"
        ),
    }

    for task_name, spec in SPECS.items():
        dataset_name, config = spec["dataset"]
        ds = (
            load_dataset(dataset_name, config)
            if config is not None
            else load_dataset(dataset_name)
        )
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

        baseline_losses, baseline_correct, baseline_labels, baseline_preds = evaluate(
            model, eval_loader, args.device
        )

        # A: current formulation — fidelity between aggregated mean states.
        mean_states = collect_mean_density_states(
            model, cal_loader, args.device
        )

        # B/C: retain the calibration examples and their per-example geometry.
        per_sample_states = collect_per_sample_density_states(
            model, cal_loader, args.device
        )
        per_sample_kernels = conditional_fidelity_kernels(
            per_sample_states,
            max_pairs_per_chunk=64,
        )

        # Shared gradient salience for the weighted variants.
        michel = michel_head_importance(
            model, cal_loader, args.device
        ).detach().cpu()

        layers = len(mean_states)
        heads = int(mean_states[0].shape[0])

        selections = {
            "MeanStateQFC": {},
            "AverageFidelityQFC": {},
            "ConditionalQFC": {},
            "AverageFidelityIWQFC": {},
            "ConditionalIWQFC": {},
        }

        for layer in range(layers):
            mean_sim = pairwise_fidelity(mean_states[layer])
            selections["MeanStateQFC"][layer], _ = greedy_select(
                mean_sim, args.heads_to_keep
            )

            kernel = per_sample_kernels[layer]
            selections["AverageFidelityQFC"][layer] = select_average_fidelity(
                kernel, args.heads_to_keep
            )
            selections["ConditionalQFC"][layer], _ = conditional_greedy_select(
                kernel, args.heads_to_keep
            )

            raw = michel[layer].clamp_min(0.0)
            weights = (
                raw / raw.sum()
                if float(raw.sum()) > 0
                else torch.full_like(raw, 1.0 / len(raw))
            )
            selections["AverageFidelityIWQFC"][layer] = select_average_weighted_fidelity(
                kernel, weights, args.heads_to_keep
            )
            selections["ConditionalIWQFC"][layer], _ = conditional_weighted_greedy_select(
                kernel, weights, args.heads_to_keep
            )

        task = {
            "model_id": spec["model_id"],
            "model_revision": spec["revision"],
            "calibration_split": "train",
            "evaluation_split": "validation_fixed_slice",
            "calibration_size": args.calibration_size,
            "evaluation_size": args.evaluation_size,
            "baseline": {
                "accuracy": float(baseline_correct.mean()),
                "loss": float(baseline_losses.mean()),
            },
            "methods": {},
        }

        if task_name == "mrpc":
            task["baseline"]["f1"] = f1_score(
                baseline_labels, baseline_preds
            )

        for name, selection in selections.items():
            mask = make_mask(selection, layers, heads, args.device)
            losses, correct, labels, preds = evaluate(
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
                "accuracy_bootstrap_delta": paired_bootstrap_delta(
                    correct,
                    baseline_correct,
                    n_boot=args.bootstrap,
                    seed=args.calibration_seed + 30000,
                ),
                "selected_heads_zero_based": selection,
            }
            if task_name == "mrpc":
                entry["f1"] = f1_score(labels, preds)
                entry["f1_delta"] = (
                    entry["f1"] - task["baseline"]["f1"]
                )
            task["methods"][name] = entry

        output["tasks"][task_name] = task
        print(
            f"{task_name}: "
            + " | ".join(
                f"{name}={task['methods'][name]['accuracy']:.4f}"
                for name in selections
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
