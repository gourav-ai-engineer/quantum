from __future__ import annotations

import argparse
import json
import os

import torch
from datasets import load_dataset

from qfc.classical_controls import vectorized_cosine_similarity
from qfc.coverage import greedy_select
from qfc.fidelity import pairwise_fidelity
from qfc.hf_experiments import (
    _move_batch,
    collect_mean_attention_matrices,
    collect_mean_density_states,
    load_sequence_classifier,
    make_text_loader,
)
from qfc.metrics import paired_bootstrap_delta
from qfc.similarities import hilbert_schmidt_similarity


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

    losses = torch.cat(losses)
    correct = torch.cat(correct)
    labels_all = torch.cat(labels_all)
    preds_all = torch.cat(preds_all)
    return losses, correct, labels_all, preds_all


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


def build_similarity_kernels(mean_states, mean_attention):
    fidelity = [pairwise_fidelity(states) for states in mean_states]
    hs = [hilbert_schmidt_similarity(states) for states in mean_states]
    cosine = [vectorized_cosine_similarity(attn) for attn in mean_attention]
    return {
        "FidelityCoverage": fidelity,
        "HSCoverage": hs,
        "CosineCoverage": cosine,
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Representation ablation: same greedy coverage optimizer, "
            "different head-to-head similarity kernels."
        )
    )
    parser.add_argument("--calibration-size", type=int, default=256)
    parser.add_argument("--calibration-seed", type=int, default=42)
    parser.add_argument("--evaluation-size", type=int, default=-1)
    parser.add_argument("--budgets", default="3,6,9")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument(
        "--output-dir", default="results/step16_representation_ablation"
    )
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
        "evaluation_size": args.evaluation_size,
        "budgets": budgets,
        "tasks": {},
        "research_question": (
            "Does the choice of similarity geometry, holding the same greedy "
            "coverage objective and calibration data fixed, explain QFC's behavior?"
        ),
    }

    for task_name, spec in SPECS.items():
        dataset_name, config = spec["dataset"]
        dataset = (
            load_dataset(dataset_name, config)
            if config is not None
            else load_dataset(dataset_name)
        )
        train = dataset["train"]
        validation = dataset["validation"]

        if args.calibration_size > len(train):
            raise ValueError(f"{task_name}: calibration-size exceeds train")
        if args.evaluation_size < 0:
            evaluation = validation
        else:
            if args.evaluation_size > len(validation):
                raise ValueError(f"{task_name}: evaluation-size exceeds validation")
            evaluation = validation.select(range(args.evaluation_size))

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
            evaluation,
            tokenizer,
            text_fields=spec["text_fields"],
            batch_size=args.batch_size,
            max_length=args.max_length,
        )

        baseline_losses, baseline_correct, baseline_labels, baseline_preds = evaluate(
            model, eval_loader, args.device
        )

        mean_states = collect_mean_density_states(
            model, cal_loader, args.device
        )
        mean_attention = collect_mean_attention_matrices(
            model, cal_loader, args.device
        )
        kernels = build_similarity_kernels(mean_states, mean_attention)

        layers = len(mean_states)
        heads = int(mean_states[0].shape[0])
        task = {
            "model_id": spec["model_id"],
            "model_revision": spec["revision"],
            "calibration_split": "train",
            "validation_size": len(validation),
            "evaluation_size": len(evaluation),
            "baseline": {
                "accuracy": float(baseline_correct.mean()),
                "loss": float(baseline_losses.mean()),
            },
            "budgets": {},
        }
        if task_name == "mrpc":
            task["baseline"]["f1"] = f1_score(
                baseline_labels, baseline_preds
            )

        for k in budgets:
            if not 1 <= k <= heads:
                raise ValueError(
                    f"{task_name}: budget must satisfy 1 <= k <= {heads}"
                )

            selections = {}
            for method_name, layer_kernels in kernels.items():
                selections[method_name] = {}
                for layer, sim in enumerate(layer_kernels):
                    selections[method_name][layer], _ = greedy_select(sim, k)

            budget_result = {
                "retention_fraction": k / heads,
                "heads_pruned_fraction": 1.0 - k / heads,
            }

            for method_name, selection in selections.items():
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
                        seed=args.calibration_seed + k,
                    ),
                    "selected_heads_zero_based": selection,
                }
                if task_name == "mrpc":
                    entry["f1"] = f1_score(labels, preds)
                    entry["f1_delta"] = (
                        entry["f1"] - task["baseline"]["f1"]
                    )
                budget_result[method_name] = entry

            task["budgets"][str(k)] = budget_result
            print(
                f"{task_name} k={k} "
                + " | ".join(
                    f"{m}={budget_result[m]['accuracy']:.4f}"
                    for m in kernels
                )
            )

        output["tasks"][task_name] = task

    os.makedirs(args.output_dir, exist_ok=True)
    with open(
        os.path.join(args.output_dir, "summary.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(output, f, indent=2)

    print(json.dumps(
        {
            task: {
                k: {
                    method: {
                        key: value
                        for key, value in data[method].items()
                        if key in {"accuracy", "f1", "accuracy_delta", "f1_delta"}
                    }
                    for method in kernels
                }
                for k, data in result["budgets"].items()
            }
            for task, result in output["tasks"].items()
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
