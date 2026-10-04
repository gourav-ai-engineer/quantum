from __future__ import annotations

import argparse
import json
import os
from statistics import mean

import torch
from datasets import load_dataset

from qfc.classical_controls import vectorized_cosine_similarity
from qfc.coverage import coverage_value, greedy_select
from qfc.fidelity import pairwise_fidelity
from qfc.hf_experiments import (
    _move_batch,
    collect_mean_attention_matrices,
    collect_mean_density_states,
    load_sequence_classifier,
    make_text_loader,
)
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
def evaluate_loss_accuracy(model, loader, device, head_mask):
    model.eval()
    losses = []
    correct = []
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
        correct.append(
            (logits.argmax(-1).cpu() == labels.cpu()).float()
        )
    losses = torch.cat(losses)
    correct = torch.cat(correct)
    return float(losses.mean()), float(correct.mean())


def rankdata(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        rank = (i + j - 1) / 2.0
        for pos in range(i, j):
            ranks[order[pos]] = rank
        i = j
    return ranks


def spearman(x, y):
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    rx = torch.tensor(rankdata(x), dtype=torch.float64)
    ry = torch.tensor(rankdata(y), dtype=torch.float64)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = rx.norm() * ry.norm()
    return 0.0 if float(denom) == 0.0 else float((rx @ ry) / denom)


def make_local_mask(layers, heads, target_layer, selected, device):
    mask = torch.ones((layers, heads), dtype=torch.float32, device=device)
    mask[target_layer, :] = 0.0
    mask[target_layer, selected] = 1.0
    return mask


def sample_subsets(heads, k, count, seed):
    g = torch.Generator().manual_seed(seed)
    subsets = []
    seen = set()
    while len(subsets) < count:
        subset = tuple(
            sorted(torch.randperm(heads, generator=g)[:k].tolist())
        )
        if subset not in seen:
            seen.add(subset)
            subsets.append(subset)
    return subsets


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Objective-alignment audit: test whether QFC coverage behaves as a "
            "surrogate for local downstream loss under head subset perturbations."
        )
    )
    parser.add_argument("--calibration-size", type=int, default=128)
    parser.add_argument("--calibration-seed", type=int, default=42)
    parser.add_argument("--evaluation-size", type=int, default=64)
    parser.add_argument("--heads-to-keep", type=int, default=6)
    parser.add_argument("--random-subsets", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--output-dir", default="results/step19_objective_alignment"
    )
    args = parser.parse_args()

    output = {
        "calibration_size": args.calibration_size,
        "calibration_seed": args.calibration_seed,
        "evaluation_size": args.evaluation_size,
        "heads_to_keep_per_layer": args.heads_to_keep,
        "random_subsets_per_layer": args.random_subsets,
        "tasks": {},
        "research_question": (
            "Does fidelity coverage correlate with downstream usefulness when "
            "head subsets are perturbed within a layer?"
        ),
        "interpretation_rule": (
            "For local pruning, higher coverage should be associated with lower "
            "loss if coverage is a useful surrogate. This is diagnostic evidence, "
            "not a proof of causal alignment."
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

        layers = int(model.config.num_hidden_layers)
        heads = int(model.config.num_attention_heads)

        all_one = torch.ones(
            (layers, heads), dtype=torch.float32, device=args.device
        )
        baseline_loss, baseline_acc = evaluate_loss_accuracy(
            model, eval_loader, args.device, all_one
        )

        states = collect_mean_density_states(
            model, cal_loader, args.device
        )
        attentions = collect_mean_attention_matrices(
            model, cal_loader, args.device
        )
        fidelity = [pairwise_fidelity(x) for x in states]
        hs = [hilbert_schmidt_similarity(x) for x in states]
        cosine = [vectorized_cosine_similarity(x) for x in attentions]

        task_result = {
            "baseline": {
                "loss": baseline_loss,
                "accuracy": baseline_acc,
            },
            "layers": {},
        }

        for layer in range(layers):
            qfc_selection, _ = greedy_select(
                fidelity[layer], args.heads_to_keep
            )
            subset_list = sample_subsets(
                heads,
                args.heads_to_keep,
                args.random_subsets,
                seed=args.calibration_seed + 1000 + layer,
            )
            if tuple(sorted(qfc_selection)) not in subset_list:
                subset_list[-1] = tuple(sorted(qfc_selection))

            records = []
            for subset in subset_list:
                mask = make_local_mask(
                    layers, heads, layer, subset, args.device
                )
                loss, accuracy = evaluate_loss_accuracy(
                    model, eval_loader, args.device, mask
                )
                records.append(
                    {
                        "subset": list(subset),
                        "fidelity_coverage": float(
                            coverage_value(fidelity[layer], subset)
                        ),
                        "hs_coverage": float(
                            coverage_value(hs[layer], subset)
                        ),
                        "cosine_coverage": float(
                            coverage_value(cosine[layer], subset)
                        ),
                        "loss": loss,
                        "accuracy": accuracy,
                        "loss_delta": loss - baseline_loss,
                        "accuracy_delta": accuracy - baseline_acc,
                    }
                )

            layer_out = {
                "qfc_selection": qfc_selection,
                "spearman_coverage_vs_loss": {},
                "spearman_coverage_vs_accuracy": {},
                "qfc_record": next(
                    r for r in records
                    if r["subset"] == sorted(qfc_selection)
                ),
                "records": records,
            }
            for name, key in (
                ("fidelity", "fidelity_coverage"),
                ("hilbert_schmidt", "hs_coverage"),
                ("cosine", "cosine_coverage"),
            ):
                layer_out["spearman_coverage_vs_loss"][name] = spearman(
                    [r[key] for r in records],
                    [r["loss"] for r in records],
                )
                layer_out["spearman_coverage_vs_accuracy"][name] = spearman(
                    [r[key] for r in records],
                    [r["accuracy"] for r in records],
                )
            task_result["layers"][str(layer)] = layer_out

        output["tasks"][task_name] = task_result

        print(
            f"{task_name} baseline_loss={baseline_loss:.4f} "
            f"baseline_acc={baseline_acc:.4f}"
        )
        for layer, data in task_result["layers"].items():
            print(
                f"layer={layer} "
                f"rho(F,loss)={data['spearman_coverage_vs_loss']['fidelity']:.3f} "
                f"rho(HS,loss)={data['spearman_coverage_vs_loss']['hilbert_schmidt']:.3f} "
                f"rho(Cos,loss)={data['spearman_coverage_vs_loss']['cosine']:.3f}"
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
