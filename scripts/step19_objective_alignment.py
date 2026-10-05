from __future__ import annotations

import argparse
import gzip
import json
import os

import torch
from datasets import load_dataset

from qfc.alignment import bootstrap_spearman, check_subset_budget
from qfc.baselines import f1_binary
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

COVERAGES = (
    ("fidelity", "fidelity_coverage"),
    ("hilbert_schmidt", "hs_coverage"),
    ("cosine", "cosine_coverage"),
)


@torch.no_grad()
def evaluate_predictions(model, loader, device, head_mask):
    model.eval()
    losses = []
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
        labels_all.append(labels.cpu())
        preds_all.append(logits.argmax(-1).cpu())
    return torch.cat(losses), torch.cat(labels_all), torch.cat(preds_all)


def evaluate_loss_accuracy(model, loader, device, head_mask):
    losses, labels, preds = evaluate_predictions(model, loader, device, head_mask)
    return float(losses.mean()), float((preds == labels).float().mean())


def make_local_mask(layers, heads, target_layer, selected, device):
    mask = torch.ones((layers, heads), dtype=torch.float32, device=device)
    mask[target_layer, :] = 0.0
    mask[target_layer, selected] = 1.0
    return mask


def sample_subsets(heads, k, count, seed):
    check_subset_budget(heads, k, count)
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


def correlation_block(records, n_boot, seed):
    """Spearman (with bootstrap CI) of every coverage vs loss and accuracy."""
    block = {
        "spearman_coverage_vs_loss": {},
        "spearman_coverage_vs_accuracy": {},
        "spearman_coverage_vs_loss_ci": {},
        "spearman_coverage_vs_accuracy_ci": {},
    }
    for name, key in COVERAGES:
        coverage = [r[key] for r in records]
        for target in ("loss", "accuracy"):
            ci = bootstrap_spearman(
                coverage,
                [r[target] for r in records],
                n_boot=n_boot,
                seed=seed,
            )
            block[f"spearman_coverage_vs_{target}"][name] = ci["estimate"]
            block[f"spearman_coverage_vs_{target}_ci"][name] = ci
    return block


def fmt_ci(ci):
    if ci["ci_low"] is None:
        return f"{ci['estimate']:.3f} [undefined: constant input]"
    return f"{ci['estimate']:.3f} [{ci['ci_low']:.3f},{ci['ci_high']:.3f}]"


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Objective-alignment audit: test whether QFC coverage behaves as a "
            "surrogate for local downstream loss under head subset perturbations."
        )
    )
    parser.add_argument("--calibration-size", type=int, default=128)
    parser.add_argument("--calibration-seed", type=int, default=42)
    parser.add_argument(
        "--evaluation-size",
        type=int,
        default=256,
        help="first N validation examples; -1 = full validation split",
    )
    parser.add_argument("--heads-to-keep", type=int, default=6)
    parser.add_argument("--random-subsets", type=int, default=300)
    parser.add_argument("--bootstrap", type=int, default=1000)
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
        "bootstrap_resamples": args.bootstrap,
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
        "statistics_note": (
            "Per-layer CIs are percentile bootstrap intervals over the sampled "
            "subsets of that layer. The pooled Spearman concatenates subsets from "
            "all layers (its CI resamples those pooled subsets) and therefore mixes "
            "between-layer and within-layer variation; read it alongside the "
            "per-layer values. A null CI (degenerate=true) means a constant series."
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
        if args.evaluation_size < 0:
            evaluation = validation
        else:
            if args.evaluation_size > len(validation):
                raise ValueError(
                    f"{task_name}: evaluation-size exceeds validation split "
                    f"({len(validation)})"
                )
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

        layers = int(model.config.num_hidden_layers)
        heads = int(model.config.num_attention_heads)

        all_one = torch.ones(
            (layers, heads), dtype=torch.float32, device=args.device
        )
        base_losses, base_labels, base_preds = evaluate_predictions(
            model, eval_loader, args.device, all_one
        )
        baseline_loss = float(base_losses.mean())
        baseline_acc = float((base_preds == base_labels).float().mean())

        # Pre-flight: head_mask must actually change the model, otherwise every
        # subset has identical loss and all correlations are vacuous (this
        # happens with Transformers versions that ignore head_mask).
        zero_loss, _ = evaluate_loss_accuracy(
            model, eval_loader, args.device, torch.zeros_like(all_one)
        )
        if abs(zero_loss - baseline_loss) < 1e-9:
            raise RuntimeError(
                f"{task_name}: masking every head leaves the loss unchanged "
                f"({baseline_loss:.6f}); head_mask is being ignored. "
                "Use the pinned transformers==4.51.3."
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
            "evaluation_size": len(evaluation),
            "baseline": {
                "loss": baseline_loss,
                "accuracy": baseline_acc,
            },
            "layers": {},
        }
        if task_name == "mrpc":
            task_result["baseline"]["f1"] = f1_binary(base_labels, base_preds)

        pooled_records = []
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
                "loss_constant_across_subsets": (
                    max(r["loss"] for r in records)
                    == min(r["loss"] for r in records)
                ),
                "qfc_record": next(
                    r for r in records
                    if r["subset"] == sorted(qfc_selection)
                ),
                "records": records,
            }
            layer_out.update(
                correlation_block(
                    records, args.bootstrap, seed=args.calibration_seed + 5000 + layer
                )
            )
            task_result["layers"][str(layer)] = layer_out
            pooled_records.extend(records)

        pooled = {"n_records": len(pooled_records)}
        pooled.update(
            correlation_block(
                pooled_records, args.bootstrap, seed=args.calibration_seed + 9999
            )
        )
        task_result["pooled"] = pooled
        output["tasks"][task_name] = task_result

        print(
            f"{task_name} eval_n={len(evaluation)} "
            f"baseline_loss={baseline_loss:.4f} "
            f"baseline_acc={baseline_acc:.4f}"
            + (
                f" baseline_f1={task_result['baseline']['f1']:.4f}"
                if task_name == "mrpc"
                else ""
            )
        )
        for layer, data in task_result["layers"].items():
            ci = data["spearman_coverage_vs_loss_ci"]
            print(
                f"layer={layer} "
                f"rho(F,loss)={fmt_ci(ci['fidelity'])} "
                f"rho(HS,loss)={fmt_ci(ci['hilbert_schmidt'])} "
                f"rho(Cos,loss)={fmt_ci(ci['cosine'])}"
            )
        ci = pooled["spearman_coverage_vs_loss_ci"]
        print(
            f"pooled(n={pooled['n_records']}) "
            f"rho(F,loss)={fmt_ci(ci['fidelity'])} "
            f"rho(HS,loss)={fmt_ci(ci['hilbert_schmidt'])} "
            f"rho(Cos,loss)={fmt_ci(ci['cosine'])}"
        )

    os.makedirs(args.output_dir, exist_ok=True)
    # Per-subset records are large (about 1 KB each). summary.json keeps everything the
    # ledger and the decision rule need; the full records go to records.json.gz so that
    # committed results stay small. Nothing is dropped.
    records_by_task = {
        task: {layer: data.pop("records") for layer, data in task_data["layers"].items()}
        for task, task_data in output["tasks"].items()
    }
    output["records_file"] = "records.json.gz"
    with gzip.open(
        os.path.join(args.output_dir, "records.json.gz"), "wt", encoding="utf-8"
    ) as f:
        json.dump(records_by_task, f)
    with open(
        os.path.join(args.output_dir, "summary.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(output, f, indent=2)


if __name__ == "__main__":
    main()
