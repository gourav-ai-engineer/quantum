from __future__ import annotations

import argparse
import json

import torch
from datasets import load_dataset

from qfc.hf_experiments import (
    collect_mean_density_states,
    evaluate_accuracy,
    load_sequence_classifier,
    make_text_loader,
    parameter_count,
    qfc_select_from_mean_states,
    save_selection,
    structured_prune,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Step 2: QFC calibration on SST-2")
    parser.add_argument(
        "--model-id",
        default="textattack/bert-base-uncased-SST-2",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument(
        "--calibration-size",
        type=int,
        default=256,
        help="Use 256 for the first smoke run; set 872 for the full SST-2 dev experiment.",
    )
    parser.add_argument("--heads-to-keep", type=int, default=6)
    parser.add_argument("--output-dir", default="results/step2_sst2")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.heads_to_keep < 1:
        raise ValueError("--heads-to-keep must be positive")

    print(f"Loading {args.model_id} on {args.device}...")
    model, tokenizer = load_sequence_classifier(args.model_id, args.device)

    dataset = load_dataset("stanfordnlp/sst2", split="validation")
    print(f"SST-2 validation examples available: {len(dataset)}")

    loader = make_text_loader(
        dataset,
        tokenizer,
        text_fields=("sentence",),
        batch_size=args.batch_size,
        max_length=args.max_length,
        max_samples=args.calibration_size,
    )

    # Baseline score on the exact same examples used for calibration.
    baseline = evaluate_accuracy(model, loader, device=args.device)
    baseline_params = parameter_count(model)
    print(f"Baseline accuracy ({args.calibration_size} samples): {baseline:.4f}")
    print(f"Baseline parameters: {baseline_params:,}")

    print("Collecting mean density operators...")
    mean_states = collect_mean_density_states(model, loader, device=args.device)
    num_heads = mean_states[0].shape[0]
    if args.heads_to_keep > num_heads:
        raise ValueError(f"--heads-to-keep cannot exceed model head count ({num_heads})")

    print("Running QFC greedy coverage selection...")
    selected, history = qfc_select_from_mean_states(mean_states, args.heads_to_keep)

    # Evaluate a structured copy only after the selection is frozen.
    pruned_model, pruned_heads = structured_prune(model, selected)
    pruned_accuracy = evaluate_accuracy(pruned_model, loader, device=args.device)
    pruned_params = parameter_count(pruned_model)

    save_selection(args.output_dir, selected, history, mean_states)
    summary = {
        "model_id": args.model_id,
        "dataset": "stanfordnlp/sst2/validation",
        "calibration_examples": len(loader.dataset),
        "max_length": args.max_length,
        "heads_to_keep_per_layer": args.heads_to_keep,
        "baseline_accuracy": baseline,
        "qfc_structured_accuracy": pruned_accuracy,
        "accuracy_change": pruned_accuracy - baseline,
        "baseline_parameters": baseline_params,
        "qfc_structured_parameters": pruned_params,
        "parameter_fraction_remaining": pruned_params / baseline_params,
        "pruned_heads_zero_based": pruned_heads,
        "selected_heads_zero_based": selected,
    }
    with open(f"{args.output_dir}/summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print("\nThis is a calibration/smoke result, not the final paper result.")
    print("The final experiment must use the full validation split and matched baselines.")


if __name__ == "__main__":
    main()
