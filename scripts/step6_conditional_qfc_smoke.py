"""step6: conditional (per-example) QFC smoke test on SST-2.

STATUS: SMOKE/EXPLORATORY, DO NOT CITE.
Standing rules (CLAUDE.md), checked 2026-10-04:
  Rule 1 (exactly k heads per layer, printed/asserted): NOT satisfied (not printed/asserted).
  Rule 2 (Random as a distribution, >= 30 masks): not applicable (no Random baseline).
  Rule 3 (entropy baselines keep-high AND keep-low): not applicable (no entropy baselines).
  Rule 4 (calibration from train, evaluation on validation): VIOLATED. Calibration and
         evaluation are slices of the validation split.
Superseded by step17 (V9) and step18 (V10).
"""

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
)
from qfc.hf_experiments import (
    evaluate_accuracy,
    load_sequence_classifier,
    make_text_loader,
)
from qfc.hf_experiments import _move_batch


@torch.no_grad()
def evaluate_loss_accuracy(model, loader, device, head_mask=None):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    for batch in loader:
        batch = _move_batch(batch, device)
        labels = batch.pop("labels")
        out = model(
            **batch,
            labels=labels,
            head_mask=head_mask,
            return_dict=True,
        )
        n = int(labels.numel())
        total_loss += float(out.loss.item()) * n
        total_correct += int((out.logits.argmax(-1) == labels).sum().item())
        total += n
    return total_correct / total, total_loss / total


def mask_from_selection(selection, layers, heads, device):
    mask = torch.zeros((layers, heads), dtype=torch.float32, device=device)
    for layer, hs in selection.items():
        mask[layer, hs] = 1.0
    return mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-id",
        default="textattack/bert-base-uncased-SST-2",
    )
    parser.add_argument("--calibration-size", type=int, default=16)
    parser.add_argument("--evaluation-size", type=int, default=32)
    parser.add_argument("--heads-to-keep", type=int, default=9)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=32)
    parser.add_argument("--output-dir", default="results/step6_conditional")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    dataset = load_dataset("stanfordnlp/sst2", split="validation")
    need = args.calibration_size + args.evaluation_size
    if need > len(dataset):
        raise ValueError("requested more examples than available")

    calibration = dataset.select(range(args.calibration_size))
    evaluation = dataset.select(range(args.calibration_size, need))

    model, tokenizer = load_sequence_classifier(args.model_id, args.device)

    cal_loader = make_text_loader(
        calibration,
        tokenizer,
        text_fields=("sentence",),
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    eval_loader = make_text_loader(
        evaluation,
        tokenizer,
        text_fields=("sentence",),
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    baseline_acc, baseline_loss = evaluate_loss_accuracy(
        model, eval_loader, args.device
    )
    print(f"baseline acc={baseline_acc:.6f} loss={baseline_loss:.6f}")

    per_sample = collect_per_sample_density_states(
        model, cal_loader, args.device
    )
    kernels = conditional_fidelity_kernels(
        per_sample,
        max_pairs_per_chunk=48,
    )

    layers = len(kernels)
    heads = int(kernels[0].shape[-1])
    selected = {}
    histories = {}

    for layer, sim in enumerate(kernels):
        sel, hist = conditional_greedy_select(
            sim,
            args.heads_to_keep,
        )
        selected[layer] = sel
        histories[layer] = [float(x) for x in hist]

    mask = mask_from_selection(
        selected, layers, heads, args.device
    )
    qfc_acc, qfc_loss = evaluate_loss_accuracy(
        model, eval_loader, args.device, head_mask=mask
    )

    result = {
        "model_id": args.model_id,
        "calibration_size": args.calibration_size,
        "evaluation_size": args.evaluation_size,
        "max_length": args.max_length,
        "heads_to_keep_per_layer": args.heads_to_keep,
        "baseline": {
            "accuracy": baseline_acc,
            "loss": baseline_loss,
        },
        "conditional_qfc_masked": {
            "accuracy": qfc_acc,
            "loss": qfc_loss,
            "accuracy_change": qfc_acc - baseline_acc,
            "loss_change": qfc_loss - baseline_loss,
            "selected_heads_zero_based": selected,
            "coverage_history": histories,
        },
        "research_note": (
            "Small engineering experiment for input-conditioned QFC. "
            "Do not treat this as a final paper result."
        ),
    }

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
