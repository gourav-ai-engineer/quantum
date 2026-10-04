"""step11: conditional IWQFC smoke test on SST-2.

STATUS: SMOKE/EXPLORATORY, DO NOT CITE.
Standing rules (CLAUDE.md), checked 2026-10-04:
  Rule 1 (exactly k heads per layer, printed/asserted): NOT satisfied (not printed/asserted).
  Rule 2 (Random as a distribution, >= 30 masks): not applicable (no Random baseline).
  Rule 3 (entropy baselines keep-high AND keep-low): not applicable (no entropy baselines).
  Rule 4 (calibration from train, evaluation on validation): satisfied (train calibration,
         validation slice evaluation; the slice is not the full validation split).
Superseded by step18 (V10), which compares against the corrected baselines.
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
    conditional_weighted_greedy_select,
)
from qfc.hf_experiments import _move_batch, load_sequence_classifier, make_text_loader, michel_head_importance


@torch.no_grad()
def evaluate(model, loader, device, head_mask=None):
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
        losses.append(torch.nn.functional.cross_entropy(logits, labels, reduction="none").cpu())
        correct.append((logits.argmax(-1) == labels).float().cpu())
    losses = torch.cat(losses)
    correct = torch.cat(correct)
    return float(correct.mean()), float(losses.mean())


def mask_from_selection(selected, layers, heads, device):
    mask = torch.zeros((layers, heads), dtype=torch.float32, device=device)
    for layer, hs in selected.items():
        mask[layer, hs] = 1.0
    return mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="textattack/bert-base-uncased-SST-2")
    parser.add_argument(
        "--model-revision",
        default="205ffbd1bc5c5b89802266f4948a601f53556b00",
    )
    parser.add_argument("--calibration-size", type=int, default=8)
    parser.add_argument("--evaluation-size", type=int, default=64)
    parser.add_argument("--heads-to-keep", type=int, default=9)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=32)
    parser.add_argument("--output-dir", default="results/step11_conditional_iwqfc")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    validation = load_dataset("stanfordnlp/sst2", split="validation")
    if args.evaluation_size > len(validation):
        raise ValueError("evaluation-size exceeds validation split")

    train = load_dataset("stanfordnlp/sst2", split="train")
    calibration = train.shuffle(seed=42).select(range(args.calibration_size))
    evaluation = validation.select(range(args.evaluation_size))

    model, tokenizer = load_sequence_classifier(
        args.model_id,
        args.device,
        revision=args.model_revision,
    )

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

    baseline_acc, baseline_loss = evaluate(model, eval_loader, args.device)

    states = collect_per_sample_density_states(model, cal_loader, args.device)
    kernels = conditional_fidelity_kernels(states, max_pairs_per_chunk=48)
    michel = michel_head_importance(model, cal_loader, args.device).detach().cpu()

    conditional = {}
    weighted = {}
    for layer, sim in enumerate(kernels):
        qfc, _ = conditional_greedy_select(sim, args.heads_to_keep)
        conditional[layer] = qfc

        raw = michel[layer].clamp_min(0.0)
        if float(raw.sum()) > 0:
            weights = raw / raw.sum()
        else:
            weights = torch.full_like(raw, 1.0 / len(raw))
        iw, _ = conditional_weighted_greedy_select(
            sim,
            weights,
            args.heads_to_keep,
        )
        weighted[layer] = iw

    layers = len(kernels)
    heads = int(kernels[0].shape[-1])
    q_mask = mask_from_selection(conditional, layers, heads, args.device)
    iw_mask = mask_from_selection(weighted, layers, heads, args.device)

    q_acc, q_loss = evaluate(model, eval_loader, args.device, q_mask)
    iw_acc, iw_loss = evaluate(model, eval_loader, args.device, iw_mask)

    result = {
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "calibration_split": "train",
        "evaluation_split": "validation_slice",
        "calibration_size": args.calibration_size,
        "evaluation_size": args.evaluation_size,
        "heads_to_keep_per_layer": args.heads_to_keep,
        "baseline": {"accuracy": baseline_acc, "loss": baseline_loss},
        "conditional_QFC": {
            "accuracy": q_acc,
            "loss": q_loss,
            "accuracy_change": q_acc - baseline_acc,
            "loss_change": q_loss - baseline_loss,
            "selected_heads_zero_based": conditional,
        },
        "conditional_IWQFC": {
            "accuracy": iw_acc,
            "loss": iw_loss,
            "accuracy_change": iw_acc - baseline_acc,
            "loss_change": iw_loss - baseline_loss,
            "selected_heads_zero_based": weighted,
        },
        "research_note": (
            "Small smoke test of exact input-conditioned fidelity combined with "
            "gate-sensitivity weighting. Engineering evidence only."
        ),
    }

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
