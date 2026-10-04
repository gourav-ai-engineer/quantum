from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from typing import Iterable

import torch
from datasets import load_dataset

from qfc.coverage import greedy_select
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
from qfc.states import density_operator


@dataclass
class Metrics:
    accuracy: float
    loss: float


@torch.no_grad()
def evaluate_metrics(
    model,
    loader: Iterable[dict[str, torch.Tensor]],
    device: str,
    head_mask: torch.Tensor | None = None,
) -> Metrics:
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0

    for batch in loader:
        batch = _move_batch(batch, device)
        labels = batch.pop("labels")
        outputs = model(
            **batch,
            labels=labels,
            head_mask=head_mask,
            return_dict=True,
        )
        n = int(labels.numel())
        loss_sum += float(outputs.loss.item()) * n
        correct += int((outputs.logits.argmax(dim=-1) == labels).sum().item())
        total += n

    if total == 0:
        raise ValueError("evaluation loader contains no examples")
    return Metrics(accuracy=correct / total, loss=loss_sum / total)


@torch.no_grad()
def collect_mean_shannon_entropy(
    model,
    loader: Iterable[dict[str, torch.Tensor]],
    device: str,
) -> list[torch.Tensor]:
    """Mean token-level Shannon entropy for each layer/head."""
    model.eval()
    sums = None
    counts = None

    for batch in loader:
        batch = _move_batch(batch, device)
        labels = batch.pop("labels", None)
        attention_mask = batch["attention_mask"].to(torch.float32)

        outputs = model(
            **batch,
            output_attentions=True,
            return_dict=True,
        )
        attentions = outputs.attentions
        if attentions is None:
            raise RuntimeError("Model did not return attentions")

        if sums is None:
            sums = [
                torch.zeros(attn.shape[1], dtype=torch.float64)
                for attn in attentions
            ]
            counts = [
                torch.zeros(attn.shape[1], dtype=torch.float64)
                for attn in attentions
            ]

        query_mask = attention_mask > 0
        key_mask = attention_mask > 0

        for layer_idx, attn in enumerate(attentions):
            probs = attn.float().clamp_min(1e-12)
            probs = probs * key_mask[:, None, None, :].to(probs.dtype)
            entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)
            entropy = entropy * query_mask[:, None, :].to(entropy.dtype)

            valid_queries = query_mask.sum(dim=-1).clamp_min(1)
            per_sample_head = entropy.sum(dim=-1) / valid_queries[:, None]
            sums[layer_idx] += per_sample_head.sum(dim=0).cpu().double()
            counts[layer_idx] += torch.full(
                (attn.shape[1],),
                float(attn.shape[0]),
                dtype=torch.float64,
            )

    if sums is None or counts is None:
        raise ValueError("calibration loader contains no examples")

    return [s / c.clamp_min(1.0) for s, c in zip(sums, counts)]


def von_neumann_entropy(rho: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    rho = 0.5 * (rho + rho.T)
    eigenvalues = torch.linalg.eigvalsh(rho).clamp_min(eps)
    return -(eigenvalues * eigenvalues.log()).sum()


def top_k_selection(scores: list[torch.Tensor], k: int) -> dict[int, list[int]]:
    selected = {}
    for layer_idx, score in enumerate(scores):
        if score.ndim != 1:
            raise ValueError("each layer score must be [heads]")
        selected[layer_idx] = torch.topk(score, k=k, largest=True).indices.tolist()
    return selected


def random_selection(
    num_layers: int,
    num_heads: int,
    k: int,
    seed: int,
) -> dict[int, list[int]]:
    generator = torch.Generator().manual_seed(seed)
    return {
        layer: torch.randperm(num_heads, generator=generator)[:k].tolist()
        for layer in range(num_layers)
    }


def qfc_selection(mean_states: list[torch.Tensor], k: int):
    selected = {}
    histories = {}
    for layer, states in enumerate(mean_states):
        sim = pairwise_fidelity(states)
        heads, history = greedy_select(sim, k)
        selected[layer] = heads
        histories[layer] = [float(x) for x in history]
    return selected, histories


def mask_from_selection(
    selected: dict[int, list[int]],
    num_layers: int,
    num_heads: int,
    device: str,
) -> torch.Tensor:
    mask = torch.zeros(
        (num_layers, num_heads),
        dtype=torch.float32,
        device=device,
    )
    for layer, heads in selected.items():
        mask[layer, heads] = 1.0
    return mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="textattack/bert-base-uncased-SST-2")
    parser.add_argument("--calibration-size", type=int, default=32)
    parser.add_argument("--evaluation-size", type=int, default=64)
    parser.add_argument("--heads-to-keep", type=int, default=9)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=32)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--output-dir", default="results/step3_selection")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    model, tokenizer = load_sequence_classifier(args.model_id, args.device)
    dataset = load_dataset("stanfordnlp/sst2", split="validation")

    minimum_required = args.calibration_size + args.evaluation_size
    if minimum_required > len(dataset):
        raise ValueError(
            f"requested {minimum_required} examples but validation split has only {len(dataset)}"
        )

    calibration_ds = dataset.select(range(args.calibration_size))
    evaluation_ds = dataset.select(
        range(args.calibration_size, minimum_required)
    )

    calibration_loader = make_text_loader(
        calibration_ds,
        tokenizer,
        text_fields=("sentence",),
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    evaluation_loader = make_text_loader(
        evaluation_ds,
        tokenizer,
        text_fields=("sentence",),
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    base_metrics = evaluate_metrics(model, evaluation_loader, args.device)
    base_params = parameter_count(model)

    print(f"Baseline: accuracy={base_metrics.accuracy:.4f}, loss={base_metrics.loss:.6f}")
    print(f"Parameters: {base_params:,}")

    mean_states = collect_mean_density_states(model, calibration_loader, args.device)
    shannon_scores = collect_mean_shannon_entropy(model, calibration_loader, args.device)
    vn_scores = [torch.stack([von_neumann_entropy(r) for r in states]) for states in mean_states]

    num_layers = len(mean_states)
    num_heads = mean_states[0].shape[0]
    if not 1 <= args.heads_to_keep <= num_heads:
        raise ValueError(f"heads-to-keep must be in [1, {num_heads}]")

    qfc_selected, qfc_history = qfc_selection(mean_states, args.heads_to_keep)
    shannon_selected = top_k_selection(shannon_scores, args.heads_to_keep)
    vn_selected = top_k_selection(vn_scores, args.heads_to_keep)
    michel = michel_head_importance(
        model,
        calibration_loader,
        device=args.device,
    ).detach().cpu()
    michel_scores = [michel[layer] for layer in range(num_layers)]
    michel_selected = top_k_selection(michel_scores, args.heads_to_keep)
    random_selected = random_selection(
        num_layers,
        num_heads,
        args.heads_to_keep,
        args.random_seed,
    )

    selections = {
        "QFC": qfc_selected,
        "Shannon": shannon_selected,
        "VonNeumann": vn_selected,
        "MichelGate": michel_selected,
        "Random": random_selected,
    }

    results = {}
    for name, selection in selections.items():
        mask = mask_from_selection(
            selection,
            num_layers,
            num_heads,
            args.device,
        )
        metrics = evaluate_metrics(
            model,
            evaluation_loader,
            args.device,
            head_mask=mask,
        )
        results[name] = {
            "accuracy": metrics.accuracy,
            "loss": metrics.loss,
            "accuracy_change_vs_full": metrics.accuracy - base_metrics.accuracy,
            "loss_change_vs_full": metrics.loss - base_metrics.loss,
            "selected_heads_zero_based": selection,
        }

    # A separate physical-pruning check for QFC.
    physical_model, pruned_heads = structured_prune(
        model,
        qfc_selected,
    )
    physical_metrics = evaluate_metrics(
        physical_model,
        evaluation_loader,
        args.device,
    )
    results["QFC_structured"] = {
        "accuracy": physical_metrics.accuracy,
        "loss": physical_metrics.loss,
        "accuracy_change_vs_full": physical_metrics.accuracy - base_metrics.accuracy,
        "loss_change_vs_full": physical_metrics.loss - base_metrics.loss,
        "parameters": parameter_count(physical_model),
        "parameter_fraction_remaining": parameter_count(physical_model) / base_params,
        "pruned_heads_zero_based": pruned_heads,
    }

    summary = {
        "model_id": args.model_id,
        "calibration_size": args.calibration_size,
        "evaluation_size": args.evaluation_size,
        "heads_to_keep_per_layer": args.heads_to_keep,
        "max_length": args.max_length,
        "device": args.device,
        "baseline": {
            "accuracy": base_metrics.accuracy,
            "loss": base_metrics.loss,
            "parameters": base_params,
        },
        "results": results,
        "qfc_coverage_history": qfc_history,
        "research_note": (
            "This is an engineering validation. Accuracy improvement is not assumed; "
            "the final paper claim requires full validation splits, corrected baselines, "
            "repeated runs, and statistical analysis."
        ),
    }

    out = args.output_dir
    import os

    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
