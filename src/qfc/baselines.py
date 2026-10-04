"""Shared baseline head selectors and the Random-mask distribution helper.

Every selector returns ``{layer: [head, ...]}`` with exactly ``k`` distinct
heads per layer. ``validate_selection`` enforces this so that a baseline can
never silently keep all heads (the V10 ``Random`` bug) or too few.
"""

from __future__ import annotations

from statistics import mean, pstdev
from typing import Callable, Mapping, Sequence

import torch

from .conditional import conditional_greedy_select, conditional_weighted_greedy_select
from .coverage import greedy_select, weighted_greedy_select
from .fidelity import pairwise_fidelity
from .hf_experiments import head_mask_from_selection

Selection = dict[int, list[int]]


def random_selection(layers: int, heads: int, k: int, seed: int) -> Selection:
    """Uniformly random k-of-H heads per layer (no calibration data)."""
    if not 1 <= k <= heads:
        raise ValueError(f"k must be in [1, {heads}], got {k}")
    g = torch.Generator().manual_seed(seed)
    return {
        layer: sorted(torch.randperm(heads, generator=g)[:k].tolist())
        for layer in range(layers)
    }


def topk_selection(scores: Sequence[torch.Tensor], k: int, largest: bool = True) -> Selection:
    """Keep the k highest (largest=True) or lowest (largest=False) scores per layer."""
    return {
        layer: sorted(torch.topk(score, k=k, largest=largest).indices.tolist())
        for layer, score in enumerate(scores)
    }


def entropy_baseline_selections(
    von_neumann: Sequence[torch.Tensor],
    shannon: Sequence[torch.Tensor],
    k: int,
) -> dict[str, Selection]:
    """Entropy baselines evaluated in BOTH directions, labelled explicitly."""
    return {
        "VonNeumann_keep_high": topk_selection(von_neumann, k, largest=True),
        "VonNeumann_keep_low": topk_selection(von_neumann, k, largest=False),
        "Shannon_keep_high": topk_selection(shannon, k, largest=True),
        "Shannon_keep_low": topk_selection(shannon, k, largest=False),
    }


def validate_selection(
    selection: Mapping[int, Sequence[int]],
    layers: int,
    heads: int,
    k: int,
    name: str = "selection",
) -> None:
    """Raise unless every layer keeps exactly k distinct in-range heads."""
    if sorted(selection) != list(range(layers)):
        raise ValueError(f"{name}: expected layers 0..{layers - 1}, got {sorted(selection)}")
    for layer, kept in selection.items():
        kept = list(kept)
        if len(kept) != k or len(set(kept)) != k:
            raise ValueError(
                f"{name}: layer {layer} keeps {len(kept)} heads "
                f"({len(set(kept))} distinct), expected exactly {k}"
            )
        if any(h < 0 or h >= heads for h in kept):
            raise ValueError(f"{name}: layer {layer} has a head index outside [0, {heads})")


def validate_all(
    selections: Mapping[str, Mapping[int, Sequence[int]]],
    layers: int,
    heads: int,
    k: int,
) -> None:
    for name, selection in selections.items():
        validate_selection(selection, layers, heads, k, name=name)


def heads_kept_per_layer(selection: Mapping[int, Sequence[int]]) -> list[int]:
    return [len(selection[layer]) for layer in sorted(selection)]


def summarize(values: Sequence[float]) -> dict[str, float]:
    """mean / population std (as elsewhere in this repo) / min / max."""
    values = [float(v) for v in values]
    if not values:
        raise ValueError("cannot summarize an empty sequence")
    return {
        "mean": mean(values),
        "std": pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "n": len(values),
    }


def f1_binary(labels: torch.Tensor, predictions: torch.Tensor) -> float:
    labels = labels.long()
    predictions = predictions.long()
    tp = ((predictions == 1) & (labels == 1)).sum().item()
    fp = ((predictions == 1) & (labels == 0)).sum().item()
    fn = ((predictions == 0) & (labels == 1)).sum().item()
    denom = 2 * tp + fp + fn
    return 0.0 if denom == 0 else float(2 * tp / denom)


EvalFn = Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]


def evaluate_random_distribution(
    eval_fn: EvalFn,
    layers: int,
    heads: int,
    k: int,
    n_masks: int,
    base_seed: int,
    device: str,
    with_f1: bool = False,
    baseline: Mapping[str, float] | None = None,
) -> dict:
    """Evaluate ``n_masks`` independent random masks (seed = base_seed + i).

    ``eval_fn(mask)`` must return ``(losses, correct, labels, predictions)``.
    The returned dict has top-level ``accuracy`` / ``loss`` (/ ``f1``) equal to
    the mean over draws, so it can sit in a ``methods`` table next to
    single-mask methods, plus the full ``distribution`` and per-draw records.
    """
    if n_masks < 1:
        raise ValueError("n_masks must be >= 1")
    draws = []
    first_selection = None
    for i in range(n_masks):
        seed = base_seed + i
        selection = random_selection(layers, heads, k, seed)
        validate_selection(selection, layers, heads, k, name=f"Random[seed={seed}]")
        if first_selection is None:
            first_selection = selection
        mask = head_mask_from_selection(layers, heads, selection, device)
        losses, correct, labels, predictions = eval_fn(mask)
        draw = {
            "seed": seed,
            "accuracy": float(correct.mean()),
            "loss": float(losses.mean()),
        }
        if with_f1:
            draw["f1"] = f1_binary(labels, predictions)
        draws.append(draw)

    keys = ["accuracy", "loss"] + (["f1"] if with_f1 else [])
    distribution = {key: summarize([d[key] for d in draws]) for key in keys}
    out = {
        "n_masks": n_masks,
        "base_seed": base_seed,
        "std_convention": "population (pstdev)",
        "accuracy": distribution["accuracy"]["mean"],
        "loss": distribution["loss"]["mean"],
        "distribution": distribution,
        "draws": draws,
        "selected_heads_zero_based": first_selection,
        "selected_heads_note": "selection of draw 0 only; see draws[].seed to regenerate the rest",
    }
    if with_f1:
        out["f1"] = distribution["f1"]["mean"]
    if baseline is not None:
        for key in keys:
            if key in baseline:
                out[f"{key}_delta_mean"] = distribution[key]["mean"] - float(baseline[key])
    return out


def michel_weights(michel_layer: torch.Tensor) -> torch.Tensor:
    """Nonnegative gate-sensitivity weights, normalised to sum to one."""
    raw = michel_layer.clamp_min(0.0)
    if float(raw.sum()) > 0:
        return raw / raw.sum()
    return torch.full_like(raw, 1.0 / len(raw))


def calibrated_selections(
    mean_states: Sequence[torch.Tensor],
    conditional_kernels: Sequence[torch.Tensor],
    michel: torch.Tensor,
    von_neumann: Sequence[torch.Tensor],
    shannon: Sequence[torch.Tensor],
    k: int,
) -> dict[str, Selection]:
    """All calibration-based selectors used by the V10 confirmatory run.

    Random is deliberately absent: it is a distribution, see
    ``evaluate_random_distribution``. The result is validated to keep exactly
    ``k`` heads per layer for every method.
    """
    layers = len(mean_states)
    heads = int(mean_states[0].shape[0])
    sel: dict[str, Selection] = {
        name: {}
        for name in (
            "MeanStateQFC",
            "ConditionalQFC",
            "MeanStateIWQFC",
            "ConditionalIWQFC",
        )
    }
    for layer in range(layers):
        sim = pairwise_fidelity(mean_states[layer])
        w = michel_weights(michel[layer])
        sel["MeanStateQFC"][layer], _ = greedy_select(sim, k)
        sel["ConditionalQFC"][layer], _ = conditional_greedy_select(conditional_kernels[layer], k)
        sel["MeanStateIWQFC"][layer], _ = weighted_greedy_select(sim, w, k)
        sel["ConditionalIWQFC"][layer], _ = conditional_weighted_greedy_select(
            conditional_kernels[layer], w, k
        )
    sel["MichelGate"] = topk_selection([michel[i] for i in range(layers)], k, largest=True)
    sel.update(entropy_baseline_selections(von_neumann, shannon, k))
    validate_all(sel, layers, heads, k)
    return sel
