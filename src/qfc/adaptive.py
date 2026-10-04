from __future__ import annotations

from collections.abc import Sequence

import torch


def _validate_layer_similarities(similarities: Sequence[torch.Tensor]) -> list[torch.Tensor]:
    if not similarities:
        raise ValueError("similarities must contain at least one layer")
    validated = []
    for sim in similarities:
        if not isinstance(sim, torch.Tensor):
            raise TypeError("each layer similarity must be a torch.Tensor")
        if sim.ndim != 2 or sim.shape[0] != sim.shape[1]:
            raise ValueError("each layer similarity must have shape [H, H]")
        if torch.any(sim < -1e-6) or torch.any(sim > 1 + 1e-6):
            raise ValueError("similarities must lie in [0, 1]")
        validated.append(sim.clamp(0.0, 1.0))
    return validated


def layer_adaptive_greedy(
    similarities: Sequence[torch.Tensor],
    total_budget: int,
    *,
    weights: Sequence[torch.Tensor] | None = None,
    min_per_layer: int = 1,
) -> tuple[dict[int, list[int]], list[float]]:
    """Allocate one global head budget across layers using fidelity coverage.

    Each layer has its own facility-location objective. The procedure first
    seeds every layer with its best singleton (when min_per_layer=1), then
    greedily assigns the remaining global budget to the largest marginal
    coverage gain across all layer/head candidates.

    The seeded variant is a practical heuristic; the cardinality
    1-1/e guarantee for plain global greedy does not directly apply once
    mandatory per-layer seeds are imposed.
    """
    sims = _validate_layer_similarities(similarities)
    layers = len(sims)

    if min_per_layer < 0:
        raise ValueError("min_per_layer must be nonnegative")
    if total_budget < layers * min_per_layer:
        raise ValueError("total_budget is too small for min_per_layer")
    total_heads = sum(int(sim.shape[0]) for sim in sims)
    if total_budget > total_heads:
        raise ValueError("total_budget exceeds the number of available heads")

    if weights is None:
        layer_weights = [
            torch.ones(sim.shape[0], dtype=sim.dtype, device=sim.device)
            for sim in sims
        ]
    else:
        if len(weights) != layers:
            raise ValueError("weights must provide one tensor per layer")
        layer_weights = []
        for sim, q in zip(sims, weights):
            if q.ndim != 1 or q.shape[0] != sim.shape[0]:
                raise ValueError("each weight tensor must have shape [H]")
            if torch.any(q < 0):
                raise ValueError("weights must be nonnegative")
            layer_weights.append(q.to(sim.device, dtype=sim.dtype))

    selected = {layer: [] for layer in range(layers)}
    current = [
        torch.zeros(sim.shape[0], dtype=sim.dtype, device=sim.device)
        for sim in sims
    ]
    history: list[float] = []

    for layer, sim in enumerate(sims):
        for _ in range(min_per_layer):
            remaining = [h for h in range(sim.shape[0]) if h not in selected[layer]]
            best_head = None
            best_gain = None
            for candidate in remaining:
                candidate_cov = torch.maximum(current[layer], sim[:, candidate])
                gain = ((candidate_cov - current[layer]) * layer_weights[layer]).sum()
                if best_gain is None or gain.item() > best_gain.item():
                    best_head = candidate
                    best_gain = gain
            if best_head is None:
                raise RuntimeError("failed to find a valid singleton seed")
            selected[layer].append(best_head)
            current[layer] = torch.maximum(current[layer], sim[:, best_head])
            history.append(float((current[layer] * layer_weights[layer]).sum()))

    chosen = layers * min_per_layer
    while chosen < total_budget:
        best_pair = None
        best_gain = None
        for layer, sim in enumerate(sims):
            for candidate in range(sim.shape[0]):
                if candidate in selected[layer]:
                    continue
                candidate_cov = torch.maximum(current[layer], sim[:, candidate])
                gain = ((candidate_cov - current[layer]) * layer_weights[layer]).sum()
                if best_gain is None or gain.item() > best_gain.item():
                    best_pair = (layer, candidate)
                    best_gain = gain

        if best_pair is None:
            raise RuntimeError("failed to allocate the remaining budget")

        layer, candidate = best_pair
        selected[layer].append(candidate)
        current[layer] = torch.maximum(current[layer], sims[layer][:, candidate])
        history.append(float((current[layer] * layer_weights[layer]).sum()))
        chosen += 1

    return selected, history
