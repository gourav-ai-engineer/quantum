from __future__ import annotations

import torch


def vectorized_cosine_similarity(
    matrices: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Cosine similarity between flattened attention matrices.

    This is a classical representation control: it preserves the full
    attention matrix but does not use quantum-state geometry.
    """
    if matrices.ndim != 3 or matrices.shape[-1] != matrices.shape[-2]:
        raise ValueError("matrices must have shape [heads, N, N]")

    vectors = matrices.reshape(matrices.shape[0], -1).to(torch.float64)
    vectors = vectors / vectors.norm(dim=-1, keepdim=True).clamp_min(eps)
    sim = (vectors @ vectors.T).clamp(-1.0, 1.0)
    # Attention probabilities are nonnegative, so similarities should be >= 0.
    sim = sim.clamp(0.0, 1.0)
    sim = 0.5 * (sim + sim.T)
    sim.fill_diagonal_(1.0)
    return sim


def attention_coverage_control(
    mean_attention: torch.Tensor,
    k: int,
):
    """Return a classical cosine-coverage baseline selection."""
    from .coverage import greedy_select

    sim = vectorized_cosine_similarity(mean_attention)
    return greedy_select(sim, k)
