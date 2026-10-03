from __future__ import annotations

import torch


def _validate_similarity(similarity: torch.Tensor) -> torch.Tensor:
    if not isinstance(similarity, torch.Tensor):
        raise TypeError("similarity must be a torch.Tensor")
    if similarity.ndim not in (2, 3) or similarity.shape[-1] != similarity.shape[-2]:
        raise ValueError("similarity must have shape [H, H] or [M, H, H]")
    if torch.any(similarity < -1e-6) or torch.any(similarity > 1 + 1e-6):
        raise ValueError("similarities must lie in [0, 1]")
    return similarity.clamp(0.0, 1.0)


def coverage_value(similarity: torch.Tensor, selected: list[int] | tuple[int, ...]) -> torch.Tensor:
    """Compute input-conditioned fidelity coverage.

    For [M,H,H]:
        sum_m sum_i max_{j in S} sim[m,i,j]
    """
    similarity = _validate_similarity(similarity)
    if len(selected) == 0:
        return similarity.new_zeros(())
    selected_idx = torch.as_tensor(selected, dtype=torch.long, device=similarity.device)
    selected_sims = similarity[..., :, selected_idx]
    return selected_sims.max(dim=-1).values.sum()


def greedy_select(similarity: torch.Tensor, k: int) -> tuple[list[int], torch.Tensor]:
    """Greedily maximize fidelity coverage under |S| <= k."""
    similarity = _validate_similarity(similarity)
    heads = similarity.shape[-1]
    if not 1 <= k <= heads:
        raise ValueError(f"k must satisfy 1 <= k <= {heads}")

    selected: list[int] = []
    remaining = set(range(heads))
    history: list[torch.Tensor] = []
    current = torch.zeros_like(similarity[..., 0])

    for _ in range(k):
        best_head = None
        best_gain = None
        for candidate in sorted(remaining):
            gain = torch.maximum(current, similarity[..., :, candidate]).sum() - current.sum()
            if best_gain is None or gain.item() > best_gain.item():
                best_head = candidate
                best_gain = gain
        assert best_head is not None
        selected.append(best_head)
        remaining.remove(best_head)
        current = torch.maximum(current, similarity[..., :, best_head])
        history.append(current.sum().detach())

    return selected, torch.stack(history)
