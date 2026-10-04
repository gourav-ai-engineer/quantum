from __future__ import annotations

from typing import Iterable

import torch

from .fidelity import pairwise_fidelity_batched
from .states import density_operator


@torch.no_grad()
def collect_per_sample_density_states(
    model,
    loader: Iterable[dict[str, torch.Tensor]],
    device: str = "cpu",
) -> list[torch.Tensor]:
    """Collect per-example density operators.

    Returns a list of L tensors, each shaped [M, H, T, T].
    """
    model.eval()
    collected: list[list[torch.Tensor]] | None = None

    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        batch.pop("labels", None)
        attention_mask = batch["attention_mask"].to(torch.float32)

        outputs = model(
            **batch,
            output_attentions=True,
            return_dict=True,
        )
        attentions = outputs.attentions
        if attentions is None:
            raise RuntimeError("Model did not return attention tensors")

        if collected is None:
            collected = [[] for _ in attentions]

        token_mask = attention_mask.to(attentions[0].dtype)
        for layer_index, attn in enumerate(attentions):
            masked = attn * token_mask[:, None, :, None] * token_mask[:, None, None, :]
            rho = density_operator(masked)
            collected[layer_index].append(rho.cpu().float())

    if collected is None:
        raise ValueError("calibration loader contains no examples")

    return [torch.cat(parts, dim=0) for parts in collected]


def conditional_fidelity_kernels(
    per_sample_states: list[torch.Tensor],
    max_pairs_per_chunk: int = 64,
) -> list[torch.Tensor]:
    """Compute exact per-example head fidelity kernels."""
    return [
        pairwise_fidelity_batched(
            states,
            max_pairs_per_chunk=max_pairs_per_chunk,
        ).cpu()
        for states in per_sample_states
    ]


def conditional_coverage_value(
    similarity: torch.Tensor,
    selected: list[int] | tuple[int, ...],
) -> torch.Tensor:
    """Exact empirical coverage E_x[sum_i max_j F_ij(x)]."""
    if similarity.ndim != 3:
        raise ValueError("similarity must have shape [samples, heads, heads]")
    if not selected:
        return similarity.new_zeros(())
    idx = torch.as_tensor(selected, dtype=torch.long)
    return similarity[:, :, idx].max(dim=-1).values.sum()


def conditional_greedy_select(
    similarity: torch.Tensor,
    k: int,
) -> tuple[list[int], torch.Tensor]:
    """Greedy optimization of empirical input-conditioned fidelity coverage."""
    if similarity.ndim != 3 or similarity.shape[-1] != similarity.shape[-2]:
        raise ValueError("similarity must have shape [samples, heads, heads]")

    samples, heads, _ = similarity.shape
    if not 1 <= k <= heads:
        raise ValueError(f"k must be in [1, {heads}]")

    selected: list[int] = []
    remaining = set(range(heads))
    current = torch.zeros((samples, heads), dtype=similarity.dtype)
    history = []

    for _ in range(k):
        best_head = None
        best_gain = None
        for candidate in sorted(remaining):
            candidate_cov = torch.maximum(
                current,
                similarity[:, :, candidate],
            )
            gain = (candidate_cov - current).sum()
            if best_gain is None or gain.item() > best_gain.item():
                best_head = candidate
                best_gain = gain

        assert best_head is not None
        selected.append(best_head)
        remaining.remove(best_head)
        current = torch.maximum(
            current,
            similarity[:, :, best_head],
        )
        history.append(current.sum().item())

    return selected, torch.tensor(history, dtype=similarity.dtype)
