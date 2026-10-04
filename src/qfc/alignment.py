"""Rank-correlation utilities for the V11 objective-alignment audit.

Diagnostic statistics only: a Spearman correlation between coverage and loss
across head subsets is associational, not evidence of causal alignment.
"""

from __future__ import annotations

from math import comb
from typing import Sequence

import torch


def rankdata(values: Sequence[float]) -> torch.Tensor:
    """Zero-based average ranks (ties share the mean rank)."""
    x = torch.as_tensor(values, dtype=torch.float64)
    _, inverse, counts = torch.unique(x, return_inverse=True, return_counts=True)
    ends = counts.cumsum(0).to(torch.float64)
    average = (2.0 * ends - counts.to(torch.float64) - 1.0) / 2.0
    return average[inverse]


def _is_degenerate(x: torch.Tensor, y: torch.Tensor) -> bool:
    return x.numel() < 2 or bool(x.max() == x.min()) or bool(y.max() == y.min())


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rho; 0.0 when undefined (constant input or n < 2)."""
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    rx = rankdata(x)
    ry = rankdata(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = rx.norm() * ry.norm()
    return 0.0 if float(denom) == 0.0 else float((rx @ ry) / denom)


def bootstrap_spearman(
    x: Sequence[float],
    y: Sequence[float],
    n_boot: int = 1000,
    seed: int = 0,
    confidence: float = 0.95,
) -> dict:
    """Percentile bootstrap CI for Spearman rho, resampling (x, y) pairs.

    ``degenerate`` is True when rho is undefined for the original sample
    (constant x or y); the estimate is then 0.0 and the CI is null, so a
    constant series can never masquerade as "no correlation".
    """
    if len(x) != len(y):
        raise ValueError("x and y must have equal length")
    tx = torch.as_tensor(x, dtype=torch.float64)
    ty = torch.as_tensor(y, dtype=torch.float64)
    n = int(tx.numel())
    out = {
        "estimate": spearman(x, y),
        "n": n,
        "n_boot": n_boot,
        "confidence": confidence,
        "degenerate": _is_degenerate(tx, ty),
        "ci_low": None,
        "ci_high": None,
    }
    if out["degenerate"] or n_boot < 1:
        return out
    g = torch.Generator().manual_seed(seed)
    rhos = torch.empty(n_boot, dtype=torch.float64)
    for b in range(n_boot):
        idx = torch.randint(0, n, (n,), generator=g)
        rhos[b] = spearman(tx[idx], ty[idx])
    alpha = (1.0 - confidence) / 2.0
    out["ci_low"] = float(torch.quantile(rhos, alpha))
    out["ci_high"] = float(torch.quantile(rhos, 1.0 - alpha))
    return out


def check_subset_budget(heads: int, k: int, count: int) -> None:
    """Fail fast if ``count`` distinct k-subsets of ``heads`` cannot exist."""
    available = comb(heads, k)
    if count > available:
        raise ValueError(
            f"cannot draw {count} distinct subsets: C({heads},{k}) = {available}"
        )
