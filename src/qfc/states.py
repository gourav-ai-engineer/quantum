from __future__ import annotations

import torch


def _check_attention(attention: torch.Tensor) -> torch.Tensor:
    if not isinstance(attention, torch.Tensor):
        raise TypeError("attention must be a torch.Tensor")
    if attention.ndim < 2 or attention.shape[-1] != attention.shape[-2]:
        raise ValueError("attention must have shape [..., N, N]")
    if not torch.is_floating_point(attention):
        attention = attention.float()
    return attention


def density_operator(attention: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Convert an attention matrix into a normalized PSD density operator.

    rho = A A^T / Tr(A A^T)

    Works with tensors shaped [..., N, N]. The input need not be non-negative;
    the construction is valid for any non-zero real matrix A.
    """
    attention = _check_attention(attention)
    gram = attention @ attention.transpose(-1, -2)
    trace = torch.diagonal(gram, dim1=-2, dim2=-1).sum(-1, keepdim=True)
    if torch.any(trace <= eps):
        raise ValueError("attention contains a zero-energy matrix; cannot normalize")
    rho = gram / trace.unsqueeze(-1)
    return 0.5 * (rho + rho.transpose(-1, -2))


def aggregate_density_operators(rhos: torch.Tensor) -> torch.Tensor:
    """Average density operators across samples.

    Input:  [M, H, N, N]
    Output: [H, N, N]

    The arithmetic mean of valid density operators remains PSD and trace one.
    """
    if not isinstance(rhos, torch.Tensor):
        raise TypeError("rhos must be a torch.Tensor")
    if rhos.ndim != 4 or rhos.shape[-1] != rhos.shape[-2]:
        raise ValueError("rhos must have shape [samples, heads, N, N]")
    rho_bar = rhos.mean(dim=0)
    return 0.5 * (rho_bar + rho_bar.transpose(-1, -2))
