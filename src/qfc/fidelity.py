from __future__ import annotations

import torch


def _sqrt_psd(matrix: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    matrix = 0.5 * (matrix + matrix.transpose(-1, -2))
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    eigenvalues = eigenvalues.clamp_min(eps)
    sqrt_values = eigenvalues.sqrt()
    return (eigenvectors * sqrt_values.unsqueeze(-2)) @ eigenvectors.transpose(-1, -2)


def fidelity(rho: torch.Tensor, sigma: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Uhlmann-Jozsa fidelity using the squared-fidelity convention."""
    if rho.shape != sigma.shape or rho.ndim < 2 or rho.shape[-1] != rho.shape[-2]:
        raise ValueError("rho and sigma must have identical shape [..., N, N]")
    root_rho = _sqrt_psd(rho, eps=eps)
    middle = root_rho @ sigma @ root_rho
    middle = 0.5 * (middle + middle.transpose(-1, -2))
    eigenvalues = torch.linalg.eigvalsh(middle).clamp_min(0.0)
    root_trace = eigenvalues.sqrt().sum(-1)
    return root_trace.square().clamp(0.0, 1.0)


def pairwise_fidelity(rhos: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Compute pairwise fidelity for head states.

    Input:  [H, N, N]
    Output: [H, H]
    """
    if rhos.ndim != 3 or rhos.shape[-1] != rhos.shape[-2]:
        raise ValueError("rhos must have shape [heads, N, N]")
    h = rhos.shape[0]
    rho_i = rhos[:, None, :, :].expand(h, h, -1, -1)
    rho_j = rhos[None, :, :, :].expand(h, h, -1, -1)
    sim = fidelity(rho_i, rho_j, eps=eps)
    sim = 0.5 * (sim + sim.transpose(-1, -2))
    sim.fill_diagonal_(1.0)
    return sim
