from __future__ import annotations

import torch


def hilbert_schmidt_similarity(rhos: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Normalized Hilbert-Schmidt similarity between density operators.

    HS(i,j) = Tr(rho_i rho_j) /
              sqrt(Tr(rho_i^2) Tr(rho_j^2))

    For PSD density operators this lies in [0,1].
    """
    if rhos.ndim != 3 or rhos.shape[-1] != rhos.shape[-2]:
        raise ValueError("rhos must have shape [heads, N, N]")

    rhos = 0.5 * (rhos + rhos.transpose(-1, -2))
    numer = torch.einsum("aij,bij->ab", rhos, rhos)
    norms = torch.einsum("aij,aij->a", rhos, rhos).clamp_min(eps)
    denom = torch.sqrt(norms[:, None] * norms[None, :])
    sim = (numer / denom).clamp(0.0, 1.0)
    sim = 0.5 * (sim + sim.T)
    sim.fill_diagonal_(1.0)
    return sim
