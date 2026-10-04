from __future__ import annotations

import torch


def _sqrt_psd(matrix: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    matrix = 0.5 * (matrix + matrix.transpose(-1, -2))
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    eigenvalues = eigenvalues.clamp_min(eps)
    sqrt_values = eigenvalues.sqrt()
    return (eigenvectors * sqrt_values.unsqueeze(-2)) @ eigenvectors.transpose(-1, -2)


def fidelity(rho: torch.Tensor, sigma: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Uhlmann-Jozsa fidelity using the squared-fidelity convention.

    Computation is carried out in float64 to reduce eigendecomposition drift.
    """
    if rho.shape != sigma.shape or rho.ndim < 2 or rho.shape[-1] != rho.shape[-2]:
        raise ValueError("rho and sigma must have identical shape [..., N, N]")

    original_dtype = rho.dtype
    work_dtype = torch.float64
    rho64 = rho.to(work_dtype)
    sigma64 = sigma.to(work_dtype)

    root_rho = _sqrt_psd(rho64, eps=eps)
    middle = root_rho @ sigma64 @ root_rho
    middle = 0.5 * (middle + middle.transpose(-1, -2))
    eigenvalues = torch.linalg.eigvalsh(middle).clamp_min(0.0)
    root_trace = eigenvalues.sqrt().sum(-1)
    result = root_trace.square().clamp(0.0, 1.0)

    return result.to(original_dtype) if original_dtype in (
        torch.float16, torch.bfloat16, torch.float32
    ) else result


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


def pairwise_fidelity_batched(
    rhos: torch.Tensor,
    eps: float = 1e-12,
    max_pairs_per_chunk: int = 64,
) -> torch.Tensor:
    """Compute per-sample pairwise fidelity.

    Input:  [B, H, N, N]
    Output: [B, H, H]

    This intentionally computes the exact squared Uhlmann-Jozsa fidelity. To
    control memory, (head_i, head_j) pairs are processed in chunks.
    """
    if rhos.ndim != 4 or rhos.shape[-1] != rhos.shape[-2]:
        raise ValueError("rhos must have shape [batch, heads, N, N]")

    b, h, n, _ = rhos.shape
    device = rhos.device
    dtype = rhos.dtype
    work = rhos.to(torch.float64)
    roots = _sqrt_psd(work, eps=eps)

    result = torch.empty((b, h, h), dtype=torch.float64, device=device)
    pairs = [(i, j) for i in range(h) for j in range(h)]

    for start in range(0, len(pairs), max_pairs_per_chunk):
        chunk = pairs[start:start + max_pairs_per_chunk]
        i_idx = torch.tensor([p[0] for p in chunk], device=device)
        j_idx = torch.tensor([p[1] for p in chunk], device=device)

        left = roots[:, i_idx]          # [B,P,N,N]
        sigma = work[:, j_idx]          # [B,P,N,N]
        middle = left @ sigma @ left
        middle = 0.5 * (middle + middle.transpose(-1, -2))
        eigenvalues = torch.linalg.eigvalsh(middle).clamp_min(0.0)
        root_trace = eigenvalues.sqrt().sum(-1)
        values = root_trace.square().clamp(0.0, 1.0)
        for p, (i, j) in enumerate(chunk):
            result[:, i, j] = values[:, p]

    result = 0.5 * (result + result.transpose(-1, -2))
    diag = torch.arange(h, device=device)
    result[:, diag, diag] = 1.0
    return result.to(dtype if dtype in (torch.float32, torch.float64) else torch.float32)
