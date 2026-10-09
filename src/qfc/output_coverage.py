"""V14 / H9: output-space coverage with compensation (docs/PROJECT_STATE.md, "Pre-registration V14").

Per layer every quantity is a function of G = Z^T Z, the Gram matrix of the concatenated head context vectors
(input of the attention output projection) over all non-pad calibration tokens, and of W = W_O stacked by head
(768 x 768, torch ``dense.weight.T``). Head h owns rows/columns [h*d, (h+1)*d) of both.
"""
from __future__ import annotations

from contextlib import contextmanager
from itertools import combinations

import torch


def _blk(h: int, d: int) -> slice:
    return slice(h * d, (h + 1) * d)


def output_fidelity(G: torch.Tensor, num_heads: int) -> torch.Tensor:
    """F_ij = (||Z_i^T Z_j||_* / (||Z_i||_F ||Z_j||_F))^2, the Uhlmann fidelity of rho_{Z_i} and rho_{Z_j}."""
    d = G.shape[0] // num_heads
    G = G.double()
    tr = torch.stack([torch.trace(G[_blk(h, d), _blk(h, d)]) for h in range(num_heads)])
    F = torch.empty(num_heads, num_heads, dtype=torch.float64)
    for i in range(num_heads):
        for j in range(i, num_heads):
            nuc = torch.linalg.svdvals(G[_blk(i, d), _blk(j, d)]).sum()
            F[i, j] = F[j, i] = (nuc**2 / (tr[i] * tr[j]).clamp_min(1e-30)).clamp(0.0, 1.0)
    return F


def bound_weights(G: torch.Tensor, W: torch.Tensor, num_heads: int) -> torch.Tensor:
    """w_j = ||W_O^j||_2 ||Z_j||_F."""
    d = G.shape[0] // num_heads
    G, W = G.double(), W.double()
    return torch.stack([
        torch.linalg.matrix_norm(W[_blk(h, d)], ord=2) * torch.trace(G[_blk(h, d), _blk(h, d)]).clamp_min(0).sqrt()
        for h in range(num_heads)
    ])


def coverage_bound(F: torch.Tensor, w: torch.Tensor, S) -> float:
    """B(S) = sum_j w_j min_{i in S} sqrt(1 - F_ji) (kept heads contribute 0)."""
    d = (1.0 - F.double()).clamp_min(0).sqrt()
    return float((w.double() * d[:, list(S)].min(dim=1).values).sum())


def bcm_select(F: torch.Tensor, w: torch.Tensor, k: int) -> tuple[list[int], dict[int, int]]:
    """Greedy maximisation of f(S) = sum_j w_j max_{i in S} (1 - sqrt(1 - F_ji)); returns (sorted S, assignment)."""
    H = F.shape[0]
    if not 1 <= k <= H:
        raise ValueError(f"k must be in [1, {H}]")
    s = 1.0 - (1.0 - F.double()).clamp_min(0).sqrt()
    w = w.double()
    cur = torch.zeros(H, dtype=torch.float64)
    S: list[int] = []
    for _ in range(k):
        c = max((h for h in range(H) if h not in S),
                key=lambda h: (float((w * (torch.maximum(cur, s[:, h]) - cur)).sum()), -h))
        S.append(c)
        cur = torch.maximum(cur, s[:, c])
    S = sorted(S)
    pi = {j: max(S, key=lambda i: (float(F[j, i]), -i)) for j in range(H) if j not in S}
    return S, pi


def procrustes_map(G: torch.Tensor, i: int, j: int, num_heads: int, kind: str = "scaled_orthogonal",
                   ridge_rel: float = 1e-4) -> torch.Tensor:
    """R (d x d) such that Z_j ~ Z_i R, fitted on the calibration tokens."""
    d = G.shape[0] // num_heads
    G = G.double()
    Gii, Gij = G[_blk(i, d), _blk(i, d)], G[_blk(i, d), _blk(j, d)]
    if kind == "scalar":
        return torch.eye(d, dtype=torch.float64) * (torch.trace(Gij) / torch.trace(Gii).clamp_min(1e-30))
    if kind in ("orthogonal", "scaled_orthogonal"):
        P, sv, Qh = torch.linalg.svd(Gij)
        U = P @ Qh
        return U if kind == "orthogonal" else U * (sv.sum() / torch.trace(Gii).clamp_min(1e-30))
    if kind == "ridge":
        lam = ridge_rel * torch.diagonal(Gii).mean()
        return torch.linalg.solve(Gii + lam * torch.eye(d, dtype=torch.float64), Gij)
    raise ValueError(f"unknown kind {kind!r}")


def merge_weights(G: torch.Tensor, W: torch.Tensor, S, pi: dict[int, int], num_heads: int,
                  kind: str = "scaled_orthogonal") -> torch.Tensor:
    """W with each pruned head j folded into pi(j): W_O^{pi(j)} += R_j W_O^j; pruned rows zeroed."""
    d = G.shape[0] // num_heads
    W = W.double()
    out = torch.zeros_like(W)
    for i in S:
        out[_blk(i, d)] = W[_blk(i, d)]
    for j, i in pi.items():
        out[_blk(i, d)] += procrustes_map(G, i, j, num_heads, kind) @ W[_blk(j, d)]
    return out


def _rows(S, d: int) -> torch.Tensor:
    return torch.cat([torch.arange(h * d, (h + 1) * d) for h in S])


def ls_refit(G: torch.Tensor, C: torch.Tensor, S, num_heads: int, ridge_rel: float = 1e-4) -> torch.Tensor:
    """argmin_{W_S} ||Y - Z_S W_S||^2 + lam ||W_S||^2 with C = Z^T Y; returns a full-size W with pruned rows zero.

    One-shot: C = G @ W (Y = Z W). Sequential/teacher: C = Z_student^T Y_teacher.
    """
    d = G.shape[0] // num_heads
    G, C = G.double(), C.double()
    r = _rows(S, d)
    Gss = G[r][:, r]
    lam = ridge_rel * torch.diagonal(Gss).mean() if ridge_rel > 0 else 0.0
    Ws = torch.linalg.solve(Gss + lam * torch.eye(len(r), dtype=torch.float64), C[r])
    out = torch.zeros(G.shape[0], C.shape[1], dtype=torch.float64)
    out[r] = Ws
    return out


def layer_error(G: torch.Tensor, W: torch.Tensor, W_hat: torch.Tensor) -> float:
    """||Z W - Z W_hat||_F computed from G."""
    D = W.double() - W_hat.double()
    return float(torch.einsum("ij,ik,kj->", D, G.double(), D).clamp_min(0).sqrt())


def delete_weights(W: torch.Tensor, S, num_heads: int) -> torch.Tensor:
    d = W.shape[0] // num_heads
    out = torch.zeros_like(W.double())
    r = _rows(S, d)
    out[r] = W.double()[r]
    return out


def greedy_ls_select(G: torch.Tensor, W: torch.Tensor, k: int, num_heads: int, ridge_rel: float = 1e-4) -> list[int]:
    """Forward selection on the exact one-shot LS residual of the layer."""
    C = G.double() @ W.double()
    S: list[int] = []
    for _ in range(k):
        best = min((h for h in range(num_heads) if h not in S),
                   key=lambda h: (layer_error(G, W, ls_refit(G, C, S + [h], num_heads, ridge_rel)), h))
        S.append(best)
    return sorted(S)


def exhaustive_ls_best(G: torch.Tensor, W: torch.Tensor, k: int, num_heads: int,
                       ridge_rel: float = 1e-4) -> tuple[list[int], float]:
    """Exact minimiser of the one-shot LS residual over all C(H, k) subsets (H = 12 -> at most 924)."""
    C = G.double() @ W.double()
    err, S = min((layer_error(G, W, ls_refit(G, C, list(S), num_heads, ridge_rel)), list(S))
                 for S in combinations(range(num_heads), k))
    return S, err


# ---- model plumbing (HF BERT) ----------------------------------------------------------------------------------
def _layers(model):
    return getattr(model, model.base_model_prefix).encoder.layer


@contextmanager
def _capture(model, layer_ids, want_out: bool = False):
    """Record inputs (context) and optionally outputs (minus bias) of attention.output.dense for the given layers."""
    store: dict[int, dict[str, torch.Tensor]] = {l: {} for l in layer_ids}
    handles = []
    for l in layer_ids:
        dense = _layers(model)[l].attention.output.dense
        handles.append(dense.register_forward_pre_hook(lambda m, a, l=l: store[l].__setitem__("x", a[0].detach())))
        if want_out:
            handles.append(dense.register_forward_hook(
                lambda m, a, o, l=l: store[l].__setitem__("y", (o - m.bias).detach())))
    try:
        yield store
    finally:
        for h in handles:
            h.remove()


@torch.no_grad()
def collect_context_grams(model, loader, device: str) -> list[torch.Tensor]:
    """Per layer G = sum over non-pad tokens of x^T x (x = 768-dim head-context vector), float64 on CPU."""
    model.eval()
    L = len(_layers(model))
    Gs: list[torch.Tensor | None] = [None] * L
    with _capture(model, range(L)) as store:
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            batch.pop("labels", None)
            model(**batch, return_dict=True)
            m = batch["attention_mask"].bool()
            for l in range(L):
                x = store[l]["x"][m].double().cpu()
                Gs[l] = x.T @ x if Gs[l] is None else Gs[l] + x.T @ x
    return Gs


def output_weight(model, layer: int) -> torch.Tensor:
    """W (in x out) = dense.weight.T of the attention output projection."""
    return _layers(model)[layer].attention.output.dense.weight.detach().double().cpu().T.contiguous()


@torch.no_grad()
def set_output_weight(model, layer: int, W: torch.Tensor) -> None:
    dense = _layers(model)[layer].attention.output.dense
    dense.weight.copy_(W.T.to(dense.weight.dtype).to(dense.weight.device))


@torch.no_grad()
def sequential_ls_refit(student, teacher, loader, device: str, selections: dict[int, list[int]],
                        num_heads: int, ridge_rel: float = 1e-4) -> None:
    """Layer by layer (first to last): refit the kept heads' output projection of ``student`` so that its attention
    output matches the unpruned ``teacher``'s, using inputs from the already-pruned student (Kwon-style).
    ``selections`` maps layer -> kept heads; the student is evaluated with the matching head mask."""
    from .hf_experiments import head_mask_from_selection
    student.eval()
    teacher.eval()
    L = len(_layers(student))
    for l in range(L):
        sel = {q: (selections[q] if q <= l else list(range(num_heads))) for q in range(L)}
        mask = head_mask_from_selection(L, num_heads, sel, device)
        G = C = None
        with _capture(student, [l]) as s_store, _capture(teacher, [l], want_out=True) as t_store:
            for batch in loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                batch.pop("labels", None)
                student(**batch, head_mask=mask, return_dict=True)
                teacher(**batch, return_dict=True)
                m = batch["attention_mask"].bool()
                x = s_store[l]["x"][m].double().cpu()
                y = t_store[l]["y"][m].double().cpu()
                G = x.T @ x if G is None else G + x.T @ x
                C = x.T @ y if C is None else C + x.T @ y
        set_output_weight(student, l, ls_refit(G, C, selections[l], num_heads, ridge_rel))
