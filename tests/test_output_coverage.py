import math
from itertools import combinations

import pytest
import torch

from qfc.hf_experiments import head_mask_from_selection
from qfc.output_coverage import (
    _capture,
    bcm_select,
    bound_weights,
    collect_context_grams,
    coverage_bound,
    delete_weights,
    exhaustive_ls_best,
    greedy_ls_select,
    layer_error,
    ls_refit,
    merge_weights,
    output_fidelity,
    output_weight,
    procrustes_map,
    sequential_ls_refit,
    set_output_weight,
)


def _psd_sqrt(M):
    w, V = torch.linalg.eigh(0.5 * (M + M.T))
    return (V * w.clamp_min(0).sqrt()) @ V.T


def _toy_layer(H=6, d=8, N=80, D=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    common = torch.randn(N, 1, generator=g, dtype=torch.float64)
    Z = [torch.randn(N, d, generator=g, dtype=torch.float64) + 0.8 * common for _ in range(H)]
    Q, _ = torch.linalg.qr(torch.randn(d, d, generator=g, dtype=torch.float64))
    Z[3] = 1.3 * Z[1] @ Q + 0.05 * torch.randn(N, d, generator=g, dtype=torch.float64)  # near-copy of head 1
    Zc = torch.cat(Z, dim=1)
    W = torch.randn(H * d, D, generator=g, dtype=torch.float64)
    return Zc, W, Zc.T @ Zc


def test_output_fidelity_equals_uhlmann_fidelity():
    Zc, _, G = _toy_layer()
    d = 8
    F = output_fidelity(G, 6)
    for i, j in [(0, 1), (1, 3), (2, 5)]:
        X, Y = Zc[:, i * d:(i + 1) * d], Zc[:, j * d:(j + 1) * d]
        rho, sig = X @ X.T / (X**2).sum(), Y @ Y.T / (Y**2).sum()
        r = _psd_sqrt(rho)
        ref = torch.trace(_psd_sqrt(r @ sig @ r)) ** 2
        assert float(F[i, j]) == pytest.approx(float(ref), abs=1e-6)
    assert torch.allclose(torch.diagonal(F), torch.ones(6, dtype=torch.float64))
    assert float(F[1, 3]) > 0.99  # the planted near-copy is found despite the rotation


def test_scaled_procrustes_residual_identity():
    Zc, _, G = _toy_layer()
    d = 8
    F = output_fidelity(G, 6)
    for i, j in [(0, 2), (1, 3), (4, 5)]:
        R = procrustes_map(G, i, j, 6)
        res = torch.linalg.norm(Zc[:, j * d:(j + 1) * d] - Zc[:, i * d:(i + 1) * d] @ R)
        ref = torch.linalg.norm(Zc[:, j * d:(j + 1) * d]) * math.sqrt(1 - float(F[j, i]))
        assert float(res) == pytest.approx(ref, rel=1e-6)


def test_layer_error_from_gram_matches_direct():
    Zc, W, G = _toy_layer()
    W_hat = delete_weights(W, [0, 2, 4], 6)
    assert layer_error(G, W, W_hat) == pytest.approx(float(torch.linalg.norm(Zc @ W - Zc @ W_hat)), rel=1e-9)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_bound_merge_ls_ordering(seed):
    Zc, W, G = _toy_layer(seed=seed)
    F, w = output_fidelity(G, 6), bound_weights(G, W, 6)
    for S in ([0, 1, 2], [1, 4], [0, 2, 4, 5]):
        pi = {j: max(S, key=lambda i: float(F[j, i])) for j in range(6) if j not in S}
        e_merge = layer_error(G, W, merge_weights(G, W, S, pi, 6))
        e_ls = layer_error(G, W, ls_refit(G, G @ W, S, 6, ridge_rel=0.0))
        e_del = layer_error(G, W, delete_weights(W, S, 6))
        assert e_merge <= coverage_bound(F, w, S) + 1e-9
        assert e_ls <= e_merge + 1e-6
        assert e_ls <= e_del + 1e-6


def test_bcm_select_objective_and_guarantee():
    _, W, G = _toy_layer()
    F, w = output_fidelity(G, 6), bound_weights(G, W, 6)
    s = 1 - (1 - F).clamp_min(0).sqrt()
    f = lambda S: float((w * s[:, S].max(dim=1).values).sum())
    for k in (1, 2, 3, 4):
        S, pi = bcm_select(F, w, k)
        assert len(S) == k == len(set(S)) and S == sorted(S)
        assert set(pi) == set(range(6)) - set(S) and all(t in S for t in pi.values())
        assert coverage_bound(F, w, S) == pytest.approx(float(w.sum()) - f(S), rel=1e-9)
        best = max(f(list(c)) for c in combinations(range(6), k))
        assert f(S) >= (1 - 1 / math.e) * best - 1e-12
    S2, _ = bcm_select(F, w, 2)
    assert not {1, 3} <= set(S2)  # never keeps both members of the near-duplicate pair


def test_greedy_and_exhaustive_ls():
    _, W, G = _toy_layer()
    for k in (2, 3):
        S_ex, e_ex = exhaustive_ls_best(G, W, k, 6)
        S_gr = greedy_ls_select(G, W, k, 6)
        C = G @ W
        e_gr = layer_error(G, W, ls_refit(G, C, S_gr, 6))
        assert len(S_gr) == k and e_ex <= e_gr + 1e-9
        assert all(layer_error(G, W, ls_refit(G, C, list(c), 6)) >= e_ex - 1e-9 for c in combinations(range(6), k))


# ---- integration on a tiny random BERT (no download) -----------------------------------------------------------
def _tiny_bert():
    from transformers import BertConfig, BertForSequenceClassification
    cfg = BertConfig(vocab_size=50, hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
                     intermediate_size=64, max_position_embeddings=32, num_labels=2)
    cfg._attn_implementation = "eager"
    torch.manual_seed(0)
    return BertForSequenceClassification(cfg).eval()


def _loader(n=6, T=10):
    g = torch.Generator().manual_seed(1)
    batches = []
    for b in range(2):
        ids = torch.randint(5, 50, (n, T), generator=g)
        mask = torch.ones(n, T, dtype=torch.long)
        mask[:, T - 1 - b * 3:] = 0  # some padding
        batches.append({"input_ids": ids, "attention_mask": mask, "token_type_ids": torch.zeros_like(ids),
                        "labels": torch.zeros(n, dtype=torch.long)})
    return batches


def _attn_out(model, loader, layer, head_mask=None):
    ys = []
    with _capture(model, [layer], want_out=True) as st:
        for b in loader:
            bb = {k: v for k, v in b.items() if k != "labels"}
            model(**bb, head_mask=head_mask, return_dict=True)
            ys.append(st[layer]["y"][b["attention_mask"].bool()])
    return torch.cat(ys).double()


def test_merge_on_real_bert_layer_matches_gram_prediction():
    model, loader = _tiny_bert(), _loader()
    Gs = collect_context_grams(model, loader, "cpu")
    W0 = output_weight(model, 0)
    y = _attn_out(model, loader, 0)
    assert torch.allclose(Gs[0], Gs[0].T) and torch.linalg.eigvalsh(Gs[0]).min() > -1e-6
    F, w = output_fidelity(Gs[0], 4), bound_weights(Gs[0], W0, 4)
    S, pi = bcm_select(F, w, 2)
    W_hat = merge_weights(Gs[0], W0, S, pi, 4)
    set_output_weight(model, 0, W_hat)
    mask = head_mask_from_selection(2, 4, {0: S, 1: [0, 1, 2, 3]}, "cpu")
    y_hat = _attn_out(model, loader, 0, head_mask=mask)
    actual = float(torch.linalg.norm(y - y_hat))
    assert actual == pytest.approx(layer_error(Gs[0], W0, W_hat), rel=1e-3)  # float32 forward vs float64 Gram
    assert actual <= coverage_bound(F, w, S) * (1 + 1e-3)


def test_sequential_ls_refit_reduces_layer_error():
    teacher, student, loader = _tiny_bert(), _tiny_bert(), _loader()
    sel = {0: [0, 2], 1: [1, 3]}
    mask = head_mask_from_selection(2, 4, sel, "cpu")
    y = _attn_out(teacher, loader, 0)
    e_del = float(torch.linalg.norm(y - _attn_out(student, loader, 0, head_mask=mask)))
    sequential_ls_refit(student, teacher, loader, "cpu", sel, 4, ridge_rel=0.0)
    e_ls = float(torch.linalg.norm(y - _attn_out(student, loader, 0, head_mask=mask)))
    assert e_ls < e_del
    logits_s = student(**{k: v for k, v in loader[0].items() if k != "labels"}, head_mask=mask).logits
    assert torch.isfinite(logits_s).all()
