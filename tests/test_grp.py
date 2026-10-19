import copy
import math

import pytest
import torch

from qfc.grp import (
    ModuleState,
    _aug,
    _hooks,
    collect_grams,
    grp_prune,
    grp_select,
    mac_per_head,
    mac_per_neuron,
    model_mac,
    module_sensitivity,
    prune_model,
    sequential_refit,
    teacher_weight,
)


def _ridge_obj(X, Y, cols, lam):
    """min_V ||Y - X[:, cols] V||^2 + lam ||V_nonbias||^2, bias = last column of X, never penalised."""
    Xs = X[:, cols]
    pen = torch.tensor([0.0 if c == X.shape[1] - 1 else lam for c in cols], dtype=torch.float64)
    V = torch.linalg.solve(Xs.T @ Xs + torch.diag(pen), Xs.T @ Y)
    return float(((Y - Xs @ V) ** 2).sum() + (pen[:, None] * V**2).sum())


@pytest.mark.parametrize("unit", [1, 3])
def test_removal_cost_and_downdate_match_brute_force(unit):
    g = torch.Generator().manual_seed(0)
    n_units, N, D = 6, 120, 5
    X = _aug(torch.randn(N, n_units * unit, generator=g, dtype=torch.float64))
    W = torch.randn(n_units * unit + 1, D, generator=g, dtype=torch.float64)
    Y = X @ W + 0.1 * torch.randn(N, D, generator=g, dtype=torch.float64)
    G = X.T @ X
    W_aug = torch.linalg.lstsq(X, Y).solution  # so that C = G W_aug = X^T Y exactly
    st = ModuleState(G, W_aug, unit, ridge_rel=1e-2, refresh_every=1000)

    def cols(units):
        return [f for j in units for f in range(j * unit, (j + 1) * unit)] + [n_units * unit]

    base = _ridge_obj(X, Y, cols(st.kept), st.lam)
    for i, j in enumerate(st.kept):
        rest = [u for u in st.kept if u != j]
        assert float(st.costs()[i]) == pytest.approx(_ridge_obj(X, Y, cols(rest), st.lam) - base, rel=1e-6)
    st.remove([1, 4])
    fresh = ModuleState(G, W_aug, unit, ridge_rel=1e-2)
    fresh.kept = [0, 2, 3, 5]
    fresh._refresh()
    assert torch.allclose(st.P, fresh.P, atol=1e-8) and torch.allclose(st.V, fresh.V, atol=1e-8)


def test_kwon_mac_formula():
    assert mac_per_head(128, 768, 64) == 3 * 128 * 768 * 64 + 2 * 128 * 128 * 64 + 128 * 64 * 768
    assert mac_per_neuron(128, 768) == 2 * 128 * 768
    assert model_mac([12] * 12, [3072] * 12, 128, 768, 64) == pytest.approx(
        12 * (12 * mac_per_head(128, 768, 64) + 3072 * mac_per_neuron(128, 768)))


# ---- tiny random BERT (no download) ----------------------------------------------------------------------------
def _tiny_bert(seed=0):
    from transformers import BertConfig, BertForSequenceClassification
    cfg = BertConfig(vocab_size=50, hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
                     intermediate_size=64, max_position_embeddings=32, num_labels=3)
    cfg._attn_implementation = "eager"
    torch.manual_seed(seed)
    return BertForSequenceClassification(cfg).eval()


def _batches(n=8, T=12, nb=3):
    g = torch.Generator().manual_seed(1)
    out = []
    for _ in range(nb):
        ids = torch.randint(5, 50, (n, T), generator=g)
        mask = torch.ones(n, T, dtype=torch.long)
        mask[: n // 2, T - 4:] = 0  # some padding
        out.append({"input_ids": ids, "attention_mask": mask, "token_type_ids": torch.zeros_like(ids)})
    return out


def _logits(model, batches, head_mask=None):
    return torch.cat([model(**b, head_mask=head_mask).logits for b in batches])


@torch.no_grad()
def test_physical_pruning_equals_masking():
    model, bs = _tiny_bert(), _batches()
    heads, neurons = {0: [1, 3], 1: [0, 1, 2]}, {0: list(range(0, 64, 2)), 1: list(range(40))}
    masked = copy.deepcopy(model)
    for l, keep in neurons.items():
        dense = masked.bert.encoder.layer[l].output.dense
        dense.weight[:, [j for j in range(64) if j not in keep]] = 0
    hm = torch.zeros(2, 4)
    for l, keep in heads.items():
        hm[l, keep] = 1
    pruned = prune_model(copy.deepcopy(model), heads, neurons)
    assert torch.allclose(_logits(pruned, bs), _logits(masked, bs, hm), atol=1e-5)


@torch.no_grad()
def test_layer_with_zero_heads_runs():
    pruned = prune_model(_tiny_bert(), {0: [], 1: [2]}, {0: [5], 1: list(range(64))})
    assert torch.isfinite(_logits(pruned, _batches())).all()


def test_sensitivity_is_finite_and_positive():
    s = module_sensitivity(_tiny_bert(), _batches(), "cpu", seed=0)
    assert set(s) == {(l, k) for l in range(2) for k in ("att", "ffn")}
    assert all(math.isfinite(v) and v > 0 for v in s.values())


def test_grp_select_meets_budget_and_constraints():
    model, bs = _tiny_bert(), _batches()
    G = collect_grams(model, bs, "cpu")
    W = {(l, k): teacher_weight(model, l, k) for l in range(2) for k in ("att", "ffn")}
    for constraint in (0.7, 0.4):
        heads, neurons, info = grp_select(G, W, None, 2, 4, 10.0, 32, constraint)
        assert info["mac_ratio"] <= constraint + 1e-12
        assert all(len(v) >= 1 for v in neurons.values())
        assert (heads, neurons) == grp_select(G, W, None, 2, 4, 10.0, 32, constraint)[:2]  # deterministic


@torch.no_grad()
def _module_errors(student, teacher, bs):
    total = 0.0
    specs = [(l, k, "out", (l, k)) for l in range(2) for k in ("att", "ffn")]
    with _hooks(student, specs) as s, _hooks(teacher, specs) as t:
        for b in bs:
            student(**b)
            teacher(**b)
            m = b["attention_mask"].bool()
            total += sum(float(((s[k][m] - t[k][m]) ** 2).sum()) for k in s)
    return total


def test_sequential_refit_reduces_error_and_kl():
    teacher, bs = _tiny_bert(), _batches(nb=6)
    heads, neurons = {0: [0, 2], 1: [1]}, {0: list(range(0, 64, 3)), 1: list(range(10))}
    plain = prune_model(copy.deepcopy(teacher), heads, neurons)
    refit = prune_model(copy.deepcopy(teacher), heads, neurons)
    sequential_refit(refit, teacher, bs, "cpu", ridge_rel=1e-6)
    assert _module_errors(refit, teacher, bs) < _module_errors(plain, teacher, bs)
    with torch.no_grad():
        pt = _logits(teacher, bs).log_softmax(-1)

        def kl(m):
            return float((pt.exp() * (pt - _logits(m, bs).log_softmax(-1))).sum(-1).mean())
        assert kl(refit) < kl(plain)


def test_random_select_budget_and_determinism():
    from qfc.grp import random_select
    a = random_select(2, 4, 64, 10.0, 32, 0.5, seed=3)
    assert a[2]["mac_ratio"] <= 0.5 + 1e-12 and all(len(v) >= 1 for v in a[1].values())
    assert a[:2] == random_select(2, 4, 64, 10.0, 32, 0.5, seed=3)[:2]
    assert a[:2] != random_select(2, 4, 64, 10.0, 32, 0.5, seed=4)[:2]


@torch.no_grad()
def test_permute_classifier_reorders_logits():
    from qfc.grp import permute_classifier
    model, bs = _tiny_bert(), _batches()
    before = _logits(model, bs)
    permute_classifier(model, [1, 2, 0])
    assert torch.allclose(_logits(model, bs), before[:, [1, 2, 0]], atol=1e-6)


def test_grp_runs_on_question_answering_model():
    from transformers import BertConfig, BertForQuestionAnswering
    cfg = BertConfig(vocab_size=50, hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
                     intermediate_size=64, max_position_embeddings=32)
    cfg._attn_implementation = "eager"
    torch.manual_seed(0)
    qa = BertForQuestionAnswering(cfg).eval()
    bs = _batches()
    s = module_sensitivity(qa, bs, "cpu", seed=0)
    assert all(math.isfinite(v) and v > 0 for v in s.values())
    student, heads, neurons, info = grp_prune(qa, bs, "cpu", seq_len=10.0, constraint=0.5)
    out = student(**bs[0])
    assert torch.isfinite(out.start_logits).all() and info["mac_ratio"] <= 0.5 + 1e-12


def test_grp_prune_end_to_end():
    teacher, bs = _tiny_bert(), _batches()
    student, heads, neurons, info = grp_prune(teacher, bs, "cpu", seq_len=10.0, constraint=0.5)
    assert info["mac_ratio"] <= 0.5 + 1e-12 and info["sensitivity"]
    assert torch.isfinite(_logits(student, bs)).all()
    assert sum(p.numel() for p in student.parameters()) < sum(p.numel() for p in teacher.parameters())


# ---- V16 Stage A ---------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("unit", [1, 3])
def test_block_cost_matches_brute_force_joint_removal(unit):
    g = torch.Generator().manual_seed(3)
    n_units, N, D = 7, 150, 4
    X = _aug(torch.randn(N, n_units * unit, generator=g, dtype=torch.float64))
    Y = X @ torch.randn(n_units * unit + 1, D, generator=g, dtype=torch.float64) + 0.1 * torch.randn(N, D, generator=g, dtype=torch.float64)
    st = ModuleState(X.T @ X, torch.linalg.lstsq(X, Y).solution, unit, ridge_rel=1e-2, refresh_every=1000)
    st.remove([2])

    def cols(units):
        return [f for j in units for f in range(j * unit, (j + 1) * unit)] + [n_units * unit]

    base = _ridge_obj(X, Y, cols(st.kept), st.lam)
    rest = [u for u in st.kept if u not in (0, 5)]
    assert st.block_cost([0, 5]) == pytest.approx(_ridge_obj(X, Y, cols(rest), st.lam) - base, rel=1e-6)


def test_grp_select_counts_hits_counts_and_matches_single_module_greedy():
    from qfc.grp import grp_select_counts
    model, bs = _tiny_bert(), _batches()
    G = collect_grams(model, bs, "cpu")
    W = {(l, k): teacher_weight(model, l, k) for l in range(2) for k in ("att", "ffn")}
    counts = {(0, "att"): 2, (0, "ffn"): 10, (1, "att"): 4, (1, "ffn"): 63}
    heads, neurons = grp_select_counts(G, W, counts, 4)
    assert [len(heads[0]), len(neurons[0]), len(heads[1]), len(neurons[1])] == [2, 10, 4, 63]
    st = ModuleState(G[(1, "ffn")], W[(1, "ffn")], 1)  # one removal: the single cheapest neuron goes
    assert set(range(64)) - set(neurons[1]) == {int(torch.argmin(st.costs()))}


# ---- V17 GRP-R (drift-corrected refit) -----------------------------------------------------------------------------
def _refit(drift, heads, neurons):
    teacher, bs = _tiny_bert(), _batches()
    s = prune_model(copy.deepcopy(teacher), heads, neurons)
    sequential_refit(s, teacher, bs, "cpu", ridge_rel=1e-4, drift_correct=drift)
    return s, teacher, bs


@torch.no_grad()
def _preln_error(student, teacher, bs, l, kind):
    """Squared error of the pre-LayerNorm sum dense(x) + h at sublayer (l, kind), non-padding tokens."""
    err = 0.0
    spec = [(l, kind, "out", "y"), (l, kind, "res", "h")]
    with _hooks(student, spec) as s_st, _hooks(teacher, spec) as t_st:
        for b in bs:
            student(**b)
            teacher(**b)
            m = b["attention_mask"].bool()
            err += float(((s_st["y"] + s_st["h"] - t_st["y"] - t_st["h"])[m].double() ** 2).sum())
    return err


def test_drift_correct_equals_plain_where_no_drift_exists():
    # the first sublayer (layer 0 attention) adds the embedding output, identical in student and teacher: h_s == h_t
    heads, neurons = {0: [0, 2]}, {0: list(range(0, 64, 2))}
    a, _, _ = _refit(False, heads, neurons)
    b, _, _ = _refit(True, heads, neurons)
    da, db = a.bert.encoder.layer[0].attention.output.dense, b.bert.encoder.layer[0].attention.output.dense
    assert torch.allclose(da.weight, db.weight, atol=1e-6) and torch.allclose(da.bias, db.bias, atol=1e-6)


def test_drift_correct_lowers_downstream_preln_error():
    heads, neurons = {0: [1], 1: list(range(4))}, {0: list(range(8)), 1: list(range(64))}  # heavy layer-0 pruning
    plain, t, bs = _refit(False, heads, neurons)
    drift, _, _ = _refit(True, heads, neurons)
    assert _preln_error(drift, t, bs, 1, "att") < _preln_error(plain, t, bs, 1, "att")
