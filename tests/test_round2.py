import numpy as np
import pytest
import torch

from qfc.baselines import (
    delimiter_mass_from_attention,
    michel_global_selection,
    topk_selection,
    validate_global_selection,
    validate_selection,
)
from qfc.metrics import (
    auroc,
    bootstrap_p_le,
    cluster_bootstrap_delta,
    holm_adjust,
    mcc_binary,
)


# ---- DelimiterMass --------------------------------------------------------------------
def test_delimiter_mass_excludes_padding():
    # one example, 1 head, T=5: real tokens 0..2 ([CLS], word, [SEP]); positions 3,4 are padding.
    attn = torch.zeros(1, 1, 5, 5)
    attn[0, 0, 0] = torch.tensor([0.0, 0.5, 0.5, 0.0, 0.0])  # [CLS] query: half on [SEP]
    attn[0, 0, 1] = torch.tensor([0.5, 0.0, 0.5, 0.0, 0.0])  # word: 0.5 on CLS + 0.5 on SEP = 1.0
    attn[0, 0, 2] = torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0])  # [SEP] query: none on delimiters
    attn[0, 0, 3:] = 1.0  # padding queries: garbage that must be ignored
    mask = torch.tensor([[1, 1, 1, 0, 0]])
    delim = torch.tensor([[1, 0, 1, 0, 0]])
    out = delimiter_mass_from_attention(attn, mask, delim)
    # delimiter mass per real query: [0.5, 1.0, 0.0] -> mean 0.5; padding queries ignored.
    assert out.shape == (1, 1)
    assert out[0, 0].item() == pytest.approx(0.5)


def test_delimiter_mass_ignores_padding_keys_even_if_flagged():
    attn = torch.full((1, 1, 3, 3), 1.0 / 3)
    mask = torch.tensor([[1, 1, 0]])
    delim = torch.tensor([[0, 0, 1]])  # a padding position wrongly flagged as delimiter
    assert delimiter_mass_from_attention(attn, mask, delim)[0, 0].item() == pytest.approx(0.0)


# ---- MichelGlobal ---------------------------------------------------------------------
def test_michel_global_total_and_topk_correctness():
    torch.manual_seed(0)
    layers, heads, total = 12, 12, 72
    michel = torch.rand(layers, heads)
    sel = michel_global_selection(michel, total)
    validate_global_selection(sel, layers, heads, total)
    assert sum(len(v) for v in sel.values()) == total
    # globally correct: every kept normalised score >= every dropped normalised score
    norm = michel / michel.norm(dim=1, keepdim=True)
    kept = torch.tensor([norm[l, h] for l, hs in sel.items() for h in hs])
    mask = torch.ones(layers, heads, dtype=torch.bool)
    for l, hs in sel.items():
        mask[l, hs] = False
    assert kept.min() >= norm[mask].max()


def test_michel_global_can_empty_a_layer_and_min1_floor_prevents_it():
    # layers 1-3 have two strong heads each; layer 0 is flat, so its normalised scores are all low.
    michel = torch.tensor([[1.0, 1.0, 1.0, 1.0],
                           [9.0, 8.0, 0.1, 0.1],
                           [9.0, 8.0, 0.1, 0.1],
                           [9.0, 8.0, 0.1, 0.1]])
    sel = michel_global_selection(michel, total=6)
    validate_global_selection(sel, 4, 4, 6)
    assert sel[0] == []  # empty layer is allowed without a floor
    floor = michel_global_selection(michel, total=6, min_per_layer=1)
    validate_global_selection(floor, 4, 4, 6, min_per_layer=1)
    assert all(len(v) >= 1 for v in floor.values())


def test_michel_global_l2_normalisation_changes_ranking():
    # raw scores say layer 1 dominates; per-layer normalisation puts both layers' best heads on top
    michel = torch.tensor([[1.0, 0.1], [100.0, 10.0]])
    assert michel_global_selection(michel, total=2) == {0: [0], 1: [0]}


def test_michel_global_is_deterministic_on_ties():
    michel = torch.ones(3, 4)
    assert michel_global_selection(michel, 5) == michel_global_selection(michel, 5)


def test_validate_global_selection_rejects_bad_totals():
    with pytest.raises(ValueError):
        validate_global_selection({0: [0, 1], 1: [0]}, 2, 4, total=4)
    with pytest.raises(ValueError):
        validate_global_selection({0: [0, 0], 1: [0]}, 2, 4, total=3)
    with pytest.raises(ValueError):
        validate_global_selection({0: [0], 1: []}, 2, 4, total=1, min_per_layer=1)


def test_uniform_delimiter_keep_low_has_exact_k_per_layer():
    scores = [torch.rand(12) for _ in range(12)]
    sel = topk_selection(scores, 6, largest=False)
    validate_selection(sel, 12, 12, 6)
    for l, hs in sel.items():
        dropped = [h for h in range(12) if h not in hs]
        assert scores[l][hs].max() <= scores[l][dropped].min()


# ---- statistics -----------------------------------------------------------------------
def test_auroc_and_mcc_known_values():
    assert auroc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert auroc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == 0.0
    assert auroc([0, 1], [0.5, 0.5]) == 0.5
    assert np.isnan(auroc([1, 1], [0.1, 0.2]))
    assert mcc_binary([1, 1, 0, 0], [1, 1, 0, 0]) == pytest.approx(1.0)
    assert mcc_binary([1, 1, 0, 0], [0, 0, 1, 1]) == pytest.approx(-1.0)
    assert mcc_binary([1, 0, 1, 0], [1, 1, 1, 1]) == 0.0  # one class predicted: undefined -> 0


def test_holm_adjust_known_example():
    adj = holm_adjust([0.01, 0.04, 0.03, 0.005])
    assert adj == pytest.approx([0.03, 0.06, 0.06, 0.02])


def _noisy(rng, labels, seeds, acc):
    keep = rng.random((seeds, len(labels))) < acc
    return np.where(keep, labels[None, :], 1 - labels[None, :])


def test_cluster_bootstrap_identical_methods_gives_zero():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, 200)
    preds = _noisy(rng, labels, 5, 0.8)
    out = cluster_bootstrap_delta(preds, preds, labels, "accuracy", n_boot=500)
    assert out["observed_delta"] == 0.0 and out["ci_low"] == 0.0 and out["ci_high"] == 0.0


def test_cluster_bootstrap_detects_a_real_gap_and_matches_direct_metric():
    rng = np.random.default_rng(1)
    labels = rng.integers(0, 2, 400)
    good, bad = _noisy(rng, labels, 5, 0.9), _noisy(rng, labels, 5, 0.7)
    out = cluster_bootstrap_delta(good, bad, labels, "accuracy", n_boot=1000, seed=3)
    assert out["observed_delta"] == pytest.approx((good == labels).mean() - (bad == labels).mean())
    assert out["ci_low"] > 0.1
    assert bootstrap_p_le(out["boot"], 0.0) < 0.01


def test_cluster_bootstrap_mcc_observed_matches_pooled_mcc():
    rng = np.random.default_rng(4)
    labels = rng.integers(0, 2, 300)
    a, b = _noisy(rng, labels, 3, 0.85), _noisy(rng, labels, 3, 0.6)
    out = cluster_bootstrap_delta(a, b, labels, "mcc", n_boot=200, seed=6)
    y = np.tile(labels, 3)
    assert out["observed_delta"] == pytest.approx(mcc_binary(y, a.reshape(-1)) - mcc_binary(y, b.reshape(-1)))


def test_cluster_bootstrap_validates_inputs():
    labels = np.zeros(10, dtype=int)
    with pytest.raises(ValueError):
        cluster_bootstrap_delta(np.zeros((2, 10)), np.zeros((3, 10)), labels)
    with pytest.raises(ValueError):
        cluster_bootstrap_delta(np.zeros((2, 10)), np.zeros((2, 10)), labels, "f1")
