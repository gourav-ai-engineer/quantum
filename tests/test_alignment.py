import pytest
import torch

from qfc.alignment import bootstrap_spearman, check_subset_budget, rankdata, spearman


def test_rankdata_averages_ties_zero_based():
    assert rankdata([10.0, 20.0, 20.0, 30.0]).tolist() == [0.0, 1.5, 1.5, 3.0]
    assert rankdata([3.0, 1.0, 2.0]).tolist() == [2.0, 0.0, 1.0]


def test_spearman_known_values():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    # monotone but nonlinear is still rho = 1
    assert spearman([1, 2, 3, 4], [1, 8, 27, 64]) == pytest.approx(1.0)
    # hand-computed with ties: ranks x=[0,1,2,3], y=[0,1.5,1.5,3]
    assert spearman([1, 2, 3, 4], [1, 2, 2, 3]) == pytest.approx(0.9486832980505138)


def test_spearman_constant_input_is_zero_and_flagged_degenerate():
    assert spearman([1, 2, 3], [5, 5, 5]) == 0.0
    out = bootstrap_spearman([1, 2, 3], [5, 5, 5], n_boot=50, seed=0)
    assert out["degenerate"] is True
    assert out["estimate"] == 0.0
    assert out["ci_low"] is None and out["ci_high"] is None


def test_bootstrap_ci_is_deterministic_and_brackets_strong_correlation():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(200, generator=g)
    y = x + 0.1 * torch.randn(200, generator=g)
    a = bootstrap_spearman(x.tolist(), y.tolist(), n_boot=200, seed=1)
    b = bootstrap_spearman(x.tolist(), y.tolist(), n_boot=200, seed=1)
    assert a == b
    assert a["degenerate"] is False
    assert a["ci_low"] <= a["estimate"] <= a["ci_high"] or abs(a["estimate"] - a["ci_low"]) < 0.05
    assert a["ci_low"] > 0.9 and a["ci_high"] <= 1.0
    assert a["n"] == 200 and a["n_boot"] == 200 and a["confidence"] == 0.95


def test_bootstrap_ci_includes_zero_for_independent_data():
    g = torch.Generator().manual_seed(3)
    x = torch.randn(100, generator=g).tolist()
    y = torch.randn(100, generator=g).tolist()
    out = bootstrap_spearman(x, y, n_boot=300, seed=2)
    assert out["ci_low"] < 0 < out["ci_high"]


def test_check_subset_budget():
    check_subset_budget(12, 6, 300)  # C(12,6) = 924
    with pytest.raises(ValueError):
        check_subset_budget(12, 6, 925)
