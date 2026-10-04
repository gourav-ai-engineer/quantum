import math
import pytest

import torch

from qfc.states import density_operator, aggregate_density_operators
from qfc.fidelity import fidelity, pairwise_fidelity, pairwise_fidelity_batched
from qfc.coverage import coverage_value, greedy_select


def test_density_operator_is_valid():
    torch.manual_seed(0)
    a = torch.rand(12, 12)
    rho = density_operator(a)
    assert torch.allclose(rho, rho.T, atol=1e-6)
    assert torch.linalg.eigvalsh(rho).min().item() >= -1e-7
    assert math.isclose(torch.trace(rho).item(), 1.0, rel_tol=0.0, abs_tol=1e-6)


def test_fidelity_identity_and_range():
    torch.manual_seed(1)
    rho = density_operator(torch.rand(8, 8))
    f = fidelity(rho, rho).item()
    assert abs(f - 1.0) < 1e-5

    sigma = density_operator(torch.rand(8, 8))
    f2 = fidelity(rho, sigma).item()
    assert 0.0 <= f2 <= 1.0 + 1e-6


def test_aggregation_preserves_density_properties():
    torch.manual_seed(2)
    rhos = density_operator(torch.rand(5, 4, 10, 10))
    mean_rho = aggregate_density_operators(rhos)
    traces = torch.diagonal(mean_rho, dim1=-2, dim2=-1).sum(-1)
    eigmins = torch.linalg.eigvalsh(mean_rho).min(dim=-1).values
    assert torch.allclose(traces, torch.ones_like(traces), atol=1e-6)
    assert eigmins.min().item() >= -1e-6


def test_pairwise_fidelity_is_symmetric():
    torch.manual_seed(3)
    rhos = density_operator(torch.rand(6, 7, 7))
    mat = pairwise_fidelity(rhos)
    assert torch.allclose(mat, mat.T, atol=1e-5)
    assert torch.allclose(torch.diag(mat), torch.ones(6), atol=1e-5)


def test_coverage_is_monotone_and_greedy_has_diminishing_returns():
    sim = torch.tensor([
        [1.00, 0.95, 0.10, 0.10],
        [0.95, 1.00, 0.10, 0.10],
        [0.10, 0.10, 1.00, 0.90],
        [0.10, 0.10, 0.90, 1.00],
    ])
    c1 = coverage_value(sim, [0]).item()
    c2 = coverage_value(sim, [0, 2]).item()
    c3 = coverage_value(sim, [0, 2, 3]).item()
    assert c1 <= c2 <= c3 + 1e-7

    selected, history = greedy_select(sim, 2)
    assert len(selected) == 2
    assert history[1] >= history[0]


def test_input_conditioned_coverage():
    sim = torch.stack([
        torch.tensor([[1.0, 0.9, 0.1], [0.9, 1.0, 0.2], [0.1, 0.2, 1.0]]),
        torch.tensor([[1.0, 0.8, 0.2], [0.8, 1.0, 0.3], [0.2, 0.3, 1.0]]),
    ])
    selected, history = greedy_select(sim, 2)
    assert len(selected) == 2
    assert history[-1] >= history[0]


def test_batched_fidelity_matches_single_sample():
    torch.manual_seed(4)
    rhos = density_operator(torch.rand(3, 4, 4))
    batched = pairwise_fidelity_batched(rhos)
    for sample in range(3):
        expected = pairwise_fidelity(rhos[sample])
        assert torch.allclose(batched[sample], expected, atol=2e-5)


def test_conditional_greedy_is_monotone():
    from qfc.conditional import conditional_greedy_select, conditional_coverage_value

    sim = torch.tensor(
        [
            [[1.0, 0.9, 0.1], [0.9, 1.0, 0.2], [0.1, 0.2, 1.0]],
            [[1.0, 0.2, 0.8], [0.2, 1.0, 0.3], [0.8, 0.3, 1.0]],
        ]
    )
    selected, history = conditional_greedy_select(sim, 2)
    assert len(selected) == 2
    assert history[1] >= history[0]
    assert conditional_coverage_value(sim, selected).item() == pytest.approx(
        history[-1].item(), abs=1e-5
    )
