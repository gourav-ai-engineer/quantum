import itertools
import pytest

import torch

from qfc.coverage import coverage_value


def test_toy_objective_is_exhaustively_submodular():
    # Verify the diminishing-returns inequality for every valid S subset T
    # and candidate h on a small similarity matrix.
    sim = torch.tensor(
        [
            [1.0, 0.8, 0.2, 0.1],
            [0.8, 1.0, 0.3, 0.2],
            [0.2, 0.3, 1.0, 0.9],
            [0.1, 0.2, 0.9, 1.0],
        ]
    )
    universe = set(range(sim.shape[0]))
    for r in range(4):
        for s_tuple in itertools.combinations(universe, r):
            S = set(s_tuple)
            for t_size in range(r, 4):
                for t_tuple in itertools.combinations(universe - S, t_size - r):
                    T = S | set(t_tuple)
                    if not S.issubset(T):
                        continue
                    remaining = universe - T
                    for h in remaining:
                        lhs = (
                            coverage_value(sim, sorted(S | {h}))
                            - coverage_value(sim, sorted(S))
                        ).item()
                        rhs = (
                            coverage_value(sim, sorted(T | {h}))
                            - coverage_value(sim, sorted(T))
                        ).item()
                        assert lhs + 1e-5 >= rhs


def test_empty_coverage_is_zero():
    sim = torch.eye(4)
    assert coverage_value(sim, []).item() == 0.0


def test_weighted_coverage_is_monotone_and_has_diminishing_returns():
    from qfc.coverage import weighted_coverage_value, weighted_greedy_select

    sim = torch.tensor([
        [1.0, 0.9, 0.1],
        [0.9, 1.0, 0.2],
        [0.1, 0.2, 1.0],
    ])
    weights = torch.tensor([0.7, 0.2, 0.1])

    selected, history = weighted_greedy_select(sim, weights, 2)
    assert len(selected) == 2
    assert history[1] >= history[0]
    assert weighted_coverage_value(sim, weights, selected).item() == pytest.approx(
        history[-1].item(), abs=1e-6
    )
