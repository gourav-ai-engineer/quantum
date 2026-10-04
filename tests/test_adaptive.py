import pytest
import torch

from qfc.adaptive import layer_adaptive_greedy


def test_layer_adaptive_greedy_respects_budget_and_minimums():
    sims = [
        torch.eye(4),
        torch.tensor(
            [
                [1.0, 0.9, 0.1, 0.0],
                [0.9, 1.0, 0.2, 0.1],
                [0.1, 0.2, 1.0, 0.8],
                [0.0, 0.1, 0.8, 1.0],
            ]
        ),
    ]
    selected, history = layer_adaptive_greedy(
        sims, total_budget=5, min_per_layer=1
    )
    assert sum(len(v) for v in selected.values()) == 5
    assert all(len(v) >= 1 for v in selected.values())
    assert len(history) == 5


def test_layer_adaptive_greedy_can_use_weights():
    sims = [torch.eye(3)]
    weights = [torch.tensor([0.0, 0.0, 10.0])]
    selected, _ = layer_adaptive_greedy(
        sims, total_budget=1, weights=weights, min_per_layer=1
    )
    assert selected == {0: [2]}


def test_layer_adaptive_greedy_validates_budget():
    with pytest.raises(ValueError):
        layer_adaptive_greedy([torch.eye(3), torch.eye(3)], total_budget=1)
