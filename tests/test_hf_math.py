import sys
sys.path.insert(0, "src")

import torch

from qfc.coverage import greedy_select


def test_greedy_coverage_exact_toy_case():
    # Two clear clusters. A good 2-head representative subset should cover both.
    sim = torch.tensor(
        [
            [1.00, 0.98, 0.05, 0.04],
            [0.98, 1.00, 0.06, 0.05],
            [0.05, 0.06, 1.00, 0.97],
            [0.04, 0.05, 0.97, 1.00],
        ]
    )
    selected, history = greedy_select(sim, 2)
    assert len(selected) == 2
    assert len(set(selected)) == 2
    assert history[-1] > history[0]
    # The selected set must contain one representative from each cluster.
    assert ({0, 2} <= set(selected)) or ({0, 3} <= set(selected)) or (
        {1, 2} <= set(selected)
    ) or ({1, 3} <= set(selected))
