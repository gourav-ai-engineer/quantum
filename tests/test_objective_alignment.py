import torch

from qfc.coverage import coverage_value


def test_objective_alignment_fixture():
    sim = torch.eye(3)
    selected = [0, 1]
    assert float(coverage_value(sim, selected)) == 2.0
