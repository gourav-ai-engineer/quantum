import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from beat_run import heldout_dist, valid_metrics  # noqa: E402


def test_accuracy_and_mcc_known_values():
    logits = torch.tensor([[2., 0.], [0., 2.], [0., 2.], [2., 0.]])
    labels = torch.tensor([0, 1, 0, 1])  # tp=1 (idx1), tn=1 (idx0), fp=1 (idx2), fn=1 (idx3)
    m = valid_metrics("cola", logits, labels)
    assert m["accuracy"] == 0.5 and abs(m["mcc"]) < 1e-12
    perfect = valid_metrics("cola", torch.tensor([[1., 0.], [0., 1.]]), torch.tensor([0, 1]))
    assert perfect["mcc"] == 1.0


def test_stsb_pearson_and_mse():
    y = torch.tensor([0., 1., 2., 3.])
    assert abs(valid_metrics("stsb", (2 * y + 1)[:, None], y)["pearson"] - 1.0) < 1e-9
    assert heldout_dist("stsb", torch.zeros(3, 1), torch.ones(3, 1)) == (1.0, None)
