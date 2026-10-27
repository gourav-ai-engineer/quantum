import copy
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_grp import _batches, _logits, _tiny_bert  # noqa: E402

from qfc.align import sequential_refit_io  # noqa: E402
from qfc.grp import prune_model, sequential_refit  # noqa: E402

HEADS, NEURONS = {0: [1, 3], 1: [0, 1, 2]}, {0: list(range(0, 64, 2)), 1: list(range(40))}


@torch.no_grad()
def test_no_align_equals_drift_corrected_refit():
    t, bs = _tiny_bert(), _batches()
    a = prune_model(copy.deepcopy(t), HEADS, NEURONS)
    b = prune_model(copy.deepcopy(t), HEADS, NEURONS)
    sequential_refit(a, t, bs, "cpu", 1e-4, True)
    sequential_refit_io(b, t, bs, "cpu", HEADS, NEURONS, 1e-4, qkv=False, ffn_in=False)
    assert torch.allclose(_logits(a, bs), _logits(b, bs), atol=1e-5)


@torch.no_grad()
def test_unpruned_student_is_recovered():
    t, bs = _tiny_bert(), _batches()
    heads, neurons = {0: [0, 1, 2, 3], 1: [0, 1, 2, 3]}, {0: list(range(64)), 1: list(range(64))}
    s = prune_model(copy.deepcopy(t), heads, neurons)
    sequential_refit_io(s, t, bs, "cpu", heads, neurons, 1e-6)
    assert torch.allclose(_logits(s, bs), _logits(t, bs), atol=1e-3)


@torch.no_grad()
def test_align_runs_with_empty_head_layer_and_is_finite():
    t, bs = _tiny_bert(), _batches()
    heads, neurons = {0: [], 1: [2]}, {0: [5, 9, 11], 1: list(range(64))}
    s = prune_model(copy.deepcopy(t), heads, neurons)
    sequential_refit_io(s, t, bs, "cpu", heads, neurons)
    assert torch.isfinite(_logits(s, bs)).all()


def test_sensitivity_regression_head_is_finite_and_positive():
    import math

    from transformers import BertConfig, BertForSequenceClassification

    from qfc.grp import module_sensitivity
    cfg = BertConfig(vocab_size=50, hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
                     intermediate_size=64, max_position_embeddings=32, num_labels=1)
    cfg._attn_implementation = "eager"
    torch.manual_seed(0)
    s = module_sensitivity(BertForSequenceClassification(cfg).eval(), _batches(), "cpu", seed=0)
    assert len(s) == 4 and all(math.isfinite(v) and v > 0 for v in s.values())
