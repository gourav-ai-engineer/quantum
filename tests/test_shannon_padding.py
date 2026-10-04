import importlib.util
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _PaddedAttentionModel:
    """Attention is exactly 0 at padded keys, like BERT's additive -inf mask."""

    def __init__(self, layers=2, heads=4, seed=0):
        self.layers, self.heads = layers, heads
        self.g = torch.Generator().manual_seed(seed)

    def eval(self):
        return self

    def __call__(self, input_ids, attention_mask, output_attentions=True, return_dict=True, **_):
        b, t = attention_mask.shape
        atts = []
        for _layer in range(self.layers):
            logits = torch.randn(b, self.heads, t, t, generator=self.g)
            logits = logits.masked_fill(attention_mask[:, None, None, :] == 0, float("-inf"))
            atts.append(torch.softmax(logits, dim=-1))
        return SimpleNamespace(attentions=tuple(atts))


def _loader():
    mask = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1], [1, 1, 0, 0, 0, 0]])
    return [{"input_ids": torch.zeros_like(mask), "attention_mask": mask, "labels": torch.zeros(3, dtype=torch.long)}]


@pytest.mark.parametrize(
    "name", ["step12_qfc_stability", "step13_mrpc_stability", "step14_budget_response", "step15_layer_adaptive_smoke"]
)
def test_shannon_scores_are_finite_with_padding(name):
    """Regression: p=0 at padded keys gave 0 * log(0) = NaN for every head."""
    scores = _load(name).shannon_scores(_PaddedAttentionModel(), _loader(), "cpu")
    assert len(scores) == 2
    for s in scores:
        assert torch.isfinite(s).all()
        assert s.std() > 0  # heads are distinguishable, so topk is meaningful


def test_no_script_uses_unclamped_log_of_masked_probabilities():
    pattern = re.compile(r"\*\s*(p|probs)\.log\(\)")
    offenders = [
        f"{path.name}:{i}"
        for path in sorted(SCRIPTS.glob("*.py"))
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if pattern.search(line)
    ]
    assert offenders == []
