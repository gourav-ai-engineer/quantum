import pytest
import torch

from qfc.preflight import (
    EXPECTED_VERSIONS,
    PreflightError,
    check_cuda,
    check_head_mask_effective,
    check_versions,
)

GOOD = {**EXPECTED_VERSIONS, "numpy": "2.0.0", "accelerate": "1.0.0"}


def test_expected_versions_are_the_documented_pins():
    assert EXPECTED_VERSIONS == {
        "torch": "2.6.0",
        "transformers": "4.51.3",
        "datasets": "3.6.0",
        "pyarrow": "24.0.0",
    }


def test_check_versions_accepts_pins_and_ignores_local_suffix():
    assert check_versions(installed=GOOD) == []
    assert check_versions(installed={**GOOD, "torch": "2.6.0+cu124"}) == []


def test_check_versions_rejects_transformers_5_and_missing():
    problems = check_versions(installed={**GOOD, "transformers": "5.18.0"})
    assert len(problems) == 1 and "transformers==5.18.0" in problems[0]
    problems = check_versions(installed={**GOOD, "pyarrow": None})
    assert len(problems) == 1 and "pyarrow" in problems[0]


def test_check_cuda_requires_gpu_unless_allowed(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(PreflightError):
        check_cuda(allow_cpu=False)
    assert "cpu" in check_cuda(allow_cpu=True)


def test_head_mask_check_detects_ignored_mask():
    """The transformers-5.x failure mode: forward does not depend on head_mask."""
    param = torch.tensor(0.7)

    def ignores_mask(head_mask):
        return param * 1.0

    with pytest.raises(PreflightError, match="no effect"):
        check_head_mask_effective(ignores_mask, layers=2, heads=4)


def test_head_mask_check_accepts_a_model_that_uses_the_mask():
    w = torch.linspace(0.1, 0.8, 8).reshape(2, 4)

    def uses_mask(head_mask):
        return (head_mask * w).sum()

    out = check_head_mask_effective(uses_mask, layers=2, heads=4)
    assert out["abs_diff"] > 0


def test_head_mask_check_detects_missing_gradient():
    def no_grad_path(head_mask):
        return head_mask.detach().sum() * torch.tensor(1.0, requires_grad=True)

    with pytest.raises(PreflightError, match="gradient"):
        check_head_mask_effective(no_grad_path, layers=2, heads=4)


def test_real_tiny_bert_with_pinned_transformers():
    if check_versions(expected={"transformers": EXPECTED_VERSIONS["transformers"]}):
        pytest.skip("pinned transformers not installed; head_mask is unreliable here")
    out = check_head_mask_effective()
    assert out["abs_diff"] > 1e-6
