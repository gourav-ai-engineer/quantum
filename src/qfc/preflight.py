"""Environment pre-flight checks that must pass before any experiment is trusted.

The key failure mode: transformers 5.x silently ignores ``head_mask``, so every
masked/pruned evaluation equals the unpruned model and all correlations and
comparisons are vacuous. ``check_head_mask_effective`` detects that on a tiny
randomly initialised BERT (no downloads, runs on CPU in about a second).
"""

from __future__ import annotations

import platform
import sys
from typing import Callable

import torch

EXPECTED_VERSIONS = {
    "torch": "2.6.0",
    "transformers": "4.51.3",
    "datasets": "3.6.0",
    "pyarrow": "24.0.0",
}


class PreflightError(RuntimeError):
    pass


def _version(module_name: str) -> str | None:
    try:
        module = __import__(module_name)
    except Exception:  # noqa: BLE001 - report any import failure as "missing"
        return None
    return getattr(module, "__version__", None)


def installed_versions() -> dict[str, str | None]:
    return {
        name: _version(name)
        for name in ("torch", "transformers", "datasets", "pyarrow", "numpy", "accelerate")
    }


def check_versions(
    expected: dict[str, str] = EXPECTED_VERSIONS,
    installed: dict[str, str | None] | None = None,
) -> list[str]:
    """Return a list of problems (empty = OK). A local suffix such as +cu124 is ignored."""
    installed = installed_versions() if installed is None else installed
    problems = []
    for name, want in expected.items():
        have = installed.get(name)
        if have is None:
            problems.append(f"{name} is not importable (expected =={want})")
        elif have.split("+")[0] != want:
            problems.append(f"{name}=={have}, expected =={want}")
    return problems


def check_cuda(allow_cpu: bool = False) -> str:
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    if allow_cpu:
        return "cpu (allowed for smoke tests only)"
    raise PreflightError("torch.cuda.is_available() is False; full runs require a GPU")


def _tiny_bert_forward() -> Callable[[torch.Tensor], torch.Tensor]:
    """Return forward(head_mask) -> loss for a tiny random BERT (2 layers, 4 heads)."""
    from transformers import BertConfig, BertForSequenceClassification

    torch.manual_seed(0)
    config = BertConfig(
        vocab_size=100,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=16,
        num_labels=2,
    )
    try:
        model = BertForSequenceClassification._from_config(config, attn_implementation="eager")
    except TypeError:
        model = BertForSequenceClassification(config)
    model.eval()
    ids = torch.randint(0, 100, (4, 8))
    labels = torch.tensor([0, 1, 0, 1])

    def forward(head_mask: torch.Tensor) -> torch.Tensor:
        out = model(input_ids=ids, labels=labels, head_mask=head_mask, return_dict=True)
        return out.loss

    return forward


def check_head_mask_effective(
    forward: Callable[[torch.Tensor], torch.Tensor] | None = None,
    layers: int = 2,
    heads: int = 4,
    tol: float = 1e-6,
) -> dict[str, float]:
    """Raise unless masking all heads changes the loss AND gradients reach head_mask."""
    forward = _tiny_bert_forward() if forward is None else forward
    ones = torch.ones(layers, heads)
    zeros = torch.zeros(layers, heads)
    with torch.no_grad():
        loss_ones = float(forward(ones))
        loss_zeros = float(forward(zeros))
    diff = abs(loss_ones - loss_zeros)
    if diff < tol:
        raise PreflightError(
            f"head_mask has no effect (loss all-ones={loss_ones:.8f}, all-zeros="
            f"{loss_zeros:.8f}). This transformers version ignores head_mask; "
            "install the pinned versions from requirements-ci.txt."
        )
    gate = torch.ones(layers, heads, requires_grad=True)
    forward(gate).backward()
    if gate.grad is None or float(gate.grad.abs().sum()) == 0.0:
        raise PreflightError(
            "no gradient reaches head_mask; Michel gate sensitivity would be vacuous"
        )
    return {"loss_all_ones": loss_ones, "loss_all_zeros": loss_zeros, "abs_diff": diff}


def collect_environment() -> dict:
    cuda = torch.cuda.is_available()
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "versions": installed_versions(),
        "cuda_available": cuda,
        "gpu_name": torch.cuda.get_device_name(0) if cuda else None,
    }
