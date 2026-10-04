from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def evaluate_per_example(model, loader, device: str, head_mask=None):
    model.eval()
    losses = []
    correct = []

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch.pop("labels")
        out = model(
            **batch,
            labels=labels,
            head_mask=head_mask,
            return_dict=True,
        )

        logits = out.logits.float()
        example_loss = torch.nn.functional.cross_entropy(
            logits,
            labels,
            reduction="none",
        )
        pred_correct = (logits.argmax(dim=-1) == labels).to(torch.float32)
        losses.append(example_loss.cpu())
        correct.append(pred_correct.cpu())

    if not losses:
        raise ValueError("empty evaluation loader")

    return torch.cat(losses), torch.cat(correct)


def mean_and_bootstrap_ci(
    values: np.ndarray | torch.Tensor,
    n_boot: int = 2000,
    seed: int = 1234,
    confidence: float = 0.95,
):
    """Return mean and percentile bootstrap CI for iid per-example values."""
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("values must be a non-empty 1-D array")

    rng = np.random.default_rng(seed)
    index = rng.integers(0, len(values), size=(n_boot, len(values)))
    boot = values[index].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return {
        "mean": float(values.mean()),
        "ci_low": float(np.quantile(boot, alpha)),
        "ci_high": float(np.quantile(boot, 1.0 - alpha)),
    }


def paired_bootstrap_delta(
    values_a: np.ndarray | torch.Tensor,
    values_b: np.ndarray | torch.Tensor,
    n_boot: int = 2000,
    seed: int = 1234,
    confidence: float = 0.95,
):
    """Bootstrap CI for mean(values_a - values_b) on paired examples."""
    if isinstance(values_a, torch.Tensor):
        values_a = values_a.detach().cpu().numpy()
    if isinstance(values_b, torch.Tensor):
        values_b = values_b.detach().cpu().numpy()

    a = np.asarray(values_a, dtype=np.float64)
    b = np.asarray(values_b, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("paired arrays must be 1-D with identical shapes")

    delta = a - b
    rng = np.random.default_rng(seed)
    index = rng.integers(0, len(delta), size=(n_boot, len(delta)))
    boot = delta[index].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return {
        "mean_delta": float(delta.mean()),
        "ci_low": float(np.quantile(boot, alpha)),
        "ci_high": float(np.quantile(boot, 1.0 - alpha)),
    }
