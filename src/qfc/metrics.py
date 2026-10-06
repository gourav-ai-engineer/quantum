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


def auroc(labels, scores) -> float:
    """Rank-based AUROC (ties get average ranks). Returns nan if one class is absent."""
    y = np.asarray(labels).astype(int)
    s = np.asarray(scores, dtype=np.float64)
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    sorted_s = s[order]
    ranks = np.empty(len(s), dtype=np.float64)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def mcc_from_counts(tp, tn, fp, fn):
    """Matthews correlation from confusion counts (scalars or arrays); 0 where undefined."""
    tp, tn, fp, fn = (np.asarray(v, dtype=np.float64) for v in (tp, tn, fp, fn))
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return np.where(den > 0, (tp * tn - fp * fn) / np.where(den > 0, den, 1.0), 0.0)


def mcc_binary(labels, preds) -> float:
    y, p = np.asarray(labels).astype(int), np.asarray(preds).astype(int)
    return float(mcc_from_counts(((p == 1) & (y == 1)).sum(), ((p == 0) & (y == 0)).sum(),
                                 ((p == 1) & (y == 0)).sum(), ((p == 0) & (y == 1)).sum()))


def _grid_stats(preds: np.ndarray, labels: np.ndarray, metric: str) -> list[np.ndarray]:
    y = labels[None, :].astype(int)
    p = preds.astype(int)
    if metric == "accuracy":
        return [(p == y).astype(np.float64)]
    if metric == "mcc":
        return [((p == 1) & (y == 1)), ((p == 0) & (y == 0)), ((p == 1) & (y == 0)), ((p == 0) & (y == 1))]
    raise ValueError(f"unknown metric {metric!r}")


def _grid_metric(totals: list[np.ndarray], metric: str, cells: float):
    if metric == "accuracy":
        return totals[0] / cells
    return mcc_from_counts(*totals)


def cluster_bootstrap_delta(preds_a, preds_b, labels, metric: str = "accuracy",
                            n_boot: int = 10000, seed: int = 1234, chunk: int = 250) -> dict:
    """Paired cluster bootstrap over (calibration seed x example).

    ``preds_a`` and ``preds_b`` are [S, N] predicted classes of two methods on the SAME
    N examples and S calibration seeds; ``labels`` is [N]. Each resample draws S seeds
    and, independently, N examples with replacement, and computes metric(A) - metric(B)
    on that resampled grid (accuracy, or MCC from the pooled confusion counts). Both
    methods use the same resample, so the difference is paired. Returns the observed
    pooled delta, the percentile 95% CI and the bootstrap deltas.
    """
    a, b = np.asarray(preds_a), np.asarray(preds_b)
    labels = np.asarray(labels)
    if a.shape != b.shape or a.ndim != 2 or labels.shape != (a.shape[1],):
        raise ValueError("preds must be [S, N] with identical shapes and labels [N]")
    seeds, examples = a.shape
    xa = [x.astype(np.float64) for x in _grid_stats(a, labels, metric)]
    xb = [x.astype(np.float64) for x in _grid_stats(b, labels, metric)]
    cells = float(seeds * examples)
    observed = float(_grid_metric([x.sum() for x in xa], metric, cells)
                     - _grid_metric([x.sum() for x in xb], metric, cells))
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=np.float64)
    for start in range(0, n_boot, chunk):
        size = min(chunk, n_boot - start)
        w = rng.multinomial(seeds, np.full(seeds, 1.0 / seeds), size=size).astype(np.float64)
        v = rng.multinomial(examples, np.full(examples, 1.0 / examples), size=size).astype(np.float64)
        ta = [((w @ x) * v).sum(1) for x in xa]
        tb = [((w @ x) * v).sum(1) for x in xb]
        boot[start : start + size] = _grid_metric(ta, metric, cells) - _grid_metric(tb, metric, cells)
    return {"observed_delta": observed, "ci_low": float(np.quantile(boot, 0.025)),
            "ci_high": float(np.quantile(boot, 0.975)), "boot": boot}


def bootstrap_p_le(boot: np.ndarray, margin: float) -> float:
    """One-sided bootstrap p-value for H0: delta <= margin (small when the bootstrap sits above margin)."""
    return float((1 + int((boot <= margin).sum())) / (len(boot) + 1))


def bootstrap_p_ge(boot: np.ndarray, margin: float) -> float:
    """One-sided bootstrap p-value for H0: delta >= margin."""
    return float((1 + int((boot >= margin).sum())) / (len(boot) + 1))


def holm_adjust(pvalues) -> list[float]:
    """Holm step-down adjusted p-values (same order as the input)."""
    p = np.asarray(pvalues, dtype=np.float64)
    order = np.argsort(p)
    m = len(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted.tolist()
