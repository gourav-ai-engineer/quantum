import pytest
import torch

from qfc.baselines import (
    calibrated_selections,
    entropy_baseline_selections,
    evaluate_random_distribution,
    f1_binary,
    heads_kept_per_layer,
    random_selection,
    summarize,
    topk_selection,
    validate_selection,
)

LAYERS, HEADS = 3, 12
KS = (1, 3, 6, 9, 12)


def _inputs(seed=0):
    g = torch.Generator().manual_seed(seed)
    n = 5
    states = torch.rand(LAYERS, HEADS, 4, 4, generator=g).double()
    states = states @ states.transpose(-1, -2)
    states = states / states.diagonal(dim1=-2, dim2=-1).sum(-1)[..., None, None]
    kernels = torch.rand(LAYERS, n, HEADS, HEADS, generator=g).double().clamp(0.01, 1.0)
    kernels = 0.5 * (kernels + kernels.transpose(-1, -2))
    kernels = kernels.clone()
    kernels.diagonal(dim1=-2, dim2=-1).fill_(1.0)
    return (
        list(states),
        list(kernels),
        torch.rand(LAYERS, HEADS, generator=g),
        [torch.rand(HEADS, generator=g) for _ in range(LAYERS)],
        [torch.rand(HEADS, generator=g) for _ in range(LAYERS)],
    )


@pytest.mark.parametrize("k", KS)
def test_random_keeps_exactly_k_heads(k):
    """Regression: V10 Random kept ALL heads (no [:k]) == unpruned model."""
    for seed in range(5):
        sel = random_selection(LAYERS, HEADS, k, seed)
        validate_selection(sel, LAYERS, HEADS, k, name="Random")
        assert heads_kept_per_layer(sel) == [k] * LAYERS


@pytest.mark.parametrize("k", KS)
def test_every_baseline_keeps_exactly_k_heads(k):
    states, kernels, michel, vn, sh = _inputs()
    sel = calibrated_selections(states, kernels, michel, vn, sh, k)
    expected = {
        "MeanStateQFC", "ConditionalQFC", "MeanStateIWQFC", "ConditionalIWQFC",
        "MichelGate", "VonNeumann_keep_high", "VonNeumann_keep_low",
        "Shannon_keep_high", "Shannon_keep_low",
    }
    assert set(sel) == expected
    for name, selection in sel.items():
        assert heads_kept_per_layer(selection) == [k] * LAYERS, name
    sel["Random"] = random_selection(LAYERS, HEADS, k, 2027 + k)
    assert heads_kept_per_layer(sel["Random"]) == [k] * LAYERS


def test_random_seeds_give_different_masks():
    masks = {tuple(map(tuple, random_selection(LAYERS, HEADS, 6, s).values())) for s in range(30)}
    assert len(masks) > 1


def test_validate_selection_rejects_all_heads_and_duplicates():
    all_heads = {l: list(range(HEADS)) for l in range(LAYERS)}
    with pytest.raises(ValueError):
        validate_selection(all_heads, LAYERS, HEADS, 6)
    dup = {l: [0, 0, 1] for l in range(LAYERS)}
    with pytest.raises(ValueError):
        validate_selection(dup, LAYERS, HEADS, 3)


def test_entropy_both_directions_are_complementary_extremes():
    scores = [torch.tensor([0.1, 0.9, 0.5, 0.3, 0.7, 0.2])]
    sel = entropy_baseline_selections(scores, scores, 2)
    assert sel["VonNeumann_keep_high"][0] == [1, 4]
    assert sel["VonNeumann_keep_low"][0] == [0, 5]
    assert topk_selection(scores, 2, largest=False)[0] == [0, 5]


def test_summarize_and_f1():
    s = summarize([1.0, 2.0, 3.0])
    assert s["mean"] == 2.0 and s["min"] == 1.0 and s["max"] == 3.0 and s["n"] == 3
    assert abs(s["std"] - (2 / 3) ** 0.5) < 1e-12
    assert f1_binary(torch.tensor([1, 1, 0, 0]), torch.tensor([1, 0, 1, 0])) == 0.5


def test_random_distribution_helper_uses_distinct_seeds_and_k_heads():
    seen = []

    def eval_fn(mask):
        assert mask.shape == (LAYERS, HEADS)
        assert torch.all(mask.sum(-1) == 4)
        seen.append(mask.clone())
        n = 8
        acc = float(mask[0, 0])
        return torch.full((n,), 0.5), torch.full((n,), acc), torch.zeros(n, dtype=torch.long), torch.zeros(n, dtype=torch.long)

    out = evaluate_random_distribution(
        eval_fn, LAYERS, HEADS, 4, n_masks=30, base_seed=100, device="cpu",
        with_f1=True, baseline={"accuracy": 1.0, "loss": 0.5, "f1": 0.0},
    )
    assert out["n_masks"] == 30 and len(out["draws"]) == 30
    assert [d["seed"] for d in out["draws"]] == list(range(100, 130))
    assert len({tuple(m.flatten().tolist()) for m in seen}) > 1
    for key in ("accuracy", "loss", "f1"):
        assert set(out["distribution"][key]) == {"mean", "std", "min", "max", "n"}
    assert out["accuracy"] == out["distribution"]["accuracy"]["mean"]
    assert out["accuracy_delta_mean"] == out["accuracy"] - 1.0
