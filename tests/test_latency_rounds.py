"""scripts/v15_latency.measure: shuffled rounds (drift control). No model is built: the timer is mocked."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from v15_latency import measure  # noqa: E402

SHAPES = {"a": ([1], [10]), "b": ([2], [20]), "c": ([3], [30])}


def fake_timer(calls):
    def t(h, n, b):
        calls.append((h[0], b))
        return float(h[0] * 10 + len(calls)), 100 * h[0]  # drifts upward with call count
    return t


def test_default_is_original_order_once():
    calls = []
    out, orders = measure(SHAPES, [1, 32], timer=fake_timer(calls))
    assert orders == [["a", "b", "c"]]
    assert calls == [(1, 1), (1, 32), (2, 1), (2, 32), (3, 1), (3, 32)]
    assert out["a"]["median_ms_b1"] == 11.0 and "round_medians_ms_b1" not in out["a"]
    assert out["c"]["params_encoder"] == 300


def test_rounds_shuffle_reproducible_and_aggregate():
    calls = []
    out, orders = measure(SHAPES, [1], rounds=5, order_seed=3, timer=fake_timer(calls))
    assert len(orders) == 5 and all(sorted(o) == ["a", "b", "c"] for o in orders)
    assert len(set(map(tuple, orders))) > 1  # order actually changes between rounds
    _, again = measure(SHAPES, [1], rounds=5, order_seed=3, timer=fake_timer([]))
    assert again == orders
    for n in SHAPES:
        r = out[n]["round_medians_ms_b1"]
        assert len(r) == 5
        assert out[n]["median_ms_b1"] == sorted(r)[2]
        assert out[n]["max_over_min_b1"] == max(r) / min(r)
