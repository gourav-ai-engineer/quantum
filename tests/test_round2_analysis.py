import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "round2_analysis.py"
spec = importlib.util.spec_from_file_location("round2_analysis", SCRIPT)
r2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r2)


def _r(delta, low, high, p_le=0.001, p_ge=1.0):
    return {"observed_delta": delta, "ci_low": low, "ci_high": high,
            "p_adj_le_margin": p_le, "p_adj_ge_zero": p_ge}


def test_h1_rule():
    ok = {t: _r(0.0, -0.005, 0.005) for t in r2.ALL_TASKS}
    assert r2.h1_verdict(ok)["verdict"] == "supported"
    one_replication_failure = dict(ok, mrpc=_r(-0.03, -0.05, -0.01, p_le=0.9))
    assert r2.h1_verdict(one_replication_failure)["verdict"] == "supported"  # at most one failure allowed
    two_failures = dict(one_replication_failure, sst2=_r(-0.03, -0.05, -0.01, p_le=0.9))
    assert r2.h1_verdict(two_failures)["verdict"] == "not_supported"
    confirmatory_failure = dict(ok, qnli=_r(-0.03, -0.05, -0.01, p_le=0.9))
    assert r2.h1_verdict(confirmatory_failure)["verdict"] == "not_supported"  # all 3 confirmatory must pass
    ci_ok_but_holm_fails = dict(ok, rte=_r(0.0, -0.005, 0.005, p_le=0.2))
    assert r2.h1_verdict(ci_ok_but_holm_fails)["verdict"] == "not_supported"  # both criteria must agree


def test_h2_outcomes_are_recorded_not_exclusive():
    better = {t: _r(0.10, 0.05, 0.15) for t in r2.ALL_TASKS}
    assert r2.h2_verdict(better)["outcomes"] == ["VNE informative"]
    close = {t: _r(0.005, -0.02, 0.03, p_le=0.4) for t in r2.ALL_TASKS}
    assert r2.h2_verdict(close)["outcomes"] == ["VNE adds nothing beyond the heuristic"]
    dm_wins = {t: _r(-0.08, -0.12, -0.04, p_le=1.0) for t in r2.ALL_TASKS}
    assert any(o.startswith("DelimiterMass_keep_low beats VNE") for o in r2.h2_verdict(dm_wins)["outcomes"])
    mixed = {t: _r(0.05, -0.01, 0.1, p_le=0.3) for t in r2.ALL_TASKS}
    assert r2.h2_verdict(mixed)["outcomes"] == ["inconclusive"]


def test_h6_rule():
    pos = {t: _r(0.02, 0.01, 0.03, p_le=0.01, p_ge=1.0) for t in ("sst2", "mrpc", "rte")}
    neutral = {t: _r(0.0, -0.01, 0.01, p_le=0.6, p_ge=0.6) for t in ("qnli", "cola")}
    assert r2.h6_verdict({**pos, **neutral})["verdict"] == "global_better"
    worse = dict(neutral, cola=_r(-0.03, -0.05, -0.01, p_le=1.0, p_ge=0.01))
    assert r2.h6_verdict({**pos, **worse})["verdict"] == "not_better"  # one significant negative blocks it
    assert r2.h6_verdict({t: neutral["qnli"] for t in r2.ALL_TASKS})["verdict"] == "not_better"


def _flip(rng, labels, acc):
    return np.where(rng.random(len(labels)) < acc, labels, 1 - labels).astype(np.int8)


def test_end_to_end_synthetic(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    for task in r2.ALL_TASKS:
        labels = rng.integers(0, 2, 300)
        for s in range(5):
            michel = _flip(rng, labels, 0.85)
            np.savez_compressed(
                tmp_path / f"round2_{task}_seed{s}.npz", labels=labels.astype(np.int8),
                VonNeumann_keep_high__pred=michel, MichelGate__pred=michel,
                MichelGlobal__pred=michel, MichelGlobal_min1__pred=michel,
                DelimiterMass_keep_low__pred=_flip(rng, labels, 0.60))
    monkeypatch.setattr(sys, "argv", ["round2_analysis", str(tmp_path), "--n-boot", "300"])
    r2.main()
    out = json.loads((tmp_path / "round2_verdicts.json").read_text())
    assert out["complete_protocol"] is True
    assert out["metrics"]["cola"] == "mcc" and out["metrics"]["sst2"] == "accuracy"
    assert out["verdicts"]["H1"]["verdict"] == "supported"
    assert out["verdicts"]["H2"]["outcomes"] == ["VNE informative"]
    assert out["verdicts"]["H6"]["verdict"] == "not_better"


def test_incomplete_protocol_gives_no_verdict(tmp_path, monkeypatch):
    rng = np.random.default_rng(1)
    labels = rng.integers(0, 2, 100)
    for s in range(2):
        arrays = {f"{m}__pred": _flip(rng, labels, 0.8) for m in (
            "VonNeumann_keep_high", "MichelGate", "DelimiterMass_keep_low", "MichelGlobal", "MichelGlobal_min1")}
        np.savez_compressed(tmp_path / f"round2_rte_seed{s}.npz", labels=labels.astype(np.int8), **arrays)
    monkeypatch.setattr(sys, "argv", ["round2_analysis", str(tmp_path), "--n-boot", "100"])
    r2.main()
    out = json.loads((tmp_path / "round2_verdicts.json").read_text())
    assert out["complete_protocol"] is False
    assert "insufficient_data" in out["verdicts"]["all"]
