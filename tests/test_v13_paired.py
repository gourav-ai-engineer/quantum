import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "v13_paired_test.py"
spec = importlib.util.spec_from_file_location("v13_paired_test", SCRIPT)
v13 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v13)


def _seeds(*lows):
    return [{"ci_low": x} for x in lows]


def test_verdict_rule():
    assert v13.v13_verdict({"sst2": _seeds(0.01, 0.02, -0.01), "mrpc": _seeds(0.01, 0.01, 0.0)}) == "supported"
    assert v13.v13_verdict({"sst2": _seeds(0.0, -0.01, 0.01), "mrpc": _seeds(-0.02, -0.01, 0.0)}) == "not_supported"
    assert v13.v13_verdict({"sst2": _seeds(0.01, 0.02, 0.03), "mrpc": _seeds(-0.02, -0.01, 0.0)}) == "inconclusive"
    assert v13.v13_verdict({"sst2": _seeds(0.1), "mrpc": _seeds(0.1)}) == "insufficient_data"
    assert v13.v13_verdict({"sst2": _seeds(0.1, 0.1, 0.1)}) == "insufficient_data"
    with pytest.raises(ValueError):
        v13.v13_verdict({})


def test_end_to_end(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    for task in ("sst2", "mrpc"):
        for seed in (7, 42, 77):
            b = rng.random(400) < 0.7
            a = b | (rng.random(400) < 0.5)  # A is right whenever B is, plus more: clear positive delta
            np.savez_compressed(tmp_path / f"{task}_seed{seed}_correct.npz", MeanStateIWQFC=a, MichelGate=b)
    monkeypatch.setattr(sys, "argv", ["v13", str(tmp_path), "--bootstrap", "500"])
    v13.main()
    out = json.loads((tmp_path / "v13_paired.json").read_text())
    assert out["verdict"] == "supported"
    assert sorted(out["tasks"]) == ["mrpc", "sst2"] and all(len(v) == 3 for v in out["tasks"].values())
