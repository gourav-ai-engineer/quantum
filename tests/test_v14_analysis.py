import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "v14_analysis.py"
spec = importlib.util.spec_from_file_location("v14_analysis", SCRIPT)
v14 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v14)


def _flip(rng, labels, acc):
    return np.where(rng.random(len(labels)) < acc, labels, 1 - labels).astype(np.int8)


def test_end_to_end_synthetic(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    for task in v14.ALL_TASKS:
        labels = rng.integers(0, 2, 400).astype(np.int8)
        for s in range(5):
            arrays = {"labels": labels}
            for sel in v14.SELECTORS:
                arrays[f"{sel}|none__pred"] = _flip(rng, labels, 0.60)
                arrays[f"{sel}|ls_seq__pred"] = _flip(rng, labels, 0.95 if sel == "bcm" else 0.80)
            arrays["bcm|merge__pred"] = _flip(rng, labels, 0.75)
            for r in range(3):
                arrays[f"bcm|merge_randtarget{r}__pred"] = _flip(rng, labels, 0.75)
            np.savez_compressed(tmp_path / f"v14_{task}_seed{s}.npz", **arrays)
    monkeypatch.setattr(sys, "argv", ["v14_analysis", str(tmp_path), "--n-boot", "300"])
    v14.main()
    out = json.loads((tmp_path / "v14_verdicts.json").read_text())
    assert out["complete_protocol"]
    assert all(v == "pass" for v in out["verdicts"]["H9a"].values())
    assert out["verdicts"]["H9b"]["verdict"] == "supported"
    assert out["verdicts"]["H9c"]["verdict"] == "supported"
    assert out["verdicts"]["H9d"]["verdict"] == "not_supported"  # same accuracy as random targets


def test_incomplete_gives_no_verdict(tmp_path, monkeypatch):
    rng = np.random.default_rng(1)
    labels = rng.integers(0, 2, 50).astype(np.int8)
    arrays = {"labels": labels}
    for sel in v14.SELECTORS:
        arrays[f"{sel}|none__pred"] = arrays[f"{sel}|ls_seq__pred"] = labels
    arrays["bcm|merge__pred"] = labels
    np.savez_compressed(tmp_path / "v14_mrpc_seed7.npz", **arrays)
    monkeypatch.setattr(sys, "argv", ["v14_analysis", str(tmp_path), "--n-boot", "50"])
    v14.main()
    assert not json.loads((tmp_path / "v14_verdicts.json").read_text())["complete_protocol"]
