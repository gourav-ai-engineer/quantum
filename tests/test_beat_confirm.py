import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import beat_confirm_analysis as bca  # noqa: E402


def _fake_root(tmp_path, gain):
    """d2i_ri equals kprune plus `gain` extra fraction of correct examples; kwon is much worse."""
    n = 400
    labels = np.random.default_rng(0).integers(0, 2, n)

    def logits(acc, seed):
        pred = np.where(np.random.default_rng(seed).random(n) < acc, labels, 1 - labels)
        return np.eye(2)[pred] * 3.0

    for task in bca.CONF:
        (tmp_path / "d2i_ri").mkdir(exist_ok=True)
        for s in bca.SEEDS:
            kp = logits(0.80, s)
            fix = (np.random.default_rng(s + 1).random(n) < gain)[:, None]
            better = np.where(fix, np.eye(2)[labels] * 3.0, kp)
            json.dump({"valid_logits": better.tolist()}, open(tmp_path / "d2i_ri" / f"d2i_ri_{task}_seed{s}_c0.5.json", "w"))
            for sub, acc in ((f"kprune_out/kprune_{task}_seed{s}_c0.5", 0.80), (f"kwon_{task}_seed{s}_c0.5", 0.60)):
                d = tmp_path / sub
                d.mkdir(parents=True, exist_ok=True)
                torch.save({"logits": torch.tensor(logits(acc, s), dtype=torch.float32), "labels": torch.tensor(labels)},
                           d / "v15_student_validation.pt")


def _tests(root):
    return {"vs_best": {t: bca.compare(root, "d2i_ri", "kprune", t, 300, i) for i, t in enumerate(bca.CONF)},
            "vs_other": {t: bca.compare(root, "d2i_ri", "kwon", t, 300, 10 + i) for i, t in enumerate(bca.CONF)}}


def test_load_shapes_and_baseline_choice(tmp_path):
    _fake_root(tmp_path, 0.1)
    p, lab = bca.load(str(tmp_path), "kprune", "qnli")
    assert p.shape == (5, 400) and lab.shape == (400,)
    assert bca.load(str(tmp_path), "d2i_ri", "qnli")[0].shape == (5, 400)
    assert bca.better_baseline()[0] == "kprune"  # fixed from exploration data (baselines_v15.json)


def test_clear_gain_supported_and_no_gain_not(tmp_path):
    _fake_root(tmp_path, 0.15)
    wins, losses, verdict = bca.decide(_tests(str(tmp_path)))
    assert verdict == "supported" and len(wins) == 3 and not losses
    _fake_root(tmp_path, 0.0)  # d2i_ri == kprune predictions: nothing to win
    assert bca.decide(_tests(str(tmp_path)))[2] == "not_supported"
