import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import beat_confirm_analysis as bca  # noqa: E402

N = 400
KEPT_H, KEPT_N = {"0": [0, 2], "1": [1]}, {"0": [0, 3], "1": [1, 2]}


def _fake_roots(tmp_path, gain):
    """Kernel layout. d2i_ri = kprune predictions plus `gain` extra fraction correct; kwon is much worse."""
    labels = np.random.default_rng(0).integers(0, 2, N)

    def logits(acc, seed):
        pred = np.where(np.random.default_rng(seed).random(N) < acc, labels, 1 - labels)
        return np.eye(2)[pred] * 3.0

    def dump(d, lg):
        d.mkdir(parents=True, exist_ok=True)
        for which in ("student", "teacher"):
            torch.save({"logits": torch.tensor(lg if which == "student" else logits(0.9, 99), dtype=torch.float32),
                        "labels": torch.tensor(labels)}, d / f"v15_{which}_validation.pt")

    roots, masks = [], {}
    for task in bca.CONF:
        root = tmp_path / f"conf_{task}"
        (root / "beat").mkdir(parents=True, exist_ok=True)
        roots.append(str(root))
        for s in bca.SEEDS:
            kp = logits(0.80, s)
            fix = (np.random.default_rng(s + 1).random(N) < gain)[:, None]
            better = np.where(fix, np.eye(2)[labels] * 3.0, kp)
            acc = float((better.argmax(-1) == labels).mean())
            t_acc = float((logits(0.9, 99).argmax(-1) == labels).mean())
            json.dump({"valid_logits": better.tolist(), "evaluation_size": N, "valid": {"accuracy": acc},
                       "unpruned_valid": {"accuracy": t_acc}, "heads_kept": KEPT_H, "neurons_kept": KEPT_N},
                      open(root / "beat" / f"d2i_ri_{task}_seed{s}_c0.5.json", "w"))
            dump(root / "bl" / "kprune_out" / f"kprune_{task}_seed{s}_c0.5", kp)
            dump(root / "bl" / f"kwon_{task}_seed{s}_c0.5", logits(0.60, s))
            hm, nm = np.zeros((2, 3), np.float32), np.zeros((2, 4), np.float32)
            for l, v in KEPT_H.items():
                hm[int(l), v] = 1
            for l, v in KEPT_N.items():
                nm[int(l), v] = 1
            masks[f"kprune_{task}_seed{s}_c0.5__head"], masks[f"kprune_{task}_seed{s}_c0.5__neuron"] = hm, nm
        np.savez(root / "bl" / "kprune_masks.npz", **{k: v for k, v in masks.items() if f"_{task}_" in k})
    return roots


def _tests(roots):
    root_of = bca.find_roots(roots)
    return {"vs_best": {t: bca.compare(root_of[t], "d2i_ri", "kprune", t, 300, i) for i, t in enumerate(bca.CONF)},
            "vs_other": {t: bca.compare(root_of[t], "d2i_ri", "kwon", t, 300, 10 + i) for i, t in enumerate(bca.CONF)}}


def test_layout_integrity_and_baseline_choice(tmp_path):
    roots = _fake_roots(tmp_path, 0.1)
    root_of = bca.find_roots(roots)
    for t in bca.CONF:
        for s in bca.SEEDS:
            bca.check_integrity(root_of[t], t, s)
    p, lab = bca.load(root_of["qnli"], "kprune", "qnli")
    assert p.shape == (5, N) and lab.shape == (N,)
    assert bca.better_baseline()[0] == "kprune"  # fixed from exploration data (baselines_v15.json)


def test_shuffled_rows_and_mask_mismatch_abort(tmp_path):
    roots = _fake_roots(tmp_path, 0.1)
    f = Path(roots[0]) / "beat" / "d2i_ri_qnli_seed7_c0.5.json"
    r = json.load(open(f))
    r["valid_logits"] = r["valid_logits"][::-1]  # row order differs from the labels of the official dump
    json.dump(r, open(f, "w"))
    with pytest.raises(bca.PipelineMismatch, match="row order"):
        bca.check_integrity(roots[0], "qnli", 7)
    f = Path(roots[1]) / "beat" / "d2i_ri_mnli_seed7_c0.5.json"
    r = json.load(open(f))
    r["heads_kept"] = {"0": [0, 1], "1": [1]}
    json.dump(r, open(f, "w"))
    with pytest.raises(bca.PipelineMismatch, match="K-prune masks"):
        bca.check_integrity(roots[1], "mnli", 7)


def test_clear_gain_supported_and_no_gain_not(tmp_path):
    wins, losses, verdict = bca.decide(_tests(_fake_roots(tmp_path, 0.15)))
    assert verdict == "supported" and len(wins) == 3 and not losses
    assert bca.decide(_tests(_fake_roots(tmp_path, 0.0)))[2] == "not_supported"
