"""Synthetic check of scripts/e7_highcomp_analysis.py: PASS when GRP-RI clearly wins everywhere, FAIL otherwise, gate raises."""
import json
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import e7_highcomp_analysis as e7  # noqa: E402

N = 100


def _logits(task, y, n_wrong):
    if task == "stsb":  # Pearson falls as noise grows
        g = torch.Generator().manual_seed(n_wrong)
        return (y.double() + 0.05 * n_wrong * torch.randn(N, generator=g, dtype=torch.float64)).unsqueeze(1)
    pred = y.clone()
    pred[:n_wrong] = 1 - pred[:n_wrong]
    return torch.nn.functional.one_hot(pred, 2).double()


def _write(root, task, grp_wrong, kp_wrong=20):
    y = torch.rand(N, dtype=torch.float64) * 5 if task == "stsb" else (torch.arange(N) % 2)
    masks = {}
    for c in e7.BUDGETS:
        for s in e7.SEEDS:
            p = e7.paths(root, task, s, c)
            for name, wrong in (("kprune", kp_wrong), ("kwon", 40)):
                os.makedirs(p[name], exist_ok=True)
                torch.save({"logits": _logits(task, y, wrong), "labels": y}, os.path.join(p[name], "v15_student_validation.pt"))
            torch.save({"logits": _logits(task, y, 0), "labels": y}, os.path.join(p["kprune"], "v15_teacher_validation.pt"))
            hm, nm = np.zeros((12, 12), int), np.zeros((12, 3072), int)
            hm[:, :2], nm[:, :300] = 1, 1
            masks[f"kprune_{task}_seed{s}_c{c}__head"], masks[f"kprune_{task}_seed{s}_c{c}__neuron"] = hm, nm
            os.makedirs(os.path.dirname(p["grp_ri"]), exist_ok=True)
            for m, wrong in (("grp_ri", grp_wrong), ("d2i_ri", kp_wrong)):
                lg = _logits(task, y, wrong)
                json.dump({"method": m, "evaluation_size": N, "valid_logits": lg.tolist(), "mac_ratio": c - 0.01,
                           "valid": e7.valid_metrics(task, lg, y), "unpruned_valid": e7.valid_metrics(task, _logits(task, y, 0), y),
                           "heads_kept": {str(l): [0, 1] for l in range(12)},
                           "neurons_kept": {str(l): list(range(300)) for l in range(12)}}, open(p[m], "w"))
    np.savez(os.path.join(root, "bl", "kprune_masks.npz"), **masks)


def _roots(tmp_path, grp_wrong):
    roots = []
    for t in e7.TASKS:
        r = str(tmp_path / t)
        _write(r, t, grp_wrong)
        roots.append(r)
    return roots


def test_pass_and_fail(tmp_path):
    assert e7.analyse(_roots(tmp_path / "a", grp_wrong=10))["verdict"] == "PASS"
    assert e7.analyse(_roots(tmp_path / "b", grp_wrong=20))["verdict"] == "FAIL"


def test_gate_rejects_mask_mismatch(tmp_path):
    roots = _roots(tmp_path, grp_wrong=10)
    p = e7.paths(roots[0], "mrpc", 7, 0.3)["d2i_ri"]
    r = json.load(open(p))
    r["heads_kept"]["0"] = [5]
    json.dump(r, open(p, "w"))
    with pytest.raises(e7.PipelineMismatch):
        e7.analyse(roots)
