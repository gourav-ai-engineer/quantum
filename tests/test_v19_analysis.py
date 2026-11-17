import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import v19_analysis as va  # noqa: E402

from qfc.metrics import cluster_bootstrap_delta  # noqa: E402

N = 400
KEPT_H, KEPT_N = {"0": [0, 2], "1": [1]}, {"0": [0, 3], "1": [1, 2]}


def _fake_roots(tmp_path, gain, kwon_gap=0.2):
    """Kernel layout for the 7 V19 tasks. d2i_ri = K-prune predictions improved by `gain` (fraction of rows fixed for
    classification; shrinkage of the error for STS-B; +gain for F1); Kwon is `kwon_gap` worse."""
    rng = np.random.default_rng(0)
    roots = []
    for task in va.TASKS:
        root = tmp_path / f"conf_{task}"
        (root / "beat").mkdir(parents=True, exist_ok=True)
        roots.append(str(root))
        labels = rng.integers(0, 2, N)
        y = rng.normal(size=N)  # STS-B gold
        masks = {}
        for s in va.SEEDS:
            g = np.random.default_rng(s)
            if task == "squad":
                kp = np.clip(g.random(N) * 0.4 + 0.5, 0, 1)
                d2 = np.clip(kp + gain, 0, 1)
                kw = np.clip(kp - kwon_gap, 0, 1)
                dump = {"kp": {"f1": torch.tensor(kp)}, "kw": {"f1": torch.tensor(kw)},
                        "t": {"f1": torch.tensor(np.full(N, 0.9))}}
                rec = {"valid_f1": d2.tolist(), "valid": {"f1": float(d2.mean())}, "unpruned_valid": {"f1": 0.9}}
            elif task == "stsb":
                kp = y + g.normal(size=N) * 0.5
                d2 = y + (kp - y) * (1 - gain)
                kw = y + (kp - y) * (1 + kwon_gap * 4)
                lab = torch.tensor(y)
                tl = y + 0.1 * g.normal(size=N)
                dump = {"kp": {"logits": torch.tensor(kp)[:, None], "labels": lab},
                        "kw": {"logits": torch.tensor(kw)[:, None], "labels": lab},
                        "t": {"logits": torch.tensor(tl)[:, None], "labels": lab}}
                rec = {"valid_logits": torch.tensor(d2)[:, None].tolist(),
                       "valid": {"pearson": float(np.corrcoef(d2, y)[0, 1])},
                       "unpruned_valid": {"pearson": float(np.corrcoef(tl, y)[0, 1])}}
            else:
                def lg(acc):
                    pred = np.where(g.random(N) < acc, labels, 1 - labels)
                    return np.eye(2)[pred] * 3.0
                kp = lg(0.80)
                fix = (g.random(N) < gain)[:, None]
                d2 = np.where(fix, np.eye(2)[labels] * 3.0, kp)
                tl = lg(0.9)
                lab = torch.tensor(labels)
                dump = {"kp": {"logits": torch.tensor(kp, dtype=torch.float32), "labels": lab},
                        "kw": {"logits": torch.tensor(lg(0.80 - kwon_gap), dtype=torch.float32), "labels": lab},
                        "t": {"logits": torch.tensor(tl, dtype=torch.float32), "labels": lab}}
                rec = {"valid_logits": d2.tolist(), "valid": {"accuracy": float((d2.argmax(-1) == labels).mean())},
                       "unpruned_valid": {"accuracy": float((tl.argmax(-1) == labels).mean())}}
            rec.update(evaluation_size=N, heads_kept=KEPT_H, neurons_kept=KEPT_N)
            json.dump(rec, open(root / "beat" / f"d2i_ri_{task}_seed{s}_c0.5.json", "w"))
            for sub, key in (("bl/kprune_out/kprune_%s_seed%d_c0.5", "kp"), ("bl/kwon_%s_seed%d_c0.5", "kw")):
                d = root / (sub % (task, s))
                d.mkdir(parents=True, exist_ok=True)
                torch.save(dump[key], d / "v15_student_validation.pt")
                if key == "kp":
                    torch.save(dump["t"], d / "v15_teacher_validation.pt")
            hm, nm = np.zeros((2, 3), np.float32), np.zeros((2, 4), np.float32)
            for l, v in KEPT_H.items():
                hm[int(l), v] = 1
            for l, v in KEPT_N.items():
                nm[int(l), v] = 1
            masks[f"kprune_{task}_seed{s}_c0.5__head"], masks[f"kprune_{task}_seed{s}_c0.5__neuron"] = hm, nm
        np.savez(root / "bl" / "kprune_masks.npz", **masks)
    return roots


def _tests(roots, n_boot=300):
    root_of = va.find_roots(roots)
    return {"vs_kprune": {t: va.compare(root_of[t], "d2i_ri", "kprune", t, n_boot, i) for i, t in enumerate(va.TASKS)},
            "vs_kwon": {t: va.compare(root_of[t], "d2i_ri", "kwon", t, n_boot, 10 + i) for i, t in enumerate(va.TASKS)}}


def test_pearson_bootstrap_observed_matches_pooled_corrcoef():
    rng = np.random.default_rng(1)
    y = rng.normal(size=60)
    a = y + 0.3 * rng.normal(size=(5, 60))
    b = y + 0.6 * rng.normal(size=(5, 60))
    r = cluster_bootstrap_delta(a, b, y, "pearson", 200, 5)

    def pooled(x):
        return np.corrcoef(x.ravel(), np.tile(y, 5))[0, 1]
    assert r["observed_delta"] == pytest.approx(pooled(a) - pooled(b), abs=1e-10)
    assert r["ci_low"] > 0  # a is clearly closer to the gold scores


def test_gate_passes_and_loads_all_seven_tasks(tmp_path):
    roots = _fake_roots(tmp_path, 0.1)
    root_of = va.find_roots(roots)
    for t in va.TASKS:
        for s in va.SEEDS:
            va.check_integrity(root_of[t], t, s)
    p, lab = va.load(root_of["mrpc"], "kprune", "mrpc")
    assert p.shape == (5, N) and lab.shape == (N,)
    p, lab = va.load(root_of["squad"], "d2i_ri", "squad")
    assert p.shape == (5, N) and lab is None
    m = va.means(root_of)
    assert m["stsb"]["d2i_ri"] > m["stsb"]["kwon"]


def test_shuffled_rows_mask_mismatch_and_teacher_mismatch_abort(tmp_path):
    roots = _fake_roots(tmp_path, 0.1)
    root_of = va.find_roots(roots)

    def edit(task, fn):
        f = Path(root_of[task]) / "beat" / f"d2i_ri_{task}_seed7_c0.5.json"
        r = json.load(open(f))
        fn(r)
        json.dump(r, open(f, "w"))

    edit("mrpc", lambda r: r.update(valid_logits=r["valid_logits"][::-1]))
    with pytest.raises(va.PipelineMismatch, match="row order"):
        va.check_integrity(root_of["mrpc"], "mrpc", 7)
    edit("qnli", lambda r: r.update(heads_kept={"0": [0, 1], "1": [1]}))
    with pytest.raises(va.PipelineMismatch, match="K-prune masks"):
        va.check_integrity(root_of["qnli"], "qnli", 7)
    edit("squad", lambda r: r["unpruned_valid"].update(f1=0.5))
    with pytest.raises(va.PipelineMismatch, match="teacher F1"):
        va.check_integrity(root_of["squad"], "squad", 7)
    edit("stsb", lambda r: r["unpruned_valid"].update(pearson=0.1))
    with pytest.raises(va.PipelineMismatch, match="teacher Pearson"):
        va.check_integrity(root_of["stsb"], "stsb", 7)


def test_clear_gain_supports_h19a_and_h19b(tmp_path):
    v = va.decide(_tests(_fake_roots(tmp_path, 0.12)))
    assert v["H19a"] == "supported" and len(v["wins"]) >= 5 and not v["losses"] and v["H19b"] == "supported"


def test_no_gain_is_not_h19a_but_noninferior(tmp_path):
    v = va.decide(_tests(_fake_roots(tmp_path, 0.0)))
    assert v["H19a"] == "not_supported" and v["wins"] == [] and v["H19b"] == "supported"


def test_loss_against_kprune_blocks_h19a_and_h19b(tmp_path):
    v = va.decide(_tests(_fake_roots(tmp_path, -0.1)))
    assert v["H19a"] == "not_supported" and v["losses"] and v["H19b"] == "not_supported"


def test_wins_needed_is_two_thirds_rounded_up():
    assert [va.wins_needed(n) for n in (6, 7)] == [4, 5]


def test_six_tasks_four_wins_supports_h19a_three_do_not():
    win = {"ci_low": 0.01, "p_le_zero": 0.001, "p_ge_zero": 1.0, "p_le_margin": 0.001}
    tie = {"ci_low": -0.001, "p_le_zero": 0.4, "p_ge_zero": 0.6, "p_le_margin": 0.001}
    six = [t for t in va.TASKS if t != "squad"]
    kw = {t: dict(win) for t in six}
    for n_win, want in ((4, "supported"), (3, "not_supported")):
        kp = {t: dict(win if i < n_win else tie) for i, t in enumerate(six)}
        assert va.decide({"vs_kprune": kp, "vs_kwon": kw}, six)["H19a"] == want


def test_four_wins_of_seven_is_not_enough():
    win = {"ci_low": 0.01, "p_le_zero": 0.001, "p_ge_zero": 1.0, "p_le_margin": 0.001}
    tie = {"ci_low": -0.001, "p_le_zero": 0.4, "p_ge_zero": 0.6, "p_le_margin": 0.001}
    kw = {t: dict(win) for t in va.TASKS}
    kp4 = {t: dict(win if i < 4 else tie) for i, t in enumerate(va.TASKS)}
    v = va.decide({"vs_kprune": kp4, "vs_kwon": kw})
    assert len(v["wins"]) == 4 and v["H19a"] == "not_supported"
    kp5 = {t: dict(win if i < 5 else tie) for i, t in enumerate(va.TASKS)}
    assert va.decide({"vs_kprune": kp5, "vs_kwon": {t: dict(win) for t in va.TASKS}})["H19a"] == "supported"
