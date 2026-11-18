"""scripts/v21_analysis.py on synthetic units (docs/V21_PREREG_DRAFT_2026-11-18.md, section 3)."""
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import v21_analysis as va  # noqa: E402

LANGS = ("en",) + va.T
N = 600


def _preds(rng, labels, acc):
    return np.where(rng.random(len(labels)) < acc, labels, (labels + 1) % 3).tolist()


def _unit(root, name, acc, seeds=(7, 42, 77), unpruned=0.8, labels_override=None):
    """acc: condition -> (en accuracy, target accuracy)."""
    d = root / name
    d.mkdir(parents=True)
    rng = np.random.default_rng(zlib.crc32(name.encode()))
    labels = rng.integers(0, 3, N)
    up = {L: _preds(rng, labels, unpruned) for L in LANGS}
    for c, (a_en, a_t) in acc.items():
        for s in seeds:
            lab = labels if labels_override is None or c != "kprune" else labels_override
            preds = {L: _preds(rng, labels, a_en if L == "en" else a_t) for L in LANGS}
            json.dump({"condition": c, "seed": s, "labels": lab.tolist(), "preds": preds, "unpruned_preds": up},
                      open(d / f"{c}_seed{s}.json", "w"))


BASE = {"mask_only": (0.70, 0.40), "kprune": (0.76, 0.70), "kwon": (0.72, 0.60), "d2i_r_en": (0.77, 0.70),
        "d2i_r_mixed": (0.79, 0.74), "d2i_ri_en": (0.77, 0.71)}


def test_strong_selection_disparity_supported_and_secondaries(tmp_path):
    for m in ("muril", "mbert"):
        for t in (42, 43, 44):
            _unit(tmp_path, f"{m}_t{t}", BASE)
    r = va.analyse(str(tmp_path), n_boot=300)
    assert r["H21-1_overall"] == {"decisive_model": "muril", "supported": True}
    mu = r["muril"]
    assert mu["stats"]["S_mask"]["observed"] == pytest.approx(0.30, abs=0.04)
    assert mu["H21-2"]["supported"] and mu["H21-3(i)"]["supported"] and mu["H21-3(ii)"]["supported"]
    assert mu["H21-4(i)"]["supported"] and mu["H21-4(ii)"]["supported"]
    assert all(mu[h]["holm_p"] >= mu[h]["p"] for h in ("H21-2", "H21-3(i)", "H21-3(ii)", "H21-4(i)", "H21-4(ii)"))
    assert set(mu["units"]) == {"muril_t42", "muril_t43", "muril_t44"}  # no pooling across models
    assert "N" in mu["robustness"]["S_mask"] and not mu["robustness"]["S_mask"]["sign_flip_N_vs_D"]


def test_sign_rule_blocks_when_one_teacher_disagrees(tmp_path):
    _unit(tmp_path, "muril_t42", BASE)
    _unit(tmp_path, "muril_t43", BASE)
    _unit(tmp_path, "muril_t44", {**BASE, "mask_only": (0.40, 0.70)})  # opposite disparity
    r = va.analyse(str(tmp_path), n_boot=300)
    st = r["muril"]["stats"]["S_mask"]
    assert st["per_unit"]["muril_t44"] < 0 < st["observed"]
    assert not r["muril"]["H21-1"]["supported"] and not r["H21-1_overall"]["supported"]


def test_no_disparity_not_supported(tmp_path):
    for t in (42, 43, 44):
        _unit(tmp_path, f"muril_t{t}", {**BASE, "mask_only": (0.55, 0.55)})
    assert not va.analyse(str(tmp_path), n_boot=300)["muril"]["H21-1"]["supported"]


def test_kwon_missing_is_not_run_and_leaves_the_family(tmp_path):
    acc = {k: v for k, v in BASE.items() if k != "kwon"}
    for t in (42, 43, 44):
        _unit(tmp_path, f"muril_t{t}", acc)
    mu = va.analyse(str(tmp_path), n_boot=300)["muril"]
    assert mu["H21-4(ii)"] == {"run": False} and mu["stats"]["SUP_kwon_T"] is None
    assert mu["H21-4(i)"]["run"]


def test_label_mismatch_breaks_pairing(tmp_path):
    _unit(tmp_path, "muril_t42", BASE, labels_override=np.zeros(N, dtype=int))
    with pytest.raises(SystemExit):
        va.load(str(tmp_path))


def test_pooled_ci_averages_independent_unit_draws(tmp_path):
    for t in (42, 43):
        _unit(tmp_path, f"muril_t{t}", BASE)
    units = va.load(str(tmp_path))
    names = sorted(units)
    idx = {u: i for i, u in enumerate(names)}
    p = va.pooled(units, names, "S_mask", va.T, 200, idx)
    assert p["observed"] == pytest.approx(np.mean(list(p["per_unit"].values())))
    assert p["ci_low"] < p["observed"] < p["ci_high"]
