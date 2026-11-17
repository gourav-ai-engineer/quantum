import json
import sys
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import v20_pilot as vp  # noqa: E402
import v20_pilot_analysis as va  # noqa: E402


def test_mixed_assignment_round_robin_balanced():
    a = vp.mixed_assignment(2048)
    assert a[:5] == ["en", "as", "bn", "hi", "en"]
    assert {l: a.count(l) for l in vp.MIX} == {"en": 512, "as": 512, "bn": 512, "hi": 512}


def test_calibration_rows_content_matched():
    tables = {l: [{"premise": f"{l}-p{i}", "hypothesis": f"{l}-h{i}"} for i in range(10)] for l in vp.MIX}
    rows = vp.calibration_rows(lambda l: tables[l], [3, 7, 1, 9], "mixed")
    assert rows["premise"] == ["en-p3", "as-p7", "bn-p1", "hi-p9"]
    rows = vp.calibration_rows(lambda l: tables[l], [3, 7], "as")
    assert rows["hypothesis"] == ["as-h3", "as-h7"]
    rows = vp.calibration_rows(lambda l: tables[l], [3, 7], "as", lambda l, i: i + 1)  # row map applied
    assert rows["premise"] == ["as-p4", "as-p8"]
    assert "label" not in rows.column_names


def test_align_rows_recovers_drops_and_skips_insertions():
    rng = np.random.default_rng(1)
    en = rng.integers(0, 3, 3000).tolist()
    keep = [i for i in range(3000) if i not in (500, 1200, 1201, 2500)]          # dropped MNLI rows
    other = [en[i] for i in keep[:1500]] + rng.integers(0, 3, 40).tolist() + [en[i] for i in keep[1500:]]
    m = vp.align_rows(en, other, k=32, max_shift=200)
    true = {j: keep[j] for j in range(1500)} | {1540 + j: keep[1500 + j] for j in range(len(keep) - 1500)}
    wrong = [i for i, j in m.items() if true.get(i) != j]
    assert not wrong  # a wrong pair would break content matching; lost rows only cost coverage (~k per event)
    assert len(m) > 0.9 * len(true)


def test_indicxnli_test_split_is_refused():
    import pytest
    import v20_finetune as vf
    with pytest.raises(ValueError):
        vf.load_indicxnli("as", "test")


def test_linear_bootstrap_matches_observed_and_brackets_it():
    rng = np.random.default_rng(0)
    a, b = rng.random((3, 500)) < 0.8, rng.random((3, 500)) < 0.6
    r = va.linear_bootstrap([(1, a), (-1, b)], n_boot=2000)
    assert abs(r["observed"] - (a.mean() - b.mean())) < 1e-12
    assert r["ci_low"] < r["observed"] < r["ci_high"] and r["ci_low"] > 0


def _write(d, rep, cal, seed, acc_by_lang, n=600):
    rng = np.random.default_rng(zlib.crc32(f"{rep}/{cal}/{seed}".encode()))
    labels = rng.integers(0, 3, n)
    preds = {L: np.where(rng.random(n) < p, labels, (labels + 1) % 3).tolist() for L, p in acc_by_lang.items()}
    json.dump({"repair": rep, "cal_lang": cal, "seed": seed, "labels": labels.tolist(), "preds": preds},
              open(d / f"{rep}_{cal}_seed{seed}_c0.5.json", "w"))


def _fake(tmp_path, en_gain, tgt_gain_en, tgt_gain_own):
    """mask-only 0.5 everywhere; en-calibrated repair: +en_gain on en, +tgt_gain_en on targets; own-language repair:
    +tgt_gain_own on its own language."""
    langs = ("en",) + va.DEV
    for s in (7, 42, 77):
        _write(tmp_path, "none", "none", s, {L: 0.5 for L in langs})
        _write(tmp_path, "d2i_r", "en", s, {L: 0.5 + (en_gain if L == "en" else tgt_gain_en) for L in langs})
        for T in va.DEV:
            _write(tmp_path, "d2i_r", T, s, {L: 0.5 + (tgt_gain_own if L == T else tgt_gain_en) for L in langs})
    return va.load(tmp_path)[0]


def test_decide_go_when_english_repair_favours_english(tmp_path):
    g = _fake(tmp_path, en_gain=0.25, tgt_gain_en=0.05, tgt_gain_own=0.20)
    assert va.decide(g, n_boot=1000)["GO"]


def test_decide_no_go_when_repair_helps_all_languages_equally(tmp_path):
    g = _fake(tmp_path, en_gain=0.15, tgt_gain_en=0.15, tgt_gain_own=0.15)
    assert not va.decide(g, n_boot=1000)["GO"]
