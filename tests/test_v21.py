"""V21 pre-freeze engineering code (docs/V21_PREREG_DRAFT_2026-11-18.md): fine-tune defaults, E3 label check, E1/E5 report."""
import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import v20_finetune as vf  # noqa: E402
import v21_e1_report as rep  # noqa: E402
import v21_gates as gates  # noqa: E402


def test_finetune_defaults_reproduce_the_v20_teacher_meta():
    meta = json.load(open(ROOT / "results/v20_pilot/e98c3fbb/ft_meta.json"))
    assert vf.recipe() == vf.recipe(42) == meta["recipe"]
    assert vf.gate_for(vf.MURIL[0]) == vf.GATE == meta["gate"]
    assert vf.PINS[vf.MURIL[0]] == meta["revision"]
    assert vf.gate_for(vf.MBERT[0]) == {"mnli_m": 0.78, "hi": 0.55}
    assert vf.recipe(43).endswith("seed 43")


def test_indicxnli_test_needs_explicit_allow():
    with pytest.raises(ValueError):
        vf.load_indicxnli("as", "test")
    with pytest.raises(ValueError):
        vf.load_indicxnli("as", "test", allow_test=False)


def test_label_parallel():
    r = gates.label_parallel({"en": [0, 1, 2], "as": [0, 1, 2], "ta": [0, 1, 2, 2], "bn": [0, 2, 1], "hi": [0, 1]})
    assert r["as"] == {"rows": 3, "extra_rows": 0, "parallel": True}
    assert r["ta"]["parallel"] and r["ta"]["extra_rows"] == 1
    assert not r["bn"]["parallel"] and not r["hi"]["parallel"]


def _run(d, head, neuron, logits, kind="kwon"):
    d.mkdir(parents=True, exist_ok=True)
    h, n = ("head_mask.pt", "neuron_mask.pt") if kind == "kwon" else ("head_masks.pkl", "neuron_masks.pkl")
    torch.save(torch.tensor(head), d / h)
    torch.save(torch.tensor(neuron), d / n)
    for tag in ("student", "teacher"):
        torch.save({"logits": torch.tensor(logits), "labels": torch.tensor([0, 1]), "idx": [0, 1]},
                   d / f"v15_{tag}_validation.pt")


def test_choose_prefers_full_then_slow_then_none(tmp_path):
    E, IN = tmp_path / "E", tmp_path / "IN"
    for v in ("mbert", "mbert_slow", "muril"):
        (IN / "tok" / v).mkdir(parents=True)
    (IN / "ref_ids").mkdir()
    E.mkdir()
    for k in ("mbert", "muril"):
        json.dump({"ok": True, "ids": [[1, 2]]}, open(IN / "ref_ids" / f"{k}.json", "w"))
    json.dump({"ok": False, "error": "parse"}, open(E / "tok_kwon_mbert.json", "w"))
    json.dump({"ok": True, "ids": [[1, 2]]}, open(E / "tok_kwon_mbert_slow.json", "w"))
    json.dump({"ok": True, "ids": [[1, 3]]}, open(E / "tok_kwon_muril.json", "w"))  # loads, but different ids
    assert rep.choose(str(E), str(IN), "kwon", "mbert") == "v21tok_mbert_slow"
    assert rep.choose(str(E), str(IN), "kwon", "muril") is None
    json.dump({"ok": True, "ids": [[1, 2]]}, open(E / "tok_kwon_mbert.json", "w"))
    assert rep.choose(str(E), str(IN), "kwon", "mbert") == "v21tok_mbert"


def test_compare_runs_and_mac(tmp_path):
    m = [[1.0, 0.0], [1.0, 1.0]]
    _run(tmp_path / "a", m, [[1.0, 0.0, 1.0]] * 2, [[0.2, 0.8], [0.9, 0.1]])
    _run(tmp_path / "b", [[0.7, 0.0], [1.2, 1.0]], [[1.0, 0.0, 1.0]] * 2, [[0.3, 0.7], [0.8, 0.2]], kind="kprune")
    r = rep.compare_runs(str(tmp_path / "a"), str(tmp_path / "b"))
    assert r["ran"] and r["keep_pattern_equal"] and not r["mask_values_equal"] and r["val_preds_equal"]
    assert rep.compare_runs(str(tmp_path / "a"), str(tmp_path / "missing")) == {"ran": False}
    full = rep.mac((torch.ones(12, 12), torch.ones(12, 3072)), 128)
    half = rep.mac((torch.ones(12, 12), torch.cat([torch.ones(12, 1536), torch.zeros(12, 1536)], 1)), 128)
    assert full == 1.0 and 0.0 < half < 1.0


def test_e5_parity(tmp_path):
    d = tmp_path / "r"
    _run(d, [[1.0]], [[1.0]], [[0.2, 0.8], [0.9, 0.1]])
    for tag in ("student", "teacher"):
        torch.save(torch.load(d / f"v15_{tag}_validation.pt"), d / f"v15_{tag}_x_mnli_m.pt")
        for s in ("en", "as", "bn", "hi"):
            torch.save({"logits": torch.zeros(3, 3), "labels": torch.zeros(3, dtype=torch.long), "idx": [0, 1, 2]},
                       d / f"v15_{tag}_x_{s}.pt")
    p = rep.e5_parity(str(d))
    assert p["student"]["logits_equal"] and p["teacher"]["extra_sets"] == {"en": 3, "as": 3, "bn": 3, "hi": 3}
    torch.save({"logits": torch.ones(2, 2), "labels": torch.tensor([0, 1]), "idx": [0, 1]}, d / "v15_student_x_mnli_m.pt")
    assert not rep.e5_parity(str(d))["student"]["logits_equal"]
