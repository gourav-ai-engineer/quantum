import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1] / "scripts"


def _mod(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _flip(rng, labels, acc):
    return np.where(rng.random(len(labels)) < acc, labels, 1 - labels)


def _write_baseline(path, labels, preds, f1=None):
    path.mkdir(parents=True, exist_ok=True)
    d = {"labels": torch.tensor(labels), "idx": list(range(len(labels)))}
    if f1 is None:
        logits = torch.zeros(len(labels), 2)
        logits[torch.arange(len(labels)), torch.tensor(preds)] = 1.0
        d["logits"] = logits
    else:
        d["f1"] = torch.tensor(f1)
    torch.save(d, path / "v15_student_validation.pt")


def test_h11_end_to_end_synthetic(tmp_path, monkeypatch):
    m = _mod("v15_h11_analysis")
    rng = np.random.default_rng(0)
    (tmp_path / "grp").mkdir()
    for task in ("mrpc", "sst2", "qnli", "mnli", "squad"):
        labels = rng.integers(0, 2, 600)
        for s in m.SEEDS:
            acc = {"grp": 0.93, "kwon": 0.85, "kprune": 0.80}
            if task == "squad":
                vals = {k: np.clip(rng.normal(v, 0.1, 600), 0, 1) for k, v in acc.items()}
                json.dump({"valid_f1": vals["grp"].tolist()},
                          open(tmp_path / "grp" / f"v15_grp_{task}_seed{s}_c0.5.json", "w"))
                _write_baseline(tmp_path / f"kwon_{task}_seed{s}_c0.5", labels, None, vals["kwon"].tolist())
                _write_baseline(tmp_path / "kprune_out" / f"kprune_{task}_seed{s}_c0.5", labels, None,
                                vals["kprune"].tolist())
            else:
                json.dump({"valid_pred": _flip(rng, labels, acc["grp"]).tolist()},
                          open(tmp_path / "grp" / f"v15_grp_{task}_seed{s}_c0.5.json", "w"))
                _write_baseline(tmp_path / f"kwon_{task}_seed{s}_c0.5", labels, _flip(rng, labels, acc["kwon"]))
                _write_baseline(tmp_path / "kprune_out" / f"kprune_{task}_seed{s}_c0.5", labels,
                                _flip(rng, labels, acc["kprune"]))
    monkeypatch.setattr(sys, "argv", ["v15_h11_analysis", str(tmp_path), "--n-boot", "300"])
    m.main()
    out = json.loads((tmp_path / "v15_h11_verdict.json").read_text())
    assert out["best_baseline"] == "kwon"
    assert out["verdict"] == "supported" and set(out["wins"]) == {"qnli", "mnli", "squad"}


def test_official_patch_works_for_lf_and_crlf(tmp_path):
    m = _mod("v15_official_addon")
    body = ("import numpy as np\n    sample = Subset(\n        x,\n"
            "        np.random.choice(len(training_dataset), num_samples).tolist(),\n    )\n"
            "    st = time.time()\n    go()\n    # Save the model and masks\n    save()\n")
    for nl in ("\n", "\r\n"):
        p = tmp_path / ("crlf.py" if nl == "\r\n" else "lf.py")
        p.write_bytes(body.replace("\n", nl).encode())
        m.patch(str(p), "kprune")
        out = p.read_bytes().decode()
        assert out.count("v15_indices(len(training_dataset), num_samples),") == 1
        assert out.count("v15_dump(") == 2 and out.startswith("from v15_official_addon import")
        assert ("\r\n" in out) == (nl == "\r\n") and "\r\r" not in out


def test_h10_gate_synthetic(tmp_path, monkeypatch):
    m = _mod("v15_h10_analysis")
    (tmp_path / "grp").mkdir()
    for t in m.TASKS:
        for s in m.SEEDS:
            for c in m.CONS:
                teacher = torch.randn(50, 2)
                for method in ("kwon", "kprune"):
                    d = tmp_path / (f"kwon_{t}_seed{s}_c{c}" if method == "kwon"
                                    else f"kprune_out/kprune_{t}_seed{s}_c{c}")
                    d.mkdir(parents=True, exist_ok=True)
                    torch.save({"logits": teacher, "labels": torch.zeros(50), "idx": list(range(50))},
                               d / "v15_teacher_heldout.pt")
                    torch.save({"logits": teacher + 0.5 * torch.randn(50, 2), "labels": torch.zeros(50),
                                "idx": list(range(50))}, d / "v15_student_heldout.pt")
                json.dump({"heldout_kl": 0.001}, open(tmp_path / "grp" / f"v15_grp_{t}_seed{s}_c{c}.json", "w"))
    monkeypatch.setattr(sys, "argv", ["v15_h10_analysis", str(tmp_path)])
    m.main()
    out = json.loads((tmp_path / "v15_h10_verdict.json").read_text())
    assert out["complete"] and out["verdict"] == "supported"


def test_addon_extra_eval_uses_same_path_as_validation(tmp_path, monkeypatch):
    """V21 E5: V21_EXTRA_EVAL sets go through the validation code path; a copy of validation gives identical logits."""
    import datasets
    m = _mod("v15_official_addon")
    val = {"premise": ["a b", "c", "d e f"], "hypothesis": ["x", "y z", "w"], "label": [0, 2, 1]}
    train = datasets.Dataset.from_dict({k: v * 2 for k, v in val.items()})
    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **k: {
        "train": train, "validation_matched": datasets.Dataset.from_dict(val)})

    class Tok:
        def __call__(self, p, h, **kw):
            ids = [[len(a.split()) + 10 * len(b.split())] for a, b in zip(p, h)]
            return {"input_ids": torch.tensor(ids)}

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.tensor([1.0, -1.0, 0.5]))

        def forward(self, input_ids, **kw):
            return type("O", (), {"logits": input_ids.float() * self.w})()

    idx = tmp_path / "idx.json"
    idx.write_text(json.dumps({"calibration": [0], "heldout": [1, 2]}))
    extra = tmp_path / "extra.json"
    extra.write_text(json.dumps({"mnli_m": val, "as": {**val, "label": [1, 1, 1]}}))
    monkeypatch.setenv("V15_INDEX_FILE", str(idx))
    monkeypatch.setenv("V21_EXTRA_EVAL", str(extra))
    m.v15_dump(Model(), Tok(), "mnli", None, None, str(tmp_path), "student")
    v = torch.load(tmp_path / "v15_student_validation.pt")
    x = torch.load(tmp_path / "v15_student_x_mnli_m.pt")
    assert torch.equal(v["logits"], x["logits"]) and torch.equal(v["labels"], x["labels"])
    assert torch.load(tmp_path / "v15_student_x_as.pt")["labels"].tolist() == [1, 1, 1]
    monkeypatch.delenv("V21_EXTRA_EVAL")
    (tmp_path / "v15_student_x_as.pt").unlink()
    m.v15_dump(Model(), Tok(), "mnli", None, None, str(tmp_path), "student")
    assert not (tmp_path / "v15_student_x_as.pt").exists()  # unset -> original behaviour
