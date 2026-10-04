"""Tests for scripts/make_ledger_row.py using SYNTHETIC fixtures (made-up numbers, tmp dirs only)."""

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "make_ledger_row.py"
spec = importlib.util.spec_from_file_location("make_ledger_row", SCRIPT)
ledger = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ledger)

VERSIONS = {"torch": "2.6.0+cu124", "transformers": "4.51.3", "datasets": "3.6.0", "pyarrow": "24.0.0"}


def _meta(experiment, **overrides):
    meta = {
        "experiment": experiment,
        "commit": "a" * 40,
        "commit_short": "aaaaaaaaaaaa",
        "dirty": False,
        "date_utc": "2000-01-01T00:00:00+00:00",
        "smoke": False,
        "args": {"flags": "--synthetic-flag 1"},
        "python": "3.12.0",
        "versions": VERSIONS,
        "cuda_available": True,
        "gpu_name": "SYNTHETIC GPU",
    }
    meta.update(overrides)
    return meta


def _write(dir_, name, obj):
    (dir_ / name).write_text(json.dumps(obj), encoding="utf-8")


def _rand(mean, lo, hi):
    return {"distribution": {"accuracy": {"mean": mean, "min": lo, "max": hi, "n": 30}}}


def test_v10_row_reads_numbers_from_json(tmp_path):
    _write(tmp_path, "run_meta.json", _meta("v10_confirmatory"))
    for seed, acc in ((7, 0.8), (42, 0.9)):
        _write(tmp_path, f"sst2_seed{seed}.json", {
            "task": "sst2", "seed": seed, "evaluation_size": 10,
            "baseline": {"accuracy": 0.95, "loss": 0.25},
            "methods": {"QFC": {"accuracy": acc}, "Random": _rand(0.7, 0.6, 0.8)},
        })
    row, warnings = ledger.row_for(tmp_path)
    assert warnings == []
    assert row.startswith("| V10 confirmatory | step18_v10_confirmatory.py |")
    assert "QFC 0.8500+/-0.0500" in row
    assert "unpruned acc 0.9500, loss 0.2500" in row
    assert "Random 0.7000 [min 0.6000, max 0.8000, n=30]" in row
    assert "aaaaaaaaaaaa" in row and "SYNTHETIC GPU" in row and "--synthetic-flag 1" in row
    assert "\n" not in row


def test_v6_and_v5_rows(tmp_path):
    _write(tmp_path, "run_meta.json", _meta("v6_budget_response"))
    _write(tmp_path, "summary.json", {"tasks": {"mrpc": {
        "validation_size": 408, "baseline": {"accuracy": 0.8, "loss": 0.4, "f1": 0.85},
        "budgets": {"6": {"QFC": {"accuracy": 0.75}, "Random": _rand(0.6, 0.5, 0.7), "retention_fraction": 0.5}},
    }}})
    row, _ = ledger.row_for(tmp_path)
    assert "eval n=408" in row and "F1 0.8500" in row and "QFC 0.7500" in row
    assert "retention_fraction" not in row

    d5 = tmp_path / "v5"
    d5.mkdir()
    _write(d5, "run_meta.json", _meta("v5_mrpc_stability"))
    _write(d5, "summary.json", {
        "evaluation_size": 408, "seeds": [7, 42],
        "runs": [{"baseline": {"accuracy": 0.8, "loss": 0.4, "f1": 0.85}}],
        "aggregate": {"QFC": {"accuracy_mean": 0.7, "accuracy_std": 0.01, "f1_mean": 0.8, "f1_std": 0.02}},
    })
    row5, _ = ledger.row_for(d5)
    assert "QFC acc 0.7000+/-0.0100 F1 0.8000+/-0.0200" in row5


def _v11_layers(n_below, n_layers=12):
    layers = {}
    for i in range(n_layers):
        below = i < n_below
        ci = {"estimate": -0.3 if below else 0.0, "ci_low": -0.5 if below else -0.2,
              "ci_high": -0.1 if below else 0.2, "degenerate": False}
        other = {"estimate": 0.0, "ci_low": -0.2, "ci_high": 0.2, "degenerate": False}
        layers[str(i)] = {"spearman_coverage_vs_loss_ci": {
            "fidelity": ci, "hilbert_schmidt": other, "cosine": other}}
    return layers


def _v11_summary(fid_below):
    def task():
        return {
            "evaluation_size": 256, "baseline": {"accuracy": 0.9, "loss": 0.3},
            "layers": _v11_layers(fid_below),
            "pooled": {"spearman_coverage_vs_loss_ci": {"fidelity": {
                "estimate": -0.1, "ci_low": -0.2, "ci_high": 0.0, "degenerate": False}}},
        }

    return {"random_subsets_per_layer": 300, "bootstrap_resamples": 1000,
            "tasks": {"sst2": task(), "mrpc": task()}}


def test_v11_row_counts_layers_and_reports_criterion_a(tmp_path):
    _write(tmp_path, "run_meta.json", _meta("v11_objective_alignment"))
    _write(tmp_path, "summary.json", _v11_summary(fid_below=9))
    row, _ = ledger.row_for(tmp_path)
    assert "fidelity: CI entirely <0 in 9/12 layers, includes 0 in 3" in row
    assert "hilbert_schmidt: CI entirely <0 in 0/12 layers, includes 0 in 12" in row
    assert "'fidelity CI entirely below 0 in >= 8 of 12 layers on BOTH tasks': MET" in row

    d2 = tmp_path / "b"
    d2.mkdir()
    _write(d2, "run_meta.json", _meta("v11_objective_alignment"))
    _write(d2, "summary.json", _v11_summary(fid_below=7))
    row2, _ = ledger.row_for(d2)
    assert "NOT MET" in row2
    assert "final verdict are for the owner" in row2


def test_refuses_smoke_wrong_env_and_cpu(tmp_path):
    _write(tmp_path, "summary.json", {})
    _write(tmp_path, "run_meta.json", _meta("v5_mrpc_stability", smoke=True))
    with pytest.raises(ledger.LedgerError, match="smoke"):
        ledger.row_for(tmp_path)

    _write(tmp_path, "run_meta.json", _meta("v5_mrpc_stability", versions={**VERSIONS, "transformers": "5.18.0"}))
    with pytest.raises(ledger.LedgerError, match="transformers"):
        ledger.row_for(tmp_path)

    _write(tmp_path, "run_meta.json", _meta("v5_mrpc_stability", cuda_available=False))
    with pytest.raises(ledger.LedgerError, match="CUDA"):
        ledger.row_for(tmp_path)


def test_missing_meta_is_an_error_and_dirty_warns(tmp_path):
    with pytest.raises(ledger.LedgerError, match="run_meta.json"):
        ledger.row_for(tmp_path)
    _write(tmp_path, "run_meta.json", _meta("v6_budget_response", dirty=True))
    _write(tmp_path, "summary.json", {"tasks": {}})
    row, warnings = ledger.row_for(tmp_path)
    assert warnings and "dirty" in warnings[0]
    assert "(dirty)" in row
