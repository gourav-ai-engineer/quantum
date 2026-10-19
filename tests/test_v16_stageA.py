import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from v16_stageA_analysis import decide  # noqa: E402

KP = {"mrpc_c0.5": 0.05, "mrpc_c0.4": 0.07, "sst2_c0.5": 0.05, "sst2_c0.4": 0.09}


def _write(d, kl_ratio, rho_top, rho_low):
    json.dump({"table": {k: {"kprune": {"mean_kl": v}} for k, v in KP.items()}}, open(d / "verdict.json", "w"))
    for part, r in kl_ratio.items():
        for task in ("mrpc", "sst2"):
            for c in (0.5, 0.4):
                for s in (7, 42, 77, 123, 2024):
                    json.dump({"heldout_kl": r * KP[f"{task}_c{c}"]},
                              open(d / f"v16_{part}_{task}_seed{s}_c{c}.json", "w"))
    for task in ("mrpc", "sst2"):
        for s in (7, 42):
            mods = [{"layer": l, "module": m, "rho": rho_top if l >= 9 else rho_low}
                    for l in range(12) for m in ("att", "ffn")]
            json.dump({"modules": mods}, open(d / f"v16_d1_{task}_seed{s}_c0.5.json", "w"))
    return decide(str(d), str(d / "verdict.json"))


def test_branch_a1(tmp_path):
    out = _write(tmp_path, {"d2i": 0.8, "d2ii": 0.85, "no_sens": 1.2}, rho_top=3.0, rho_low=1.0)
    assert out["branch"] == "A-1"


def test_d2ii_beats_without_underpricing_is_a3_when_d2i_beats_too(tmp_path):
    out = _write(tmp_path, {"d2i": 0.8, "d2ii": 0.85, "no_sens": 1.2}, rho_top=1.5, rho_low=1.0)
    assert out["branch"] == "A-3"  # d2ii beats, so A-2 cannot fire; no underpricing, so not A-1


def test_branch_a2(tmp_path):
    out = _write(tmp_path, {"d2i": 0.8, "d2ii": 0.95, "no_sens": 1.2}, rho_top=3.0, rho_low=1.0)
    assert out["branch"] == "A-2"


def test_ten_percent_margin_is_inclusive_and_strict_above(tmp_path):
    out = _write(tmp_path, {"d2i": 0.9, "d2ii": 0.91, "no_sens": 1.0}, rho_top=3.0, rho_low=1.0)
    assert out["beats"]["d2i"] and not out["beats"]["d2ii"] and out["branch"] == "A-2"


def test_v17_gate_and_mechanism(tmp_path):
    from v17_rdev_analysis import decide as decide17
    json.dump({"table": {k: {"kprune": {"mean_kl": v}} for k, v in KP.items()}}, open(tmp_path / "v.json", "w"))

    def put(part, mrpc, sst2):
        for task, r in (("mrpc", mrpc), ("sst2", sst2)):
            for c in (0.5, 0.4):
                for s in (7, 42, 77, 123, 2024):
                    json.dump({"heldout_kl": r * KP[f"{task}_c{c}"]}, open(tmp_path / f"v17_{part}_{task}_seed{s}_c{c}.json", "w"))
    put("grp_r", 1.0, 0.9)
    put("d2i_r", 1.1, 1.2)
    out = decide17(str(tmp_path), str(tmp_path / "v.json"))
    assert out["gate_pass"] and not out["mechanism_confirmed"]
    put("grp_r", 1.01, 0.5)
    assert not decide17(str(tmp_path), str(tmp_path / "v.json"))["gate_pass"]
