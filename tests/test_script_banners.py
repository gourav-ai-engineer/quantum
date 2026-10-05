import ast
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"

LEGACY = [
    "step2_calibrate_sst2",
    "step3_compare_selection",
    "step4_split_stability",
    "step5_full_sst2_budget_sweep",
    "step6_conditional_qfc_smoke",
    "step7_mrpc_smoke",
    "step11_conditional_iwqfc_smoke",
    "step15_layer_adaptive_smoke",
    "step16_representation_ablation",
    "step17_input_conditioned",
]


@pytest.mark.parametrize("name", LEGACY)
def test_legacy_script_has_status_banner(name):
    """Each legacy script must say whether it satisfies rules 1-4 and whether it may be cited."""
    doc = ast.get_docstring(ast.parse((SCRIPTS / f"{name}.py").read_text(encoding="utf-8")))
    assert doc, f"{name} has no module docstring"
    assert "STATUS:" in doc
    assert "DO NOT CITE" in doc or "Do not cite" in doc
    for rule in ("Rule 1", "Rule 2", "Rule 3", "Rule 4"):
        assert rule in doc, f"{name} banner does not address {rule}"
    # Any script that is not a citable experiment must say so explicitly.
    first_status = next(line for line in doc.splitlines() if line.startswith("STATUS:"))
    assert any(tag in first_status for tag in ("SMOKE", "EXPLORATORY", "ABLATION ONLY"))
