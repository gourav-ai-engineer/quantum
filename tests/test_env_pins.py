import re
from pathlib import Path

import pytest

from qfc.preflight import EXPECTED_VERSIONS

ROOT = Path(__file__).resolve().parents[1]


def _requirement_pins():
    pins = {}
    for line in (ROOT / "requirements-ci.txt").read_text(encoding="utf-8").splitlines():
        m = re.match(r"^([A-Za-z0-9_.-]+)==([^\s#]+)", line.strip())
        if m:
            pins[m.group(1).lower().replace("_", "-")] = m.group(2)
    return pins


def test_requirements_pin_the_four_critical_libraries():
    pins = _requirement_pins()
    for name, version in EXPECTED_VERSIONS.items():
        assert pins.get(name) == version, name


def test_pyproject_transformers_extra_matches_requirements():
    tomllib = pytest.importorskip("tomllib")
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extra = data["project"]["optional-dependencies"]["transformers"]
    pins = _requirement_pins()
    for name in ("transformers", "datasets", "pyarrow"):
        assert f"{name}=={pins[name]}" in extra, name
    assert not any(re.search(r"transformers\s*>=", dep) for dep in extra), (
        "an unpinned transformers lower bound would allow transformers 5.x"
    )
