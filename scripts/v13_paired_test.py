"""V13: paired bootstrap of per-example correctness, method A minus method B, per task and seed.

Reads <task>_seed<seed>_correct.npz written by step18 and applies the rule pre-registered in
docs/PROJECT_STATE.md (section "Pre-registered hypothesis V13"). Analysis only: no selector is touched.

    python scripts/v13_paired_test.py <results_dir> [--a MeanStateIWQFC] [--b MichelGate]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from qfc.metrics import paired_bootstrap_delta

MIN_SEEDS_FOR_PASS = 2  # of 3 calibration seeds, per task


def v13_verdict(cis: dict[str, list[dict]]) -> str:
    """cis: task -> one {"ci_low": ...} dict per seed. See the pre-registered rule."""
    if not cis:
        raise ValueError("no tasks")
    if any(len(seeds) != 3 for seeds in cis.values()) or len(cis) < 2:
        return "insufficient_data"  # rule is defined for 3 seeds on both tasks (e.g. smoke runs)
    passed = [sum(c["ci_low"] > 0 for c in seeds) >= MIN_SEEDS_FOR_PASS for seeds in cis.values()]
    if all(passed):
        return "supported"
    if not any(passed):
        return "not_supported"
    return "inconclusive"  # exactly one task passes: task-dependent, do not read as support


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("root")
    p.add_argument("--a", default="MeanStateIWQFC")
    p.add_argument("--b", default="MichelGate")
    p.add_argument("--bootstrap", type=int, default=10000)
    args = p.parse_args()

    cis: dict[str, list[dict]] = {}
    for f in sorted(Path(args.root).rglob("*_correct.npz")):
        m = re.fullmatch(r"(?P<task>[a-z0-9]+)_seed(?P<seed>\d+)_correct\.npz", f.name)
        if m is None:
            continue
        z = np.load(f)
        ci = paired_bootstrap_delta(z[args.a], z[args.b], n_boot=args.bootstrap, seed=int(m["seed"]) + 40000)
        cis.setdefault(m["task"], []).append({"seed": int(m["seed"]), "n": int(len(z[args.a])), **ci})
    out = {"a": args.a, "b": args.b, "metric": "accuracy (paired, per example)", "tasks": cis, "verdict": v13_verdict(cis)}
    (Path(args.root) / "v13_paired.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
