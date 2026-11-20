"""ESM-2 pilot (E6): K1 and K2 exactly as pre-registered. Usage: python scripts/esm2_pilot_analysis.py results/esm2_pilot/<commit>
K1 (budget 0.225, selector grp): mean(B) - mean(A) >= 0.05 and B > A (strict) on >= 14 of 20 assays.
K2 (budget 0.225, selector grp): mean(B) >= mean(C) and percentile-95% bootstrap lower bound of mean(B - C) > -0.01.
Bootstrap over assays, 10,000 resamples, seed 1234. CONTINUE only if both pass; otherwise STOP. Others are descriptive."""
import json
import sys
from pathlib import Path

import numpy as np

from qfc.esm_pilot import bootstrap_ci

d = Path(sys.argv[1])
arms = json.loads((d / "results.json").read_text())["arms"]
ids = list(arms["D_dense150M"]["rho"])


def v(name):
    return np.array([arms[name]["rho"][a] for a in ids])


def diff(x, y):
    dd = x - y
    lo, hi = bootstrap_ci(dd)
    return {"mean": float(dd.mean()), "ci95": [lo, hi], "x_gt_y": int((x > y).sum()), "n": len(dd)}


A, B, C = v("A|grp|0.225"), v("B|grp|0.225"), v("C_dense35M")
ba, bc = diff(B, A), diff(B, C)
k1 = bool(ba["n"] == 20 and ba["mean"] >= 0.05 and ba["x_gt_y"] >= 14)
k2 = bool(bc["n"] == 20 and bc["mean"] >= 0 and bc["ci95"][0] > -0.01)
out = {"K1": {"B_minus_A": ba, "pass": k1}, "K2": {"B_minus_C": bc, "pass": k2},
       "verdict": "CONTINUE" if (k1 and k2) else "STOP",
       "means": {k: float(np.mean(v(k))) for k in arms},
       "descriptive": {f"{x}-{y}": diff(v(x), v(y)) for x, y in [
           ("B|grp|0.225", "D_dense150M"), ("B|grp|0.5", "A|grp|0.5"), ("B|grp|0.5", "D_dense150M"), ("B|grp|0.5", "C_dense35M"),
           ("B|magnitude|0.225", "A|magnitude|0.225"), ("B|magnitude|0.225", "C_dense35M"), ("B|grp|0.225", "B|magnitude|0.225"),
           ("E_int8_150M", "D_dense150M"), ("C_dense35M", "D_dense150M")]}}
(d / "verdict.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
