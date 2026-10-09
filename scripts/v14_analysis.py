"""Apply the pre-registered V14 rules (docs/PROJECT_STATE.md, "Pre-registration V14") to step21 output.

    python scripts/v14_analysis.py <results_dir> [--n-boot 10000] [--seed 1234]

Paired cluster bootstrap over (calibration seed x example), 95% percentile CI, one-sided bootstrap p-values, Holm
across the 5 tasks within a test. CoLA uses MCC, the other tasks accuracy. Writes v14_verdicts.json.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from qfc.metrics import bootstrap_p_ge, bootstrap_p_le, cluster_bootstrap_delta, holm_adjust

ALL_TASKS = ["sst2", "mrpc", "rte", "qnli", "cola"]
CONFIRMATORY = ["rte", "qnli", "cola"]
SELECTORS = ["bcm", "fisher", "magnitude", "greedy_ls", "qfc_attn"]
ALPHA, MIN_SEEDS = 0.05, 5


def load_task(root: Path, task: str):
    files = sorted(root.glob(f"v14_{task}_seed*.npz"), key=lambda f: int(re.search(r"seed(\d+)", f.name)[1]))
    if not files:
        return None
    preds, labels = {}, None
    for f in files:
        z = np.load(f)
        labels = z["labels"] if labels is None else labels
        if not np.array_equal(labels, z["labels"]):
            raise ValueError(f"{f.name}: labels differ across seeds")
        for key in z.files:
            if key.endswith("__pred"):
                preds.setdefault(key[: -len("__pred")], []).append(z[key])
    return {m: np.stack(v) for m, v in preds.items()}, labels, len(files)


def compare(preds, labels, a, b, metric, n_boot, seed):
    """a minus b; b may be a list of arms (delta against their mean, bootstrapped with shared resamples)."""
    bs = b if isinstance(b, list) else [b]
    rs = [cluster_bootstrap_delta(preds[a], preds[x], labels, metric, n_boot=n_boot, seed=seed) for x in bs]
    boot = np.mean([r["boot"] for r in rs], axis=0)  # same seed -> same resamples, so the average is a valid bootstrap
    return {"observed_delta": float(np.mean([r["observed_delta"] for r in rs])),
            "ci_low": float(np.quantile(boot, 0.025)), "ci_high": float(np.quantile(boot, 0.975)),
            "p_le_zero": bootstrap_p_le(boot, 0.0), "p_ge_zero": bootstrap_p_ge(boot, 0.0)}


def with_holm(res):
    tasks = list(res)
    for key in ("p_le_zero", "p_ge_zero"):
        for t, adj in zip(tasks, holm_adjust([res[t][key] for t in tasks])):
            res[t]["holm_" + key] = adj
    return res


def superiority(res):
    wins = [t for t in CONFIRMATORY if t in res and res[t]["ci_low"] > 0 and res[t]["holm_p_le_zero"] < ALPHA]
    losses = [t for t, r in res.items() if r["holm_p_ge_zero"] < ALPHA]
    return {"confirmatory_wins": wins, "significant_losses": losses,
            "verdict": "supported" if len(wins) >= 2 and not losses else "not_supported"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()
    root = Path(args.root)
    data = {t: d for t in ALL_TASKS if (d := load_task(root, t)) is not None}
    metric = {t: "mcc" if t == "cola" else "accuracy" for t in data}

    def run(a, b, i):
        return {t: compare(data[t][0], data[t][1], a, b, metric[t], args.n_boot, args.seed + i) for t in data}

    out = {"tasks_present": {t: data[t][2] for t in data}, "tests": {}}
    h9a = {}
    for i, s in enumerate(SELECTORS):
        out["tests"][f"H9a_{s}"] = res = run(f"{s}|ls_seq", f"{s}|none", i)
        h9a[s] = sum(r["ci_low"] > 0 for r in res.values())
    out["tests"]["H9b"] = with_holm(run("bcm|ls_seq", "fisher|ls_seq", 10))
    out["tests"]["H9c"] = with_holm(run("bcm|ls_seq", "greedy_ls|ls_seq", 11))
    rand = sorted(m for m in next(iter(data.values()))[0] if m.startswith("bcm|merge_randtarget")) if data else []
    out["tests"]["H9d"] = run("bcm|merge", rand, 12) if rand else {}
    complete = all(t in data and data[t][2] >= MIN_SEEDS for t in ALL_TASKS)
    if complete:
        h9d_pos = [t for t, r in out["tests"]["H9d"].items() if r["ci_low"] > 0]
        out["verdicts"] = {
            "H9a": {s: ("pass" if n >= 4 else "fail") for s, n in h9a.items()},
            "H9b": superiority(out["tests"]["H9b"]),
            "H9c": superiority(out["tests"]["H9c"]),
            "H9d": {"tasks_ci_above_zero": h9d_pos, "verdict": "supported" if len(h9d_pos) >= 3 else "not_supported"},
        }
    else:
        out["verdicts"] = {"all": "insufficient_data: need all 5 tasks with >= 5 calibration seeds"}
    out["complete_protocol"] = complete
    (root / "v14_verdicts.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({"complete_protocol": complete, "verdicts": out["verdicts"]}, indent=2))


if __name__ == "__main__":
    main()
