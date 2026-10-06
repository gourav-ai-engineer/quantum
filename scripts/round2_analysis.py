"""Apply the pre-registered Round 2 rules (docs/PROJECT_STATE.md) to step20 output; write round2_verdicts.json.

    python scripts/round2_analysis.py <results_dir> [--n-boot 10000] [--seed 1234]

Statistics: paired cluster bootstrap over (calibration seed x example), 95% percentile CI, one-sided bootstrap
p-values, Holm correction across the 5 tasks within each hypothesis (alpha = 0.05). A task passes a test only if
BOTH the 95% CI bound and the Holm-adjusted p-value agree. CoLA uses MCC, the other tasks accuracy; margins are
0.015 (1.5 pp / 0.015 MCC). Nothing here chooses a method or direction on evaluation data.
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
MARGIN = 0.015
ALPHA = 0.05
MIN_SEEDS = 5


def load_task(root: Path, task: str):
    """Return ({method: [S, N] predictions}, labels [N], seeds) for one task, or None if absent."""
    files = sorted(root.glob(f"round2_{task}_seed*.npz"), key=lambda f: int(re.search(r"seed(\d+)", f.name)[1]))
    if not files:
        return None
    seeds, preds, labels = [], {}, None
    for f in files:
        z = np.load(f)
        seeds.append(int(re.search(r"seed(\d+)", f.name)[1]))
        labels = z["labels"] if labels is None else labels
        if not np.array_equal(labels, z["labels"]):
            raise ValueError(f"{f.name}: labels differ across seeds (evaluation set changed)")
        for key in z.files:
            if key.endswith("__pred"):
                preds.setdefault(key[: -len("__pred")], []).append(z[key])
    return {m: np.stack(v) for m, v in preds.items()}, labels, seeds


def compare(preds, labels, a, b, metric, margin, n_boot, seed):
    r = cluster_bootstrap_delta(preds[a], preds[b], labels, metric, n_boot=n_boot, seed=seed)
    return {"observed_delta": r["observed_delta"], "ci_low": r["ci_low"], "ci_high": r["ci_high"],
            "p_le_margin": bootstrap_p_le(r["boot"], margin), "p_ge_zero": bootstrap_p_ge(r["boot"], 0.0)}


def add_holm(results: dict[str, dict]) -> None:
    tasks = list(results)
    for key in ("p_le_margin", "p_ge_zero"):
        for t, adj in zip(tasks, holm_adjust([results[t][key] for t in tasks])):
            results[t]["p_adj_" + key[2:]] = adj  # p_adj_le_margin, p_adj_ge_zero


def h1_verdict(res: dict[str, dict]) -> dict:
    """VNE_keep_high - MichelGate (margin -1.5 pp). Needs res[task] with ci_low and p_adj_le_margin."""
    passes = {t: bool(r["ci_low"] > -MARGIN and r["p_adj_le_margin"] < ALPHA) for t, r in res.items()}
    failures = sum(not v for v in passes.values())
    supported = all(passes.get(t, False) for t in CONFIRMATORY) and failures <= 1
    return {"passes": passes, "failures": failures, "verdict": "supported" if supported else "not_supported"}


def h2_verdict(res: dict[str, dict]) -> dict:
    """VNE_keep_high - DelimiterMass_keep_low (margin 0). Outcomes are recorded, not mutually exclusive."""
    informative_tasks = [t for t in CONFIRMATORY if t in res and res[t]["ci_low"] > 0 and res[t]["p_adj_le_margin"] < ALPHA]
    outcomes = []
    if len(informative_tasks) >= 2:
        outcomes.append("VNE informative")
    if all(r["observed_delta"] <= MARGIN for r in res.values()):
        outcomes.append("VNE adds nothing beyond the heuristic")
    beats = [t for t, r in res.items() if r["ci_high"] < 0]
    if beats:
        outcomes.append("DelimiterMass_keep_low beats VNE on: " + ",".join(beats))
    return {"informative_confirmatory_tasks": informative_tasks, "delimiter_beats_vne_tasks": beats,
            "outcomes": outcomes or ["inconclusive"]}


def h6_verdict(res: dict[str, dict]) -> dict:
    """MichelGlobal - MichelGate (margin 0): better if Holm-significant positive on >= 3 of 5 and none negative."""
    positive = [t for t, r in res.items() if r["p_adj_le_margin"] < ALPHA]
    negative = [t for t, r in res.items() if r["p_adj_ge_zero"] < ALPHA]
    return {"significantly_better_tasks": positive, "significantly_worse_tasks": negative,
            "verdict": "global_better" if len(positive) >= 3 and not negative else "not_better"}


def method_tables(root: Path) -> dict:
    """Per task and method: mean over seeds of the task metric, loss, AUROC; kept-per-layer counts; Random quantiles."""
    out: dict = {}
    for task in ALL_TASKS:
        records = [json.loads(f.read_text()) for f in sorted(root.glob(f"round2_{task}_seed*.json"))]
        if not records:
            continue
        metric = records[0]["metric"]
        methods: dict = {}
        for name in records[0]["methods"]:
            vals = [r["methods"][name] for r in records]
            methods[name] = {m: float(np.mean([v[m] for v in vals])) for m in (metric, "loss", "auroc")}
            methods[name]["heads_kept_per_layer_seed0"] = vals[0]["heads_kept_per_layer"]
            methods[name]["total_heads_kept"] = vals[0]["total_heads_kept"]
        entry = {"metric": metric, "seeds": [r["seed"] for r in records], "evaluation_size": records[0]["evaluation_size"],
                 "unpruned": records[0]["unpruned"], "methods": methods}
        rf = root / f"round2_{task}_random.json"
        if rf.exists():
            entry["random_quantiles"] = json.loads(rf.read_text())["quantiles"]
        out[task] = entry
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()
    root = Path(args.root)

    data, metrics = {}, {}
    for task in ALL_TASKS:
        loaded = load_task(root, task)
        if loaded is not None:
            data[task] = loaded
            metrics[task] = "mcc" if task == "cola" else "accuracy"

    specs = {  # hypothesis -> (method A, method B, margin)
        "H1": ("VonNeumann_keep_high", "MichelGate", -MARGIN),
        "H2": ("VonNeumann_keep_high", "DelimiterMass_keep_low", 0.0),
        "H6": ("MichelGlobal", "MichelGate", 0.0),
        "H6_min1": ("MichelGlobal_min1", "MichelGate", 0.0),
    }
    comparisons: dict[str, dict[str, dict]] = {}
    for h, (a, b, margin) in specs.items():
        comparisons[h] = {}
        for i, (task, (preds, labels, _)) in enumerate(data.items()):
            comparisons[h][task] = compare(preds, labels, a, b, metrics[task], margin, args.n_boot, args.seed + i)
        add_holm(comparisons[h])

    complete = all(t in data and len(data[t][2]) >= MIN_SEEDS for t in ALL_TASKS)
    if complete:
        verdicts = {"H1": h1_verdict(comparisons["H1"]), "H2": h2_verdict(comparisons["H2"]),
                    "H6": h6_verdict(comparisons["H6"]), "H6_min1": h6_verdict(comparisons["H6_min1"])}
    else:
        verdicts = {"all": "insufficient_data: need all 5 tasks with >= 5 calibration seeds"}
    out = {"complete_protocol": complete, "margin": MARGIN, "alpha": ALPHA, "n_boot": args.n_boot,
           "metrics": metrics, "comparisons": comparisons, "verdicts": verdicts, "tables": method_tables(root)}
    (root / "round2_verdicts.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({"complete_protocol": complete, "verdicts": verdicts}, indent=2))


if __name__ == "__main__":
    main()
