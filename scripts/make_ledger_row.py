"""Print the docs/PROJECT_STATE.md Results-ledger row for one results directory.

    python scripts/make_ledger_row.py results/<experiment>/<commit>

Every number is read from the JSON files in that directory (plus run_meta.json written by
scripts/run_gpu.sh); nothing is typed by hand. The verdict column only records what the
files support; applying the decision rule/tree is the owner's step.

Refuses: smoke runs (unless --allow-smoke), runs whose recorded library versions are not
the pinned ones (head_mask unreliable), and CPU runs of full experiments.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean, pstdev

PINS = {"torch": "2.6.0", "transformers": "4.51.3", "datasets": "3.6.0", "pyarrow": "24.0.0"}
NOT_A_RESULT = ("run_meta.json", "aggregate.json")


class LedgerError(RuntimeError):
    pass


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _f(x: float) -> str:
    return f"{x:.4f}"


def _ms(values) -> str:
    values = list(values)
    return f"{_f(mean(values))}+/-{_f(pstdev(values) if len(values) > 1 else 0.0)}"


def _random_cell(entry: dict) -> str:
    d = entry["distribution"]["accuracy"]
    return f"{_f(d['mean'])} [min {_f(d['min'])}, max {_f(d['max'])}, n={d['n']}]"


def validate_meta(meta: dict, allow_smoke: bool) -> list[str]:
    """Raise LedgerError for unusable runs; return warnings."""
    if meta.get("smoke") and not allow_smoke:
        raise LedgerError("smoke run: engineering check, never a result (use --allow-smoke to inspect)")
    warnings = []
    for name, want in PINS.items():
        have = (meta.get("versions", {}).get(name) or "").split("+")[0]
        if have != want:
            raise LedgerError(f"recorded {name}=={have or 'missing'}, expected =={want}; results not trustworthy")
    if not meta.get("cuda_available") and not meta.get("smoke"):
        raise LedgerError("full experiment recorded without CUDA")
    if meta.get("dirty"):
        warnings.append("WARNING: run was made from a dirty working tree")
    return warnings


def key_numbers_v10(run_dir: Path) -> str:
    files = sorted(p for p in run_dir.glob("*_seed*.json") if p.name not in NOT_A_RESULT)
    if not files:
        raise LedgerError("no <task>_seed<N>.json files found")
    by_task: dict[str, list[dict]] = {}
    for p in files:
        r = _load(p)
        by_task.setdefault(r["task"], []).append(r)
    lines = []
    for task, runs in sorted(by_task.items()):
        seeds = ",".join(str(r["seed"]) for r in sorted(runs, key=lambda r: r["seed"]))
        base = runs[0]["baseline"]
        head = f"{task} (eval n={runs[0]['evaluation_size']}, seeds {seeds}): unpruned acc {_f(base['accuracy'])}, loss {_f(base['loss'])}"
        if "f1" in base:
            head += f", F1 {_f(base['f1'])}"
        parts = []
        for m in runs[0]["methods"]:
            if m == "Random":
                parts.append(f"Random {_random_cell(runs[0]['methods'][m])}")
            else:
                parts.append(f"{m} {_ms(r['methods'][m]['accuracy'] for r in runs)}")
        lines.append(head + ". Accuracy mean+/-std over seeds: " + "; ".join(parts))
    return "<br>".join(lines)


def key_numbers_v6(run_dir: Path) -> str:
    s = _load(run_dir / "summary.json")
    lines = []
    for task, td in s["tasks"].items():
        base = td["baseline"]
        head = f"{task} (eval n={td['validation_size']}): unpruned acc {_f(base['accuracy'])}, loss {_f(base['loss'])}"
        if "f1" in base:
            head += f", F1 {_f(base['f1'])}"
        lines.append(head)
        for k, bd in td["budgets"].items():
            parts = []
            for m, e in bd.items():
                if not isinstance(e, dict):
                    continue
                parts.append(f"Random {_random_cell(e)}" if m == "Random" else f"{m} {_f(e['accuracy'])}")
            lines.append(f"{task} k={k} accuracy: " + "; ".join(parts))
    return "<br>".join(lines)


def key_numbers_v5(run_dir: Path) -> str:
    s = _load(run_dir / "summary.json")
    base = s["runs"][0]["baseline"]
    head = (
        f"mrpc (eval n={s['evaluation_size']}, seeds {','.join(map(str, s['seeds']))}): unpruned acc "
        f"{_f(base['accuracy'])}, loss {_f(base['loss'])}, F1 {_f(base['f1'])}"
    )
    parts = [
        f"{m} acc {_f(a['accuracy_mean'])}+/-{_f(a['accuracy_std'])} F1 {_f(a['f1_mean'])}+/-{_f(a['f1_std'])}"
        for m, a in s["aggregate"].items()
    ]
    return head + ". Mean+/-std over calibration seeds: " + "; ".join(parts)


def _layer_counts(layers: dict, name: str) -> dict:
    below = include0 = degenerate = 0
    for data in layers.values():
        ci = data["spearman_coverage_vs_loss_ci"][name]
        if ci["degenerate"]:
            degenerate += 1
        elif ci["ci_high"] < 0:
            below += 1
        elif ci["ci_low"] <= 0 <= ci["ci_high"]:
            include0 += 1
    return {"below0": below, "includes0": include0, "degenerate": degenerate, "layers": len(layers)}


def key_numbers_v11(run_dir: Path) -> tuple[str, str]:
    s = _load(run_dir / "summary.json")
    lines = []
    criterion_a = []
    for task, td in s["tasks"].items():
        counts = {n: _layer_counts(td["layers"], n) for n in ("fidelity", "hilbert_schmidt", "cosine")}
        criterion_a.append(counts["fidelity"]["below0"] >= 8 and counts["fidelity"]["layers"] == 12)
        base = td["baseline"]
        head = (
            f"{task} (eval n={td['evaluation_size']}, {s['random_subsets_per_layer']} subsets/layer, "
            f"{s['bootstrap_resamples']} bootstrap): unpruned acc {_f(base['accuracy'])}, loss {_f(base['loss'])}"
        )
        if "f1" in base:
            head += f", F1 {_f(base['f1'])}"
        parts = [
            f"{n}: CI entirely <0 in {c['below0']}/{c['layers']} layers, includes 0 in {c['includes0']}, degenerate {c['degenerate']}"
            for n, c in counts.items()
        ]
        pooled = td["pooled"]["spearman_coverage_vs_loss_ci"]["fidelity"]
        pooled_txt = (
            f"pooled fidelity rho {_f(pooled['estimate'])} [{_f(pooled['ci_low'])}, {_f(pooled['ci_high'])}]"
            if pooled["ci_low"] is not None
            else f"pooled fidelity rho {_f(pooled['estimate'])} [CI undefined]"
        )
        lines.append(head + ". " + "; ".join(parts) + "; " + pooled_txt)
    met = "MET" if criterion_a and all(criterion_a) else "NOT MET"
    verdict = (
        f"RULE INPUTS RECORDED. Criterion 'fidelity CI entirely below 0 in >= 8 of 12 layers on BOTH tasks': {met}. "
        "The classical-similarity comparison and the final verdict are for the owner (see decision rule)"
    )
    return "<br>".join(lines), verdict


def key_numbers_round2(run_dir: Path) -> tuple[str, str]:
    """Per-task method table and the verdicts that scripts/round2_analysis.py wrote (rules applied there)."""
    v = _load(run_dir / "round2_verdicts.json")
    lines = []
    for task, t in v["tables"].items():
        m, un = t["metric"], t["unpruned"]
        parts = [f"{n} {_f(e[m])} (loss {_f(e['loss'])}, AUROC {_f(e['auroc'])})" for n, e in t["methods"].items()]
        rq = t.get("random_quantiles", {}).get(m)
        rand = f"; Random median {_f(rq['q50'])} [2.5% {_f(rq['q2.5'])}, 97.5% {_f(rq['q97.5'])}]" if rq else ""
        lines.append(
            f"{task} (eval n={t['evaluation_size']}, seeds {','.join(str(s) for s in t['seeds'])}): unpruned {m} "
            f"{_f(un[m])}, loss {_f(un['loss'])}, AUROC {_f(un['auroc'])}. Mean over seeds, {m}: " + "; ".join(parts) + rand
        )
    verdict = "ROUND 2 RULES APPLIED by scripts/round2_analysis.py (complete_protocol=%s): %s" % (
        v["complete_protocol"], json.dumps(v["verdicts"], sort_keys=True))
    return "<br>".join(lines), verdict


EXPERIMENTS = {
    "round2_h1_h2_h6": ("step20_round2.py", "Round 2 (H1/H2/H6)"),
    "v10_confirmatory": ("step18_v10_confirmatory.py", "V10 confirmatory"),
    "v6_budget_response": ("step14_budget_response.py", "V6 budget response"),
    "v5_mrpc_stability": ("step13_mrpc_stability.py", "V5 MRPC stability"),
    "v11_objective_alignment": ("step19_objective_alignment.py", "V11 objective alignment"),
}


def row_for(run_dir: str | Path, allow_smoke: bool = False) -> tuple[str, list[str]]:
    run_dir = Path(run_dir)
    meta_path = run_dir / "run_meta.json"
    if not meta_path.exists():
        raise LedgerError(f"{meta_path} not found (was this directory produced by scripts/run_gpu.sh?)")
    meta = _load(meta_path)
    warnings = validate_meta(meta, allow_smoke)
    experiment = meta["experiment"]
    if experiment not in EXPERIMENTS:
        raise LedgerError(f"unknown experiment {experiment!r}")
    script, label = EXPERIMENTS[experiment]
    verdict = "RECORDED; verdict pending owner review (apply the decision tree in docs/PROJECT_STATE.md)"
    if experiment == "v10_confirmatory":
        numbers = key_numbers_v10(run_dir)
    elif experiment == "v6_budget_response":
        numbers = key_numbers_v6(run_dir)
    elif experiment == "v5_mrpc_stability":
        numbers = key_numbers_v5(run_dir)
    elif experiment == "round2_h1_h2_h6":
        numbers, verdict = key_numbers_round2(run_dir)
    else:
        numbers, verdict = key_numbers_v11(run_dir)
    env = meta["versions"]
    config = (
        f"{meta['args'].get('flags', '')} ; env: torch {env['torch']}, transformers {env['transformers']}, "
        f"datasets {env['datasets']}, pyarrow {env['pyarrow']}; device {meta.get('gpu_name') or 'cpu'}; "
        f"run {meta['date_utc']}; JSON dir {run_dir.as_posix()}"
    )
    commit = meta["commit_short"] + (" (dirty)" if meta.get("dirty") else "")
    cells = [label, script, config, commit, numbers, verdict]
    return "| " + " | ".join(c.replace("|", "/") for c in cells) + " |", warnings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir")
    parser.add_argument("--allow-smoke", action="store_true")
    args = parser.parse_args()
    try:
        row, warnings = row_for(args.run_dir, args.allow_smoke)
    except (LedgerError, FileNotFoundError, KeyError) as exc:
        print(f"make_ledger_row: {exc}", file=sys.stderr)
        return 1
    for w in warnings:
        print(w, file=sys.stderr)
    print(row)
    return 0


if __name__ == "__main__":
    sys.exit(main())
