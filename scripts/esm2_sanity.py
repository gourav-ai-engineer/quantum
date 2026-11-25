"""E6-S implementation sanity check (docs/EXPLORATION_LOG.md, pre-specified 2026-11-25). Not a result; E6 verdict stays STOP.
Same code path as scripts/esm2_pilot.py run(): first 5 assays, all 1,024 calibration sequences, GRP, budgets 1.0, 0.5, 0.9, 0.8.
Usage: python scripts/esm2_sanity.py [--threads 8]"""
import argparse
import gc
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import esm2_pilot as P  # noqa: E402

BUDGETS, N_ASSAYS, PILOT = (1.0, 0.5, 0.9, 0.8), 5, P.RES / "53c73f70" / "results.json"


def main():
    from qfc import esm_pilot as ep, grp
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=8)
    torch.set_num_threads(ap.parse_args().threads)
    inputs = json.loads((P.RES / "inputs.json").read_text())
    calib = json.loads((P.RES / "calibration.json").read_text())
    assert P.sha256(P.RES / "calibration.json") == inputs["uniref"]["calibration_sha256"]
    out = P.RES / f"sanity_{P.git('rev-parse', '--short', 'HEAD')}"
    out.mkdir(parents=True, exist_ok=True)
    teacher, _small, tok = P.load_models(inputs)
    _, df = P.load_pg(inputs["proteingym"]["revision"], list(inputs["proteingym"]["files"]))
    ids = [m["DMS_id"] for m in inputs["assays"]][:N_ASSAYS]
    data = {a: ep.single_mutants(df[df.DMS_id == a]) for a in ids}
    del df
    batches = ep.make_batches([s for _, s in calib], tok, 8)
    res = {"assays": ids, "n_calibration": len(calib), "arms": {}}

    def record(name, model, extra=None):
        rho = {}
        for a, (wt, muts, y) in data.items():
            s = ep.wt_marginal_scores(model, tok, wt, muts)
            rho[a] = float(np.corrcoef(pd.Series(s).rank().to_numpy(), pd.Series(y).rank().to_numpy())[0, 1])
        res["arms"][name] = {"rho": rho, "mean_rho": float(np.mean(list(rho.values()))), **(extra or {})}
        (out / "results.json").write_text(json.dumps(res, indent=1))
        print(f"{name:12s} mean rho {res['arms'][name]['mean_rho']:.4f}  {[round(v, 4) for v in rho.values()]}", flush=True)

    record("D", teacher)
    cache = Path(tempfile.mkdtemp(prefix="esm2_grams_"))
    try:
        stats = grp.teacher_stats(teacher, batches, "cpu", seed=0, store_dir=cache)
        for b in BUDGETS:
            heads, neurons, _ = ep.select_grp(stats, teacher, b)
            kept = {"heads": sum(map(len, heads.values())), "neurons": sum(map(len, neurons.values()))}
            print(f"budget {b}: kept {kept}", flush=True)
            student = ep.build_student(teacher, heads, neurons)
            extra = {"budget": b, "kept": kept, "mac_ratio": ep.mac_per_token(student) / ep.mac_per_token(teacher)}
            record(f"A|{b}", student, extra)
            grp.sequential_refit(student, teacher, batches, "cpu", 1e-4, drift_correct=True)
            record(f"B|{b}", student, extra)
            del student
            gc.collect()
    finally:
        stats = None
        gc.collect()
        shutil.rmtree(cache, ignore_errors=True)

    # checks exactly as pre-specified in E6-S
    pilot, arm = json.loads(PILOT.read_text())["arms"], res["arms"]
    diff = lambda x, y: max(abs(x["rho"][a] - y["rho"][a]) for a in ids)  # noqa: E731
    cfg = teacher.config
    full = {"heads": cfg.num_hidden_layers * cfg.num_attention_heads, "neurons": cfg.num_hidden_layers * cfg.intermediate_size}
    s0 = {k: diff(arm[f"{k}|0.5"], pilot[f"{k}|grp|0.5"]) for k in "AB"}
    s1 = {"keeps_all": arm["A|1.0"]["kept"] == full, "A_vs_D": diff(arm["A|1.0"], arm["D"]), "B_vs_D": diff(arm["B|1.0"], arm["D"])}
    s2 = {b: arm[f"B|{b}"]["mean_rho"] - arm["B|0.5"]["mean_rho"] for b in (0.8, 0.9)}
    res["checks"] = {
        "S0": {"max_abs_diff": s0, "pass": max(s0.values()) <= 1e-3},
        "S1": {**s1, "pass": s1["keeps_all"] and s1["A_vs_D"] <= 1e-4 and s1["B_vs_D"] <= 0.01},
        "S2": {"B_minus_B05": s2, "pass": min(s2.values()) >= -0.02},
    }
    (out / "results.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res["checks"], indent=1))


if __name__ == "__main__":
    main()
