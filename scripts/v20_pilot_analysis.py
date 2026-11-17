"""V20 PILOT decision (docs/PROJECT_STATE.md, "V20 PILOT pre-registration"). Validation only; never a paper result.

M = mask only, R_L = repair calibrated in language L. Accuracy grids are [seed, example]; examples are parallel across
languages (same ids), so every contrast is paired and uses ONE resample of (seeds, examples) for all its terms.
  P1(T) = [R_en(en) - M(en)] - [R_en(T) - M(T)]       pooled P1 = mean over T in {as, bn, hi}
  P2(T) = R_T(T) - R_en(T)
GO (for d2i_r): pooled P1 95% CI > 0 and point >= 0.01, AND P2 CI > 0 for >= 2 of {as, bn, hi}.

    python scripts/v20_pilot_analysis.py <pilot output dir> [--out decision.json]
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

DEV = ("as", "bn", "hi")
MIN_P1 = 0.01


def linear_bootstrap(terms, n_boot=10000, seed=1234, chunk=250):
    """terms: list of (coef, correct[S, N] 0/1). Statistic = sum coef * accuracy(grid). Paired cluster bootstrap over
    (seed x example): each resample draws S seeds and N examples with replacement, shared by every term."""
    grids = [np.asarray(g, dtype=np.float64) for _, g in terms]
    coefs = np.array([c for c, _ in terms], dtype=np.float64)
    S, N = grids[0].shape
    if any(g.shape != (S, N) for g in grids):
        raise ValueError("all grids must be [S, N]")
    obs = float(sum(c * g.mean() for c, g in zip(coefs, grids)))
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for start in range(0, n_boot, chunk):
        k = min(chunk, n_boot - start)
        w = rng.multinomial(S, np.full(S, 1.0 / S), size=k).astype(np.float64)
        v = rng.multinomial(N, np.full(N, 1.0 / N), size=k).astype(np.float64)
        boot[start:start + k] = sum(c * ((w @ g) * v).sum(1) / (S * N) for c, g in zip(coefs, grids))
    return {"observed": obs, "ci_low": float(np.quantile(boot, 0.025)), "ci_high": float(np.quantile(boot, 0.975))}


def load(d):
    """(repair, cal_lang) -> {eval_lang: correct[S, N]} with seeds sorted; checks the seed sets agree."""
    runs = {}
    for f in glob.glob(os.path.join(d, "*_seed*_c*.json")):
        r = json.load(open(f))
        lab = np.asarray(r["labels"])
        runs.setdefault((r["repair"], r["cal_lang"]), {})[r["seed"]] = {k: np.asarray(v) == lab for k, v in r["preds"].items()}
    out, seeds = {}, None
    for key, by_seed in runs.items():
        s = sorted(by_seed)
        seeds = seeds or s
        if s != seeds:
            raise SystemExit(f"{key}: seeds {s} != {seeds}")
        out[key] = {lang: np.stack([by_seed[x][lang] for x in s]) for lang in by_seed[s[0]]}
    return out, seeds


def p1_terms(g, rep, T):
    return [(1, g[(rep, "en")]["en"]), (-1, g[("none", "none")]["en"]), (-1, g[(rep, "en")][T]), (1, g[("none", "none")][T])]


def decide(g, rep="d2i_r", n_boot=10000):
    pooled = [(c / len(DEV), x) for T in DEV for c, x in p1_terms(g, rep, T)]
    res = {"P1": {T: linear_bootstrap(p1_terms(g, rep, T), n_boot) for T in DEV},
           "P1_pooled": linear_bootstrap(pooled, n_boot),
           "P2": {T: linear_bootstrap([(1, g[(rep, T)][T]), (-1, g[(rep, "en")][T])], n_boot) for T in DEV}}
    p1_ok = res["P1_pooled"]["ci_low"] > 0 and res["P1_pooled"]["observed"] >= MIN_P1
    p2_ok = sum(res["P2"][T]["ci_low"] > 0 for T in DEV) >= 2
    res["GO"] = bool(p1_ok and p2_ok)
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dir")
    p.add_argument("--out", default=None)
    p.add_argument("--n-boot", type=int, default=10000)
    a = p.parse_args()
    g, seeds = load(a.dir)
    out = {"seeds": seeds, "rule_for": "d2i_r", "d2i_r": decide(g, "d2i_r", a.n_boot)}
    if all(("d2i_ri", L) in g for L in ("en",) + DEV):
        out["d2i_ri_descriptive"] = decide(g, "d2i_ri", a.n_boot)
    out["mean_accuracy"] = {f"{r}/{c}": {lang: float(x.mean()) for lang, x in v.items()} for (r, c), v in sorted(g.items())}
    print(json.dumps({"seeds": seeds, "GO": out["d2i_r"]["GO"], "P1_pooled": out["d2i_r"]["P1_pooled"],
                      "P2": out["d2i_r"]["P2"]}, indent=1))
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
