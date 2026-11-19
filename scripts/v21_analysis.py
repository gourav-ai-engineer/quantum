"""V21 confirmatory analysis (docs/V21_PREREG_DRAFT_2026-11-18.md, section 3). Run ONCE, after every unit passes the
integrity gate. Committed, with synthetic tests (tests/test_v21_analysis.py), before any V21 kernel runs.

Input: <root>/<model>_t<teacher seed>/<condition>_seed<s>.json, one per (unit, condition, calibration seed), each
{"condition", "seed", "labels": [N], "preds": {lang: [N]}, "unpruned_preds": {lang: [N]}}; rows are the same
label-parallel examples in every language (paired by example id = row index). Conditions: mask_only, kprune, kwon,
d2i_r_en, d2i_r_mixed, d2i_ri_en.

Every statistic is a linear combination of accuracies. Per unit: paired cluster bootstrap over (calibration seed x
example), 10,000 resamples, seed 1234 + unit index (units sorted by name). Pooled per model: mean over the model's units of
the unit statistics and of the unit bootstrap draws (units resampled independently; teacher variance is NOT in the CI,
hence the all-teachers sign rule). No pooling across models.

    python scripts/v21_analysis.py <root> --out <json>
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

from qfc.metrics import bootstrap_p_le, holm_adjust

D = ("as", "bn", "hi")
N_ = ("gu", "kn", "ml", "mr", "or", "pa", "ta", "te")
T = D + N_
MARGIN, ALPHA = 0.015, 0.05
H1_CI, H1_POINT = 0.02, 0.05
CONDITIONS = ("mask_only", "kprune", "kwon", "d2i_r_en", "d2i_r_mixed", "d2i_ri_en")


def load(root):
    """{unit: {"cond": {condition: {lang: correct[S, N]}}, "U": {lang: correct[N]}, "seeds": [...], "n": N}}."""
    units = {}
    for udir in sorted(d for d in glob.glob(os.path.join(root, "*_t*")) if os.path.isdir(d)):
        recs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(udir, "*_seed*.json")))]
        if not recs:
            continue
        labels = np.asarray(recs[0]["labels"])
        if any(np.asarray(r["labels"]).tolist() != labels.tolist() for r in recs):
            raise SystemExit(f"{udir}: label vectors differ between files (example pairing broken)")
        by = {}
        for r in recs:
            by.setdefault(r["condition"], {})[r["seed"]] = {k: np.asarray(v) == labels for k, v in r["preds"].items()}
        seeds = sorted(by[next(iter(by))])
        cond = {}
        for c, s_map in by.items():
            if sorted(s_map) != seeds:
                raise SystemExit(f"{udir}/{c}: seeds {sorted(s_map)} != {seeds}")
            cond[c] = {L: np.stack([s_map[s][L] for s in seeds]) for L in s_map[seeds[0]]}
        U = {k: np.asarray(v) == labels for k, v in recs[0]["unpruned_preds"].items()}
        units[os.path.basename(udir)] = {"cond": cond, "U": U, "seeds": seeds, "n": len(labels)}
    return units


def model_of(unit):
    return unit.rsplit("_t", 1)[0]


def boot_linear(terms, n_boot, seed, chunk=250):
    """terms: (coef, correct[S, N]); statistic = sum coef * accuracy. One shared (seed x example) resample per draw."""
    grids = [np.asarray(g, dtype=np.float64) for _, g in terms]
    coefs = [c for c, _ in terms]
    S, N = grids[0].shape
    obs = float(sum(c * g.mean() for c, g in zip(coefs, grids)))
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for i in range(0, n_boot, chunk):
        k = min(chunk, n_boot - i)
        w = rng.multinomial(S, np.full(S, 1.0 / S), size=k).astype(np.float64)
        v = rng.multinomial(N, np.full(N, 1.0 / N), size=k).astype(np.float64)
        out[i:i + k] = sum(c * ((w @ g) * v).sum(1) / (S * N) for c, g in zip(coefs, grids))
    return obs, out


def _u(unit, L, S):
    return np.tile(unit["U"][L], (S, 1))


def terms_for(name, unit, langs):
    """Linear terms of statistic `name` on one unit, target languages `langs`."""
    c, S, k = unit["cond"], len(unit["seeds"]), 1.0 / len(langs)
    def disparity(X):  # mean_L r_X(L) - r_X(en), r_X(L) = U(L) - X(L)
        t = [(k, _u(unit, L, S)) for L in langs] + [(-k, c[X][L]) for L in langs]
        return t + [(-1.0, _u(unit, "en", S)), (1.0, c[X]["en"])]
    def mean_diff(A, B):
        return [(k, c[A][L]) for L in langs] + [(-k, c[B][L]) for L in langs]
    return {"S_mask": lambda: disparity("mask_only"),
            "R_repair": lambda: disparity("d2i_r_en"),
            "G_mixed_gain": lambda: mean_diff("d2i_r_mixed", "d2i_r_en"),
            "E_mixed_en_cost": lambda: [(1.0, c["d2i_r_mixed"]["en"]), (-1.0, c["d2i_r_en"]["en"])],
            "NI_kprune_en": lambda: [(1.0, c["d2i_ri_en"]["en"]), (-1.0, c["kprune"]["en"])],
            "NI_kprune_T": lambda: mean_diff("d2i_ri_en", "kprune"),
            "SUP_kwon_T": lambda: mean_diff("d2i_ri_en", "kwon")}[name]()


NEEDS = {"S_mask": ("mask_only",), "R_repair": ("d2i_r_en",), "G_mixed_gain": ("d2i_r_mixed", "d2i_r_en"),
         "E_mixed_en_cost": ("d2i_r_mixed", "d2i_r_en"), "NI_kprune_en": ("d2i_ri_en", "kprune"),
         "NI_kprune_T": ("d2i_ri_en", "kprune"), "SUP_kwon_T": ("d2i_ri_en", "kwon")}


def pooled(units, names, stat, langs, n_boot, index):
    """Model-level pooled statistic: mean of unit statistics and of unit bootstrap draws."""
    obs, boots, per = [], [], {}
    for u in names:
        o, b = boot_linear(terms_for(stat, units[u], langs), n_boot, 1234 + index[u])
        obs.append(o)
        boots.append(b)
        per[u] = o
    b = np.mean(boots, 0)
    return {"observed": float(np.mean(obs)), "ci_low": float(np.quantile(b, 0.025)),
            "ci_high": float(np.quantile(b, 0.975)), "per_unit": per,
            "p_le_zero": bootstrap_p_le(b, 0.0), "p_le_margin": bootstrap_p_le(b, -MARGIN)}


def analyse_model(units, names, n_boot, index):
    if len(names) < 2:
        return {"units": names, "stats": {s: None for s in NEEDS},
                **{"H21-1": {"run": False, "supported": False},
                   "H21-2": {"run": False}, "H21-3(i)": {"run": False},
                   "H21-3(ii)": {"run": False}, "H21-4(i)": {"run": False},
                   "H21-4(ii)": {"run": False},
                   "H21-3": {"supported": False}, "robustness": {},
                   "descriptive": {}, "worst_language_residual": {}}}
    have = lambda stat: all(all(c in units[u]["cond"] for c in NEEDS[stat]) for u in names)
    st = {s: (pooled(units, names, s, T, n_boot, index) if have(s) else None) for s in NEEDS}
    signs = lambda s: all(v > 0 for v in st[s]["per_unit"].values())
    out = {"units": names, "stats": st}
    s1 = st["S_mask"]
    out["H21-1"] = {"supported": bool(s1 and s1["ci_low"] > H1_CI and s1["observed"] >= H1_POINT and signs("S_mask"))}
    # secondary family (Holm over the members that were run); H21-4(i) needs both parts -> max p (intersection-union)
    fam = {}
    if st["R_repair"]:
        fam["H21-2"] = (st["R_repair"]["p_le_zero"], st["R_repair"]["ci_low"] > 0 and signs("R_repair"))
    if st["G_mixed_gain"]:
        fam["H21-3(i)"] = (st["G_mixed_gain"]["p_le_zero"], st["G_mixed_gain"]["ci_low"] > 0 and signs("G_mixed_gain"))
        fam["H21-3(ii)"] = (st["E_mixed_en_cost"]["p_le_margin"], st["E_mixed_en_cost"]["ci_low"] > -MARGIN)
    if st["NI_kprune_en"]:
        fam["H21-4(i)"] = (max(st["NI_kprune_en"]["p_le_margin"], st["NI_kprune_T"]["p_le_margin"]),
                           st["NI_kprune_en"]["ci_low"] > -MARGIN and st["NI_kprune_T"]["ci_low"] > -MARGIN)
    if st["SUP_kwon_T"]:
        fam["H21-4(ii)"] = (st["SUP_kwon_T"]["p_le_zero"], st["SUP_kwon_T"]["ci_low"] > 0)
    adj = dict(zip(fam, holm_adjust([p for p, _ in fam.values()]))) if fam else {}
    for h in ("H21-2", "H21-3(i)", "H21-3(ii)", "H21-4(i)", "H21-4(ii)"):
        out[h] = ({"run": False} if h not in fam else
                  {"run": True, "p": fam[h][0], "holm_p": adj[h], "ci_and_sign_ok": bool(fam[h][1]),
                   "supported": bool(fam[h][1] and adj[h] < ALPHA)})
    out["H21-3"] = {"supported": bool(out["H21-3(i)"].get("supported") and out["H21-3(ii)"].get("supported"))}
    # robustness (reported, not decisive): every pooled statistic on N only and on D only
    rob = {}
    for s in NEEDS:
        if st[s] is None or s in ("E_mixed_en_cost", "NI_kprune_en"):
            continue
        on = {g: float(np.mean([boot_linear(terms_for(s, units[u], langs), 1, 0)[0] for u in names]))
              for g, langs in (("N", N_), ("D", D))}
        rob[s] = {**on, "sign_flip_N_vs_D": bool(np.sign(on["N"]) != np.sign(on["D"]))}
    out["robustness"] = rob
    out["descriptive"] = {u: {c: {L: float(x.mean()) for L, x in v.items()} for c, v in units[u]["cond"].items()}
                          | {"unpruned": {L: float(x.mean()) for L, x in units[u]["U"].items()}} for u in names}
    out["worst_language_residual"] = {
        u: {c: max(((float(units[u]["U"][L].mean() - v[L].mean()), L) for L in T if L in v)) for c, v in
            units[u]["cond"].items()} for u in names}
    return out


def analyse(root, n_boot=10000):
    units = load(root)
    index = {u: i for i, u in enumerate(sorted(units))}
    # Keep both preregistered models in the report, including a model with no
    # gate-passing unit folders. The caller supplies only units that passed
    # the integrity/teacher gates.
    models = sorted({"muril", "mbert"} | {model_of(u) for u in units})
    res = {m: analyse_model(units, sorted(u for u in units if model_of(u) == m), n_boot, index) for m in models}
    muril = next((m for m in models if "muril" in m), None)
    res["H21-1_overall"] = {"decisive_model": muril, "supported": bool(muril and res[muril]["H21-1"]["supported"])}
    res["n_boot"] = n_boot
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root")
    p.add_argument("--out", required=True)
    p.add_argument("--n-boot", type=int, default=10000)
    a = p.parse_args()
    r = analyse(a.root, a.n_boot)
    json.dump(r, open(a.out, "w"), indent=1)
    print(json.dumps({m: {h: v.get("supported") for h, v in r[m].items() if h.startswith("H21")}
                      for m in r if isinstance(r[m], dict) and "units" in r[m]}, indent=1))


if __name__ == "__main__":
    main()
