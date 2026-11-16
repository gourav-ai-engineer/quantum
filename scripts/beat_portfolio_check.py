"""Dev-only check (existing JSONs, no new runs): per (task, seed, c) pick d2i_ri vs grp_ri by held-out-TRAIN
distance (KL; MSE for STS-B); compare the pick's validation metric to official K-prune / Kwon per seed."""
import json, os
B = json.load(open("results/beat_explore/baselines_v15.json"))["runs"]
DIRS = {"mrpc": "results/beat_explore/E1_align/e0063df", "sst2": "results/beat_explore/E1_align/e0063df",
        "stsb": "results/beat_explore/D_stsb/ed9f196"}
SEEDS = [7, 42, 77, 123, 2024]
def metric(v):
    for k in ("accuracy", "pearson"):
        if k in v: return v[k]
    raise KeyError(list(v))
for task, d in DIRS.items():
    for c in (0.5, 0.4):
        kp, kw = B[f"kprune|{task}|{c}"], B[f"kwon|{task}|{c}"]
        rows, inconsistent = [], 0
        for i, s in enumerate(SEEDS):
            m = {}
            for meth in ("d2i_ri", "grp_ri"):
                r = json.load(open(os.path.join(d, f"{meth}_{task}_seed{s}_c{c}.json")))
                m[meth] = (r["heldout_dist"], metric(r["valid"]))
            pick = min(m, key=lambda k: m[k][0])
            other = "grp_ri" if pick == "d2i_ri" else "d2i_ri"
            if m[pick][1] < m[other][1] - 1e-12: inconsistent += 1
            assert kp["seeds"][i] == s
            rows.append((s, pick, m[pick][1], kp["acc"][i], kw["acc"][i], m["d2i_ri"][1], m["grp_ri"][1]))
        pf = sum(r[2] for r in rows) / 5; kpm = sum(r[3] for r in rows) / 5; kwm = sum(r[4] for r in rows) / 5
        wins = sum(r[2] > r[3] + 1e-12 for r in rows); ties = sum(abs(r[2] - r[3]) <= 1e-12 for r in rows)
        picks = ",".join(("D" if r[1] == "d2i_ri" else "G") for r in rows)
        print(f"{task:5s} c{c}: portfolio {pf:.4f} | K-prune {kpm:.4f} | Kwon {kwm:.4f} | delta vs KP {pf-kpm:+.4f} | "
              f"seed W/T/L vs KP {wins}/{ties}/{5-wins-ties} | picks {picks} | KL-pick worse-acc {inconsistent}/5")
