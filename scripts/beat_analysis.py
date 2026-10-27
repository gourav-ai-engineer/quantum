"""Compare beat_run.py results with the official baselines (results/beat_explore/baselines_v15.json) per task and
budget, over the same 5 calibration seeds (EXPLORATION_LOG "beats both": mean strictly better than BOTH baselines on
held-out KL and on validation accuracy at 0.5 AND 0.4).

    python scripts/beat_analysis.py results/beat_explore/<run_dir> [more dirs] > results/beat_explore/<run>_table.txt
"""
import glob
import json
import sys
from collections import defaultdict

base = json.load(open("results/beat_explore/baselines_v15.json"))["runs"]
rec = defaultdict(dict)  # (method, task, c) -> {seed: record}
for d in sys.argv[1:]:
    for f in glob.glob(d + "/*.json"):
        r = json.load(open(f))
        if r.get("mode") == "full":
            rec[(r["method"], r["task"], r["constraint"])][r["seed"]] = r
mean = lambda v: sum(v) / len(v)
tasks = sorted({k[1] for k in rec})
beat = defaultdict(dict)
for task in tasks:
    for c in (0.5, 0.4):
        kp, kw = base[f"kprune|{task}|{c}"], base[f"kwon|{task}|{c}"]
        teacher = next(r for k, v in rec.items() if k[1] == task for r in v.values())["unpruned_valid"]["accuracy"]
        print(f"\n== {task} c={c}   teacher acc {teacher:.4f}")
        print(f"{'method':10s} {'n':>2s} {'acc':>7s} {'dAcc/kprune':>11s} {'seedwins':>8s} {'KL':>8s} {'KL/kprune':>9s}")
        print(f"{'kprune':10s} {len(kp['acc']):2d} {kp['mean']:7.4f} {'':>11s} {'':>8s} {kp['heldout_kl_mean']:8.4f} {1:9.2f}")
        print(f"{'kwon':10s} {len(kw['acc']):2d} {kw['mean']:7.4f} {'':>11s} {'':>8s} {kw['heldout_kl_mean']:8.4f} "
              f"{kw['heldout_kl_mean'] / kp['heldout_kl_mean']:9.2f}")
        for (m, t, cc), runs in sorted(rec.items()):
            if t != task or cc != c:
                continue
            seeds = sorted(runs)
            acc = [runs[s]["valid"]["accuracy"] for s in seeds]
            kl = [runs[s]["heldout_dist"] for s in seeds]
            kacc = {s: a for s, a in zip(kp["seeds"], kp["acc"])}
            wins = sum(a > kacc[s] for s, a in zip(seeds, acc))
            print(f"{m:10s} {len(seeds):2d} {mean(acc):7.4f} {mean(acc) - kp['mean']:+11.4f} {wins:>5d}/{len(seeds)} "
                  f"{mean(kl):8.4f} {mean(kl) / kp['heldout_kl_mean']:9.2f}")
            ok = (mean(acc) > max(kp["mean"], kw["mean"])) and (mean(kl) < min(kp["heldout_kl_mean"], kw["heldout_kl_mean"]))
            beat[m].setdefault(task, {})[c] = ok
print("\nBEATS BOTH (acc and KL, same seeds) per method/task at 0.5 and 0.4:")
for m, d in sorted(beat.items()):
    print(m, {t: all(v.values()) and len(v) == 2 for t, v in d.items()})
