"""POST-HOC, UNREGISTERED (V18): held-out-train logit KL(teacher||student) of d2i-RI vs K-prune vs Kwon on QNLI/MNLI-m/QQP at
0.5 MACs, from dumps already produced by the V18 kernels. No new runs. Held-out = train rows not used for calibration, but the
teacher was fine-tuned on them, so this is a calibration-side quantity, not generalization evidence. Exploratory.

    python scripts/v18_heldout_posthoc.py results/beat_confirm/h18_posthoc_heldout.json
"""
import json
import sys

import torch

D = "results/beat_confirm"
ROOT = {"qnli": f"{D}/raw_qnli/conf_qnli", "mnli": f"{D}/merged/conf_mnli", "qqp": f"{D}/merged/conf_qqp"}
SEEDS = [7, 42, 77, 123, 2024]


def kl(d):
    lt = torch.load(f"{d}/v15_teacher_heldout.pt", weights_only=True)["logits"].log_softmax(-1)
    ls = torch.load(f"{d}/v15_student_heldout.pt", weights_only=True)["logits"].log_softmax(-1)
    return float((lt.exp() * (lt - ls)).sum(-1).mean())


out = {"note": "post-hoc, unregistered, exploratory; KL(teacher||student) on 512 held-out TRAIN rows (not validation)", "tasks": {}}
for t, r in ROOT.items():
    row = {"d2i_ri": [], "kprune": [], "kwon": []}
    for s in SEEDS:
        row["d2i_ri"].append(json.load(open(f"{r}/beat/d2i_ri_{t}_seed{s}_c0.5.json"))["heldout_dist"])
        row["kprune"].append(kl(f"{r}/bl/kprune_out/kprune_{t}_seed{s}_c0.5"))
        row["kwon"].append(kl(f"{r}/bl/kwon_{t}_seed{s}_c0.5"))
    m = {k: sum(v) / len(v) for k, v in row.items()}
    row["mean"] = m
    row["ratio_d2i_over_kprune_per_seed"] = [a / b for a, b in zip(row["d2i_ri"], row["kprune"])]
    row["ratio_means"] = m["d2i_ri"] / m["kprune"]
    out["tasks"][t] = row
    print(t, {k: round(v, 4) for k, v in m.items()}, "ratio", round(row["ratio_means"], 3),
          "seeds<1:", sum(x < 1 for x in row["ratio_d2i_over_kprune_per_seed"]), "/5")
json.dump(out, open(sys.argv[1], "w"), indent=1)

# validation accuracy means (descriptive table in the paper; the registered verdict is beat_confirm_verdict.json)
acc = {}
for t, r in ROOT.items():
    a = {"d2i_ri": [], "kprune": [], "kwon": [], "unpruned": []}
    for s in SEEDS:
        j = json.load(open(f"{r}/beat/d2i_ri_{t}_seed{s}_c0.5.json"))
        a["d2i_ri"].append(j["valid"]["accuracy"])
        a["unpruned"].append(j["unpruned_valid"]["accuracy"])
        for m, p in (("kprune", f"{r}/bl/kprune_out/kprune_{t}_seed{s}_c0.5"), ("kwon", f"{r}/bl/kwon_{t}_seed{s}_c0.5")):
            v = torch.load(p + "/v15_student_validation.pt", weights_only=True)
            a[m].append(float((v["logits"].argmax(-1) == v["labels"]).float().mean()))
    acc[t] = {k: 100 * sum(v) / len(v) for k, v in a.items()}
    acc[t]["evaluation_size"] = j["evaluation_size"]
    print(t, {k: round(v, 2) for k, v in acc[t].items()})
out["mean_validation_accuracy_pct"] = acc
json.dump(out, open(sys.argv[1], "w"), indent=1)
