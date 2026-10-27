"""Figures from committed verdict JSONs only.
fig_h10: held-out logit KL vs fraction of MACs kept (V15 H10), Kwon / K-prune / GRP, one panel per task.
fig_factorial: selection x compensation, KL ratio to official K-prune (V16 d2i, V17 grp_r / d2i_r, V15 grp).
    python paper/figs/fig_kl.py                (run from the repo root; needs matplotlib)
Colors: reference categorical slots 1-3 (#2a78d6, #eb6834, #1baf7a); markers/linestyle repeat the identity.
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

H10 = "results/v15_h10/c43dd3e/h10/v15_h10_verdict.json"
V16 = "results/v16_stageA/b2b444e/v16_stageA_verdict.json"
V17 = "results/v17_rdev/195c18f/v17_rdev_verdict.json"
BLUE, ORANGE, AQUA, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#1f1f1e", "#6b6a63"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
TASKS = [("mrpc", "MRPC"), ("sst2", "SST-2")]
BUDGETS = [0.7, 0.6, 0.5, 0.4]


def fig_h10():
    t = json.load(open(H10))["table"]
    fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.5), sharey=False)
    for ax, (task, name) in zip(axs, TASKS):
        for key, lab, col, mk, ls in [("kwon", "Kwon et al.", AQUA, "s", ":"), ("kprune", "K-prune", BLUE, "o", "-"),
                                      ("grp", "GRP", ORANGE, "^", "--")]:
            ax.plot(BUDGETS, [t[f"{task}_c{c}"][key]["mean_kl"] for c in BUDGETS], ls, color=col, marker=mk, ms=4,
                    lw=1.5, label=lab)
        ax.set_yscale("log"); ax.set_title(name, color=INK, fontsize=9); ax.invert_xaxis()
        ax.set_xlabel("fraction of MACs kept"); ax.set_xticks(BUDGETS)
    axs[0].set_ylabel("mean logit KL (held-out train)"); axs[0].legend(frameon=False)
    fig.tight_layout(); fig.savefig("paper/figs/fig_h10.pdf"); fig.savefig("paper/figs/fig_h10.png", dpi=200)


def fig_factorial():
    h10, v16, v17 = (json.load(open(p)) for p in (H10, V16, V17))
    cells = [(t, c) for t in ("mrpc", "sst2") for c in (0.5, 0.4)]
    rows = {  # (selection, refit) -> ratio per cell
        ("K-prune", "module-output"): [v16["table"][f"d2i_{t}_c{c}"]["ratio"] for t, c in cells],
        ("K-prune", "drift-corrected"): [v17["table"][f"d2i_r_{t}_c{c}"]["ratio"] for t, c in cells],
        ("GRP", "module-output"): [h10["table"][f"{t}_c{c}"]["grp"]["mean_kl"] / h10["table"][f"{t}_c{c}"]["kprune"]["mean_kl"]
                                   for t, c in cells],
        ("GRP", "drift-corrected"): [v17["table"][f"grp_r_{t}_c{c}"]["ratio"] for t, c in cells],
    }
    fig, ax = plt.subplots(figsize=(6.4, 2.7))
    w = 0.18
    for i, ((sel, refit), vals) in enumerate(rows.items()):
        col = BLUE if sel == "K-prune" else ORANGE
        filled = refit == "drift-corrected"
        xs = [j + (i - 1.5) * w for j in range(len(cells))]
        ax.scatter(xs, vals, s=34, marker="o" if sel == "K-prune" else "^", facecolors=col if filled else "none",
                   edgecolors=col, linewidths=1.5, label=f"{sel}, {refit} LS", zorder=3)
    ax.axhline(1, color=MUTED, lw=1, ls="--"); ax.set_yscale("log")
    ax.set_xticks(range(len(cells))); ax.set_xticklabels([f"{dict(TASKS)[t]}\n{c}" for t, c in cells])
    ax.set_xlabel("task, fraction of MACs kept"); ax.set_ylabel("KL / official K-prune KL")
    ax.legend(frameon=False, ncol=2, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, 1.28))
    fig.tight_layout(); fig.savefig("paper/figs/fig_factorial.pdf"); fig.savefig("paper/figs/fig_factorial.png", dpi=200)


if __name__ == "__main__":
    fig_h10(); fig_factorial()
