"""Figure: selection vs compensation (V14). One panel per task; for each selector, the mean validation metric over
5 seeds without compensation (open circle) and with sequential LS refit (filled), joined by a line. Dashed line:
unpruned; shaded band: Random mean +- 1 sd (100 masks, no compensation).

    python paper/figs/fig_v14.py results/v14_bcm/9c4b999 paper/figs/fig_v14     (needs matplotlib)
Colors: reference categorical slot 1 (#2a78d6) for the data; text in ink tokens. Open vs filled markers carry the
compensation identity (no second hue needed).
"""
import glob
import json
import os
import statistics as st
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BLUE, INK, MUTED, BAND = "#2a78d6", "#1f1f1e", "#6b6a63", "#ecebe6"
TASKS = [("sst2", "SST-2", "accuracy"), ("mrpc", "MRPC", "accuracy"), ("rte", "RTE", "accuracy"),
         ("qnli", "QNLI", "accuracy"), ("cola", "CoLA", "mcc")]
SEL = [("qfc_attn", "Attn. cov."), ("fisher", "Fisher"), ("magnitude", "Magnitude"), ("greedy_ls", "Greedy LS"),
       ("bcm", "BCM")]


def main():
    root, out = sys.argv[1], sys.argv[2]
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 5, figsize=(7.2, 2.9))
    for ax, (task, name, metric) in zip(axes, TASKS):
        runs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(root, f"v14_{task}_seed*.json")))]
        mean = lambda arm: st.mean(r["arms"][arm][metric] for r in runs)  # noqa: E731
        unpruned = st.mean(r["unpruned"][metric] for r in runs)
        rnd = [m[metric] for m in json.load(open(os.path.join(root, f"v14_{task}_random.json")))["per_mask"]]
        ax.axhspan(st.mean(rnd) - st.pstdev(rnd), st.mean(rnd) + st.pstdev(rnd), color=BAND, lw=0, zorder=0)
        ax.axhline(unpruned, color=MUTED, lw=0.8, ls=(0, (3, 2)), zorder=1)
        for x, (sel, _) in enumerate(SEL):
            lo, hi = mean(f"{sel}|none"), mean(f"{sel}|ls_seq")
            ax.plot([x, x], [lo, hi], color=BLUE, lw=1.2, zorder=2)
            ax.scatter([x], [lo], s=18, facecolors="white", edgecolors=BLUE, linewidths=1.2, zorder=3,
                       label="no compensation" if x == 0 else None)
            ax.scatter([x], [hi], s=18, color=BLUE, zorder=3, label="sequential LS refit" if x == 0 else None)
        ax.set_xticks(range(len(SEL)), [s[1] for s in SEL], fontsize=6.5, rotation=55, ha="right",
                      rotation_mode="anchor")
        ax.set_xlim(-0.6, len(SEL) - 0.4)
        ax.set_title(name + (" (MCC)" if metric == "mcc" else ""), fontsize=8.5, color=INK, loc="left")
        ax.grid(True, axis="y", color="#e4e3dc", lw=0.5, zorder=0)
    axes[0].set_ylabel("validation metric (mean of 5 seeds)")
    h, l = axes[0].get_legend_handles_labels()
    h.append(plt.Line2D([], [], color=MUTED, lw=0.8, ls=(0, (3, 2))))
    l.append("unpruned")
    h.append(plt.Rectangle((0, 0), 1, 1, color=BAND))
    l.append("Random, mean $\\pm$ sd")
    fig.legend(h, l, loc="upper center", ncol=4, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out + ".pdf")
    fig.savefig(out + ".png", dpi=200)
    print("wrote", out + ".pdf/.png")


if __name__ == "__main__":
    main()
