"""Figure: attention-pattern fidelity vs output fidelity for every within-layer head pair (BERT-base, calibration data).

    python paper/figs/fig_pairs.py results/v14_pairs/<commit> paper/figs/fig_pairs
(needs matplotlib; run it outside the pinned .venv-ci). Palette: reference categorical slots 1-2 (#2a78d6, #eb6834),
validated (CVD dE 24.7, normal-vision dE 33.6). Flagged pairs also differ by marker shape, so identity is not carried
by color alone.
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#1f1f1e", "#6b6a63"


def main():
    root, out = sys.argv[1], sys.argv[2]
    tasks = [("sst2", "SST-2"), ("mrpc", "MRPC")]
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1), sharey=True)
    for ax, (task, name) in zip(axes, tasks):
        pts = json.load(open(os.path.join(root, f"pairs_{task}_seed7.json")))["pairs"]
        is_flag = [p["attn_F"] > 0.99 and p["out_F"] < 0.5 for p in pts]
        flag = [p for p, f in zip(pts, is_flag) if f]
        rest = [p for p, f in zip(pts, is_flag) if not f]
        gap = lambda p: max(1.0 - p["attn_F"], 1e-6)  # noqa: E731  (log scale spreads the pairs near F = 1)
        ax.axvspan(1e-6, 0.01, ymin=0, ymax=0.5, color="#ecebe6", zorder=0, lw=0)
        ax.axvline(0.01, color=MUTED, lw=0.6, ls=(0, (3, 2)), zorder=1)
        ax.scatter([gap(p) for p in rest], [p["out_F"] for p in rest], s=9, c=BLUE, alpha=0.55, lw=0,
                   label="head pair", zorder=2)
        ax.scatter([gap(p) for p in flag], [p["out_F"] for p in flag], s=22, c=ORANGE, marker="^",
                   edgecolors="white", linewidths=0.6, label="attention F > 0.99 but output F < 0.5", zorder=3)
        n_hi = sum(p["attn_F"] > 0.99 for p in pts)
        ax.set_title(f"{name}: {len(flag)} of {n_hi} pairs with attention F > 0.99", fontsize=9, color=INK, loc="left")
        ax.set_xscale("log")
        ax.invert_xaxis()  # more similar attention to the right
        ax.set_xlabel("1 - attention-pattern fidelity (log scale)")
        ax.grid(True, color="#e4e3dc", lw=0.5, zorder=1)
        ax.set_xlim(max(gap(p) for p in pts) * 1.3, min(gap(p) for p in pts) / 1.3)
        ax.set_ylim(0, 1.0)
    axes[0].set_ylabel("output fidelity")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out + ".pdf")
    fig.savefig(out + ".png", dpi=200)
    print("wrote", out + ".pdf/.png")


if __name__ == "__main__":
    main()
