"""Summarize V14 diagnostics D1-D3 (step21 --diagnostics-only JSONs) over tasks x seeds. Descriptive only.

    python scripts/v14_diag_summary.py results/v14_bcm_diag/full_<commit>
Writes <dir>/diag_summary.json and prints one line per task.
"""
from __future__ import annotations

import glob
import json
import os
import statistics as st
import sys


def main():
    root = sys.argv[1]
    by_task: dict[str, list] = {}
    for f in sorted(glob.glob(os.path.join(root, "v14_*_seed*_diag.json"))):
        d = json.load(open(f))
        by_task.setdefault(d["task"], []).append(d)
    out = {}
    for task, runs in by_task.items():
        pairs = hi = hi_low = 0
        sp_ao, sp_b, slack, rel = [], [], [], {}
        for d in runs:
            for lay in d["layers"].values():
                pairs += lay["D1_pairs"]
                hi += lay["D1_attnF_gt_0.99"]
                hi_low += lay["D1_attnF_gt_0.99_and_outF_lt_0.5"]
                sp_ao.append(lay["D1_spearman_attnF_vs_outF"])
                sp_b.append(lay["D2_spearman_bound_vs_merge_error"])
                slack.append(lay["D2_median_slack_bound_over_merge"])
                for name, v in lay["D3"].items():
                    rel.setdefault(name, []).append(v["ls_error"] / lay["total_output_norm"])
        rec = {"seeds": sorted(d["seed"] for d in runs), "D1_pairs": pairs, "D1_attnF_gt_0.99": hi,
               "D1_of_those_outF_lt_0.5": hi_low, "D1_median_spearman_attn_vs_out": st.median(sp_ao),
               "D2_median_spearman_bound_vs_merge": st.median(sp_b), "D2_median_slack": st.median(slack),
               "D3_mean_relative_ls_error": {k: st.mean(v) for k, v in rel.items()}}
        out[task] = rec
        print(f"{task} seeds={rec['seeds']} attnF>0.99: {hi}/{pairs}, of which outF<0.5: {hi_low}; "
              f"spearman(attn,out)={rec['D1_median_spearman_attn_vs_out']:.3f}; "
              f"spearman(B,merge)={rec['D2_median_spearman_bound_vs_merge']:.3f}; slack={rec['D2_median_slack']:.2f}; "
              + " ".join(f"{k}={v:.3f}" for k, v in sorted(rec["D3_mean_relative_ls_error"].items())))
    json.dump(out, open(os.path.join(root, "diag_summary.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
