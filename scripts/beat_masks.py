"""Collect K-prune masks (head_masks.pkl / neuron_masks.pkl per run dir kprune_<task>_seed<s>_c<c>) into one npz with
the key layout step23/beat_run read: kprune_<task>_seed<s>_c<c>__head [L,H] and __neuron [L,I] (nonzero = kept).

    python scripts/beat_masks.py <dir containing kprune_* run dirs> <out.npz>
"""
import glob
import os
import re
import sys

import numpy as np
import torch

root, out = sys.argv[1], sys.argv[2]
arrays = {}
for d in sorted(glob.glob(os.path.join(root, "**", "kprune_*_seed*_c*"), recursive=True)):
    name = os.path.basename(d)
    if not re.fullmatch(r"kprune_[a-z0-9]+_seed\d+_c[\d.]+", name) or not os.path.exists(os.path.join(d, "head_masks.pkl")):
        continue
    hm = torch.load(os.path.join(d, "head_masks.pkl"), weights_only=True)
    nm = torch.load(os.path.join(d, "neuron_masks.pkl"), weights_only=True)
    arrays[name + "__head"] = (hm != 0).numpy().astype(np.float32)
    arrays[name + "__neuron"] = (nm != 0).numpy().astype(np.float32)
np.savez_compressed(out, **arrays)
print(len(arrays) // 2, "runs ->", out)
