"""V19 fix1: merge K-prune mask npz files (original kernel + fix kernel) into one file for beat_run.py --kprune-masks.
Usage: python scripts/v19_merge_masks.py OUT.npz IN1.npz IN2.npz ...
A key present in two inputs must hold identical arrays (a mask is never silently replaced)."""
import sys

import numpy as np


def merge(out, ins):
    merged = {}
    for p in ins:
        z = np.load(p)
        for k in z.files:
            if k in merged and not np.array_equal(merged[k], z[k]):
                raise SystemExit(f"conflicting mask {k} in {p}")
            merged[k] = z[k]
    np.savez(out, **merged)
    return sorted(merged)


if __name__ == "__main__":
    keys = merge(sys.argv[1], sys.argv[2:])
    print(f"{len(keys)} arrays -> {sys.argv[1]}")
    for k in keys:
        print(" ", k)
