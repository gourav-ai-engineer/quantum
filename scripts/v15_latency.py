"""V15 H12 (descriptive, rule 8): measured latency of physically pruned BERT-base shapes.

Latency depends only on the per-layer numbers of kept heads and FFN neurons, so every method's result is reduced to
that shape and timed in ONE environment on one GPU: FP16, batch 1 and 32, sequence length 128, median of 100 timed
runs after 20 warm-up runs. Shapes come from GRP JSONs (step22), Kwon et al. masks (head_mask.pt / neuron_mask.pt,
zero = pruned) and K-prune masks (head_masks.pkl / neuron_masks.pkl).

    python scripts/v15_latency.py --grp 'results/.../v15_grp_*_c0.5.json' --kwon <dir>... --kprune <dir>... --out lat.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics

import torch

from qfc.grp import prune_model


def shape_from_grp(path):
    r = json.load(open(path))
    L = len(r["heads_kept"])
    return [len(r["heads_kept"][str(l)]) for l in range(L)], [len(r["neurons_kept"][str(l)]) for l in range(L)]


def shape_from_masks(head_path, neuron_path):
    hm = torch.load(head_path, weights_only=True)
    nm = torch.load(neuron_path, weights_only=True)
    return (hm != 0).sum(1).tolist(), (nm != 0).sum(1).tolist()


@torch.no_grad()
def time_shape(heads, neurons, batch, seq_len, warmup=20, runs=100):
    """Median ms of one forward pass of a BERT-base classifier keeping heads[l] heads and neurons[l] neurons."""
    from transformers import BertConfig, BertForSequenceClassification
    cfg = BertConfig()  # BERT-base dimensions; weights are random (they do not affect latency)
    cfg._attn_implementation = "eager"
    model = BertForSequenceClassification(cfg)
    prune_model(model, {l: list(range(h)) for l, h in enumerate(heads)},
                {l: list(range(n)) for l, n in enumerate(neurons)})
    model = model.half().cuda().eval()
    ids = torch.randint(1000, 2000, (batch, seq_len), device="cuda")
    mask = torch.ones_like(ids)
    for _ in range(warmup):
        model(input_ids=ids, attention_mask=mask)
    torch.cuda.synchronize()
    ts = []
    for _ in range(runs):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        model(input_ids=ids, attention_mask=mask)
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    params = sum(p.numel() for n, p in model.named_parameters() if ".encoder." in n)
    return statistics.median(ts), params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grp", nargs="*", default=[], help="glob(s) of step22 JSONs")
    ap.add_argument("--kwon", nargs="*", default=[], help="dirs with head_mask.pt and neuron_mask.pt")
    ap.add_argument("--kprune", nargs="*", default=[], help="dirs with head_masks.pkl and neuron_masks.pkl")
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("H12 latency must be measured on a GPU (rule 8)")
    shapes = {"unpruned": ([12] * 12, [3072] * 12)}
    for p in [q for g in a.grp for q in sorted(glob.glob(g))]:
        shapes["grp:" + os.path.basename(p)] = shape_from_grp(p)
    for d in a.kwon:
        shapes["kwon:" + d] = shape_from_masks(os.path.join(d, "head_mask.pt"), os.path.join(d, "neuron_mask.pt"))
    for d in a.kprune:
        shapes["kprune:" + d] = shape_from_masks(os.path.join(d, "head_masks.pkl"), os.path.join(d, "neuron_masks.pkl"))
    out = {"device": torch.cuda.get_device_name(0), "torch": torch.__version__, "seq_len": a.seq_len, "dtype": "fp16",
           "results": {}}
    for name, (h, n) in shapes.items():
        rec = {"heads": h, "neurons": n}
        for b in (1, 32):
            ms, params = time_shape(h, n, b, a.seq_len)
            rec[f"median_ms_b{b}"], rec["params_encoder"] = ms, params
        out["results"][name] = rec
        print(name, round(rec["median_ms_b1"], 3), round(rec["median_ms_b32"], 3), rec["params_encoder"], flush=True)
    json.dump(out, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
