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
    hm = torch.load(head_path, weights_only=True, map_location="cpu")
    nm = torch.load(neuron_path, weights_only=True, map_location="cpu")
    return (hm != 0).sum(1).tolist(), (nm != 0).sum(1).tolist()


@torch.no_grad()
def time_shape(heads, neurons, batch, seq_len, warmup=20, runs=100, device="cuda"):
    """Median ms of one forward pass of a BERT-base classifier keeping heads[l] heads and neurons[l] neurons."""
    from transformers import BertConfig, BertForSequenceClassification
    cfg = BertConfig(num_hidden_layers=len(heads))  # BERT-base dimensions (V19: 6 layers); random weights
    cfg._attn_implementation = "eager"
    model = BertForSequenceClassification(cfg)
    prune_model(model, {l: list(range(h)) for l, h in enumerate(heads)},
                {l: list(range(n)) for l, n in enumerate(neurons)})
    # GPU: FP16 (rule 8 protocol). CPU (descriptive, separate): FP32.
    model = (model.half() if device == "cuda" else model).to(device).eval()
    ids = torch.randint(1000, 2000, (batch, seq_len), device=device)
    mask = torch.ones_like(ids)
    for _ in range(warmup):
        model(input_ids=ids, attention_mask=mask)
    ts = []
    if device == "cuda":
        torch.cuda.synchronize()
        for _ in range(runs):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            model(input_ids=ids, attention_mask=mask)
            e.record()
            torch.cuda.synchronize()
            ts.append(s.elapsed_time(e))
    else:
        import time
        for _ in range(runs):
            t0 = time.perf_counter()
            model(input_ids=ids, attention_mask=mask)
            ts.append((time.perf_counter() - t0) * 1000)
    params = sum(p.numel() for n, p in model.named_parameters() if ".encoder." in n)
    return statistics.median(ts), params


def measure(shapes, batches, rounds=1, order_seed=None, timer=None):
    """{name: rec}. rounds=1, order_seed=None: each shape timed once in the given order (the original protocol).
    Otherwise every round times every shape once, in a seeded random order (a model is rebuilt per round), and
    median_ms_b<b> is the median of the per-round medians, so slow drift of the machine cannot favour one shape.
    timer(heads, neurons, batch) -> (median_ms, params)."""
    import random
    names = list(shapes)
    out = {n: {"heads": shapes[n][0], "neurons": shapes[n][1]} for n in names}
    rng = random.Random(order_seed)
    per = {n: {b: [] for b in batches} for n in names}
    orders = []
    for r in range(rounds):
        order = names[:] if order_seed is None else rng.sample(names, len(names))
        orders.append(order)
        for n in order:
            for b in batches:
                ms, params = timer(shapes[n][0], shapes[n][1], b)
                per[n][b].append(ms)
                out[n]["params_encoder"] = params
            print(f"round {r} {n}", {b: round(per[n][b][-1], 3) for b in batches}, flush=True)
    for n in names:
        for b in batches:
            out[n][f"median_ms_b{b}"] = statistics.median(per[n][b])
            if rounds > 1:
                out[n][f"round_medians_ms_b{b}"] = per[n][b]
                out[n][f"max_over_min_b{b}"] = max(per[n][b]) / min(per[n][b])
    return out, orders


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grp", nargs="*", default=[], help="glob(s) of step22 JSONs")
    ap.add_argument("--kwon", nargs="*", default=[], help="dirs with head_mask.pt and neuron_mask.pt")
    ap.add_argument("--kprune", nargs="*", default=[], help="dirs with head_masks.pkl and neuron_masks.pkl")
    ap.add_argument("--shapes", help="JSON {name: [heads, neurons]} (from --dump-shapes); replaces --grp/--kwon/--kprune")
    ap.add_argument("--dump-shapes", help="write the shapes read from --kwon/--kprune/--grp to this JSON and exit")
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--batches", type=int, nargs="*", default=[1, 32])
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--rounds", type=int, default=1, help="time every shape once per round (default 1: original protocol)")
    ap.add_argument("--order-seed", type=int, default=None, help="shuffle the shape order in every round with this seed")
    ap.add_argument("--threads", type=int, default=None, help="torch.set_num_threads (CPU)")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    if a.device == "cuda" and not a.dump_shapes and not torch.cuda.is_available():
        raise SystemExit("H12 latency must be measured on a GPU (rule 8); use --device cpu for the separate CPU measurement")
    shapes = {"unpruned": ([12] * 12, [3072] * 12)}
    for p in [q for g in a.grp for q in sorted(glob.glob(g))]:
        shapes["grp:" + os.path.basename(p)] = shape_from_grp(p)
    for d in a.kwon:
        shapes["kwon:" + d] = shape_from_masks(os.path.join(d, "head_mask.pt"), os.path.join(d, "neuron_mask.pt"))
    for d in a.kprune:
        shapes["kprune:" + d] = shape_from_masks(os.path.join(d, "head_masks.pkl"), os.path.join(d, "neuron_masks.pkl"))
    if a.dump_shapes:
        json.dump({k: list(v) for k, v in shapes.items()}, open(a.dump_shapes, "w"), indent=1)
        return
    if a.shapes:
        shapes = {k: tuple(v) for k, v in json.load(open(a.shapes)).items()}
    if a.device == "cpu":
        import platform
        cpu = platform.processor()
        if os.path.exists("/proc/cpuinfo"):  # Linux (Kaggle): platform.processor() is often empty
            cpu = next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), cpu)
        dev = f"cpu: {cpu}, {os.cpu_count()} logical cpus, torch threads {torch.get_num_threads()}"
    else:
        dev = torch.cuda.get_device_name(0)
    out = {"device": dev, "torch": torch.__version__, "seq_len": a.seq_len,
           "dtype": "fp16" if a.device == "cuda" else "fp32", "runs": a.runs, "warmup": a.warmup,
           "rounds": a.rounds, "order_seed": a.order_seed, "torch_threads": torch.get_num_threads()}
    out["results"], out["round_orders"] = measure(
        shapes, a.batches, a.rounds, a.order_seed,
        lambda h, n, b: time_shape(h, n, b, a.seq_len, a.warmup, a.runs, a.device))
    for name, rec in out["results"].items():
        print(name, {k: round(v, 3) for k, v in rec.items() if k.startswith("median")}, rec["params_encoder"], flush=True)
    json.dump(out, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
