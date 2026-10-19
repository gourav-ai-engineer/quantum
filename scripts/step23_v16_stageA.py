"""V16 Stage A (docs/PROJECT_STATE.md, "Pre-registration V16"): diagnostics on the development tasks MRPC/SST-2.

Parts (held-out-train KL, the V15 protocol: calibration = train.shuffle(seed)[:2048], held-out = the next 512):
  d2i     : K-prune's official masks (V15 H10 runs) -> physical pruning -> GRP's joint sequential refit.
  d2ii    : per-module kept counts from K-prune's masks; GRP's refit-aware greedy inside each module; joint refit.
  no_sens : GRP with s_m = 1.
  d2i_r   : V17 R-dev mechanism check: K-prune's masks + drift-corrected refit (GRP-R's compensation).
  grp_r   : V17 GRP-R: GRP selection + drift-corrected joint refit.
  d1      : exchange-rate probe at GRP's V15 c0.5 solution: per module, remove the cheapest units worth 2% of the
            module's current MACs (max(1, round(0.02 * kept)) units), refit, KL increase on the CALIBRATION rows,
            against s_m * DeltaE_m (exact joint removal cost).
Validation data is never loaded.

    PYTHONPATH=scripts python scripts/step23_v16_stageA.py --parts d2i,d2ii,no_sens --output-dir results/v16_stageA
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import time

import numpy as np
import torch
from datasets import load_dataset

from qfc.grp import (AVG_SEQ_LEN, ModuleState, grp_prune, grp_select_counts, model_mac, prune_model,
                     sequential_refit, teacher_stats)
from step22_grp import SPECS, compare_outputs, load_model, logits_of, make_batches


def kprune_masks(npz, task, seed, c):
    z = np.load(npz)
    hm, nm = z[f"kprune_{task}_seed{seed}_c{c}__head"], z[f"kprune_{task}_seed{seed}_c{c}__neuron"]
    return ({l: np.nonzero(hm[l])[0].tolist() for l in range(hm.shape[0])},
            {l: np.nonzero(nm[l])[0].tolist() for l in range(nm.shape[0])})


def mac_ratio(heads, neurons, cfg, n_seq):
    hs = cfg.hidden_size // cfg.num_attention_heads
    full = model_mac([cfg.num_attention_heads] * cfg.num_hidden_layers, [cfg.intermediate_size] * cfg.num_hidden_layers,
                     n_seq, cfg.hidden_size, hs)
    return model_mac([len(heads[l]) for l in heads], [len(neurons[l]) for l in neurons], n_seq, cfg.hidden_size, hs) / full


def refit_student(teacher, heads, neurons, cb, device, ridge_rel, drift_correct=False):
    assert all(len(v) >= 1 for v in neurons.values()), "a layer with 0 FFN neurons is not supported"
    s = prune_model(copy.deepcopy(teacher), heads, neurons)
    sequential_refit(s, teacher, cb, device, ridge_rel, drift_correct)
    return s


def d1_probe(teacher, stats, heads, neurons, cb, lt_cal, device, ridge_rel, probe_frac=0.02):
    cfg = teacher.config
    hs = cfg.hidden_size // cfg.num_attention_heads
    base = refit_student(teacher, heads, neurons, cb, device, ridge_rel)
    kl_base = compare_outputs({}, lt_cal, logits_of(base, cb, device))[0]
    del base
    rows = []
    for l in range(cfg.num_hidden_layers):
        for kind, kept, n_all in (("att", heads[l], cfg.num_attention_heads), ("ffn", neurons[l], cfg.intermediate_size)):
            st = ModuleState(stats["G"][(l, kind)], stats["W"][(l, kind)], hs if kind == "att" else 1, ridge_rel)
            drop = [j for j in range(n_all) if j not in set(kept)]
            if drop:
                st.remove(drop)
                st._refresh()
            k = max(1, round(probe_frac * len(st.kept)))
            if kind == "ffn":
                k = min(k, len(st.kept) - 1)
            if not st.kept or k < 1:
                rows.append({"layer": l, "module": kind, "skipped": "nothing removable"})
                continue
            R = [st.kept[i] for i in torch.argsort(st.costs())[:k].tolist()]
            dE = st.block_cost(R)
            h2, n2 = dict(heads), dict(neurons)
            if kind == "att":
                h2[l] = [j for j in kept if j not in set(R)]
            else:
                n2[l] = [j for j in kept if j not in set(R)]
            s = refit_student(teacher, h2, n2, cb, device, ridge_rel)
            dkl = compare_outputs({}, lt_cal, logits_of(s, cb, device))[0] - kl_base
            del s
            sm = stats["sens"][(l, kind)]
            rows.append({"layer": l, "module": kind, "units_removed": R, "k": k, "kept_before": len(st.kept),
                         "s_m": sm, "delta_E": dE, "predicted": sm * dE, "delta_kl_calibration": dkl,
                         "rho": dkl / (sm * dE) if sm * dE > 0 else None})
            print(f"  d1 layer {l} {kind}: k={k} pred={sm * dE:.3e} dKL={dkl:.3e}", flush=True)
    return kl_base, rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", default="mrpc,sst2")
    p.add_argument("--seeds", default="7,42,77,123,2024")
    p.add_argument("--constraints", default="0.5,0.4")
    p.add_argument("--parts", default="d2i,d2ii,no_sens,d1")
    p.add_argument("--d1-seeds", default="7,42")
    p.add_argument("--kprune-masks", default="results/v15_h10/c43dd3e/h10/kprune_masks.npz")
    p.add_argument("--grp-dir", default="results/v15_h10/c43dd3e/h10/grp")
    p.add_argument("--calibration-size", type=int, default=2048)
    p.add_argument("--heldout-size", type=int, default=512)
    p.add_argument("--ridge-rel", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output-dir", default="results/v16_stageA")
    a = p.parse_args()
    os.makedirs(a.output_dir, exist_ok=True)
    parts = a.parts.split(",")
    for task in a.tasks.split(","):
        spec = SPECS[task]
        assert task in ("mrpc", "sst2"), "Stage A uses the development tasks only"
        ds = load_dataset(*spec["dataset"])
        teacher, tok = load_model(spec, a.device)
        cfg, n_seq = teacher.config, AVG_SEQ_LEN[task]
        for seed in [int(s) for s in a.seeds.split(",")]:
            shuf, n = ds["train"].shuffle(seed=seed), a.calibration_size
            cb = make_batches(shuf.select(range(n)), tok, spec["text_fields"], a.batch_size, a.max_length)
            hb = make_batches(shuf.select(range(n, n + a.heldout_size)), tok, spec["text_fields"], a.batch_size,
                              a.max_length)
            lt_held = logits_of(teacher, hb, a.device)
            stats = None
            for c in [float(x) for x in a.constraints.split(",")]:
                for part in [x for x in parts if x != "d1"]:
                    prefix = "v17" if part.endswith("_r") else "v16"
                    path = os.path.join(a.output_dir, f"{prefix}_{part}_{task}_seed{seed}_c{c}.json")
                    if os.path.exists(path):
                        continue
                    t0 = time.time()
                    if part in ("no_sens", "grp_r"):
                        stats = stats or teacher_stats(teacher, cb, a.device, seed)
                        student, heads, neurons, _ = grp_prune(teacher, cb, a.device, n_seq, c, a.ridge_rel,
                                                               use_sensitivity=part != "no_sens", seed=seed,
                                                               stats=stats, drift_correct=part == "grp_r")
                    else:
                        kh, kn = kprune_masks(a.kprune_masks, task, seed, c)
                        if part in ("d2i", "d2i_r"):
                            heads, neurons = kh, kn
                        else:
                            stats = stats or teacher_stats(teacher, cb, a.device, seed)
                            counts = {**{(l, "att"): len(kh[l]) for l in kh}, **{(l, "ffn"): len(kn[l]) for l in kn}}
                            heads, neurons = grp_select_counts(stats["G"], stats["W"], counts,
                                                               cfg.num_attention_heads, a.ridge_rel)
                        student = refit_student(teacher, heads, neurons, cb, a.device, a.ridge_rel,
                                                drift_correct=part.endswith("_r"))
                    kl, agree = compare_outputs(spec, lt_held, logits_of(student, hb, a.device))
                    rec = {"task": task, "seed": seed, "constraint": c, "part": part, "calibration_size": n,
                           "heldout": f"train.shuffle({seed})[{n}:{n + a.heldout_size}]", "seq_len_for_macs": n_seq,
                           "ridge_rel": a.ridge_rel, "mac_ratio": mac_ratio(heads, neurons, cfg, n_seq),
                           "heads_kept": {str(l): h for l, h in heads.items()},
                           "neurons_kept": {str(l): x for l, x in neurons.items()},
                           "heldout_kl": kl, "heldout_teacher_agreement": agree, "seconds": time.time() - t0,
                           "model_id": spec["model_id"], "model_revision": spec["revision"]}
                    print(f"{task} seed={seed} c={c} {part}: mac={rec['mac_ratio']:.3f} heads/layer="
                          f"{[len(h) for h in heads.values()]} kl={kl:.4f} agree={agree:.3f} ({rec['seconds']:.0f}s)",
                          flush=True)
                    json.dump(rec, open(path, "w"))
                    del student
            if "d1" in parts and str(seed) in a.d1_seeds.split(","):
                path = os.path.join(a.output_dir, f"v16_d1_{task}_seed{seed}_c0.5.json")
                if not os.path.exists(path):
                    t0 = time.time()
                    stats = stats or teacher_stats(teacher, cb, a.device, seed)
                    g = json.load(open(os.path.join(a.grp_dir, f"v15_grp_{task}_seed{seed}_c0.5.json")))
                    heads = {int(l): v for l, v in g["heads_kept"].items()}
                    neurons = {int(l): v for l, v in g["neurons_kept"].items()}
                    lt_cal = logits_of(teacher, cb, a.device)
                    kl_base, rows = d1_probe(teacher, stats, heads, neurons, cb, lt_cal, a.device, a.ridge_rel)
                    json.dump({"task": task, "seed": seed, "constraint": 0.5, "part": "d1", "probe_frac": 0.02,
                               "grp_solution": f"{a.grp_dir}/v15_grp_{task}_seed{seed}_c0.5.json",
                               "calibration_kl_base": kl_base, "modules": rows, "seconds": time.time() - t0,
                               "model_id": spec["model_id"], "model_revision": spec["revision"]}, open(path, "w"))
                    print(f"{task} seed={seed} d1 done ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
