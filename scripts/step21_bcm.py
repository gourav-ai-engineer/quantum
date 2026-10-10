"""V14 / H9 (docs/PROJECT_STATE.md, "Pre-registration V14"): output-space coverage with compensation.

--diagnostics-only : D1-D3 from calibration data only (one forward pass + small matrix algebra). Deterministic;
                     allowed on CPU as an engineering diagnostic, never an accuracy result.
default            : accuracy factorial selector x compensation on the full validation split (GPU required unless
                     --smoke --allow-cpu). Saves per-example predictions for paired statistics.

    PYTHONPATH=scripts python scripts/step21_bcm.py --tasks sst2,mrpc --seeds 7 --diagnostics-only --output-dir results/v14_bcm_diag
    PYTHONPATH=scripts python scripts/step21_bcm.py --tasks sst2,mrpc,rte,qnli,cola --output-dir results/v14_bcm
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random

import numpy as np
import torch
from datasets import load_dataset

from qfc.alignment import spearman
from qfc.baselines import heads_kept_per_layer, random_selection, topk_selection, validate_selection
from qfc.fidelity import pairwise_fidelity
from qfc.hf_experiments import (
    collect_mean_density_states,
    head_mask_from_selection,
    load_sequence_classifier,
    make_text_loader,
    michel_head_importance,
    qfc_select_from_mean_states,
)
from qfc.output_coverage import (
    bcm_select,
    bound_weights,
    collect_context_grams,
    coverage_bound,
    delete_weights,
    exhaustive_ls_best,
    greedy_ls_select,
    layer_error,
    ls_refit,
    merge_weights,
    output_fidelity,
    output_weight,
    sequential_ls_refit,
    set_output_weight,
)
from step20_round2 import SPECS, evaluate, summarize

SELECTORS = ("bcm", "fisher", "magnitude", "greedy_ls", "qfc_attn")


def select_all(name, Gs, Ws, Fs, wts, michel, mean_states, k, H, ridge):
    L = len(Gs)
    if name == "bcm":
        return {l: bcm_select(Fs[l], wts[l], k)[0] for l in range(L)}
    if name == "fisher":
        return topk_selection([michel[l] for l in range(L)], k, largest=True)
    if name == "magnitude":
        return topk_selection([wts[l] for l in range(L)], k, largest=True)
    if name == "greedy_ls":
        return {l: greedy_ls_select(Gs[l], Ws[l], k, H, ridge) for l in range(L)}
    if name == "qfc_attn":
        return {l: sorted(s) for l, s in qfc_select_from_mean_states(mean_states, k)[0].items()}
    raise ValueError(name)


def assignment(F, S):
    return {j: max(S, key=lambda i: (float(F[j, i]), -i)) for j in range(F.shape[0]) if j not in S}


def diagnostics(Gs, Ws, Fs, wts, mean_states, sels, k, H, ridge, n_random, rng):
    out = {"layers": {}}
    iu = torch.triu_indices(H, H, 1)
    for l, (G, W, F, w) in enumerate(zip(Gs, Ws, Fs, wts)):
        Fa = pairwise_fidelity(mean_states[l]).double()
        fa, fo = Fa[iu[0], iu[1]], F[iu[0], iu[1]]
        C = G @ W
        rec = []
        for _ in range(n_random):  # D2
            S = sorted(rng.sample(range(H), k))
            rec.append((coverage_bound(F, w, S), layer_error(G, W, merge_weights(G, W, S, assignment(F, S), H)),
                        layer_error(G, W, ls_refit(G, C, S, H, ridge))))
        B, em, el = (list(x) for x in zip(*rec))
        S_opt, e_opt = exhaustive_ls_best(G, W, k, H, ridge)  # D3
        d3 = {name: {"heads": sel[l],
                     "ls_error": layer_error(G, W, ls_refit(G, C, sel[l], H, ridge)),
                     "delete_error": layer_error(G, W, delete_weights(W, sel[l], H)),
                     "merge_error": layer_error(G, W, merge_weights(G, W, sel[l], assignment(F, sel[l]), H)),
                     "bound": coverage_bound(F, w, sel[l])} for name, sel in sels.items()}
        d3["exhaustive_opt"] = {"heads": S_opt, "ls_error": e_opt}
        out["layers"][str(l)] = {
            "D1_pairs": int(fa.numel()),
            "D1_attnF_gt_0.99_and_outF_lt_0.5": int(((fa > 0.99) & (fo < 0.5)).sum()),
            "D1_attnF_gt_0.99": int((fa > 0.99).sum()),
            "D1_spearman_attnF_vs_outF": spearman(fa.tolist(), fo.tolist()),
            "D2_spearman_bound_vs_merge_error": spearman(B, em),
            "D2_spearman_bound_vs_ls_error": spearman(B, el),
            "D2_median_slack_bound_over_merge": float(np.median(np.array(B) / np.maximum(np.array(em), 1e-12))),
            "D3": d3,
            "total_output_norm": layer_error(G, W, torch.zeros_like(W)),
        }
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", default="sst2,mrpc,rte,qnli,cola")
    p.add_argument("--seeds", default="7,42,77,123,2024")
    p.add_argument("--calibration-size", type=int, default=512)
    p.add_argument("--evaluation-size", type=int, default=-1)
    p.add_argument("--heads-to-keep", type=int, default=6)
    p.add_argument("--ridge-rel", type=float, default=1e-4)
    p.add_argument("--random-masks", type=int, default=100)
    p.add_argument("--merge-random-targets", type=int, default=10)
    p.add_argument("--diag-random-subsets", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--diagnostics-only", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--allow-cpu", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output-dir", default="results/v14_bcm")
    a = p.parse_args()
    if a.device == "cpu" and not a.diagnostics_only and not (a.smoke and a.allow_cpu):
        raise SystemExit("accuracy runs need a GPU (CLAUDE.md); use --diagnostics-only, or --smoke --allow-cpu")
    k, ridge = a.heads_to_keep, a.ridge_rel
    os.makedirs(a.output_dir, exist_ok=True)

    for task in a.tasks.split(","):
        spec = SPECS[task]
        dn, cfg = spec["dataset"]
        ds = load_dataset(dn, cfg) if cfg else load_dataset(dn)
        train, valid = ds["train"], ds["validation"]
        if a.evaluation_size >= 0:
            valid = valid.select(range(a.evaluation_size))
        teacher, tok = load_sequence_classifier(spec["model_id"], a.device, revision=spec["revision"])
        L, H = teacher.config.num_hidden_layers, teacher.config.num_attention_heads
        vl = None if a.diagnostics_only else make_text_loader(
            valid, tok, text_fields=spec["text_fields"], batch_size=a.batch_size, max_length=a.max_length)
        base = None if a.diagnostics_only else evaluate(teacher, vl, a.device)
        for seed in [int(s) for s in a.seeds.split(",") if s]:
            stem = os.path.join(a.output_dir, f"v14_{task}_seed{seed}")
            if not a.diagnostics_only and os.path.exists(stem + ".npz"):  # resume after a timeout
                print(f"  {task} seed={seed} done, skipping")
                continue
            cal = train.shuffle(seed=seed).select(range(a.calibration_size))
            cl = make_text_loader(cal, tok, text_fields=spec["text_fields"], batch_size=a.batch_size,
                                  max_length=a.max_length)
            Gs = collect_context_grams(teacher, cl, a.device)
            Ws = [output_weight(teacher, l) for l in range(L)]
            Fs = [output_fidelity(G, H) for G in Gs]
            wts = [bound_weights(G, W, H) for G, W in zip(Gs, Ws)]
            michel = michel_head_importance(teacher, cl, a.device).detach().cpu()
            mean_states = collect_mean_density_states(teacher, cl, a.device)
            sels = {n: select_all(n, Gs, Ws, Fs, wts, michel, mean_states, k, H, ridge) for n in SELECTORS}
            for n, sel in sels.items():
                validate_selection(sel, L, H, k, name=n)
                print(f"  {task} seed={seed} {n:10s} heads_kept_per_layer={heads_kept_per_layer(sel)}")
            if a.diagnostics_only:
                d = diagnostics(Gs, Ws, Fs, wts, mean_states, sels, k, H, ridge, a.diag_random_subsets,
                                random.Random(seed + 70000))
                d.update(task=task, seed=seed, k=k, calibration_size=a.calibration_size, ridge_rel=ridge,
                         model_id=spec["model_id"], model_revision=spec["revision"])
                json.dump(d, open(stem + "_diag.json", "w"), indent=2)
                continue

            arrays = {"labels": base[3], "Unpruned__pred": base[0], "Unpruned__loss": base[1], "Unpruned__p1": base[2]}
            rec = {"task": task, "seed": seed, "metric": spec["metric"], "k": k, "evaluation_size": len(valid),
                   "calibration_size": a.calibration_size, "ridge_rel": ridge, "unpruned": summarize(*base),
                   "selections": {n: {str(l): s for l, s in sel.items()} for n, sel in sels.items()}, "arms": {}}

            def run_arm(name, sel, weights=None, student=None):
                model = student if student is not None else teacher
                if weights is not None:
                    for l in range(L):
                        set_output_weight(teacher, l, weights[l])
                pred, loss, p1, labels = evaluate(model, vl, a.device, head_mask_from_selection(L, H, sel, a.device))
                if weights is not None:
                    for l in range(L):
                        set_output_weight(teacher, l, Ws[l])
                arrays[f"{name}__pred"], arrays[f"{name}__loss"], arrays[f"{name}__p1"] = pred, loss, p1
                rec["arms"][name] = summarize(pred, loss, p1, labels)
                print(f"  {task} seed={seed} {name:28s} {spec['metric']}={rec['arms'][name][spec['metric']]:.4f}")

            for n, sel in sels.items():
                run_arm(f"{n}|none", sel)
                run_arm(f"{n}|merge", sel, [merge_weights(Gs[l], Ws[l], sel[l], assignment(Fs[l], sel[l]), H)
                                            for l in range(L)])
                run_arm(f"{n}|ls_oneshot", sel, [ls_refit(Gs[l], Gs[l] @ Ws[l], sel[l], H, ridge) for l in range(L)])
                student = copy.deepcopy(teacher)
                sequential_ls_refit(student, teacher, cl, a.device, sel, H, ridge)
                run_arm(f"{n}|ls_seq", sel, student=student)
                del student
            rng = random.Random(seed + 80000)  # H9d: merge into a random kept head instead of pi(j)
            sel = sels["bcm"]
            for r in range(a.merge_random_targets):
                wr = [merge_weights(Gs[l], Ws[l], sel[l], {j: rng.choice(sel[l]) for j in range(H) if j not in sel[l]},
                                    H) for l in range(L)]
                run_arm(f"bcm|merge_randtarget{r}", sel, wr)
            np.savez_compressed(stem + ".npz", **arrays)
            json.dump(rec, open(stem + ".json", "w"), indent=2)

        rand_path = os.path.join(a.output_dir, f"v14_{task}_random.json")
        if not a.diagnostics_only and a.random_masks > 0 and not os.path.exists(rand_path):
            # Random distribution: one per task, masks independent of the calibration seed
            rand = []
            for i in range(a.random_masks):
                sel = random_selection(L, H, k, 2027 + 7000 + i)
                validate_selection(sel, L, H, k, name=f"Random[{i}]")
                rand.append(summarize(*evaluate(teacher, vl, a.device, head_mask_from_selection(L, H, sel, a.device))))
            json.dump({"task": task, "k": k, "random_masks": a.random_masks, "per_mask": rand},
                      open(rand_path, "w"), indent=2)


if __name__ == "__main__":
    main()
