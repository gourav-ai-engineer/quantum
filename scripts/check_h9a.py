"""H9a check (docs/PROJECT_STATE.md, V14 accuracy run 1): does sequential LS refit lower the per-layer attention-output
error for the BCM and Fisher selections on SST-2 and MRPC? Train data only (never validation); CPU allowed, because
this is a calibration-side diagnostic, not an accuracy result.

Reuses the selections saved by the GPU run (results/v14_bcm/9c4b999). For each (task, seed, selector) and each arm
(none = masked original weights, ls_oneshot, ls_seq), reports per layer ||Y_t - Y_s||_F / ||Y_t||_F of the attention
output projection (bias removed), measured inside the full pruned forward pass, plus the mean KL(teacher || pruned)
of the logits and the agreement of predicted labels with the unpruned teacher. It measures on the calibration set
(in-sample) and on a disjoint held-out slice of train (the next 512 examples of the same shuffle).

    PYTHONPATH=scripts python scripts/check_h9a.py --tasks sst2,mrpc --seeds 7,42,77,123,2024 \
        --selections results/v14_bcm/9c4b999 --output-dir results/v14_h9a_check/<commit>
"""
from __future__ import annotations

import argparse
import copy
import json
import os

import torch
from datasets import load_dataset

from qfc.hf_experiments import head_mask_from_selection, load_sequence_classifier, make_text_loader
from qfc.output_coverage import (_capture, collect_context_grams, ls_refit, output_weight, sequential_ls_refit,
                                 set_output_weight)
from step20_round2 import SPECS


@torch.no_grad()
def measure(model, teacher, loader, mask, L):
    """Per-layer relative output error, mean logit KL(teacher || model), label agreement with the teacher."""
    num, den = torch.zeros(L, dtype=torch.float64), torch.zeros(L, dtype=torch.float64)
    kl, agree, n = 0.0, 0, 0
    with _capture(model, range(L), want_out=True) as s, _capture(teacher, range(L), want_out=True) as t:
        for batch in loader:
            batch.pop("labels", None)
            ls = model(**batch, head_mask=mask, return_dict=True).logits.double()
            lt = teacher(**batch, return_dict=True).logits.double()
            m = batch["attention_mask"].bool()
            for l in range(L):
                ys, yt = s[l]["y"][m].double(), t[l]["y"][m].double()
                num[l] += ((ys - yt) ** 2).sum()
                den[l] += (yt ** 2).sum()
            pt = lt.log_softmax(-1)
            kl += (pt.exp() * (pt - ls.log_softmax(-1))).sum().item()
            agree += int((ls.argmax(-1) == lt.argmax(-1)).sum())
            n += lt.shape[0]
    return {"layer_rel_error": (num / den).sqrt().tolist(), "logit_kl": kl / n, "teacher_agreement": agree / n}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", default="sst2,mrpc")
    p.add_argument("--seeds", default="7,42,77,123,2024")
    p.add_argument("--selectors", default="bcm,fisher")
    p.add_argument("--selections", default="results/v14_bcm/9c4b999")
    p.add_argument("--calibration-size", type=int, default=512)
    p.add_argument("--ridge-rel", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--output-dir", default="results/v14_h9a_check")
    a = p.parse_args()
    os.makedirs(a.output_dir, exist_ok=True)
    n = a.calibration_size
    for task in a.tasks.split(","):
        spec = SPECS[task]
        dn, cfg = spec["dataset"]
        train = (load_dataset(dn, cfg) if cfg else load_dataset(dn))["train"]
        teacher, tok = load_sequence_classifier(spec["model_id"], "cpu", revision=spec["revision"])
        L, H = teacher.config.num_hidden_layers, teacher.config.num_attention_heads
        Ws = [output_weight(teacher, l) for l in range(L)]
        for seed in [int(s) for s in a.seeds.split(",")]:
            path = os.path.join(a.output_dir, f"h9a_{task}_seed{seed}.json")
            if os.path.exists(path):
                continue
            shuf = train.shuffle(seed=seed)  # same shuffle as step21: first n = calibration set
            kw = dict(text_fields=spec["text_fields"], batch_size=a.batch_size, max_length=a.max_length)
            cl = make_text_loader(shuf.select(range(n)), tok, **kw)
            hl = make_text_loader(shuf.select(range(n, 2 * n)), tok, **kw)
            saved = json.load(open(os.path.join(a.selections, f"v14_{task}_seed{seed}.json")))["selections"]
            Gs = collect_context_grams(teacher, cl, "cpu")
            out = {"task": task, "seed": seed, "calibration_size": n, "heldout": f"train.shuffle({seed})[{n}:{2*n}]",
                   "ridge_rel": a.ridge_rel, "selections_from": a.selections, "results": {}}
            for name in a.selectors.split(","):
                sel = {int(l): s for l, s in saved[name].items()}
                mask = head_mask_from_selection(L, H, sel, "cpu")
                one = copy.deepcopy(teacher)
                for l in range(L):
                    set_output_weight(one, l, ls_refit(Gs[l], Gs[l] @ Ws[l], sel[l], H, a.ridge_rel))
                seq = copy.deepcopy(teacher)
                sequential_ls_refit(seq, teacher, cl, "cpu", sel, H, a.ridge_rel)
                res = {}
                orig = copy.deepcopy(teacher)  # separate module: hooks on the reference teacher must not see it
                for arm, model in (("none", orig), ("ls_oneshot", one), ("ls_seq", seq)):
                    res[arm] = {"calibration": measure(model, teacher, cl, mask, L),
                                "heldout": measure(model, teacher, hl, mask, L)}
                    print(f"{task} seed={seed} {name:7s} {arm:10s} "
                          f"cal: err_last={res[arm]['calibration']['layer_rel_error'][-1]:.3f} "
                          f"kl={res[arm]['calibration']['logit_kl']:.4f} | held: "
                          f"err_last={res[arm]['heldout']['layer_rel_error'][-1]:.3f} "
                          f"kl={res[arm]['heldout']['logit_kl']:.4f} agree={res[arm]['heldout']['teacher_agreement']:.3f}",
                          flush=True)
                out["results"][name] = res
                del one, seq, orig
            json.dump(out, open(path, "w"), indent=2)


if __name__ == "__main__":
    main()
