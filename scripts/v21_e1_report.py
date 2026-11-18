"""V21 E1/E2/E5 report (our env; reads only the kernel's own outputs and committed V15/V20 files; VALIDATION data only).

    python scripts/v21_e1_report.py choose <E> <IN21> kwon|kprune <key>   -> prints the tokenizer dir name to use
    python scripts/v21_e1_report.py report <E> <IN21> --v20-npz <npz> --v15-dir <dir> --out <json>

Decisive E1 criteria (docs/V21_PREREG_DRAFT_2026-11-18.md): a local tokenizer dir gives token ids identical to our env's
pinned hub tokenizer, in both official envs, for bert/muril/mbert; and the same-kernel A/B on BERT-base MRPC (hub id vs
local dir) gives identical keep patterns and identical validation predictions for both codes. Raw mask values, a
hub-vs-hub rerun (GPU determinism) and the V15-era comparison are reported, not decisive. E5: the add-on's extra-set dump
of MNLI-m validation equals its own validation dump, row for row.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

KEYS = ("bert", "muril", "mbert")


def _load(p):
    return json.load(open(p)) if os.path.exists(p) else {"ok": False, "error": "missing " + os.path.basename(p)}


def tok_match(E, IN21, env, var, key):
    r, ref = _load(os.path.join(E, f"tok_{env}_{var}.json")), _load(os.path.join(IN21, "ref_ids", key + ".json"))
    return bool(r.get("ok")) and r["ids"] == ref["ids"]


def choose(E, IN21, env, key):
    for var in (key, key + "_slow"):
        if os.path.isdir(os.path.join(IN21, "tok", var)) and tok_match(E, IN21, env, var, key):
            return "v21tok_" + var
    return None


def masks(d):
    """(head, neuron) float tensors of a Kwon (head_mask.pt) or K-prune (head_masks.pkl) run dir, or None."""
    for h, n in (("head_mask.pt", "neuron_mask.pt"), ("head_masks.pkl", "neuron_masks.pkl")):
        if os.path.exists(os.path.join(d, h)):
            return tuple(torch.as_tensor(torch.load(os.path.join(d, f), map_location="cpu", weights_only=True)).float() for f in (h, n))
    return None


def mac(m, n_seq, hidden=768, heads=12, inter=3072):
    from qfc.grp import model_mac
    hm, nm = m
    hs = hidden // heads
    full = model_mac([heads] * hm.shape[0], [inter] * hm.shape[0], n_seq, hidden, hs)
    return model_mac((hm != 0).sum(1).tolist(), (nm != 0).sum(1).tolist(), n_seq, hidden, hs) / full


def dump(d, tag, split):
    p = os.path.join(d, f"v15_{tag}_{split}.pt")
    return torch.load(p, map_location="cpu", weights_only=True) if os.path.exists(p) else None


def compare_runs(a, b):
    ma, mb = masks(a), masks(b)
    va, vb = dump(a, "student", "validation"), dump(b, "student", "validation")
    if ma is None or mb is None or va is None or vb is None:
        return {"ran": False}
    return {"ran": True, "keep_pattern_equal": all(torch.equal(x != 0, y != 0) for x, y in zip(ma, mb)),
            "mask_values_equal": all(torch.equal(x, y) for x, y in zip(ma, mb)),
            "val_preds_equal": torch.equal(va["logits"].argmax(-1), vb["logits"].argmax(-1)),
            "val_logits_max_abs_diff": float((va["logits"] - vb["logits"]).abs().max())}


def e5_parity(d):
    out = {}
    for tag in ("teacher", "student"):
        v, x = dump(d, tag, "validation"), dump(d, tag, "x_mnli_m")
        if v is None or x is None:
            out[tag] = {"present": False}
            continue
        out[tag] = {"present": True, "logits_equal": torch.equal(v["logits"], x["logits"]),
                    "labels_equal": torch.equal(v["labels"], x["labels"]),
                    "extra_sets": {s: len(dump(d, tag, "x_" + s)["labels"]) for s in ("en", "as", "bn", "hi")
                                   if dump(d, tag, "x_" + s) is not None}}
    return out


def report(E, IN21, v20_npz, v15_dir):
    from qfc.grp import AVG_SEQ_LEN
    r = {"tokenizer": {}, "chosen": {}}
    for env in ("kwon", "kprune"):
        for key in KEYS:
            vars_ = [v for v in (key, key + "_slow") if os.path.isdir(os.path.join(IN21, "tok", v))]
            r["tokenizer"][f"{env}/{key}"] = {v: {"loads": bool(_load(os.path.join(E, f"tok_{env}_{v}.json")).get("ok")),
                                                  "ids_equal_ref": tok_match(E, IN21, env, v, key)} for v in vars_}
            r["chosen"][f"{env}/{key}"] = choose(E, IN21, env, key)
    r["hub_id_loads"] = {os.path.basename(p)[7:-5]: _load(p).get("ok") or _load(p).get("error")
                         for p in sorted(os.path.join(E, f) for f in os.listdir(E) if f.startswith("tokhub_"))}
    ab = {code: {"hub_vs_local": compare_runs(os.path.join(E, f"{code}_mrpc_hub"), os.path.join(E, f"{code}_mrpc_local")),
                 "hub_vs_hub_rerun": compare_runs(os.path.join(E, f"{code}_mrpc_hub"), os.path.join(E, f"{code}_mrpc_hub2"))}
          for code in ("kwon", "kprune")}
    r["ab_mrpc"] = ab
    # V15-era (results/v15_h10/c43dd3e, other environment): reported only
    v15 = {}
    for code, sub in (("kwon", "kwon_mrpc_seed7_c0.5"), ("kprune", os.path.join("kprune_out", "kprune_mrpc_seed7_c0.5"))):
        old = dump(os.path.join(v15_dir, sub), "student", "validation") if v15_dir else None
        new = dump(os.path.join(E, f"{code}_mrpc_local"), "student", "validation")
        v15[code] = None if old is None or new is None else {
            "val_pred_agreement": float((old["logits"].argmax(-1) == new["logits"].argmax(-1)).float().mean()),
            "val_logits_max_abs_diff": float((old["logits"] - new["logits"]).abs().max())}
    r["v15_era_mrpc"] = v15
    n_seq = AVG_SEQ_LEN["mnli"]
    runs = {}
    for name in ("kwon_muril_local", "kprune_muril_local", "kprune_muril_hub", "kwon_mbert_local", "kprune_mbert_local"):
        m = masks(os.path.join(E, name))
        runs[name] = {"ran": m is not None, "mac_ratio": None if m is None else mac(m, n_seq),
                      "heads_kept": None if m is None else (m[0] != 0).sum(1).tolist()}
    r["multilingual_runs"] = runs
    km = masks(os.path.join(E, "kprune_muril_local"))
    if km is not None and v20_npz and os.path.exists(v20_npz):
        z = np.load(v20_npz)
        r["kprune_muril_vs_v20_seed7"] = {
            "head_equal": bool(((km[0] != 0).numpy() == (z["kprune_mnli_seed7_c0.5__head"] != 0)).all()),
            "neuron_equal": bool(((km[1] != 0).numpy() == (z["kprune_mnli_seed7_c0.5__neuron"] != 0)).all())}
    r["kprune_muril_hub_vs_local"] = compare_runs(os.path.join(E, "kprune_muril_hub"), os.path.join(E, "kprune_muril_local"))
    r["e5_parity"] = {n: e5_parity(os.path.join(E, n)) for n in ("kwon_muril_local", "kprune_muril_local")}
    ok_mac = lambda n: runs[n]["ran"] and abs(runs[n]["mac_ratio"] - 0.5) < 0.005
    r["E1_pass"] = (all(r["chosen"][f"{e}/{k}"] for e in ("kwon", "kprune") for k in KEYS)
                    and all(ab[c]["hub_vs_local"].get("ran") and ab[c]["hub_vs_local"]["keep_pattern_equal"]
                            and ab[c]["hub_vs_local"]["val_preds_equal"] for c in ab)
                    and ok_mac("kwon_muril_local") and ok_mac("kprune_muril_local"))
    r["E2_codes_accept_vocab"] = all(ok_mac(n) for n in ("kwon_muril_local", "kprune_muril_local",
                                                         "kwon_mbert_local", "kprune_mbert_local"))
    r["E5_pass"] = all(t["present"] and t["logits_equal"] and t["labels_equal"] and len(t["extra_sets"]) == 4
                       for v in r["e5_parity"].values() for t in v.values())
    return r


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "choose":
        c = choose(*sys.argv[2:6])
        print(c or "NONE")
        sys.exit(0 if c else 1)
    p = argparse.ArgumentParser()
    p.add_argument("cmd")
    p.add_argument("E")
    p.add_argument("IN21")
    p.add_argument("--v20-npz")
    p.add_argument("--v15-dir")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    r = report(a.E, a.IN21, a.v20_npz, a.v15_dir)
    json.dump(r, open(a.out, "w"), indent=1)
    print({k: r[k] for k in ("E1_pass", "E2_codes_accept_vocab", "E5_pass", "chosen")})


if __name__ == "__main__":
    main()
