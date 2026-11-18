"""V21 confirmatory runner for ONE unit = (model, teacher seed) (docs/V21_PREREG_DRAFT_2026-11-18.md). The V20 pilot
pipeline (v20-pilot-freeze-fix2) unchanged: official English K-prune masks, content-matched English / mixed calibration
rows, repairs none / d2i_r / d2i_ri. Evaluation: XNLI-en + IndicXNLI in 11 languages, label-parallel rows; the TEST split
only with --allow-test (frozen V21 kernels), otherwise validation (smoke / engineering).

  --write-extra-eval PATH  write the evaluation sets as the add-on's V21_EXTRA_EVAL JSON and exit (run BEFORE the
                           official baselines, so K-prune / Kwon students predict on exactly these rows);
  default                  our conditions mask_only, d2i_r_en, d2i_r_mixed, d2i_ri_en -> <out>/<unit>/<cond>_seed<s>.json;
                           with --official-root, the official students' extra-set dumps -> kprune_/kwon_seed<s>.json;
                           then <out>/<unit>/integrity.json (labels / sizes / teacher parity / MAC / kept units).

    PYTHONPATH=scripts python scripts/v21_run.py --ckpt <dir> --unit muril_t42 --kprune-masks <npz> --index-dir <dir> \
        --official-root <bl dir> --ft-meta <ft_meta.json> --output-dir <root> --allow-test
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch
from datasets import load_dataset

from beat_run import kprune_masks, mac_ratio
from qfc.grp import AVG_SEQ_LEN, _layers, ffn_in_of
from qfc.hf_experiments import load_sequence_classifier
from step22_grp import logits_of, make_batches
from v20_finetune import INDIC, XNLI, load_indicxnli
from v20_pilot import FIELDS, MIX, align_rows, calibration_rows, parallel_evals, pick_calibration, repair

OURS = (("mask_only", "none", "none"), ("d2i_r_en", "d2i_r", "en"), ("d2i_r_mixed", "d2i_r", "mixed"),
        ("d2i_ri_en", "d2i_ri", "en"))
OFFICIAL = (("kprune", "kprune_out/kprune_mnli_seed{s}_c{c}"), ("kwon", "kwon_mnli_seed{s}_c{c}"))


def eval_sets(split, allow_test, smoke=False):
    if split == "test" and not allow_test:
        raise SystemExit("the V21 test split needs --allow-test (frozen V21 kernels only)")
    ev = {"en": load_dataset(XNLI[0], "en", revision=XNLI[1])[split]}
    for lang in INDIC:
        ev[lang] = load_indicxnli(lang, split, allow_test=allow_test)
    ev = parallel_evals(ev)
    return {k: v.select(range(64)) for k, v in ev.items()} if smoke else ev


def official_dumps(run_dir, tag, langs):
    """{lang: dump} of v15_<tag>_x_<lang>.pt in an official run dir; None if any is missing."""
    out = {}
    for L in langs:
        p = os.path.join(run_dir, f"v15_{tag}_x_{L}.pt")
        if not os.path.exists(p):
            return None
        out[L] = torch.load(p, map_location="cpu", weights_only=True)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--unit", required=True, help="<model>_t<teacher seed>, e.g. muril_t42")
    p.add_argument("--kprune-masks")
    p.add_argument("--index-dir", default=None)
    p.add_argument("--official-root", default=None, help="bl dir with kprune_out/ and kwon_* run dirs")
    p.add_argument("--ft-meta", default=None, help="teacher ft_meta.json (integrity: validation accuracy)")
    p.add_argument("--seeds", default="7,42,77")
    p.add_argument("--constraint", type=float, default=0.5)
    p.add_argument("--calibration-size", type=int, default=2048)
    p.add_argument("--ridge-rel", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--output-dir")
    p.add_argument("--write-extra-eval")
    p.add_argument("--allow-test", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--allow-cpu", action="store_true")
    a = p.parse_args()
    split = "test" if a.allow_test else "validation"
    if a.write_extra_eval:
        ev = eval_sets(split, a.allow_test, a.smoke)
        json.dump({k: {f: list(v[f]) for f in ("premise", "hypothesis", "label")} for k, v in ev.items()},
                  open(a.write_extra_eval, "w", encoding="utf-8"), ensure_ascii=False)
        print("extra eval", split, {k: len(v) for k, v in ev.items()})
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and not (a.smoke and a.allow_cpu):
        raise SystemExit("V21 runs need a GPU (CLAUDE.md); CPU only with --smoke --allow-cpu")
    out = os.path.join(a.output_dir, a.unit)
    os.makedirs(out, exist_ok=True)
    teacher, tok = load_sequence_classifier(a.ckpt, device)
    teacher.eval()
    cfg, n_seq = teacher.config, AVG_SEQ_LEN["mnli"]
    integ = {"unit": a.unit, "split": split, "mode": "smoke" if a.smoke else "full"}
    if a.ft_meta:  # teacher parity with its own gate (validation XNLI-en, independent of the evaluated split)
        val_en = load_dataset(XNLI[0], "en", revision=XNLI[1])["validation"]
        if a.smoke:
            val_en = val_en.select(range(64))
        pred = logits_of(teacher, make_batches(val_en, tok, FIELDS, 128, a.max_length), device).argmax(-1)
        acc = float((pred == torch.tensor(val_en["label"])).float().mean())
        ref = json.load(open(a.ft_meta))["validation_accuracy"]["en"]
        integ["teacher_val_en"] = {"ours": acc, "ft_meta": ref, "ok": abs(acc - ref) <= 0.002}
    ev = eval_sets(split, a.allow_test, a.smoke)
    langs = list(ev)
    lab = torch.tensor(ev["en"]["label"])
    vb = {k: make_batches(v, tok, FIELDS, 128, a.max_length) for k, v in ev.items()}
    up = {k: logits_of(teacher, b, device).argmax(-1) for k, b in vb.items()}
    integ["evaluation_size"] = len(lab)
    integ["unpruned_accuracy"] = {k: float((v == lab).float().mean()) for k, v in up.items()}
    cache, inv = {}, {}

    def train_of(lang):
        if lang not in cache:
            cache[lang] = load_dataset("glue", "mnli")["train"] if lang == "en" else load_indicxnli(lang, "train")
            if lang != "en":
                m = align_rows(train_of("en")["label"], cache[lang]["label"])
                inv[lang] = {j: i for i, j in m.items()}
                if len(m) < 0.98 * len(cache[lang]):
                    raise SystemExit(f"IndicXNLI-{lang} train: row map covers only {len(m)} rows")
        return cache[lang]

    need = [l for l in MIX if l != "en"]
    for l in need:
        train_of(l)
    base = {"labels": lab.tolist(), "unpruned_preds": {k: v.tolist() for k, v in up.items()}, "split": split,
            "evaluation_size": len(lab), "constraint": a.constraint, "mode": integ["mode"]}
    integ["mac_ratio"], integ["kept_equal_kprune"], integ["official"] = {}, {}, {}
    for seed in [int(s) for s in a.seeds.split(",")]:
        heads, neurons = kprune_masks(a.kprune_masks, "mnli", seed, a.constraint)
        integ["mac_ratio"][seed] = mac_ratio(heads, neurons, cfg, n_seq)
        n = 64 if a.smoke else a.calibration_size
        order = train_of("en").add_column("_row", list(range(len(train_of("en"))))).shuffle(seed=seed)["_row"]
        if a.index_dir:
            official = json.load(open(os.path.join(a.index_dir, f"mnli_seed{seed}.json")))["calibration"]
            assert order[:len(official)] == official, "shuffle differs from the official index file"
        idx, replaced = pick_calibration(order, lambda i: all(i in inv[l] for l in need), n)
        kept = [len(heads[l]) for l in sorted(heads)], [len(neurons[l]) for l in sorted(neurons)]
        for cond, method, lang in OURS:
            path = os.path.join(out, f"{cond}_seed{seed}.json")
            if os.path.exists(path):
                continue
            t0 = time.time()
            rows = None if method == "none" else calibration_rows(
                train_of, idx, lang, lambda l, i: i if l == "en" else inv[l][i])
            cb = None if rows is None else make_batches(rows, tok, FIELDS, a.batch_size, a.max_length)
            s = repair(method, teacher, heads, neurons, cb, device, a.ridge_rel)
            sh = [l.attention.self.num_attention_heads for l in _layers(s)]  # physical shape after pruning
            preds = {k: logits_of(s, b, device).argmax(-1) for k, b in vb.items()}
            rec = {**base, "condition": cond, "seed": seed, "repair": method, "cal_lang": lang,
                   "calibration_mnli_positions": idx, "calibration_rows_replaced": replaced,
                   "preds": {k: v.tolist() for k, v in preds.items()},
                   "accuracy": {k: float((v == lab).float().mean()) for k, v in preds.items()},
                   "heads_kept_per_layer": sh,
                   "neurons_kept_per_layer": [ffn_in_of(l, False).out_features for l in _layers(s)],
                   "seconds": time.time() - t0}
            integ["kept_equal_kprune"][f"{cond}_seed{seed}"] = (rec["heads_kept_per_layer"], rec["neurons_kept_per_layer"]) == kept
            print(f"{a.unit} seed={seed} {cond}: acc={ {k: round(v, 4) for k, v in rec['accuracy'].items()} } "
                  f"({rec['seconds']:.0f}s)", flush=True)
            json.dump(rec, open(path, "w"))
            del s
        for cond, pattern in (OFFICIAL if a.official_root else ()):
            key, d = f"{cond}_seed{seed}", os.path.join(a.official_root, pattern.format(s=seed, c=a.constraint))
            st, te = official_dumps(d, "student", langs), official_dumps(d, "teacher", langs)
            if st is None or te is None:
                integ["official"][key] = "missing"
                continue
            if any(not torch.equal(st[L]["labels"], lab) for L in langs):
                raise SystemExit(f"{key}: official labels differ from ours (rows not the same)")
            preds = {L: st[L]["logits"].argmax(-1) for L in langs}
            rec = {**base, "condition": cond, "seed": seed, "preds": {k: v.tolist() for k, v in preds.items()},
                   "accuracy": {k: float((v == lab).float().mean()) for k, v in preds.items()}}
            json.dump(rec, open(os.path.join(out, f"{key}.json"), "w"))
            integ["official"][key] = {"teacher_rows_differing": {
                L: int((te[L]["logits"].argmax(-1) != up[L]).sum()) for L in langs}}
    integ["gate_pass"] = bool(
        integ.get("teacher_val_en", {"ok": True})["ok"]
        and all(round(m, 3) == round(a.constraint, 3) for m in integ["mac_ratio"].values())
        and all(integ["kept_equal_kprune"].values())
        and all(v != "missing" and max(v["teacher_rows_differing"].values()) <= 2 for v in integ["official"].values()))
    json.dump(integ, open(os.path.join(out, "integrity.json"), "w"), indent=1)
    print("integrity", a.unit, "PASS" if integ["gate_pass"] else "FAIL", flush=True)


if __name__ == "__main__":
    main()
