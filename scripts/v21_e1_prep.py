"""V21 E1/E5 inputs (our pinned env; VALIDATION data only, no test split). Writes into --out:

  tok/<key>/        the hub repo's own top-level tokenizer/config files at the pinned revision (what a hub-id load
                    fetches), keys bert (google-bert/bert-base-uncased, head sha recorded), muril, mbert;
  tok/<key>_slow/   the same without tokenizer.json (for an old `tokenizers` that cannot parse a new tokenizer.json);
  pairs.json        fixed premise/hypothesis pairs (MRPC/XNLI-en/IndicXNLI as,bn,hi validation, first 64 each);
  ref_ids/<key>.json token ids under our env (transformers 4.51.3) for the hub id at the pin;
  extra_eval.json   E5 sets for the add-on (V21_EXTRA_EVAL): mnli_m = full GLUE validation_matched (parity with the
                    add-on's own validation dump) + en/as/bn/hi validation, first --n-extra rows;
  ckpt_mbert_rand/  mBERT at the pin with a seeded random 3-way head (E1/E2 loading + MAC check only, never a result);
  prep_meta.json    pins and file sha256.

    PYTHONPATH=scripts python scripts/v21_e1_prep.py --out /tmp/v21e1in
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

from v20_finetune import MBERT, MURIL, XNLI, load_indicxnli
from v21_gates import sha256

BERT = "google-bert/bert-base-uncased"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--n-extra", type=int, default=512)
    a = p.parse_args()
    import torch
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download, list_repo_files, model_info
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    os.makedirs(a.out, exist_ok=True)
    pins = {"bert": (BERT, model_info(BERT).sha), "muril": MURIL, "mbert": MBERT}
    meta = {"pins": pins, "files": {}}
    for key, (rid, rev) in pins.items():
        files = [f for f in list_repo_files(rid, revision=rev) if "/" not in f and f.endswith((".json", ".txt"))]
        for var in (key, key + "_slow"):
            if var.endswith("_slow") and "tokenizer.json" not in files:
                continue
            d = os.path.join(a.out, "tok", var)
            os.makedirs(d, exist_ok=True)
            for f in files:
                if var.endswith("_slow") and f == "tokenizer.json":
                    continue
                shutil.copy(hf_hub_download(rid, f, revision=rev), os.path.join(d, f))
            meta["files"][var] = {f: sha256(os.path.join(d, f)) for f in sorted(os.listdir(d))}
    mrpc = load_dataset("glue", "mrpc")["validation"].select(range(64))
    pairs = {"premise": list(mrpc["sentence1"]), "hypothesis": list(mrpc["sentence2"])}
    sets = {"en": load_dataset(XNLI[0], "en", revision=XNLI[1])["validation"]}
    for lang in ("as", "bn", "hi"):
        sets[lang] = load_indicxnli(lang, "validation")
    for v in sets.values():
        pairs["premise"] += list(v["premise"][:64])
        pairs["hypothesis"] += list(v["hypothesis"][:64])
    json.dump(pairs, open(os.path.join(a.out, "pairs.json"), "w", encoding="utf-8"), ensure_ascii=False)
    os.makedirs(os.path.join(a.out, "ref_ids"), exist_ok=True)
    for key, (rid, rev) in pins.items():
        tok = AutoTokenizer.from_pretrained(rid, revision=rev, use_fast=True)
        enc = tok(pairs["premise"], pairs["hypothesis"], truncation=True, max_length=128)
        json.dump({"ok": True, "ids": enc["input_ids"]}, open(os.path.join(a.out, "ref_ids", key + ".json"), "w"))
    mm = load_dataset("glue", "mnli")["validation_matched"]
    extra = {"mnli_m": {k: list(mm[k]) for k in ("premise", "hypothesis", "label")}}
    for lang, v in sets.items():
        extra[lang] = {k: list(v[k][:a.n_extra]) for k in ("premise", "hypothesis", "label")}
    json.dump(extra, open(os.path.join(a.out, "extra_eval.json"), "w", encoding="utf-8"), ensure_ascii=False)
    torch.manual_seed(0)
    m = AutoModelForSequenceClassification.from_pretrained(MBERT[0], revision=MBERT[1], num_labels=3)
    m.save_pretrained(os.path.join(a.out, "ckpt_mbert_rand"), safe_serialization=False)
    meta["ckpt_mbert_rand_sha256"] = sha256(os.path.join(a.out, "ckpt_mbert_rand", "pytorch_model.bin"))
    meta["extra_eval_rows"] = {k: len(v["label"]) for k, v in extra.items()}
    json.dump(meta, open(os.path.join(a.out, "prep_meta.json"), "w"), indent=1)
    print("prep done", meta["extra_eval_rows"], sorted(meta["files"]))


if __name__ == "__main__":
    main()
