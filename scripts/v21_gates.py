"""V21 engineering gates E2/E3 (docs/V21_PREREG_DRAFT_2026-11-18.md). No model is run; no text is printed or stored.

E2: model pins (resolved hub id + commit sha), architecture dimensions (the MAC formula `beat_run.mac_ratio` reads only
    hidden size / layers / heads / intermediate size and excludes embeddings, so equal dimensions = identical accounting),
    vocabulary sizes.
E3: test-split integrity, LABELS ONLY: XNLI-en test and IndicXNLI forward/test in 11 languages at the pinned revisions;
    row counts, label equality with XNLI-en (V20 rule: keep the first label-parallel rows), file sha256.

    python scripts/v21_gates.py --out results/v21_gates/<commit>/e2_e3.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os

from v20_finetune import INDIC, INDICXNLI, MBERT, MURIL, XNLI

DIMS = ("hidden_size", "num_hidden_layers", "num_attention_heads", "intermediate_size", "model_type")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def label_parallel(labels, ref="en"):
    """{lang: {rows, extra_rows, parallel}}: the first len(labels[ref]) rows of every language must equal ref's labels."""
    n = len(labels[ref])
    return {k: {"rows": len(v), "extra_rows": len(v) - n, "parallel": len(v) >= n and list(v[:n]) == list(labels[ref])}
            for k, v in labels.items()}


def e2():
    from huggingface_hub import hf_hub_download, model_info
    out = {}
    for legacy, (mid, rev) in (("google/muril-base-cased", MURIL), ("bert-base-multilingual-cased", MBERT)):
        info = model_info(legacy)
        cfg = json.load(open(hf_hub_download(mid, "config.json", revision=rev)))
        out[mid] = {"legacy_id": legacy, "resolved_id": info.id, "pinned_revision": rev,
                    "hub_head_revision": info.sha, "head_equals_pin": info.sha == rev,
                    "vocab_size": cfg["vocab_size"], **{k: cfg[k] for k in DIMS}}
    a, b = (out[m] for m in (MURIL[0], MBERT[0]))
    out["same_mac_accounting"] = all(a[k] == b[k] for k in DIMS)
    return out


def e3():
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    files, labels = {}, {}
    f = hf_hub_download(XNLI[0], "en/test-00000-of-00001.parquet", repo_type="dataset", revision=XNLI[1])
    files["en"], labels["en"] = sha256(f), pq.read_table(f, columns=["label"]).column("label").to_pylist()
    for lang in INDIC:
        f = hf_hub_download(INDICXNLI[0], f"forward/test/xnli_{lang}.json", repo_type="dataset", revision=INDICXNLI[1])
        data = json.load(open(f, encoding="utf-8"))
        files[lang], labels[lang] = sha256(f), [r["label"] for r in data[next(iter(data))]]
    rep = label_parallel(labels)
    return {"xnli": XNLI, "indicxnli": INDICXNLI, "file_sha256": files, "per_language": rep,
            "label_counts_en": {str(c): labels["en"].count(c) for c in sorted(set(labels["en"]))},
            "all_parallel": all(r["parallel"] for r in rep.values())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    a = p.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    rec = {"E2": e2(), "E3": e3()}
    json.dump(rec, open(a.out, "w"), indent=1)
    print("E2 same_mac_accounting:", rec["E2"]["same_mac_accounting"],
          {m: (v["vocab_size"], v["head_equals_pin"]) for m, v in rec["E2"].items() if isinstance(v, dict)})
    print("E3 all_parallel:", rec["E3"]["all_parallel"],
          {k: (v["rows"], v["extra_rows"], v["parallel"]) for k, v in rec["E3"]["per_language"].items()})


if __name__ == "__main__":
    main()
