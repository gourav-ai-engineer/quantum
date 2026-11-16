"""V19 step 1 engineering checks on the official DistilBERT checkpoints (no pruning, no validation data, rule 9):
for every task folder (a) list files, sizes and sha256 of the weights, (b) load under the installed transformers (must be
4.51.3), (c) label order on TRAIN rows (binary: identity vs swapped; MNLI: all 6 permutations; STS-B: Pearson),
(d) SQuAD folders: no-answer rate and token F1 on SQuAD v1.1 TRAIN rows, to tell `squad` (v1.1) from `squad_v2`.

    python scripts/v19_ckpt_check.py <ck_dir>/distilbert-base-uncased --out ckpt_check.json
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os

import torch
import transformers
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForQuestionAnswering, AutoModelForSequenceClassification, AutoTokenizer

from qfc.squad import best_over_truths, f1_score

GLUE = {"sst2": (("glue", "sst2"), ("sentence",)), "mrpc": (("glue", "mrpc"), ("sentence1", "sentence2")),
        "qnli": (("glue", "qnli"), ("question", "sentence")), "qqp": (("glue", "qqp"), ("question1", "question2")),
        "mnli": (("glue", "mnli"), ("premise", "hypothesis")), "stsb": (("glue", "stsb"), ("sentence1", "sentence2"))}


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def listing(d):
    out = {}
    for f in sorted(os.listdir(d)):
        p = os.path.join(d, f)
        if os.path.isfile(p):
            out[f] = {"bytes": os.path.getsize(p)}
            if f.endswith((".bin", ".safetensors")):
                out[f]["sha256"] = sha(p)
            elif os.path.getsize(p) < 3000 and f.endswith((".json", ".txt", ".md")):
                out[f]["text"] = open(p, encoding="utf8", errors="replace").read()
    return out


def tokenizer_for(d):
    try:
        return AutoTokenizer.from_pretrained(d), "folder"
    except Exception:  # noqa: BLE001 - recorded, fallback is the public base tokenizer
        return AutoTokenizer.from_pretrained("distilbert-base-uncased"), "hub distilbert-base-uncased"


@torch.no_grad()
def check_glue(task, d, n=1000):
    (ds_args, fields) = GLUE[task]
    tok, tok_src = tokenizer_for(d)
    model = AutoModelForSequenceClassification.from_pretrained(d, attn_implementation="eager").eval()
    rows = load_dataset(*ds_args)["train"].shuffle(seed=0).select(range(n))
    logits = []
    for i in range(0, n, 50):
        ch = rows[i:i + 50]
        enc = tok(*[ch[f] for f in fields], truncation=True, max_length=128, padding=True, return_tensors="pt")
        logits.append(model(**{k: v for k, v in enc.items() if k in ("input_ids", "attention_mask")}).logits)
    lg, lab = torch.cat(logits), torch.tensor(rows["label"])
    res = {"tokenizer": tok_src, "num_labels": model.config.num_labels, "id2label": model.config.id2label}
    if task == "stsb":
        res["pearson_train_1000"] = float(torch.corrcoef(torch.stack([lg[:, 0].double(), lab.double()]))[0, 1])
        return res
    k = lg.shape[1]
    res["accuracy_by_permutation"] = {str(p): float((torch.tensor(p)[lg.argmax(-1)] == lab).float().mean())
                                      for p in itertools.permutations(range(k))}
    return res


@torch.no_grad()
def check_squad(d, n=300):
    tok, tok_src = tokenizer_for(d)
    model = AutoModelForQuestionAnswering.from_pretrained(d, attn_implementation="eager").eval()
    rows = load_dataset("squad")["train"].shuffle(seed=0).select(range(n))
    f1s, noans, used = [], 0, 0
    for ex in rows:
        enc = tok(ex["question"], ex["context"], truncation="only_second", max_length=384, return_tensors="pt",
                  return_offsets_mapping=True)
        off = enc.pop("offset_mapping")[0]
        seq = enc.sequence_ids(0)
        out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        s, e = int(out.start_logits.argmax()), int(out.end_logits.argmax())
        used += 1
        if s == 0:
            noans += 1
            f1s.append(0.0)
            continue
        if seq[s] != 1 or seq[e] != 1 or e < s:
            f1s.append(0.0)
            continue
        pred = ex["context"][int(off[s][0]):int(off[e][1])]
        f1s.append(best_over_truths(f1_score, pred, ex["answers"]["text"]))
    return {"tokenizer": tok_src, "rows": used, "no_answer_rate": noans / used, "token_f1_train": sum(f1s) / len(f1s)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    out = {"transformers": transformers.__version__, "torch": torch.__version__, "tasks": {}}
    assert transformers.__version__ == "4.51.3", transformers.__version__
    for task in sorted(os.listdir(a.root)):
        d = os.path.join(a.root, task)
        if not os.path.isdir(d):
            continue
        rec = {"files": listing(d)}
        try:
            cfg = AutoConfig.from_pretrained(d)
            rec["config"] = {"model_type": cfg.model_type, "n_layers": getattr(cfg, "n_layers", None),
                             "n_heads": getattr(cfg, "n_heads", None), "dim": getattr(cfg, "dim", None),
                             "hidden_dim": getattr(cfg, "hidden_dim", None)}
            rec["check"] = check_glue(task, d) if task in GLUE else check_squad(d) if task.startswith("squad") else None
            rec["loads"] = True
        except Exception as ex:  # noqa: BLE001 - the check records the failure
            rec["loads"], rec["error"] = False, repr(ex)
        out["tasks"][task] = rec
        print(task, json.dumps({k: v for k, v in rec.items() if k != "files"})[:600], flush=True)
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
