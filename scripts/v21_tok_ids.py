"""V21 E1: token ids of fixed sentence pairs under one tokenizer, in WHATEVER environment runs it (ours, or the official
Kwon / K-prune envs with transformers 4.12 / 4.25), loaded exactly as the official codes load it:
AutoTokenizer.from_pretrained(name, use_fast=True). Old, stable APIs only.

    python v21_tok_ids.py <tokenizer dir or hub id> <pairs.json> <out.json>
pairs.json = {"premise": [...], "hypothesis": [...]}; out.json = {"ok": bool, "ids": [[...], ...] | "error": str}.
"""
import json
import sys


def main(name, pairs, out):
    p = json.load(open(pairs, encoding="utf-8"))
    try:
        import transformers
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(name, use_fast=True)
        enc = tok(p["premise"], p["hypothesis"], truncation=True, max_length=128)
        rec = {"ok": True, "transformers": transformers.__version__, "fast": bool(tok.is_fast),
               "ids": [list(map(int, x)) for x in enc["input_ids"]]}
    except Exception as e:  # recorded, never hidden: the report fails the gate on it
        rec = {"ok": False, "error": "%s: %s" % (type(e).__name__, e)}
    json.dump(rec, open(out, "w"))
    print(name, "ok" if rec["ok"] else rec["error"])


if __name__ == "__main__":
    main(*sys.argv[1:4])
