"""SQuAD v1.1 utilities for V15 H11: feature construction (max length 384, doc stride 128), n-best span
post-processing, and the official answer normalization, exact match and token F1 (per example, for the paired
bootstrap)."""
from __future__ import annotations

import collections
import re
import string

import numpy as np
import torch


def normalize_answer(s: str) -> str:
    """Lower text, remove punctuation, articles and extra whitespace (official SQuAD v1.1 evaluation)."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def f1_score(prediction: str, truth: str) -> float:
    p, t = normalize_answer(prediction).split(), normalize_answer(truth).split()
    common = collections.Counter(p) & collections.Counter(t)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision, recall = same / len(p), same / len(t)
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: str, truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(truth))


def best_over_truths(fn, prediction: str, truths) -> float:
    return max(fn(prediction, t) for t in truths)


def make_features(examples, tok, max_length: int = 384, stride: int = 128):
    """Tokenize (question, context) pairs into overlapping windows. Returns a dict of lists with input_ids,
    attention_mask, token_type_ids, offset_mapping (None outside the context) and example_index."""
    enc = tok([q.lstrip() for q in examples["question"]], examples["context"], truncation="only_second",
              max_length=max_length, stride=stride, return_overflowing_tokens=True, return_offsets_mapping=True)
    sample_map = enc.pop("overflow_to_sample_mapping")
    offsets = []
    for i, off in enumerate(enc["offset_mapping"]):
        seq_ids = enc.sequence_ids(i)
        offsets.append([o if seq_ids[k] == 1 else None for k, o in enumerate(off)])
    enc["offset_mapping"] = offsets
    enc["example_index"] = list(sample_map)
    return enc


def batches_from_features(feats, tok, batch_size: int):
    keys = ("input_ids", "attention_mask", "token_type_ids")
    out = []
    for i in range(0, len(feats["input_ids"]), batch_size):
        chunk = [{k: feats[k][j] for k in keys} for j in range(i, min(i + batch_size, len(feats["input_ids"])))]
        out.append({k: v for k, v in tok.pad(chunk, return_tensors="pt").items() if k in keys})
    return out


@torch.no_grad()
def predict_logits(model, batches, device):
    model.eval()
    starts, ends = [], []
    for b in batches:
        o = model(**{k: v.to(device) for k, v in b.items()})
        for s, e, m in zip(o.start_logits.float().cpu(), o.end_logits.float().cpu(), b["attention_mask"]):
            n = int(m.sum())
            starts.append(s[:n].numpy())
            ends.append(e[:n].numpy())
    return starts, ends


def postprocess(examples, feats, starts, ends, n_best: int = 20, max_answer_length: int = 30):
    """Best answer text per example (highest start+end score over all windows and valid spans)."""
    by_example = collections.defaultdict(list)
    for f, ex in enumerate(feats["example_index"]):
        by_example[ex].append(f)
    preds = []
    for ex in range(len(examples["context"])):
        context, best, best_score = examples["context"][ex], "", -np.inf
        for f in by_example[ex]:
            off, s_log, e_log = feats["offset_mapping"][f], starts[f], ends[f]
            for si in np.argsort(s_log)[-n_best:][::-1]:
                for ei in np.argsort(e_log)[-n_best:][::-1]:
                    if si >= len(off) or ei >= len(off) or off[si] is None or off[ei] is None:
                        continue
                    if ei < si or ei - si + 1 > max_answer_length:
                        continue
                    score = s_log[si] + e_log[ei]
                    if score > best_score:
                        best_score, best = score, context[off[si][0]:off[ei][1]]
        preds.append(best)
    return preds


def per_example_scores(examples, preds):
    truths = [a["text"] for a in examples["answers"]]
    f1 = [best_over_truths(f1_score, p, t) for p, t in zip(preds, truths)]
    em = [best_over_truths(exact_match, p, t) for p, t in zip(preds, truths)]
    return f1, em
