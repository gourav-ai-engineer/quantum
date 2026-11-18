"""V20 pilot (docs/PROJECT_STATE.md, "V20 PILOT pre-registration"): the ONE pinned fine-tuning recipe.

MuRIL (or --model-id) on GLUE MNLI train (English), 2 epochs, AdamW lr 2e-5, wd 0.01, linear warmup 10% + linear
decay, batch 32, max_len 128, fp16, seed 42. Saves pytorch_model.bin (old transformers in the official codes cannot read
safetensors) + tokenizer to <out>/mnli, then the teacher gate: MNLI validation_matched, XNLI-en and IndicXNLI validation
accuracy per language -> <out>/ft_meta.json (with the weights sha256). V20: a single fine-tune. V21: --seed 42/43/44
gives three separate teachers (units of analysis, never pooled as calibration seeds); --model-id selects the pin and gate.

    python scripts/v20_finetune.py --out /tmp/v20/ft
    python scripts/v20_finetune.py --out /tmp/ftsmoke --smoke --allow-cpu --model-id hf-internal-testing/tiny-random-BertModel
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time

import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

MURIL = ("google/muril-base-cased", "afd9f36c7923d54e97903922ff1b260d091d202f")
INDIC = ("as", "bn", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te")
INDICXNLI = ("Divyanshu/indicxnli", "7092c27872e919f31d0496fb8b9c47bd2cba3f6c")
XNLI = ("facebook/xnli", "b8dd5d7af51114dbda02c0e3f6133f332186418e")
GATE = {"mnli_m": 0.78, "as": 0.55}
# V21 (docs/V21_PREREG_DRAFT_2026-11-18.md): mBERT pinned at E2; its gate uses hi (little Assamese pretraining data).
MBERT = ("google-bert/bert-base-multilingual-cased", "3f076fdb1ab68d5b2880cb87a0886f315b8146f8")
GATES = {MURIL[0]: GATE, MBERT[0]: {"mnli_m": 0.78, "hi": 0.55}}
PINS = {MURIL[0]: MURIL[1], MBERT[0]: MBERT[1]}


def recipe(seed=42):
    return f"MNLI train, 2 ep, AdamW 2e-5 wd0.01, warmup 10% linear, bs32, len128, fp16, seed {seed}"


def gate_for(model_id):
    return GATES.get(model_id, GATE)


def load_indicxnli(lang, split, allow_test=False):
    """IndicXNLI 'forward' JSON files read directly at the pinned revision (the repo's loading script is third-party code
    and is NOT executed). Label ints: entailment 0, neutral 1, contradiction 2. The test split is reserved for the frozen
    V21 confirmatory kernels: it loads only with allow_test=True."""
    if split not in ("train", "validation") and not (split == "test" and allow_test):
        raise ValueError("IndicXNLI test is reserved for a confirmatory freeze (docs/PROJECT_STATE.md, V20 pilot)")
    from huggingface_hub import hf_hub_download
    sub = {"validation": "dev", "train": "train", "test": "test"}[split]
    f = hf_hub_download(INDICXNLI[0], f"forward/{sub}/xnli_{lang}.json", repo_type="dataset", revision=INDICXNLI[1])
    data = json.load(open(f, encoding="utf-8"))
    rows = data[next(iter(data))]
    return Dataset.from_dict({k: [r[k] for r in rows] for k in ("premise", "hypothesis", "label")})


def eval_sets(smoke=False):
    """name -> dataset with premise/hypothesis/label (label order entailment, neutral, contradiction everywhere)."""
    sets = {"mnli_m": load_dataset("glue", "mnli")["validation_matched"],
            "en": load_dataset(XNLI[0], "en", revision=XNLI[1])["validation"]}
    for lang in INDIC:
        sets[lang] = load_indicxnli(lang, "validation")
    return {k: v.select(range(64)) if smoke else v for k, v in sets.items()}


@torch.no_grad()
def predict(model, tok, rows, device, bs=128, max_len=128):
    model.eval()
    out = []
    for i in range(0, len(rows), bs):
        ch = rows[i:i + bs]
        enc = tok(ch["premise"], ch["hypothesis"], truncation=True, max_length=max_len, padding=True, return_tensors="pt")
        with torch.autocast("cuda", enabled=device == "cuda"):
            out.append(model(**{k: v.to(device) for k, v in enc.items()}).logits.float().argmax(-1).cpu())
    return torch.cat(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--model-id", default=MURIL[0])
    p.add_argument("--revision", default=None, help="default: the pinned MuRIL revision")
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=42, help="V21 teachers: 42, 43, 44")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--allow-cpu", action="store_true")
    a = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and not (a.smoke and a.allow_cpu):
        raise SystemExit("fine-tuning needs a GPU; CPU only with --smoke --allow-cpu")
    rev = a.revision or PINS.get(a.model_id)
    torch.manual_seed(a.seed)
    tok = AutoTokenizer.from_pretrained(a.model_id, revision=rev)
    model = AutoModelForSequenceClassification.from_pretrained(a.model_id, revision=rev, num_labels=3).to(device)
    train = load_dataset("glue", "mnli")["train"]
    if a.smoke:
        train = train.select(range(256))
    bs, steps_per_epoch = 32, (len(train) + 31) // 32
    total = int(steps_per_epoch * a.epochs)
    decay = [p_ for n, p_ in model.named_parameters() if not any(k in n for k in ("bias", "LayerNorm"))]
    no_decay = [p_ for n, p_ in model.named_parameters() if any(k in n for k in ("bias", "LayerNorm"))]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.01}, {"params": no_decay, "weight_decay": 0.0}], lr=2e-5)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * total), total)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    g = torch.Generator().manual_seed(a.seed)
    step, t0 = 0, time.time()
    model.train()
    while step < total:
        order = torch.randperm(len(train), generator=g).tolist()
        for i in range(0, len(order), bs):
            if step >= total:
                break
            ch = train[order[i:i + bs]]
            enc = tok(ch["premise"], ch["hypothesis"], truncation=True, max_length=128, padding=True, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.autocast("cuda", enabled=device == "cuda"):
                loss = model(**enc, labels=torch.tensor(ch["label"], device=device)).loss
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            if step % 500 == 0 or step == total:
                print(f"step {step}/{total} loss {loss.item():.4f} ({time.time() - t0:.0f}s)", flush=True)
    ck = os.path.join(a.out, "mnli")
    model.save_pretrained(ck, safe_serialization=False)
    tok.save_pretrained(ck)
    sha = hashlib.sha256(open(os.path.join(ck, "pytorch_model.bin"), "rb").read()).hexdigest()
    acc = {}
    for name, rows in eval_sets(a.smoke).items():
        acc[name] = float((predict(model, tok, rows, device) == torch.tensor(rows["label"])).float().mean())
        print(f"teacher {name}: {acc[name]:.4f} (n={len(rows)})", flush=True)
    gate_rule = gate_for(a.model_id)
    gate = all(acc[k] >= v for k, v in gate_rule.items())
    meta = {"model_id": a.model_id, "revision": rev, "recipe": recipe(a.seed), "epochs": a.epochs, "steps": total, "weights_sha256": sha,
            "validation_accuracy": acc, "gate": gate_rule, "gate_pass": gate, "mode": "smoke" if a.smoke else "full",
            "seconds": time.time() - t0}
    json.dump(meta, open(os.path.join(a.out, "ft_meta.json"), "w"), indent=1)
    print("GATE", "PASS" if gate else "FAIL", flush=True)


if __name__ == "__main__":
    main()
