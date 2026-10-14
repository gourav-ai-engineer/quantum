"""V15 add-on for the official Kwon et al. (806ac5a6) and K-prune (9ca629a1) codes. Runs in THEIR environments
(Python 3.9/3.10, transformers 4.12/4.25), so it uses only old, stable APIs.

1. `python v15_official_addon.py patch <main.py> kwon|kprune` edits exactly these places of the official main.py:
   - the calibration sampling line: `np.random.choice(len(training_dataset), N).tolist()` becomes
     `v15_indices(len(training_dataset), N)`, which returns OUR fixed calibration indices when the environment
     variable V15_INDEX_FILE is set and otherwise calls the same `np.random.choice(...)` (so without the variable
     the run is identical to the unpatched code);
   - dump points that save teacher and pruned-model logits on our held-out TRAIN rows and on the validation split
     (only when V15_INDEX_FILE is set; GLUE tasks only).
   Each anchor must occur exactly once or the patch aborts. No other line changes.
2. `v15_dump(...)` writes <out_dir>/v15_<tag>_<split>.pt = {"logits": FloatTensor, "labels": LongTensor, "idx": list}.
"""
import json
import os
import sys

import numpy as np
import torch

TASK_TO_KEYS = {"cola": ("sentence", None), "mnli": ("premise", "hypothesis"), "mrpc": ("sentence1", "sentence2"),
                "qnli": ("question", "sentence"), "qqp": ("question1", "question2"),
                "rte": ("sentence1", "sentence2"), "sst2": ("sentence", None), "stsb": ("sentence1", "sentence2")}


def v15_indices(n_total, n_samples):
    f = os.environ.get("V15_INDEX_FILE")
    if not f:
        return np.random.choice(n_total, n_samples).tolist()
    idx = json.load(open(f))["calibration"]
    assert max(idx) < n_total
    return list(idx)


@torch.no_grad()
def v15_dump(model, tokenizer, task, head_mask, neuron_mask, out_dir, tag):
    f = os.environ.get("V15_INDEX_FILE")
    if not f or task not in TASK_TO_KEYS:
        return
    from datasets import load_dataset
    spec = json.load(open(f))
    raw = load_dataset("glue", task)
    splits = {"heldout": (raw["train"].select(spec["heldout"]), spec["heldout"]),
              "validation": (raw["validation_matched" if task == "mnli" else "validation"], None)}
    k1, k2 = TASK_TO_KEYS[task]
    layers = getattr(model, model.base_model_prefix).encoder.layer
    handles = []
    if neuron_mask is not None:  # Kwon's convention: scale the input of layer.output (the FFN activations)
        for l, layer in enumerate(layers):
            handles.append(layer.output.register_forward_pre_hook(
                lambda m, inp, l=l: (inp[0] * neuron_mask[l],) + tuple(inp[1:])))
    model.eval()
    device = next(model.parameters()).device
    try:
        for name, (ds, idx) in splits.items():
            out = []
            for i in range(0, len(ds), 64):
                rows = ds[i:i + 64]
                args = (rows[k1],) if k2 is None else (rows[k1], rows[k2])
                enc = tokenizer(*args, truncation=True, max_length=128, padding=True, return_tensors="pt")
                enc = {k: v.to(device) for k, v in enc.items()}
                kw = {} if head_mask is None else {"head_mask": head_mask}
                out.append(model(**enc, **kw).logits.float().cpu())
            torch.save({"logits": torch.cat(out), "labels": torch.tensor(ds["label"]),
                        "idx": idx if idx is not None else list(range(len(ds)))},
                       os.path.join(out_dir, "v15_%s_%s.pt" % (tag, name)))
    finally:
        for h in handles:
            h.remove()


PATCHES = {
    "kwon": [
        ("np.random.choice(len(training_dataset), args.num_samples).tolist(),",
         "v15_indices(len(training_dataset), args.num_samples),"),
        ("    # Save the masks\n",
         "    v15_dump(model, tokenizer, args.task_name, None, None, args.output_dir, 'teacher')\n"
         "    v15_dump(model, tokenizer, args.task_name, head_mask, neuron_mask, args.output_dir, 'student')\n"
         "    # Save the masks\n"),
    ],
    "kprune": [
        ("np.random.choice(len(training_dataset), num_samples).tolist(),",
         "v15_indices(len(training_dataset), num_samples),"),
        ("    st = time.time()\n",
         "    v15_dump(model, tokenizer, args.task_name, None, None, OUTPUT_DIR, 'teacher')\n    st = time.time()\n"),
        ("    # Save the model and masks\n",
         "    v15_dump(model, tokenizer, args.task_name, None, None, OUTPUT_DIR, 'student')\n"
         "    # Save the model and masks\n"),
    ],
}


def patch(path, kind):
    src = open(path, newline="").read()  # keep the file's own line endings byte for byte
    if "v15_indices" in src:
        print("already patched:", path)
        return
    for old, new in PATCHES[kind]:
        n = src.count(old)
        if n != 1:
            raise SystemExit("anchor found %d times in %s: %r" % (n, path, old))
        src = src.replace(old, new)
    src = "from v15_official_addon import v15_indices, v15_dump\n" + src
    open(path, "w", newline="").write(src)
    print("patched:", path)


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "patch":
        patch(sys.argv[2], sys.argv[3])
    else:
        raise SystemExit("usage: python v15_official_addon.py patch <main.py> kwon|kprune")
