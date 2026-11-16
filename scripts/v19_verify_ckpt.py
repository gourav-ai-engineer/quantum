"""V19: abort unless every named task's pytorch_model.bin under <root>/<task>/ has the sha256 pinned in the JSON file
(scripts/V19_CKPT_SHA256.json). The official DistilBERT-weights folders have no revision, so the hash is the pin.

    python scripts/v19_verify_ckpt.py <root> <sha.json> <task> [<task> ...]
"""
import hashlib
import json
import os
import sys

root, shafile, tasks = sys.argv[1], sys.argv[2], sys.argv[3:]
pins = json.load(open(shafile))["sha256"]
for t in tasks:
    h = hashlib.sha256()
    with open(os.path.join(root, t, "pytorch_model.bin"), "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    if h.hexdigest() != pins[t]:
        raise SystemExit(f"checkpoint {t}: sha256 {h.hexdigest()} != pinned {pins[t]}")
    print(t, "sha256 OK")
