"""ESM-2 pilot (docs/EXPLORATION_LOG.md E6 + amendment E6-A1). Development-only kill test, not a result.

  prepare  : pin data, apply the assay rule, sample calibration -> results/esm2_pilot/{inputs.json,calibration.json}
  run      : arms A-E at budgets 0.225 / 0.5 for selectors grp / magnitude -> results/esm2_pilot/<commit>/
  latency  : CPU latency + parameter counts for the pruned shapes, C, D, E  -> <same dir>/latency.json
Use .venv-ci only. CPU is enough.
"""
from __future__ import annotations

import argparse
import copy
import gc
import gzip
import hashlib
import json
import platform
import random
import shutil
import statistics
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "esm2_pilot"
MODELS = {"150M": "facebook/esm2_t30_150M_UR50D", "35M": "facebook/esm2_t12_35M_UR50D"}
PG, UR = "OATML-Markslab/ProteinGym_v1", "zpn/uniref50"
BUDGETS, SELECTORS = (0.225, 0.5), ("grp", "magnitude")


def sha256(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def git(*a) -> str:
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def load_pg(rev, files):
    from huggingface_hub import hf_hub_download
    paths = [hf_hub_download(PG, f, repo_type="dataset", revision=rev) for f in files]
    cols = ["DMS_id", "mutant", "DMS_score", "target_seq"]
    return paths, pd.concat([pd.read_parquet(p, columns=cols) for p in paths], ignore_index=True)


# ---------------------------------------------------------------------------------------------------------------- prepare
def prepare(_):
    from huggingface_hub import HfApi, hf_hub_download
    from qfc import esm_pilot as ep
    api = HfApi()
    RES.mkdir(parents=True, exist_ok=True)
    info = api.repo_info(PG, repo_type="dataset")
    files = sorted(s.rfilename for s in info.siblings if s.rfilename.startswith("DMS_substitutions/"))
    paths, df = load_pg(info.sha, files)
    assays = ep.select_assays(df)
    print("assays:", assays)
    sha = json.load(urllib.request.urlopen("https://api.github.com/repos/OATML-Markslab/ProteinGym/commits/main"))["sha"]
    url = f"https://raw.githubusercontent.com/OATML-Markslab/ProteinGym/{sha}/reference_files/DMS_substitutions.csv"
    ref_path = RES / "DMS_substitutions_reference.csv"
    urllib.request.urlretrieve(url, ref_path)
    ref = pd.read_csv(ref_path).set_index("DMS_id")
    meta = []
    for a in assays:
        wt, muts, _s = ep.single_mutants(df[df.DMS_id == a])
        meta.append({"DMS_id": a, "UniProt_ID": ref.loc[a, "UniProt_ID"], "length": len(wt), "n_single": len(muts)})
    drop_ids = {m["UniProt_ID"] for m in meta}
    drop_seq = {df[df.DMS_id == a].target_seq.iloc[0] for a in assays}
    # calibration: shard-level random sample (E6-A1)
    uinfo = api.repo_info(UR, repo_type="dataset")
    shards = sorted(s.rfilename for s in uinfo.siblings if s.rfilename.endswith(".parquet"))
    rng = random.Random(7)
    order = rng.sample(shards, len(shards))  # E6-A2: first 8 shards with >= 100k eligible sequences
    chosen, skipped = [], []
    calib, excl = [], {"repid": 0, "seq": 0}
    for sh in order:
        if len(chosen) == 8:
            break
        t = pd.read_parquet(hf_hub_download(UR, sh, repo_type="dataset", revision=uinfo.sha), columns=["RepID", "sequence"])
        t = t[(t.sequence.str.len() >= 50) & (t.sequence.str.len() <= 400)]
        bad_id, bad_seq = t.RepID.isin(drop_ids), t.sequence.isin(drop_seq)
        t = t[~bad_id & ~bad_seq].reset_index(drop=True)
        if len(t) < 100_000:
            skipped.append(sh)
            continue
        excl["repid"] += int(bad_id.sum())
        excl["seq"] += int(bad_seq.sum())
        chosen.append(sh)
        calib += [(t.RepID[i], t.sequence[i]) for i in sorted(rng.sample(range(len(t)), 128))]
    (RES / "calibration.json").write_text(json.dumps(calib))
    inputs = {
        "models": {k: {"id": v, "revision": api.repo_info(v).sha} for k, v in MODELS.items()},
        "proteingym": {"repo": PG, "revision": info.sha, "files": {f: sha256(p) for f, p in zip(files, paths)}},
        "reference": {"url": url, "github_commit": sha, "sha256": sha256(ref_path)},
        "assays": meta,
        "uniref": {"repo": UR, "revision": uinfo.sha, "n_shards_total": len(shards), "shards": chosen, "skipped_shards": skipped,
                   "n_sequences": len(calib), "excluded_before_sampling": excl,
                   "calibration_sha256": sha256(RES / "calibration.json")},
        "median_wt_length": int(statistics.median(m["length"] for m in meta)),
    }
    (RES / "inputs.json").write_text(json.dumps(inputs, indent=1))
    print(json.dumps({k: inputs[k] for k in ("assays", "uniref", "median_wt_length")}, indent=1))


# -------------------------------------------------------------------------------------------------------------------- run
def load_models(inputs):
    from transformers import AutoTokenizer, EsmForMaskedLM
    m = {k: EsmForMaskedLM.from_pretrained(v["id"], revision=v["revision"], torch_dtype=torch.float32).eval()
         for k, v in inputs["models"].items()}
    t = inputs["models"]["150M"]
    return m["150M"], m["35M"], AutoTokenizer.from_pretrained(t["id"], revision=t["revision"])


def run(args):
    from qfc import esm_pilot as ep, grp
    torch.set_num_threads(args.threads)
    inputs = json.loads((RES / "inputs.json").read_text())
    calib = json.loads((RES / "calibration.json").read_text())
    assert sha256(RES / "calibration.json") == inputs["uniref"]["calibration_sha256"]
    commit = git("rev-parse", "--short", "HEAD")
    dirty = bool(git("status", "--porcelain", "--untracked-files=no", "--", "src", "scripts", "tests", "docs/EXPLORATION_LOG.md", ":(exclude)*.pyc"))
    out = RES / ("_smoke" if args.smoke else commit)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(RES / "inputs.json", out / "inputs.json")
    teacher, small, tok = load_models(inputs)
    _, df = load_pg(inputs["proteingym"]["revision"], list(inputs["proteingym"]["files"]))
    ids = [m["DMS_id"] for m in inputs["assays"]][: 2 if args.smoke else None]
    assay_data = {a: ep.single_mutants(df[df.DMS_id == a]) for a in ids}
    del df
    seqs = [s for _, s in calib][: 64 if args.smoke else None]
    batches = ep.make_batches(seqs, tok, 8)
    meta = {"commit": commit, "dirty_tracked": dirty, "smoke": args.smoke, "threads": args.threads,
            "python": platform.python_version(), "torch": torch.__version__, "n_assays": len(ids),
            "n_calibration": len(seqs), "start": time.strftime("%F %T")}
    res, all_scores = {"meta": meta, "arms": {}}, {}
    t150 = ep.mac_per_token(teacher)

    carried = {}
    if args.resume_from:  # E6-A3: D, C, E carried over from the killed run; per-variant scores re-derived and checked
        prev = json.loads((Path(args.resume_from) / "results.json").read_text())
        carried = {k: v for k, v in prev["arms"].items() if k in ("D_dense150M", "C_dense35M", "E_int8_150M")}
        assert len(carried) == 3 and prev["meta"]["n_assays"] == len(ids) and not prev["meta"]["smoke"]
        meta["carried_from"] = {"commit": prev["meta"]["commit"], "arms": sorted(carried)}

    def record(name, model, extra=None, counted=True):
        rho, scores = {}, {}
        for a, (wt, muts, y) in assay_data.items():
            s = ep.wt_marginal_scores(model, tok, wt, muts)
            rho[a], scores[a] = float(np.corrcoef(pd.Series(s).rank().to_numpy(), pd.Series(y).rank().to_numpy())[0, 1]), np.round(s, 4).tolist()  # Spearman = Pearson of average ranks
        if name in carried:
            r = dict(carried[name])
            r["rescore_max_abs_diff"] = max(abs(rho[a] - r["rho"][a]) for a in rho)
            res["arms"][name], all_scores[name] = r, scores
            (out / "results.json").write_text(json.dumps(res, indent=1))
            print(f"{name:24s} carried mean rho {r['mean_rho']:.4f}  rescore diff {r['rescore_max_abs_diff']:.2e}", flush=True)
            return
        r = {"rho": rho, "mean_rho": float(np.mean(list(rho.values()))),
             "n_variants": {a: len(v[1]) for a, v in assay_data.items()}, **(extra or {})}
        if counted:
            r["mac_ratio"], r["params"] = ep.mac_per_token(model) / t150, ep.param_counts(model)
        res["arms"][name], all_scores[name] = r, scores
        (out / "results.json").write_text(json.dumps(res, indent=1))
        print(f"{name:24s} mean rho {r['mean_rho']:.4f}", flush=True)

    record("D_dense150M", teacher)
    record("C_dense35M", small)
    q = torch.ao.quantization.quantize_dynamic(copy.deepcopy(teacher), {torch.nn.Linear}, dtype=torch.qint8)
    record("E_int8_150M", q, {"note": "dynamic qint8 of every nn.Linear incl. lm_head; MACs and params as D"}, counted=False)
    del q
    cache = Path(tempfile.mkdtemp(prefix="esm2_grams_"))
    try:
        print("teacher stats ...", flush=True)
        stats = grp.teacher_stats(teacher, batches, "cpu", seed=0, store_dir=cache)  # E6-A3: Grams on disk, same numbers
        res["sensitivity"] = {f"{l}:{k}": v for (l, k), v in stats["sens"].items()}
        for sel in SELECTORS:
            for b in BUDGETS:
                heads, neurons, _info = ep.select_grp(stats, teacher, b) if sel == "grp" else ep.select_magnitude(teacher, b)
                L = len(heads)
                for l in range(L):  # kept units distinct, in range, >= 1 FFN neuron per layer (rule 1, global allocation)
                    assert len(set(heads[l])) == len(heads[l]) and all(0 <= h < teacher.config.num_attention_heads for h in heads[l])
                    assert len(set(neurons[l])) == len(neurons[l]) >= 1
                print(f"{sel}@{b}: heads/layer {[len(heads[l]) for l in range(L)]}\n   neurons/layer {[len(neurons[l]) for l in range(L)]}", flush=True)
                student = ep.build_student(teacher, heads, neurons)
                extra = {"selector": sel, "budget": b, "heads_kept": {str(l): heads[l] for l in heads},
                         "neurons_kept": {str(l): neurons[l] for l in neurons}}
                record(f"A|{sel}|{b}", student, extra)
                grp.sequential_refit(student, teacher, batches, "cpu", 1e-4, drift_correct=True)
                record(f"B|{sel}|{b}", student, extra)
                del student
    finally:
        stats = None  # release the memmaps before deleting the files
        gc.collect()
        shutil.rmtree(cache, ignore_errors=True)
    meta["end"] = time.strftime("%F %T")
    (out / "results.json").write_text(json.dumps(res, indent=1))
    with gzip.open(out / "scores.json.gz", "wt") as f:
        json.dump(all_scores, f)
    (out / "run_meta.json").write_text(json.dumps(meta, indent=1))
    print("saved", out)


# ---------------------------------------------------------------------------------------------------------------- latency
def latency(args):
    from qfc import esm_pilot as ep
    torch.set_num_threads(args.threads)
    inputs = json.loads((RES / "inputs.json").read_text())
    out = RES / ("_smoke" if args.smoke else git("rev-parse", "--short", "HEAD"))
    res = json.loads((out / "results.json").read_text())
    teacher, small, _tok = load_models(inputs)
    L = inputs["median_wt_length"] + 2
    models = {"D_dense150M": teacher, "C_dense35M": small,
              "E_int8_150M": torch.ao.quantization.quantize_dynamic(copy.deepcopy(teacher), {torch.nn.Linear}, dtype=torch.qint8)}
    for b in BUDGETS:
        a = res["arms"][f"A|grp|{b}"]
        models[f"AB_pruned_grp_{b}"] = ep.build_student(teacher, {int(l): v for l, v in a["heads_kept"].items()},
                                                        {int(l): v for l, v in a["neurons_kept"].items()})
    table = {}
    for name, m in models.items():
        m.eval()
        row = {"params": None if name == "E_int8_150M" else ep.param_counts(m)}
        for bs in (1, 8):
            ids = torch.randint(4, 24, (bs, L))
            ids[:, 0], ids[:, -1] = 0, 2
            with torch.inference_mode():
                for _ in range(5):
                    m(input_ids=ids)
                ts = []
                for _ in range(30):
                    t0 = time.perf_counter()
                    m(input_ids=ids)
                    ts.append(time.perf_counter() - t0)
            row[f"batch{bs}_ms_median"] = 1000 * statistics.median(ts)
        table[name] = row
        print(name, row, flush=True)
    (out / "latency.json").write_text(json.dumps({"threads": args.threads, "seq_len_tokens": L, "cpu": platform.processor(),
                                                  "torch": torch.__version__, "table": table}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare", "run", "latency"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--resume-from", default=None, help="results dir of the killed run; carries over arms D, C, E")
    a = ap.parse_args()
    {"prepare": prepare, "run": run, "latency": latency}[a.cmd](a)
