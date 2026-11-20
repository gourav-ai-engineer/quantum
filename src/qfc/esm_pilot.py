"""ESM-2 pilot (docs/EXPLORATION_LOG.md E6 and amendment E6-A1): assay rule, wild-type-marginal scoring, MAC budget,
magnitude selector. The GRP selector and the drift-corrected refit are the existing qfc.grp code (ESM-2 blocks use
the same (hidden, residual) sublayer signature as BERT, so the hook path is shared)."""
from __future__ import annotations

import copy

import numpy as np
import torch

from . import grp


# ---- assays and scoring ------------------------------------------------------------------------------------------
def select_assays(df, max_len: int = 400, n: int = 20) -> list[str]:
    """E6 assay rule, mechanical: wild-type length <= max_len, >= 1 single mutant with a score; sort by DMS_id; first n.
    df needs columns DMS_id, mutant, DMS_score, target_seq."""
    ok = []
    for dms_id, g in df.groupby("DMS_id"):
        seqs = g["target_seq"].unique()
        if len(seqs) != 1 or len(seqs[0]) > max_len:
            continue
        if (~g["mutant"].str.contains(":") & g["DMS_score"].notna()).any():
            ok.append(dms_id)
    return sorted(ok)[:n]


def single_mutants(g) -> tuple[str, list[tuple[int, str, str]], np.ndarray]:
    """(wild-type sequence, [(pos 1-based, wt aa, mut aa)], scores) for the single, non-NaN mutants of one assay."""
    wt = g["target_seq"].iloc[0]
    g = g[~g["mutant"].str.contains(":") & g["DMS_score"].notna()]
    muts = []
    for m in g["mutant"]:
        a, pos, b = m[0], int(m[1:-1]), m[-1]
        assert wt[pos - 1] == a, f"wild-type mismatch {m}"
        muts.append((pos, a, b))
    return wt, muts, g["DMS_score"].to_numpy(dtype=float)


@torch.no_grad()
def wt_marginal_scores(model, tok, wt: str, muts) -> np.ndarray:
    """log p(mut) - log p(wt) at the mutated position from one unmasked forward pass; token index = pos (<cls> is 0)."""
    model.eval()
    lp = model(**tok(wt, return_tensors="pt")).logits[0].float().log_softmax(-1)
    ids = {c: tok.convert_tokens_to_ids(c) for c in set("ACDEFGHIKLMNPQRSTVWY")}
    return np.array([float(lp[p, ids[b]] - lp[p, ids[a]]) for p, a, b in muts])


# ---- MAC accounting (encoder projections + FFN, attention-score term excluded; per token) --------------------------
def mac_per_token(model) -> float:
    d, hs = model.config.hidden_size, model.config.hidden_size // model.config.num_attention_heads
    layers = grp._layers(model)
    heads = [l.attention.self.num_attention_heads for l in layers]
    neurons = [l.intermediate.dense.out_features for l in layers]
    return grp.model_mac(heads, neurons, 1, d, hs, score_term=False)


def param_counts(model) -> dict:
    total = sum(p.numel() for p in model.parameters())
    return {"total": total, "excl_embeddings": total - model.esm.embeddings.word_embeddings.weight.numel()}


# ---- selectors -----------------------------------------------------------------------------------------------------
def select_grp(stats: dict, teacher, constraint: float, ridge_rel: float = 1e-4):
    """Primary selector: existing GRP greedy with sensitivity weights, score term excluded."""
    L, H = len(grp._layers(teacher)), teacher.config.num_attention_heads
    return grp.grp_select(stats["G"], stats["W"], stats["sens"], L, H, 1, teacher.config.hidden_size, constraint,
                          ridge_rel, score_term=False)


def select_magnitude(teacher, constraint: float):
    """Descriptive selector (E6-A1): unit score = ||output-projection columns||_F^2 / unit MACs, global ascending removal
    until MACs <= constraint * original; >= 1 FFN neuron per layer."""
    cfg = teacher.config
    L, H, d = len(grp._layers(teacher)), cfg.num_attention_heads, cfg.hidden_size
    hs = d // H
    f_att, f_ffn = grp.mac_per_head(1, d, hs, False), grp.mac_per_neuron(1, d)
    units = []
    for l, layer in enumerate(grp._layers(teacher)):
        Wa, Wf = layer.attention.output.dense.weight.detach().double(), layer.output.dense.weight.detach().double()
        for h in range(H):
            units.append((float((Wa[:, h * hs:(h + 1) * hs] ** 2).sum()) / f_att, l, "att", h))
        for n, s in enumerate((Wf ** 2).sum(0).tolist()):
            units.append((s / f_ffn, l, "ffn", n))
    units.sort(key=lambda u: u[0])
    heads = {l: set(range(H)) for l in range(L)}
    neurons = {l: set(range(cfg.intermediate_size)) for l in range(L)}
    orig = grp.model_mac([H] * L, [cfg.intermediate_size] * L, 1, d, hs, False)
    total = orig
    for _, l, kind, j in units:
        if total <= constraint * orig:
            break
        if kind == "ffn":
            if len(neurons[l]) <= 1:
                continue
            neurons[l].discard(j)
            total -= f_ffn
        else:
            heads[l].discard(j)
            total -= f_att
    return ({l: sorted(v) for l, v in heads.items()}, {l: sorted(v) for l, v in neurons.items()},
            {"mac_ratio": total / orig, "constraint": constraint})


def build_student(teacher, heads, neurons):
    return grp.prune_model(copy.deepcopy(teacher), heads, neurons)


# ---- calibration batches and statistics ----------------------------------------------------------------------------
def make_batches(seqs: list[str], tok, batch_size: int = 8) -> list[dict]:
    seqs = sorted(seqs, key=len)
    out = []
    for i in range(0, len(seqs), batch_size):
        enc = tok(seqs[i:i + batch_size], return_tensors="pt", padding=True)
        out.append({"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]})
    return out


def bootstrap_ci(x: np.ndarray, n_boot: int = 10000, seed: int = 1234, stat=np.mean):
    """Percentile 2.5/97.5 bootstrap over assays (rows of x are assays)."""
    idx = np.random.default_rng(seed).integers(0, len(x), size=(n_boot, len(x)))
    vals = np.array([stat(x[i]) for i in idx])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
