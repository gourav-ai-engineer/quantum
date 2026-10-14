"""V15: global reconstruction pruning (GRP) of attention heads and FFN neurons (docs/PROJECT_STATE.md,
"Pre-registration V15"). One-shot selection from teacher statistics, then a joint sequential closed-form refit.

Modules per layer l: "att" (features = head contexts, input of attention.output.dense; unit = one head = a block of
head_dim features) and "ffn" (features = GELU activations, input of output.dense; unit = one neuron). Every Gram is
augmented with a constant-1 feature (last index) so the refit also fits the bias; the bias is never penalised and
never removed.
"""
from __future__ import annotations

import copy
import math
from contextlib import contextmanager

import torch
from torch import nn

# ---- FLOPs: Kwon et al. 2022, efficiency/mac.py @ 806ac5a6 (K-prune's capacity.py is exactly 2x) ---------------------
AVG_SEQ_LEN = {"stsb": 31.47, "mrpc": 53.24, "rte": 64.59, "sst2": 25.16, "qqp": 30.55, "qnli": 50.97,
               "cola": 11.67, "mnli": 39.05, "squad": 170.0}  # dev-set averages used by both official codes


def mac_per_head(seq_len: float, hidden_size: int, head_size: int) -> float:
    return 3 * seq_len * hidden_size * head_size + 2 * seq_len * seq_len * head_size + seq_len * head_size * hidden_size


def mac_per_neuron(seq_len: float, hidden_size: int) -> float:
    return 2 * seq_len * hidden_size


def model_mac(heads_per_layer, neurons_per_layer, seq_len: float, hidden_size: int, head_size: int) -> float:
    return sum(h * mac_per_head(seq_len, hidden_size, head_size) + n * mac_per_neuron(seq_len, hidden_size)
               for h, n in zip(heads_per_layer, neurons_per_layer))


# ---- model plumbing (HF BERT, eager attention) -----------------------------------------------------------------
def _layers(model):
    return getattr(model, model.base_model_prefix).encoder.layer


def _dense(model, l: int, kind: str) -> nn.Linear:
    layer = _layers(model)[l]
    return layer.attention.output.dense if kind == "att" else layer.output.dense


@contextmanager
def _hooks(model, specs):
    """specs: list of (layer, kind, 'in'|'out', store_key). Stores the dense layer's input or output (with bias)."""
    store, handles = {}, []
    for l, kind, where, key in specs:
        dense = _dense(model, l, kind)
        if where == "in":
            handles.append(dense.register_forward_pre_hook(lambda m, a, key=key: store.__setitem__(key, a[0])))
        else:
            handles.append(dense.register_forward_hook(lambda m, a, o, key=key: store.__setitem__(key, o)))
    try:
        yield store
    finally:
        for h in handles:
            h.remove()


def _aug(x: torch.Tensor) -> torch.Tensor:
    return torch.cat([x, torch.ones(x.shape[0], 1, dtype=x.dtype, device=x.device)], dim=1)


def _batch_inputs(batch, device):
    return {k: v.to(device) for k, v in batch.items() if k in ("input_ids", "attention_mask", "token_type_ids")}


@torch.no_grad()
def collect_grams(model, batches, device: str) -> dict:
    """Teacher Grams per layer: att (H*d+1)^2 and ffn (I+1)^2, float64 on CPU, over non-pad tokens."""
    model.eval()
    L = len(_layers(model))
    G = {(l, k): None for l in range(L) for k in ("att", "ffn")}
    specs = [(l, k, "in", (l, k)) for l in range(L) for k in ("att", "ffn")]
    with _hooks(model, specs) as st:
        for batch in batches:
            b = _batch_inputs(batch, device)
            model(**b, return_dict=True)
            m = b["attention_mask"].bool()
            for key in G:
                x = _aug(st[key][m].double())
                g = (x.T @ x).cpu()
                G[key] = g if G[key] is None else G[key] + g
    return G


def module_sensitivity(model, batches, device: str, seed: int = 0) -> dict:
    """s_m = E_x E_{y ~ p_teacher(.|x)} ||d log p(y|x) / d Y_m||_F^2 / n_tokens (one label sample per input)."""
    model.eval()
    L = len(_layers(model))
    gen = torch.Generator().manual_seed(seed)
    s = {(l, k): 0.0 for l in range(L) for k in ("att", "ffn")}
    n_tok = 0
    specs = [(l, k, "out", (l, k)) for l in range(L) for k in ("att", "ffn")]
    for batch in batches:
        b = _batch_inputs(batch, device)
        with _hooks(model, specs) as st:
            logits = model(**b, return_dict=True).logits
            for y in st.values():
                y.retain_grad()
            probs = logits.detach().float().softmax(-1).cpu()
            lab = torch.multinomial(probs, 1, generator=gen).squeeze(1).to(logits.device)
            logp = logits.log_softmax(-1).gather(1, lab[:, None]).sum()
            model.zero_grad(set_to_none=True)
            logp.backward()
            m = b["attention_mask"].bool()
            for key, y in st.items():
                s[key] += float((y.grad[m].double() ** 2).sum())
        n_tok += int(b["attention_mask"].sum())
    model.zero_grad(set_to_none=True)
    return {k: v / n_tok for k, v in s.items()}


def teacher_weight(model, l: int, kind: str) -> torch.Tensor:
    """W_aug ((in+1) x out): dense.weight.T with the bias appended as the last row, float64 CPU."""
    d = _dense(model, l, kind)
    return torch.cat([d.weight.detach().double().cpu().T, d.bias.detach().double().cpu()[None]], dim=0)


# ---- closed-form removal cost and greedy state per module ------------------------------------------------------
class ModuleState:
    """Least-squares refit of one module's output from its kept features, with exact block downdates.

    Features 0..p-1 belong to units (blocks of unit_size); feature p is the constant (bias), always kept and not
    penalised. The ridge lam is fixed at construction (ridge_rel * mean diagonal of the non-bias Gram) so that every
    downdate is exact for one objective; P is recomputed from scratch every ``refresh_every`` removals.
    """

    def __init__(self, G: torch.Tensor, W_aug: torch.Tensor, unit_size: int, ridge_rel: float = 1e-4,
                 refresh_every: int = 25):
        self.G = G.double()
        self.p = G.shape[0] - 1
        self.unit_size = unit_size
        self.lam = ridge_rel * float(torch.diagonal(self.G)[: self.p].mean()) if ridge_rel > 0 else 0.0
        self.C = self.G @ W_aug.double()
        self.kept = list(range(self.p // unit_size))
        self.refresh_every, self._since = refresh_every, 0
        self._refresh()

    def _idx(self) -> torch.Tensor:
        u = self.unit_size
        feats = [torch.arange(j * u, (j + 1) * u) for j in self.kept]
        return torch.cat(feats + [torch.tensor([self.p])])

    def _refresh(self):
        idx = self._idx()
        A = self.G[idx][:, idx].clone()
        A.diagonal()[:-1] += self.lam
        self.P = torch.linalg.inv(A)
        self.P = 0.5 * (self.P + self.P.T)
        self.V = self.P @ self.C[idx]
        self._since = 0

    def costs(self) -> torch.Tensor:
        """Increase of the ridge refit objective from removing each kept unit alone: tr(V_R^T P_RR^{-1} V_R)."""
        u, k = self.unit_size, len(self.kept)
        if k == 0:
            return torch.zeros(0, dtype=torch.float64)
        if u == 1:
            return (self.V[:k] ** 2).sum(1) / torch.diagonal(self.P)[:k]
        out = torch.empty(k, dtype=torch.float64)
        for i in range(k):
            r = slice(i * u, (i + 1) * u)
            out[i] = torch.trace(self.V[r].T @ torch.linalg.solve(self.P[r, r], self.V[r]))
        return out

    def remove(self, units) -> None:
        pos = {j: i for i, j in enumerate(self.kept)}
        u = self.unit_size
        R = torch.cat([torch.arange(pos[j] * u, (pos[j] + 1) * u) for j in units])
        keep = torch.ones(self.P.shape[0], dtype=torch.bool)
        keep[R] = False
        K = keep.nonzero().squeeze(1)
        Q = torch.linalg.solve(self.P[R][:, R], self.P[R][:, K])  # P_RR^{-1} P_RK
        self.P = self.P[K][:, K] - self.P[K][:, R] @ Q
        self.P = 0.5 * (self.P + self.P.T)
        self.V = self.V[K] - Q.T @ self.V[R]
        rm = set(units)
        self.kept = [j for j in self.kept if j not in rm]
        self._since += 1
        if self._since >= self.refresh_every:
            self._refresh()


def grp_select(G: dict, weights: dict, sens: dict | None, L: int, num_heads: int, seq_len: float,
               hidden_size: int, constraint: float, ridge_rel: float = 1e-4, max_frac: float = 0.01):
    """Global greedy elimination: repeatedly pick the module whose cheapest unit has the smallest
    s_m * DeltaE / MACs, remove its k = max(1, floor(max_frac * |kept|)) cheapest units (fewer if that would
    overshoot the budget), until total MACs <= constraint * original. sens=None gives s_m = 1.

    Returns (heads_kept {l: [...]}, neurons_kept {l: [...]}, info).
    """
    intermediate = G[(0, "ffn")].shape[0] - 1
    head_size = (G[(0, "att")].shape[0] - 1) // num_heads
    f_unit = {"att": mac_per_head(seq_len, hidden_size, head_size), "ffn": mac_per_neuron(seq_len, hidden_size)}
    orig = model_mac([num_heads] * L, [intermediate] * L, seq_len, hidden_size, head_size)
    target, total = constraint * orig, orig
    states = {(l, k): ModuleState(G[(l, k)], weights[(l, k)], head_size if k == "att" else 1, ridge_rel)
              for l in range(L) for k in ("att", "ffn")}
    score = {}

    def rescore(key):
        st = states[key]
        sc = (1.0 if sens is None else sens[key]) * st.costs() / f_unit[key[1]]
        if key[1] == "ffn" and len(st.kept) <= 1:  # keep at least one FFN neuron per layer
            sc = torch.full_like(sc, math.inf)
        score[key] = sc

    for key in states:
        rescore(key)
    rounds = 0
    while total > target:
        key = min(states, key=lambda k: float(score[k].min()) if len(score[k]) else math.inf)
        sc, st = score[key], states[key]
        if not len(sc) or not math.isfinite(float(sc.min())):
            raise RuntimeError("budget unreachable under the module constraints")
        k = max(1, int(max_frac * len(st.kept)))
        k = min(k, max(1, math.ceil((total - target) / f_unit[key[1]])))
        if key[1] == "ffn":
            k = min(k, len(st.kept) - 1)
        units = [st.kept[i] for i in torch.argsort(sc)[:k].tolist()]
        st.remove(units)
        total -= len(units) * f_unit[key[1]]
        rescore(key)
        rounds += 1
    heads = {l: list(states[(l, "att")].kept) for l in range(L)}
    neurons = {l: list(states[(l, "ffn")].kept) for l in range(L)}
    return heads, neurons, {"rounds": rounds, "mac_ratio": total / orig, "constraint": constraint}


# ---- physical pruning and joint sequential refit ---------------------------------------------------------------
def prune_model(model, heads_kept: dict, neurons_kept: dict):
    """Physically remove heads (HF prune_heads) and FFN neurons (slice intermediate/output dense). In place."""
    H = model.config.num_attention_heads
    to_prune = {l: [h for h in range(H) if h not in set(k)] for l, k in heads_kept.items()}
    model.prune_heads({l: hs for l, hs in to_prune.items() if hs})
    for l, keep in neurons_kept.items():
        layer = _layers(model)[l]
        old_in, old_out = layer.intermediate.dense, layer.output.dense
        idx = torch.tensor(sorted(keep), dtype=torch.long, device=old_in.weight.device)
        new_in = nn.Linear(old_in.in_features, len(idx)).to(old_in.weight.device, old_in.weight.dtype)
        new_out = nn.Linear(len(idx), old_out.out_features).to(old_out.weight.device, old_out.weight.dtype)
        with torch.no_grad():
            new_in.weight.copy_(old_in.weight[idx])
            new_in.bias.copy_(old_in.bias[idx])
            new_out.weight.copy_(old_out.weight[:, idx])
            new_out.bias.copy_(old_out.bias)
        layer.intermediate.dense, layer.output.dense = new_in, new_out
    return model


@torch.no_grad()
def _solve_set(dense: nn.Linear, G: torch.Tensor, C: torch.Tensor, ridge_rel: float) -> None:
    p = G.shape[0] - 1
    A = G.clone()
    if ridge_rel > 0 and p > 0:
        A.diagonal()[:p] += ridge_rel * float(torch.diagonal(G)[:p].mean())
    W = torch.linalg.solve(A, C)
    dense.weight.copy_(W[:p].T.to(dense.weight.dtype))
    dense.bias.copy_(W[p].to(dense.bias.dtype))


@torch.no_grad()
def sequential_refit(student, teacher, batches, device: str, ridge_rel: float = 1e-4) -> None:
    """Layer by layer from first to last: refit the student's attention output projection, then its FFN output
    projection (weights and bias), so each matches the unpruned teacher's module output, using inputs from the
    already-refit student. Hidden states entering each layer are cached per batch (teacher and student)."""
    student.eval()
    teacher.eval()
    base_s, base_t = getattr(student, student.base_model_prefix), getattr(teacher, teacher.base_model_prefix)
    bs = [_batch_inputs(b, device) for b in batches]
    masks = [b["attention_mask"].bool() for b in bs]
    ext = [teacher.get_extended_attention_mask(b["attention_mask"], b["attention_mask"].shape) for b in bs]
    hs = [base_s.embeddings(input_ids=b["input_ids"], token_type_ids=b.get("token_type_ids")) for b in bs]
    ht = [base_t.embeddings(input_ids=b["input_ids"], token_type_ids=b.get("token_type_ids")) for b in bs]
    for l in range(len(_layers(student))):
        ls, lt = _layers(student)[l], _layers(teacher)[l]
        for kind in ("att", "ffn"):
            G = C = None
            with _hooks(student, [(l, kind, "in", "x")]) as s_st, _hooks(teacher, [(l, kind, "out", "y")]) as t_st:
                for i in range(len(bs)):
                    ls(hs[i], attention_mask=ext[i])
                    lt(ht[i], attention_mask=ext[i])
                    x = _aug(s_st["x"][masks[i]].double())
                    y = t_st["y"][masks[i]].double()
                    g, c = (x.T @ x).cpu(), (x.T @ y).cpu()
                    G, C = (g, c) if G is None else (G + g, C + c)
            _solve_set(_dense(student, l, kind), G, C, ridge_rel)
        for i in range(len(bs)):
            hs[i] = ls(hs[i], attention_mask=ext[i])[0]
            ht[i] = lt(ht[i], attention_mask=ext[i])[0]


def teacher_stats(teacher, batches, device: str, seed: int = 0) -> dict:
    """Budget-independent statistics (computed once per calibration set): Grams, teacher weights, sensitivities."""
    L = len(_layers(teacher))
    return {"G": collect_grams(teacher, batches, device),
            "W": {(l, k): teacher_weight(teacher, l, k) for l in range(L) for k in ("att", "ffn")},
            "sens": module_sensitivity(teacher, batches, device, seed)}


def grp_prune(teacher, batches, device: str, seq_len: float, constraint: float, ridge_rel: float = 1e-4,
              use_sensitivity: bool = True, refit: bool = True, seed: int = 0, stats: dict | None = None):
    """Full GRP: statistics -> global greedy selection -> physical pruning -> joint sequential refit."""
    L, H = len(_layers(teacher)), teacher.config.num_attention_heads
    stats = stats or teacher_stats(teacher, batches, device, seed)
    G, weights = stats["G"], stats["W"]
    sens = stats["sens"] if use_sensitivity else None
    heads, neurons, info = grp_select(G, weights, sens, L, H, seq_len, teacher.config.hidden_size, constraint,
                                      ridge_rel)
    student = prune_model(copy.deepcopy(teacher), heads, neurons)
    if refit:
        sequential_refit(student, teacher, batches, device, ridge_rel)
    info["sensitivity"] = {f"{l}:{k}": v for (l, k), v in sens.items()} if sens else None
    return student, heads, neurons, info
