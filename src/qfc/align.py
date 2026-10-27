"""Beat-mission idea 1 (docs/EXPLORATION_LOG.md): input-side alignment on top of the drift-corrected output refit.

K-prune / GRP-R refit only the output projections (attention.output.dense, output.dense). After upstream pruning the
inputs of every kept unit are drifted, and the kept units' own input weights are untouched. Here, layer by layer and
with the already-refit student's hidden states as input, we also solve in closed form (ridge least squares, no
training):
  qkv    : student query/key/value rows of the kept heads  ->  the teacher's query/key/value features of those heads
  ffn_in : student intermediate.dense rows of the kept neurons -> the teacher's pre-activations of those neurons
then the existing output refits (drift-corrected target y_t + h_t - h_s) run on the aligned units.
Order per layer: qkv -> attention output refit -> ffn_in -> ffn output refit.
"""
from __future__ import annotations

import torch

from .grp import _aug, _batch_inputs, _dense, _hooks, _layers, _solve_set


class _Cap:
    """Forward hook capturing a module's input[0] ('in') or output ('out')."""

    def __init__(self, module, where):
        self.v = None
        fn = (lambda m, a: setattr(self, "v", a[0])) if where == "in" else (lambda m, a, o: setattr(self, "v", o))
        self.h = (module.register_forward_pre_hook if where == "in" else module.register_forward_hook)(fn)

    def close(self):
        self.h.remove()


def _head_cols(kept, head_size):
    return torch.tensor([h * head_size + j for h in sorted(kept) for j in range(head_size)], dtype=torch.long)


@torch.no_grad()
def sequential_refit_io(student, teacher, batches, device: str, heads_kept: dict, neurons_kept: dict,
                        ridge_rel: float = 1e-4, qkv: bool = True, ffn_in: bool = True) -> None:
    student.eval()
    teacher.eval()
    base_s, base_t = getattr(student, student.base_model_prefix), getattr(teacher, teacher.base_model_prefix)
    bs = [_batch_inputs(b, device) for b in batches]
    masks = [b["attention_mask"].bool() for b in bs]
    ext = [teacher.get_extended_attention_mask(b["attention_mask"], b["attention_mask"].shape) for b in bs]
    hs = [base_s.embeddings(input_ids=b["input_ids"], token_type_ids=b.get("token_type_ids")) for b in bs]
    ht = [base_t.embeddings(input_ids=b["input_ids"], token_type_ids=b.get("token_type_ids")) for b in bs]
    head_size = teacher.config.hidden_size // teacher.config.num_attention_heads
    for l in range(len(_layers(student))):
        ls, lt = _layers(student)[l], _layers(teacher)[l]
        kh = sorted(heads_kept[l])
        if qkv and kh:
            cols = _head_cols(kh, head_size).to(device)
            G, C = None, {"query": None, "key": None, "value": None}
            for i in range(len(bs)):
                x = _aug(hs[i][masks[i]].double())
                g = (x.T @ x).cpu()
                G = g if G is None else G + g
                for name in C:
                    y = getattr(lt.attention.self, name)(ht[i])[masks[i]].double()[:, cols]
                    c = (x.T @ y).cpu()
                    C[name] = c if C[name] is None else C[name] + c
            for name in C:
                _solve_set(getattr(ls.attention.self, name), G, C[name], ridge_rel)
        _refit_out(student, teacher, ls, lt, l, "att", bs, hs, ht, ext, masks, ridge_rel)
        if ffn_in:
            kn = torch.tensor(sorted(neurons_kept[l]), dtype=torch.long, device=device)
            G = C = None
            cs, ct = _Cap(ls.intermediate.dense, "in"), _Cap(lt.intermediate.dense, "out")
            try:
                for i in range(len(bs)):
                    ls(hs[i], attention_mask=ext[i])
                    lt(ht[i], attention_mask=ext[i])
                    x = _aug(cs.v[masks[i]].double())
                    y = ct.v[masks[i]].double()[:, kn]
                    g, c = (x.T @ x).cpu(), (x.T @ y).cpu()
                    G, C = (g, c) if G is None else (G + g, C + c)
            finally:
                cs.close()
                ct.close()
            _solve_set(ls.intermediate.dense, G, C, ridge_rel)
        _refit_out(student, teacher, ls, lt, l, "ffn", bs, hs, ht, ext, masks, ridge_rel)
        for i in range(len(bs)):
            hs[i] = ls(hs[i], attention_mask=ext[i])[0]
            ht[i] = lt(ht[i], attention_mask=ext[i])[0]


def _refit_out(student, teacher, ls, lt, l, kind, bs, hs, ht, ext, masks, ridge_rel):
    G = C = None
    s_spec = [(l, kind, "in", "x"), (l, kind, "res", "h")]
    t_spec = [(l, kind, "out", "y"), (l, kind, "res", "h")]
    with _hooks(student, s_spec) as s_st, _hooks(teacher, t_spec) as t_st:
        for i in range(len(bs)):
            ls(hs[i], attention_mask=ext[i])
            lt(ht[i], attention_mask=ext[i])
            x = _aug(s_st["x"][masks[i]].double())
            y = t_st["y"][masks[i]].double() + (t_st["h"][masks[i]] - s_st["h"][masks[i]]).double()
            g, c = (x.T @ x).cpu(), (x.T @ y).cpu()
            G, C = (g, c) if G is None else (G + g, C + c)
    _solve_set(_dense(student, l, kind), G, C, ridge_rel)
