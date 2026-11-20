"""ESM-2 pilot (E6): physical pruning == masking, refit, kept units, MAC budget, scoring and assay rule. Tiny random ESM, CPU."""
import copy

import numpy as np
import pandas as pd
import torch
from transformers import EsmConfig, EsmForMaskedLM

from qfc import esm_pilot as ep
from qfc import grp


def tiny(hidden=32, heads=4, inter=64, layers=3, seed=0):
    torch.manual_seed(seed)
    cfg = EsmConfig(vocab_size=33, hidden_size=hidden, num_hidden_layers=layers, num_attention_heads=heads,
                    intermediate_size=inter, max_position_embeddings=64, pad_token_id=1, mask_token_id=32,
                    position_embedding_type="rotary", token_dropout=True, emb_layer_norm_before=False,
                    hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0)
    return EsmForMaskedLM(cfg).eval()


def batches(n=16, length=20, seed=1):
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n // 8):
        ids = torch.randint(4, 24, (8, length), generator=g)
        ids[:, 0], ids[:, -1] = 0, 2
        out.append({"input_ids": ids, "attention_mask": torch.ones_like(ids)})
    return out


def zero_removed(model, heads, neurons):
    """Masking reference: zero the output-projection columns of removed units (bias kept)."""
    H = model.config.num_attention_heads
    hs = model.config.hidden_size // H
    with torch.no_grad():
        for l, layer in enumerate(grp._layers(model)):
            for h in range(H):
                if h not in heads[l]:
                    layer.attention.output.dense.weight[:, h * hs:(h + 1) * hs] = 0
            for n in range(model.config.intermediate_size):
                if n not in neurons[l]:
                    layer.output.dense.weight[:, n] = 0
    return model


def test_physical_pruning_equals_masking_including_empty_layer():
    t = tiny()
    heads = {0: [0, 2], 1: [], 2: [1, 2, 3]}  # layer 1 has no heads
    neurons = {0: list(range(0, 64, 2)), 1: [3, 5, 7], 2: list(range(10, 40))}
    s = ep.build_student(t, heads, neurons)
    m = zero_removed(copy.deepcopy(t), heads, neurons)
    b = batches()[0]
    with torch.no_grad():
        a, r = s(**b).logits, m(**b).logits
    assert torch.allclose(a, r, atol=1e-5)


def test_kept_units_equal_selected_masks():
    t = tiny()
    heads = {0: [1, 3], 1: [0], 2: [0, 1, 2, 3]}
    neurons = {0: [4, 9, 20], 1: [0, 1], 2: list(range(64))}
    s = ep.build_student(t, heads, neurons)
    hs = 8
    for l, layer in enumerate(grp._layers(s)):
        tl = grp._layers(t)[l]
        assert layer.attention.self.num_attention_heads == len(heads[l])
        assert layer.intermediate.dense.out_features == len(neurons[l])
        idx = torch.tensor(neurons[l])
        assert torch.equal(layer.intermediate.dense.weight, tl.intermediate.dense.weight[idx])
        assert torch.equal(layer.output.dense.weight, tl.output.dense.weight[:, idx])
        for k, h in enumerate(heads[l]):
            assert torch.equal(layer.attention.self.query.weight[k * hs:(k + 1) * hs],
                               tl.attention.self.query.weight[h * hs:(h + 1) * hs])


def test_refit_recovers_unpruned_student_from_perturbation():
    t = tiny()
    bs = batches(32, 24)
    s = copy.deepcopy(t)
    g = torch.Generator().manual_seed(3)
    with torch.no_grad():
        for layer in grp._layers(s):
            for d in (layer.attention.output.dense, layer.output.dense):
                d.weight.add_(0.05 * torch.randn(d.weight.shape, generator=g))
        err0 = (s(**bs[0]).logits - t(**bs[0]).logits).pow(2).mean()
    grp.sequential_refit(s, t, bs, "cpu", 1e-6, drift_correct=True)
    with torch.no_grad():
        err1 = (s(**bs[0]).logits - t(**bs[0]).logits).pow(2).mean()
    assert err1 < 1e-3 * err0


def test_refit_lowers_hidden_state_error_after_pruning():
    t = tiny()
    bs = batches(32, 24)
    heads = {l: [0, 1] for l in range(3)}
    neurons = {l: list(range(0, 64, 2)) for l in range(3)}
    s = ep.build_student(t, heads, neurons)

    def err(m):
        with torch.no_grad():
            return float(sum((m(**b, output_hidden_states=True).hidden_states[-1]
                              - t(**b, output_hidden_states=True).hidden_states[-1]).pow(2).mean() for b in bs))

    e0 = err(s)
    grp.sequential_refit(s, t, bs, "cpu", 1e-4, drift_correct=True)
    assert err(s) < e0


def test_mac_ratio_equals_budget_to_three_decimals():
    t = tiny(hidden=16, heads=8, inter=1024, layers=4)
    stats = grp.teacher_stats(t, batches(16, 24), "cpu")  # also exercises the MLM sensitivity branch
    assert all(v > 0 for v in stats["sens"].values())
    full = ep.mac_per_token(t)
    for c in (0.225, 0.5):
        for sel in (lambda: ep.select_grp(stats, t, c), lambda: ep.select_magnitude(t, c)):
            heads, neurons, info = sel()
            assert all(len(v) >= 1 for v in neurons.values())
            ratio = ep.mac_per_token(ep.build_student(t, heads, neurons)) / full
            assert abs(ratio - info["mac_ratio"]) < 1e-9
            assert 0 <= c - ratio < 1e-3


def test_mac_ratio_of_real_esm2_shapes_is_0_225():
    def mac(L, d, H, I):
        return grp.model_mac([H] * L, [I] * L, 1, d, d // H, False)
    assert abs(mac(12, 480, 20, 1920) / mac(30, 640, 20, 2560) - 0.225) < 1e-4


class _Tok:
    vocab = {c: i + 4 for i, c in enumerate("ACDEFGHIKLMNPQRSTVWY")}

    def __call__(self, s, return_tensors=None):
        ids = torch.tensor([[0] + [self.vocab[c] for c in s] + [2]])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def convert_tokens_to_ids(self, c):
        return self.vocab[c]


def test_wt_marginal_zero_for_identity_and_position_index():
    t, tok, wt = tiny(), _Tok(), "ACDEFGHIK"
    sc = ep.wt_marginal_scores(t, tok, wt, [(1, "A", "A"), (3, "D", "D"), (3, "D", "E")])
    assert sc[0] == 0 and sc[1] == 0 and sc[2] != 0
    with torch.no_grad():
        lp = t(**tok(wt)).logits[0].log_softmax(-1)
    assert abs(sc[2] - float(lp[3, tok.vocab["E"]] - lp[3, tok.vocab["D"]])) < 1e-6


def test_select_assays_mechanical_rule():
    rows = []

    def add(i, L, muts):
        for m, sc in muts:
            rows.append({"DMS_id": i, "mutant": m, "DMS_score": sc, "target_seq": "A" * L})

    add("B_long", 401, [("A1C", 1.0)])
    add("A_ok", 50, [("A1C", 1.0), ("A1C:A2D", 2.0)])
    add("C_multi_only", 60, [("A1C:A2D", 1.0)])
    add("D_nan", 60, [("A1C", np.nan)])
    add("E_ok", 400, [("A1C", 1.0)])
    assert ep.select_assays(pd.DataFrame(rows), 400, 20) == ["A_ok", "E_ok"]
    assert ep.select_assays(pd.DataFrame(rows), 400, 1) == ["A_ok"]


def test_bootstrap_ci_brackets_mean():
    x = np.linspace(0, 1, 20)
    lo, hi = ep.bootstrap_ci(x, 2000, 1234)
    assert lo < x.mean() < hi
    assert ep.bootstrap_ci(x, 2000, 1234) == (lo, hi)
