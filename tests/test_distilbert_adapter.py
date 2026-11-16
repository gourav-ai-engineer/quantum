"""V19 step 1: the DistilBERT adapter for the frozen d2i-RI (no method change). Tiny random models, CPU, no download."""
import copy
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_grp import _batches, _tiny_bert  # noqa: E402

from qfc.align import sequential_refit_io  # noqa: E402
from qfc.grp import (_hooks, _layers, collect_grams, ffn_width, grp_prune, module_sensitivity, prune_model,  # noqa: E402
                     sequential_refit)

HEADS, NEURONS = {0: [1, 3], 1: [0, 1, 2]}, {0: list(range(0, 64, 2)), 1: list(range(40))}


def _tiny_distil(seed=0, num_labels=3):
    from transformers import DistilBertConfig, DistilBertForSequenceClassification
    cfg = DistilBertConfig(vocab_size=50, dim=32, n_layers=2, n_heads=4, hidden_dim=64, max_position_embeddings=32,
                           num_labels=num_labels)
    cfg._attn_implementation = "eager"
    torch.manual_seed(seed)
    return DistilBertForSequenceClassification(cfg).eval()


def _dbatches():
    return [{k: v for k, v in b.items() if k != "token_type_ids"} for b in _batches()]


def _logits(model, batches, head_mask=None):
    return torch.cat([model(**b, head_mask=head_mask).logits for b in batches])


def _bert_twin(distil):
    """A tiny BERT whose encoder weights equal `distil`'s (zero token-type embedding), to compare the two code paths."""
    bert = _tiny_bert()
    b, d = bert.bert, distil.distilbert
    with torch.no_grad():
        b.embeddings.word_embeddings.weight.copy_(d.embeddings.word_embeddings.weight)
        b.embeddings.position_embeddings.weight.copy_(d.embeddings.position_embeddings.weight)
        b.embeddings.token_type_embeddings.weight.zero_()
        b.embeddings.LayerNorm.load_state_dict(d.embeddings.LayerNorm.state_dict())
        for lb, ld in zip(b.encoder.layer, d.transformer.layer):
            for name, lin in (("query", ld.attention.q_lin), ("key", ld.attention.k_lin), ("value", ld.attention.v_lin)):
                getattr(lb.attention.self, name).load_state_dict(lin.state_dict())
            lb.attention.output.dense.load_state_dict(ld.attention.out_lin.state_dict())
            lb.attention.output.LayerNorm.load_state_dict(ld.sa_layer_norm.state_dict())
            lb.intermediate.dense.load_state_dict(ld.ffn.lin1.state_dict())
            lb.output.dense.load_state_dict(ld.ffn.lin2.state_dict())
            lb.output.LayerNorm.load_state_dict(ld.output_layer_norm.state_dict())
    return bert.eval()


def test_config_helpers():
    assert ffn_width(_tiny_distil().config) == 64 and ffn_width(_tiny_bert().config) == 64
    assert _tiny_distil().config.num_attention_heads == 4 and len(_layers(_tiny_distil())) == 2


@torch.no_grad()
def test_physical_pruning_equals_masking():
    model, bs = _tiny_distil(), _dbatches()
    masked = copy.deepcopy(model)
    for l, keep in NEURONS.items():
        masked.distilbert.transformer.layer[l].ffn.lin2.weight[:, [j for j in range(64) if j not in keep]] = 0
    hm = torch.zeros(2, 4)
    for l, keep in HEADS.items():
        hm[l, keep] = 1
    pruned = prune_model(copy.deepcopy(model), HEADS, NEURONS)
    assert torch.allclose(_logits(pruned, bs), _logits(masked, bs, hm), atol=1e-5)


@torch.no_grad()
def test_layer_with_zero_heads_runs():
    pruned = prune_model(_tiny_distil(), {0: [], 1: [2]}, {0: [5], 1: list(range(64))})
    assert torch.isfinite(_logits(pruned, _dbatches())).all()


@torch.no_grad()
def test_unpruned_student_is_recovered():
    t, bs = _tiny_distil(), _dbatches()
    heads, neurons = {0: [0, 1, 2, 3], 1: [0, 1, 2, 3]}, {0: list(range(64)), 1: list(range(64))}
    s = prune_model(copy.deepcopy(t), heads, neurons)
    sequential_refit_io(s, t, bs, "cpu", heads, neurons, 1e-6)
    assert torch.allclose(_logits(s, bs), _logits(t, bs), atol=1e-3)
    s2 = prune_model(copy.deepcopy(t), heads, neurons)
    sequential_refit(s2, t, bs, "cpu", 1e-6, True)
    assert torch.allclose(_logits(s2, bs), _logits(t, bs), atol=1e-3)


@torch.no_grad()
def test_no_align_equals_drift_corrected_refit():
    t, bs = _tiny_distil(), _dbatches()
    a, b = prune_model(copy.deepcopy(t), HEADS, NEURONS), prune_model(copy.deepcopy(t), HEADS, NEURONS)
    sequential_refit(a, t, bs, "cpu", 1e-4, True)
    sequential_refit_io(b, t, bs, "cpu", HEADS, NEURONS, 1e-4, qkv=False, ffn_in=False)
    assert torch.allclose(_logits(a, bs), _logits(b, bs), atol=1e-5)


@torch.no_grad()
def test_refit_lowers_hidden_state_error():
    t, bs = _tiny_distil(), _dbatches()
    raw = prune_model(copy.deepcopy(t), HEADS, NEURONS)
    fit = prune_model(copy.deepcopy(t), HEADS, NEURONS)
    sequential_refit_io(fit, t, bs, "cpu", HEADS, NEURONS, 1e-4)

    def err(m):
        return sum(float(((m.distilbert(**b).last_hidden_state - t.distilbert(**b).last_hidden_state) ** 2).mean())
                   for b in bs)
    assert err(fit) < err(raw)


@torch.no_grad()
def test_residual_hooks_capture_the_add_and_ln_inputs():
    t, bs = _tiny_distil(), _dbatches()
    b = bs[0]
    layer0, layer1 = t.distilbert.transformer.layer[0], t.distilbert.transformer.layer[1]
    h0 = t.distilbert.embeddings(b["input_ids"])
    with _hooks(t, [(0, "att", "res", "a0"), (0, "ffn", "res", "f0"), (1, "att", "res", "a1")]) as st:
        out0 = layer0(h0, attn_mask=b["attention_mask"])[0]
        layer1(out0, attn_mask=b["attention_mask"])
    sa0 = layer0.sa_layer_norm(layer0.attention(h0, h0, h0, b["attention_mask"])[0] + h0)
    assert torch.allclose(st["a0"], h0) and torch.allclose(st["f0"], sa0, atol=1e-6)
    assert torch.allclose(st["a1"], out0)


@torch.no_grad()
def test_distilbert_path_equals_bert_path_with_copied_weights():
    """The strongest adapter check: identical weights in both architectures -> identical Grams, identical refit
    weights (every q/k/v, attention-out, FFN-up, FFN-out matrix) and identical hidden states after pruning."""
    d = _tiny_distil()
    bert = _bert_twin(d)
    bs_d, bs_b = _dbatches(), _batches()
    gd, gb = collect_grams(d, bs_d, "cpu"), collect_grams(bert, bs_b, "cpu")
    for k in gd:
        assert torch.allclose(gd[k], gb[k], atol=1e-8)
    pd_, pb = prune_model(copy.deepcopy(d), HEADS, NEURONS), prune_model(copy.deepcopy(bert), HEADS, NEURONS)
    sequential_refit_io(pd_, d, bs_d, "cpu", HEADS, NEURONS, 1e-4)
    sequential_refit_io(pb, bert, bs_b, "cpu", HEADS, NEURONS, 1e-4)
    for ld, lb in zip(pd_.distilbert.transformer.layer, pb.bert.encoder.layer):
        pairs = [(ld.attention.q_lin, lb.attention.self.query), (ld.attention.k_lin, lb.attention.self.key),
                 (ld.attention.v_lin, lb.attention.self.value), (ld.attention.out_lin, lb.attention.output.dense),
                 (ld.ffn.lin1, lb.intermediate.dense), (ld.ffn.lin2, lb.output.dense)]
        for a, b in pairs:
            assert a.weight.shape == b.weight.shape
            assert torch.allclose(a.weight, b.weight, atol=1e-5) and torch.allclose(a.bias, b.bias, atol=1e-5)
    hd = pd_.distilbert(**bs_d[0]).last_hidden_state
    hb = pb.bert(**bs_b[0]).last_hidden_state
    assert torch.allclose(hd, hb, atol=1e-4)


def test_grp_prune_end_to_end_on_distilbert():
    t = _tiny_distil()
    s, heads, neurons, info = grp_prune(t, _dbatches(), "cpu", 12, 0.6)
    assert all(len(h) <= 4 for h in heads.values()) and torch.isfinite(_logits(s, _dbatches())).all()


def test_question_answering_model_sensitivity():
    from transformers import DistilBertConfig, DistilBertForQuestionAnswering
    cfg = DistilBertConfig(vocab_size=50, dim=32, n_layers=2, n_heads=4, hidden_dim=64, max_position_embeddings=32)
    cfg._attn_implementation = "eager"
    torch.manual_seed(0)
    s = module_sensitivity(DistilBertForQuestionAnswering(cfg).eval(), _dbatches(), "cpu", seed=0)
    assert len(s) == 4 and all(v > 0 and v == v for v in s.values())
