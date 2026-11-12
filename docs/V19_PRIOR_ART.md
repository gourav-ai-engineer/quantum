# Prior-art review: multi-downstream-layer reconstruction for structured transformer pruning (V19)

**Reviewed:** 2026-11-12 (external reviewer pass; verified against paper sections/equations). Verdict for the 3–4 day
window: **V19 is NOT the fast path** — it is unverified, narrowed, and needs ESWA full-text + SNOWS code checks before
any code. The fast path stays V18-confirmatory + paper (see `docs/LIT_REVIEW_2026-11-12.md` §10).

**Question:** Is "multi-downstream-layer reconstruction for retraining-free structured pruning of attention heads +
FFN neurons" a defensible new direction? Distinguish (a) a loss comparing multiple downstream activations while
optimizing one pruned layer, from (b) sequential pruning using already-pruned upstream activations, and (c) iterative
mask search / local reconstruction.

## Finding
The broad idea of repairing a pruning against multiple downstream operations is **not new**. **SNOWS (ICLR 2025)**
§4 Eq. (3) defines a K-step reconstruction objective comparing original vs pruned network at successive downstream
operations, with nonlinear forward propagation and Hessian-free second-order optimization. It evaluates ViTs (QKV,
output-projection, MLP) but with **2:4 semi-structured weight masks, not physical removal of whole BERT heads/FFN
neurons**; authors name LLM specialization as future work. So a narrowly scoped structured-BERT adaptation may remain
untested, but "multi-layer reconstruction" by itself is not a novelty claim.

The **ESWA two-stage** paper is a second close result: cumulative/global output reconstruction for post-training
**structured channel pruning** (ResNet/ImageNet; LLM scaling = future work). Full text was inaccessible (ScienceDirect
403; ResearchGate no full text), so its exact horizon/optimization is unverified — but enough to make "multi-layer/
global reconstruction + structured pruning" already occupied outside Transformers.

STAT, LIAR, G-Pruner, Putri cover physical structures, calibration repair, sequential propagation, or cross-layer
correction, but none establishes the same explicit K-step loss over several future transformer blocks. They still
**undermine any novelty claim about iterative pruning, cross-layer drift compensation, or structured reconstruction in
general.**

**Recommendation:** Do NOT pitch V19 as inventing multi-layer reconstruction. A possible (narrow) contribution is a
specialized, validated method applying a multi-step reconstruction objective to *physical head + FFN-neuron removal in
encoder BERT*, compared to SNOWS-style sparse-mask repair and the ESWA channel method. Before implementing: verify the
ESWA full method and SNOWS ViT head-ablation/implementation details. The claim must be "this exact structured-encoder
setting/mechanism is unaddressed in the checked papers," not "prior work is only iterative" or "nobody reconstructs
multiple layers." Bounded search, not exhaustive-novelty claim.

## Paper-by-paper (repair target vs V19)
- **STAT (2406.00061)** — removes heads+MLP neurons in BERT; correction fused into the **next layer**; attention LS can
  account for error compounded in previous layers. Next-layer correction using the current pruned state, NOT a joint
  K-future-layer objective. Strong prior art for retraining-free physical Transformer pruning + drift-corrected
  next-layer repair — do not claim either as new.
- **LIAR (2407.13331)** — BERT head/FFN + LLaMA; explicitly splits into **layer-wise subproblems**, LS interpolation
  transforms, no backprop. Layer-local, not multi-downstream.
- **G-Pruner (CoNLL 2024)** — structured BERT/DistilBERT heads+FFN; PPO mask refine + CG²MT conjugate-gradient LS per
  layer, first→last, pruned+original inputs so earlier drift enters the target. NOT a K-future-layer loss. "inter-layer"
  != multi-future-layer.
- **ESWA two-stage (DOI 10.1016/j.eswa.2025.128930)** — channel pruning, ResNet/ImageNet; stage-1 transfer from
  planned-pruned channels, stage-2 models cumulative layer-wise error, "global output reconstruction." Exact graph /
  #future-layers **unconfirmed (403)**. Closest structured precedent outside Transformers.
- **SNOWS (ICLR 2025, 2411.18376)** — generic sparse-mask repair; CNN + ViT-B/16, ViT-L/16 with **2:4 masks** on
  QKV/output/MLP. §4 Eq. (3): sum of discrepancies at operations l…l+K using the fixed dense downstream network incl.
  nonlinearities; Hessian-free second-order + CG; mask-agnostic. **Decisive novelty constraint** — already owns
  multi-downstream-operation reconstruction; remaining distinction is physical head/neuron removal in encoder LMs vs 2:4
  ViT sparsity; no BERT eval in the paper.
- **OSSCAR (ICML 2024, 2403.12983)** — structured heads/neurons (OPT/GPT-2 + vision); **layer-wise** squared-output
  target via MIQP/combinatorial. Local target, not multi-layer.
- **Putri "Prune, Update and Trim" (2605.18331)** — Llama3/Qwen3; sequential FFN pruning feeding later FFNs the
  already-pruned outputs; iterative head pruning. Sequential recomputation, not an explicit horizon-K objective.
  "Iterative pruning" + sequential shift compensation already present — not V19 novelties.
- **2SSP (TMLR 2025, 2501.17771)** — LLM FFN-width + whole-attention removal; iterative by perplexity; no multi-layer
  reconstruction described.

## Next checks before any V19 code (NOT a 3–4 day task)
1. Drop broad claims (iterative / drift-aware / layerwise-structured reconstruction / looking beyond one activation) —
   all appear above.
2. Keep candidate narrow: physical head+FFN removal in encoder BERT, bounded multi-block target, no fine-tune, CPU/edge
   latency. Distinction from SNOWS = structured mask/architecture + task, not the existence of downstream-aware repair.
3. **Read full ESWA** (which layers' outputs enter the loss; simultaneous vs sequential; gradients?; which structures).
4. Check **SNOWS code + Appendix A.3** for whether K-step repair was already applied to masks that physically drop whole
   heads/FFN neurons; a mask-agnostic optimizer could make the delta small.
5. Compare vs SNOWS-repair, STAT, K-prune and strongest encoder/LLM baselines under equal calibration/masks/budget/
   timing; pre-register a light fine-tuning comparator (multi-layer objective needs iterative gradient/2nd-order opt).

## Sources
STAT arxiv.org/abs/2406.00061 · LIAR arxiv.org/abs/2407.13331 · G-Pruner aclanthology.org/2024.conll-1.5 ·
ESWA doi.org/10.1016/j.eswa.2025.128930 (abstract only) · SNOWS openreview.net/pdf?id=eNQp79A5Oz, arxiv.org/abs/2411.18376 ·
OSSCAR proceedings.mlr.press/v235/meng24a.html · Putri arxiv.org/abs/2605.18331 · 2SSP arxiv.org/abs/2501.17771
