# Literature sweep 2026-11-12: where we lag, what to try (retraining-free structured BERT pruning)

Scope: retraining-free (post-training) structured pruning of BERT-base (attention heads + FFN neurons), the family our
V14–V18 work sits in. Evidence grade is marked per item: **[full]** full text read, **[abstract]** abstract/snippet
only, **[binary]** PDF fetched but not parsed. Venues marked "unconfirmed" were not verified on the publisher site.

## 1. The baseline bar is stale — this is our biggest lag
`docs/CLI_MISSION_BEAT.md` defines "beat" against Kwon et al. (2022) and K-prune (2024). The field has moved; a Q1
reviewer will reject a win measured only against those two. Newer retraining-free BERT head+FFN pruners **with
reconstruction** that must be positioned against (and ideally run as baselines):

- **LIAR** — "Reconstruct the Pruned Model without Any Retraining", arXiv 2407.13331, IEEE JSTSP 2025 (venue
  unconfirmed). **[full]** Layer-wise least-squares recovery: pruned-channel output approximated as a linear
  interpolation of kept channels; kept weight becomes (I+QP)W_u, bias absorbs the mean term. Sequential, recollects
  inputs from already-updated earlier layers (same family as our drift-corrected refit). **BERT-base GLUE avg: Dense
  87.53; LIAR 85.71 at 50% params** (~98% of dense); beats Mask-Tuning/Kwon reconstruction (83.94) and KCM. This is
  the concrete SOTA number to beat. Code existence not verified.
- **STAT** — "Shrinking Transformers After Training", arXiv 2406.00061. **[binary]** (PDF unreadable; earlier snippet:
  prunes heads+neurons, compensates by **updating subsequent layers** — directly adjacent to our mechanism). MUST be
  read in full before any novelty claim.
- **G-Pruner** — CoNLL 2024, aclanthology 2024.conll-1.5. **[abstract]** Retraining-free head+FFN masking, global
  optimization; claims to beat other retraining-free frameworks and some retraining ones on GLUE/SQuAD, BERT-base +
  DistilBERT.
- **DP-Prune** — IPCCC 2024 (already in our V16 novelty note): global dynamic-programming allocation, retraining-free
  BERT head/FFN.

**Action (dated amendment, before any confirmatory run):** read LIAR + STAT + G-Pruner tables; pick the strongest on
BERT-base GLUE at ~50% FLOPs; if official code runs on BERT-base unmodified, add it as a baseline. Raising the bar
before seeing confirmatory data is allowed (not goalpost-moving). A win over K-prune alone is not a Q1 claim.

## 2. Our own evidence says the open axis is compensation, not selection
- V14: after refit, selector choice is statistically indistinguishable.
- V17: our drift-corrected refit reproduces K-prune's compensation.
Therefore the 2025–2026 **selection-criterion** papers are the weakest bets for us, because refit washes out the
selector: Greedy-Gnorm (2602.04491, Feb 2026 **[abstract]**), HIES (2510.13832 **[abstract]**), SAP syntactic
(2512.19125 **[abstract]**), mutual-information FFN pruning (Entropy 2025, DOI 10.3390/e27121224 **[abstract]**).
Use them as cited baselines/criteria, not as our contribution.

Compensation is the right axis — **but it is also the most crowded**, so novelty is thin and must be verified:
- **SNOWS** — arXiv 2411.18376 (NeurIPS 2024 / ICLR 2025, unconfirmed). **[abstract]** Global **multi-layer**
  reconstruction objective that accounts for nonlinear activations deeper in the network; Hessian-free Newton solver.
  Demonstrated on ResNet/ViT, **unstructured**. Not shown on BERT or structured heads/FFN.
- **Two-stage reconstruction** — "Improving post-training structured pruning via two-stage reconstruction", ESWA 2025
  (unconfirmed; ScienceDirect 403, ResearchGate 393443189). **[abstract]** Explicitly models **accumulated error
  across layers** with a global output reconstruction; structured; claims to beat LLM-Pruner and structured-Wanda. This
  is the closest prior art to our drift-correction direction — **read in full before claiming multi-layer
  reconstruction as novel.**
- **OSSCAR** — arXiv 2403.12983 (ICML 2024). Combinatorial local search on a **layer-wise** reconstruction objective,
  low-rank updates; heads+FFN, but shown on OPT/vision, not BERT GLUE encoders.
- **Greedy Output Approximation (LLM-BIP)** and **Putri** (last-FFN-linear update): **[abstract]** two-step
  reconstruction / error-reducing updates for LLMs.

## 3. The one genuinely under-occupied intersection (candidate direction, needs novelty verification)
**Global multi-layer reconstruction (SNOWS/two-stage style) applied to STRUCTURED head+FFN pruning of a BERT encoder,
with combinatorial selection (OSSCAR style) jointly optimized, and the win proven by measured latency.** Each
ingredient exists; the *combination on encoder-only structured pruning with measured hardware speedup* was not found.
BUT the two-stage ESWA paper and STAT both sit right next to it — decide only after reading both in full. The question
that settles it: **does STAT or two-stage already fit a target beyond the immediate next sublayer for head/FFN
pruning?** If yes, global reconstruction is a baseline/component for us, not the novelty.

## 4. Speed is a hard, concrete gap (the "runs on a low machine" goal)
We have never measured latency; our V18 adds per-task latency, which is the right move. The bar from the literature:
- Kwon: up to **1.56× on V100** at 50% FLOPs, ~1% accuracy drop. **[abstract]**
- SHRP (Dec 2025, **[abstract]**): mixture-of-experts head routing, **4.2× throughput** at ~93% accuracy, 48% params.
- Sources repeatedly warn **FLOPs are a poor latency proxy**. A MAC-ratio win is not a speed win.
A credible "low-machine" claim should measure latency on CPU/edge (matching the owner's goal), not just a GPU MAC count,
and beat these numbers.

## 5. Honest bottom line
The field is mature and crowded; this sweep found **no clear open win**. The compensation axis (where our edge is) is
exactly where the newest strong work clusters (SNOWS, two-stage, STAT, LIAR, OSSCAR). The credible path to Q1, in order:
1. Settle the true SOTA bar (read LIAR/STAT/G-Pruner full text; add the strongest as a baseline via dated amendment).
2. Confirm whether multi-layer reconstruction for structured encoder heads is actually unoccupied (STAT + two-stage
   full read).
3. Only then commit a direction, measured by confirmatory accuracy **and** real CPU/edge latency.
No Q1 result is promised; the evidence does not yet support one.

## Open full-text reads still needed (in priority order)
STAT (2406.00061) · two-stage ESWA (ResearchGate 393443189) · G-Pruner (2024.conll-1.5) · SNOWS (2411.18376) ·
OSSCAR applicability to BERT-base.
