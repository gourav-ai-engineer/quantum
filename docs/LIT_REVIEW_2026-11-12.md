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
two-stage ESWA (ResearchGate 393443189) · G-Pruner (2024.conll-1.5) · SNOWS (2411.18376) ·
OSSCAR applicability to BERT-base. (STAT now read in full — see Part 2.)

---

# Part 2 (deeper pass): gap analysis + what gets pruning papers rejected

## 6. STAT already publishes our V17 "mechanism" — [full read, decisive]
STAT (arXiv 2406.00061, 2024) prunes heads+neurons retraining-free and compensates by **folding an interpolation
correction into the next layer**; for deeper models it **re-runs the pruning data through the already-pruned network
and solves least squares, so the correction accounts for errors earlier pruned layers introduced into the inputs.**
That is exactly our drift-corrected refit (V17). **Our drift correction is therefore NOT novel — STAT precedes it.**
The paper must cite STAT and must not claim next-layer/residual-drift compensation as a contribution. (It does not use
the name "K-prune"; its baseline is Kwon et al. 2022; it reports BERT GLUE as a FLOPs–accuracy curve 20–90%, matching
dense within error bars at modest pruning and beating Kwon at high compression — no single 50% number given.)

## 7. STAT's own stated limitations ARE the open gaps (this is the answer to "what new thing can we try")
Quoted from STAT's limitations/future-work:
- **No multi-downstream-layer correction.** "The paper does not describe a separate mechanism that corrects errors
  across multiple downstream layers." → **Global multi-layer reconstruction for structured head/FFN encoder pruning is
  open** (SNOWS does it but unstructured/vision; two-stage ESWA may — verify).
- **One-shot only.** "Pruning is one-shot. An iterative approach could perform better, but would cost more." →
  **Iterative prune→refit→re-score for structured heads/FFN is open.** Our V14/V17 evidence (refit washes out the
  selector) makes iterative selection-under-compensation the natural, mechanism-driven idea: re-score what to prune
  *after* each compensation step, not before.
- **Architecture-dependent error metric.** "The per-layer error metric must be adapted to each architecture, since
  layer-norm placement changes how errors propagate... a systematic study is future work." → open.

**Best V19 candidate (mechanism-driven, targets stated open problems):** iterative, multi-downstream-layer
reconstruction for structured head+FFN pruning of BERT, i.e. at each round prune a small slice, refit against a target
that spans several downstream sublayers (SNOWS-style), then re-score the next slice from the compensated model.
Baselines: Kwon, K-prune, STAT, LIAR. Win shown on measured CPU/edge latency across the whole accuracy–FLOPs curve.
Novelty must still be checked against the two-stage ESWA paper (closest) before committing.

## 8. What gets pruning/compression papers rejected (reviewer criteria to pre-empt)
From Blalock et al. "What is the State of Neural Network Pruning?" (MLSys 2020) **[abstract/summary]** and Wang et al.
"Why is the State of NN Pruning so Confusing?" (arXiv 2301.05219) **[full text extracted locally]**:
- **Too few / weak baselines.** Meta-analysis of 81 papers: most are compared against by very few others; many compare
  against few baselines. Reviewers penalize this. Always include **random** and **magnitude** baselines, plus the
  current SOTA (here: STAT, LIAR), not just the 1–2 you can beat.
- **Confounded comparison — the single biggest rejection cause.** Wang et al. fairness principle: *the performance
  advantage must be attributable to the pruning stage, not to outside factors.* They revive a 6-year-old L1 baseline to
  match recent top venues **just by using a better fine-tuning LR schedule.** Lesson for us: hold the base model,
  calibration set, FLOPs budget, and evaluation protocol identical across all methods; report all settings.
- **FLOPs ≠ latency.** A MAC-ratio win is not a speed win; measure real hardware (ideally CPU/edge for the
  "low-machine" goal).
- **Single operating point.** Report the full accuracy–FLOPs curve, not one cherry-picked sparsity.
- **No standardized / reproducible eval.** They call for ShrinkBench-style standardized evaluation and full release of
  settings and code.

## 9. Our rigor is already a Q1-grade strength — lean on it
The pre-registration discipline we already follow directly answers the above rejection causes: hypotheses and
thresholds fixed before data; selection on held-out train only; paired cluster bootstrap with Holm; identical
checkpoints/calibration/budget for every method; physical pruning + per-task latency (V18). Few pruning papers do
this. Even if a raw SOTA win is thin, a **rigorous, pre-registered, reproducible comparison with an honest
analysis of what actually matters (selection vs compensation vs allocation)** is itself a defensible contribution for a
methods-and-reproducibility-friendly Q1 venue. Pair it with the iterative/multi-layer idea for the method claim.

## 10. Revised path to Q1 (ordered, honest)
1. Add STAT + LIAR as baselines (dated amendment) — non-negotiable; without them a win is not Q1-credible.
2. Verify the two-stage ESWA paper does NOT already do iterative multi-layer structured reconstruction (full read).
3. Build the V19 candidate only if step 2 leaves it open; freeze + pre-register; confirm on QNLI/MNLI-m/QQP once.
4. Measure CPU latency across the full accuracy–FLOPs curve; never claim speed from FLOPs alone.
5. If no accuracy win survives, pivot the paper to the rigorous selection-vs-compensation analysis (strength in §9),
   which is still publishable. No Q1 *method* claim is promised; this is the credible route, not a guarantee.
