# Literature sweep 2026-11-12: where we lag, what to try (retraining-free structured BERT pruning)

Scope: retraining-free (post-training) structured pruning of BERT-base (attention heads + FFN neurons), the family our
V14–V18 work sits in. Evidence grade is marked per item: **[full]** full text read, **[abstract]** abstract/snippet
only, **[binary]** PDF fetched but not parsed. Venues marked "unconfirmed" were not verified on the publisher site.

## 1. The baseline bar is stale — this is our biggest lag
`docs/CLI_MISSION_BEAT.md` defines "beat" against Kwon et al. (2022) and K-prune (2024). The field has moved; a Q1
reviewer will reject a win measured only against those two. Newer retraining-free BERT head+FFN pruners **with
reconstruction** that must be positioned against (and ideally run as baselines):

- **LIAR** — "Reconstruct the Pruned Model without Any Retraining", arXiv 2407.13331, IEEE JSTSP 2025 (venue
  unconfirmed). **[summary — via WebFetch summarizer, not full text]** Layer-wise least-squares recovery: pruned-channel
  output approximated as a linear interpolation of kept channels; kept weight becomes (I+QP)W_u, bias absorbs the mean
  term. Sequential, recollects inputs from already-updated earlier layers (same family as our refit). Reported BERT-base
  GLUE avg: Dense 87.53, LIAR 85.71 at **50% params** (~98% of dense), on LIAR's own checkpoints/protocol.
  **CAUTION: 85.71 is a cross-protocol number (their checkpoints, param ratio, GLUE average) — it is context, NOT a bar
  we can compare to directly.** A real bar needs LIAR's code run on OUR checkpoints, calibration rows and MAC budget.
  Code existence not verified.
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

## 6. STAT is close prior art for compensation — [summary via WebFetch, NOT full text; attribution UNVERIFIED]
STAT (arXiv 2406.00061, 2024) prunes heads+neurons retraining-free and compensates by folding an interpolation
correction into the next layer; the summarizer says for deeper models it re-runs the pruning data through the pruned
network and solves least squares. **Correction needed vs my earlier claim:** student-input LS on the pruned network is
GRP's *plain* refit (the one that LOST 34–60% in V16/V17), not necessarily our drift-corrected target (which carries the
h_t − h_s term we found in K-prune's code). **Whether STAT's target includes the drift term is UNVERIFIED — read STAT's
actual equations before attributing our mechanism to it.** Either way, drift correction is not ours (it is K-prune's);
the point is only to cite the right source. STAT's baseline is Kwon et al. 2022 (not "K-prune"); it reports BERT GLUE as
a FLOPs–accuracy curve 20–90%, no single 50% number.

## 7. Candidate gaps — [all summarizer-reported, NOT verbatim STAT quotes; verify before relying]
The WebFetch summary of STAT *reported* these as limitations (its wording, not STAT's verbatim text):
- possibly no mechanism correcting errors across *multiple* downstream layers;
- one-shot, "an iterative approach could perform better";
- per-layer error metric is architecture-dependent (LayerNorm placement).

**Correction — "iterative is open" is FALSE here.** K-prune already prunes sublayer-by-sublayer, re-scoring on the
partially pruned+tuned model (verified in its code during V16), and our own GRP greedy re-scores after every removal.
So iterative prune→refit→re-score is NOT novel. **The ONLY candidate open axis is the reconstruction *target* spanning
several downstream sublayers for STRUCTURED encoder heads/FFN** (SNOWS does multi-layer but unstructured/vision). Even
that is unverified against the two-stage ESWA paper, "Prune, Update and Trim" (2605.18331) and LLM-BIP, all unread.

**Two hard caveats for any such V19:**
1. A target through LayerNorm/GELU/softmax has **no closed form** → needs Newton/gradient optimization on the
   calibration rows (SNOWS-style). That invites a reviewer to ask "why not a light fine-tune / distillation on the same
   2048 rows at matched compute?" (the "Free Lunch" point, 2510.14444). Pre-register that baseline, or keep it
   closed-form.
2. **No evidence multi-layer reconstruction even helps structured BERT** (SNOWS evidence is unstructured vision). It is
   a research bet, not a sure win.

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

## 10. Go/no-go for the NEXT implementation (corrected 2026-11-12 after reading V18 freeze)
**V18 (d2i-RI) is frozen and is the real next step, not V19.** It beats both baselines on the three exploration tasks
(MRPC, SST-2, STS-B) at 0.5/0.4 — but the accuracy margins are inside dev-set noise (MRPC +0.0029/+0.0010, Pearson
+0.0005); the only clear effect is held-out KL/MSE −5..−13% vs K-prune. Its confirmatory run on QNLI, MNLI-m, QQP has
NOT happened yet. **Implications:**
- **V18 claims the three confirmatory tasks.** V19 cannot reuse them. V19 needs its OWN untouched confirmatory set,
  fixed before any V19 code: **SQuAD v1.1 validation is clean** (only `_smoke` touched it) — or RoBERTa-base GLUE.
- **V18 keeps K-prune's masks → identical shapes → identical latency.** It CANNOT deliver the owner's "runs on a smaller
  machine" goal. Best case it is an *accuracy-at-equal-compute* win, and that margin is noise. The owner must know this.
- The honest "low-machine over SOTA" target is **match K-prune's accuracy at a LOWER budget** (e.g. K-prune@0.5 reached
  at 0.4), not a raw latency number. Latency-aware head/FFN allocation is already published (ZipLM).

**Ordered path (do NOT start V19 implementation yet):**
1. Let V18's confirmatory run finish on QNLI/MNLI-m/QQP. Record the verdict. That result reshapes everything.
2. Owner decision: add STAT + LIAR as baselines via a dated amendment IF their code runs on BERT-base (check first);
   without newer baselines a win is not Q1-credible. Raising the bar before confirmatory data is allowed.
3. Only then consider V19 (multi-layer-target structured pruning at a lower budget), after reading two-stage ESWA +
   2605.18331 + LLM-BIP in full, reserving SQuAD/RoBERTa as its confirmatory set, and pre-registering a fine-tune
   baseline (§7 caveat 1).
4. Always measure CPU latency across the full accuracy–FLOPs curve; never claim speed from FLOPs alone.
5. If no accuracy win survives, pivot to the rigorous selection-vs-compensation analysis (§9) — still publishable.
No Q1 *method* claim is promised; this is the credible route, not a guarantee.
