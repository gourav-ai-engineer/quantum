# Strategy audit, 2026-11-19: failure forensics, publication route, prior art, method candidate

Written by Claude (Opus 5.5) for the owner, answering the "independent senior researcher" audit prompt pasted on
2026-11-19. Source of truth: `docs/PROJECT_STATE.md` as of commit 5b60342b. No V21 per-condition prediction file, `run_*.log`
or pruned-condition number was read for this audit. Labels: VERIFIED = primary source read by me, location given;
PLAUSIBLE = secondary/abstract/another session's reading; UNVERIFIED = not checked. Repo facts are cited by ledger section;
they are ledger records, not recomputed here.

## 1. Executive verdict

The strategy failed for one dominant reason that the project's own data showed early: in retraining-free structured
pruning of fine-tuned BERT encoders at 50% MACs, the closed-form repair carries the recovery and the remaining headroom over
K-prune is tiny (teacher minus K-prune accuracy 0.01-2.43 pp across six tasks, `EXPLORATION_LOG.md` E3), yet the project kept
building new selectors and repair variants to "beat" K-prune at exactly that operating point. V14 (selectors indistinguishable
after LS repair) and K-prune's own ablation (VERIFIED: K-prune arXiv 2308.03449 §4.5 Table 2, removing weight-tuning drops
SQuAD F1 at 80% compression from 74.16 to 16.50) both said "repair dominates" before V15-V18 were run. Novelty checks were
also done after building, not before (STAT's next-layer correction, SNOWS's multi-layer loss and the multilingual prior art were
found after the corresponding experiments were designed). V21 should be finished as frozen, because three of six units already
passed, the remaining cost is two kernels, and abandoning it would leave an incomplete registered study. But it is a bounded
confirmatory re-examination, not the paper's centre. No new head-pruning algorithm survives the novelty-and-headroom screen
(section 5). **Single recommended next action: get the analysis paper through the Rule 11 independent audit and the supervisor's
proof check now (zero GPU), while V21 waits for the Kaggle quota.**

## 2. Forensic failure audit

### 2.1 Chronology

| # | Experiment / idea | Original hypothesis | What happened (ledger) | Failure type | Root cause (evidence) | What was learned | Status |
|---|---|---|---|---|---|---|---|
| 0 | QIS score (pre-repo notebook) | 0.5 VNE + 0.3 QJSD - 0.2 F ranks heads by importance | Spearman about -0.05 with importance; identical "multi-seed" runs; t-test mislabelled paired (owner-reported, CLAUDE.md "History") | Scientific + statistical | Arbitrary hand weights; no validation of the score against importance | Never combine measures with tuned weights; seeds must vary something | Closed |
| 1 | QFC fidelity coverage (V4-V10) | Fidelity-coverage subset selection beats VNE/Michel/Random | Not consistently better; fidelity did not beat cosine or Hilbert-Schmidt (V8); input-conditioned QFC failed on full validation (V10) (owner-reported; no V8/V10 JSON committed) | Scientific | Attention-pattern similarity is the wrong redundancy object (later proved: Prop. 1) | Submodularity guarantees the objective, not accuracy | Closed |
| 2 | V12 baseline bugs | n/a | Random kept all heads; Shannon NaN under padding (all heads 0..k-1) | Engineering (baseline) | Unvalidated baselines | All earlier Random/Shannon numbers invalid; selectors must assert k distinct heads | Fixed |
| 3 | V13 diversity vs importance | IWQFC beats Michel | INCONCLUSIVE: MRPC 2/3 seeds CI > 0, SST-2 0/3 (`PROJECT_STATE` V13 row) | Scientific (null) | Effect, if any, task-dependent and small | Task dependence is the norm, not the exception | Closed |
| 4 | V14 BCM output-space coverage + compensation | Output coverage selector + merge/LS beats classical selectors | Pre-registered factorial: after LS refit four selectors statistically indistinguishable; attention-pattern coverage clearly worse; H9b/c/d not supported. Diagnostics: attention-blindness 72-97% (20% CoLA); bound ranks subsets (Spearman 0.915-0.955) but ~3x loose | Scientific (informative negative) | Repair absorbs selection differences at 50% heads/layer | **The decisive finding of the project**: selection is a poor lever under LS repair | Closed; core of paper |
| 5 | V15 GRP (global refit-aware allocation, heads + FFN) | Lower held-out-train KL than both official baselines at >=2 of 4 budgets on both dev tasks (H10) | Gate failed: SST-2 4/4 (KL 40-75% below K-prune), MRPC 0/4; registered accuracy test (H11) not run. Baseline reproduction passed (K-prune MRPC 0.8485 vs 84.80; SQuAD F1 74.20 vs 74.16). Engineering: gdown, missing sklearn, MRPC index gaps, disk full, quota | Scientific (gate) + engineering | Refit target lacked drift correction; allocation effect task-dependent | A calibration-side gate is not an accuracy result in either direction | Closed |
| 6 | V16 Stage A diagnosis | Locate why GRP is task-dependent | Branch A-3: GRP refit on K-prune masks 34-60% WORSE than K-prune's tuning; mechanism found by reading K-prune code: drift-corrected LS target | Scientific (diagnostic) | Our repair was weaker than the baseline's | The baseline's repair, not its selector, is its strength | Closed |
| 7 | V17 GRP-R (GRP + drift-corrected refit) | GRP-R <= 1.0x K-prune KL on MRPC and <= 0.9x on SST-2 | Fails: MRPC 1.19/1.29x; SST-2 0.31/0.50x. K-prune masks + drift refit = 0.97-1.00x K-prune. Registered AFTER V15/V16 dev data were seen | Scientific + statistical (post-hoc registration) | Allocation still task-dependent; repair equalised | Equal repair -> selection winner is task-dependent | Closed |
| 8 | Exploration E0-E2 (d2i-RI) | Input-side alignment + drift refit beats both baselines on MRPC/SST-2/STS-B | Freeze rule met with noise-level margins (MRPC +0.0029/+0.0010; STS-B Pearson +0.0005); clear effect only held-out KL -5..-13% | Statistical (rule permitted a noise "win") | Freeze rule did not demand an effect size above noise | Freeze rules need a minimum effect, not a sign | Closed |
| 9 | V18 d2i-RI confirmatory | d2i-RI beats better-of {Kwon, K-prune} on >=2/3 of QNLI/MNLI-m/QQP | H18a NOT supported: vs K-prune -0.0002, +0.0016, +0.0013 (QQP only significant); vs Kwon +2.8/+8.1/+5.6 pp. Kwon cupy import failure (fix1). Post-hoc held-out KL ratios 0.958/0.994/0.952: the dev mechanism mostly did not transfer | Scientific (clean negative) + engineering | Same masks as K-prune; headroom 1.27-2.43 pp | Ties are ties; "same mask, new repair" cannot be a compression win | Closed |
| 10 | H18b latency | Descriptive | CPU (i7-13700, FP32): K-prune shapes 1.65x (b1), 2.05x (b32); T4 FP16: no speedup at b1 (8.96 vs 9.2-9.8 ms), 1.2-1.3x at b32. Three GPU attempts (transformers 5.x on Kaggle) | Engineering | Platform drift | FLOPs != latency; GPU batch-1 gains absent | Done |
| 11 | E3 portfolio pre-check | Pick d2i-RI vs GRP-RI by held-out KL | MRPC -0.10 pp; KL picked the accuracy-worse mask 2/5 seeds | Scientific | 512 held-out rows cannot rank near-equal masks | Calibration-KL selection is unreliable at this headroom | Closed |
| 12 | V19 DistilBERT-weights, 6 GLUE tasks | H19a superiority >=4/6; H19b non-inferiority 1.5 pp | H19a NOT supported (1/6, QQP +0.16 pp); H19b supported 6/6; beats Kwon on 6/6 (4 Holm-significant). SQuAD dropped before any statistic: official K-prune OOM on T4, patching would weaken the baseline. HF 429, partial Drive download | Scientific (replicated tie) + engineering | Same as V18 | Replicated parity; registered non-inferiority now exists | Closed |
| 13 | V20 multilingual pilot (MuRIL, validation) | H20b: English-calibrated repair raises English more than targets | NO-GO, wrong sign: pooled -0.1357 [-0.1634, -0.1072]. Disparity entered through English-selected MASKS (mask-only targets about 0.41) | Scientific (falsified) | Wrong causal story | Pilot with a kill rule worked as designed | Closed |
| 14 | Direction A (target-language mask selection) | Language-matched masks reduce disparity | Not run; ceiling 1.0-2.5 pp after best repair; confound with calibration language; Li et al. ACL 2022 per-language pruning does not beat shared pruning (`RESEARCH_DIRECTION_AUDIT` §10) | Novelty + headroom | Prior evidence against, ceiling below threshold | Compute headroom before designing a method | Closed |
| 15 | V21 confirmatory multilingual (MuRIL + mBERT x 3 teachers, test split) | H21-1 English-selected masks hurt targets more than English; H21-2..4 secondaries | Frozen; muril_t42, mbert_t42, muril_t43 PASS integrity; mbert_t43 running; t44 blocked by weekly GPU quota. No pruned-condition number read | Pending; project-management (quota) | 6 kernels x 6-10 h against 30 GPU-h/week | n/a | Pending |
| 16 | Process | n/a | Novelty found after design (STAT, SNOWS, multilingual "no study found" claim in `Q1_GAP_RESEARCH` later closed by the 11-18 audit); failures hidden by `|| echo` and `tail` (V19 failure 1); concurrent sessions and an external assistant editing the working tree (2026-11-19: an uncommitted edit to the frozen `v21_analysis.py`, accepted only after new tests) | Project-management | No prior-art gate before design; shared mutable tree | Prior-art screen and headroom check BEFORE any design; one writer per tree | Ongoing risk |

### 2.2 The most consequential wrong strategic choices
1. **Selector-first after V14.** Reasonable at the start (QFC was the founding idea); wrong once V14 showed selectors indistinguishable after LS repair. V15 (GRP) and V17 (GRP-R) still targeted selection/allocation.
2. **Fixing the operating point at 50% MACs without computing headroom.** The headroom (0.01-2.43 pp) was first computed in E3 on 2026-11-16, after V18/V19. At this budget a "beat K-prune everywhere" goal was bounded by arithmetic.
3. **Choosing a same-mask candidate (d2i-RI) for the confirmatory slot** while the owner's goal was "runs on a smaller machine". Identical shapes guarantee zero compression or latency gain relative to K-prune.
4. **Prior-art search after building.** STAT's correction and Llama least-squares re-run (VERIFIED: arXiv 2406.00061 §3.1-3.2), SNOWS's multi-layer loss, GRAIL's selector-agnostic compensation and the multilingual literature were each found after the experiment they undercut was designed.
5. **Pivoting the question to find a gap** (monolingual method -> multilingual mechanism -> multilingual disparity confirmation) instead of finishing one paper. This split GPU quota and produced two storylines.
Reasonable at the time and still defensible: V15 as a theory-motivated test with a gate; V18 as an honest single-use confirmatory; V20 as a cheap pilot with a kill rule.

### 2.3 Mistakes not to repeat (each with ledger support)
- Do not build a selector or score when repair dominates (V14; K-prune Table 2).
- Do not read non-inferiority or a tie as a win (V18 H18a; V19 H19a vs H19b).
- Do not present "same mask, new repair" as compression (V18 do-not-claim list; H18b shapes).
- Do not accept a freeze rule without a minimum effect size (E1/E2 MRPC +0.0010).
- Do not register a hypothesis after seeing its development data without labelling it (V17 disclosure).
- Do not trust calibration-KL to rank masks at small headroom (E3).
- Do not hide failures in kernel logic (`|| echo`, `tail`; V19 failure 1).
- Do not let more than one writer change the tree, and never let an external assistant edit frozen code uncommitted (2026-11-19).

### 2.4 Which negatives strengthen the paper, which thin it
Strengthen: the pre-registered V14 factorial (encoder evidence that selection matters little after repair, matching decoder and
vision reports: Free Lunch §3(iv), GRAIL abstract); the attention-blindness proposition plus its measurement; V18/V19 parity
against official baselines with integrity gates; the headroom table; the latency reality (no GPU batch-1 speedup).
Thin or unfocus it: the V15-V17 GRP narrative (development-only, gate failures, mechanism already in K-prune's Eq. 10-11 and in
STAT); d2i-RI as a "method" (a combination that ties); the quantum framing (diagnostic only; the selector failed); and a
multilingual storyline (V20/V21) that asks a different question.

### 2.5 Framing ranking
1. **(a) Honest analysis/negative-results paper** on BERT encoders: most evidence-ready. Needs Rule 11, supervisor proofs, and
   repositioning against GRAIL, Free Lunch, STAT, K-prune Table 2.
2. **(d) Separate papers**: (a) now; V21 as a short separate confirmatory paper or a bounded appendix, decided after its single
   analysis.
3. **(b) Multilingual confirmatory study as the main paper**: weaker; incomplete, prior art dense (section 4), ceiling small.
4. **(c) New-method paper**: no candidate (section 5).

## 3. Fastest defensible route

### 3.1 Ranked actions
| Rank | Action | GPU-h | Time to decision | Risk | Impact if it succeeds |
|---|---|---|---|---|---|
| 1 | Rule 11 independent human audit + supervisor proof check of the analysis paper | 0 | 3-7 days (supervisor) | Low | Unblocks submission |
| 2 | Reposition the paper: compact V15-V17 into one "what failed" section; cite GRAIL, Free Lunch, STAT (no K-prune comparison), K-prune Table 2, NAACL 2021 attention redundancy; keep claims bounded | 0 | 2-3 days | Low | Removes the main novelty objections |
| 3 | Finish V21 (muril_t44, mbert_t44 after quota reset; wait for mbert_t43) and run its single analysis | about 12-20 | 2-4 days after reset | Medium (platform) | Completes a registered study; decides V21 placement |
| 4 | CPU latency and parameter counts for the multilingual shapes (registered descriptive) | 0 | 1 day | Low | Honest efficiency reporting |
| 5 | Optional high-compression dev kill test (section 3.3), only if a method paper is still wanted | about 8-12 | 1 week after V21 | High (likely NO-GO) | Decides the method line |

### 3.2 GO / NO-GO tree
- Now: actions 1, 2, 4 in parallel; V21 waits for quota.
- After V21's single analysis:
  - H21-1 supported for MuRIL and the pre-declared floor checks agree -> V21 as a separate short confirmatory paper (or a bounded section if the supervisor prefers one paper).
  - Otherwise -> one paragraph plus appendix in the analysis paper; no multilingual claim.
- Method line: run the kill test only if the owner still wants a method paper. If it fails -> close the method line permanently and write that in `PROJECT_STATE`.

### 3.3 Development-only kill test (only for deciding the method line; no new method is involved)
- Question: is there any selection/allocation effect beyond K-prune once headroom exists?
- Data: BERT-base MRPC, SST-2, STS-B (already development); calibration `train.shuffle(seed)[:2048]`; 5 seeds; budgets c = 0.3 and 0.2 MACs kept.
- Arms: official K-prune and Kwon (unmodified, new constraint only); GRP-RI (existing code, different allocation); d2i-RI (existing code, K-prune masks: measures repair headroom only, cannot count as a compression win).
- Metric: validation accuracy (Pearson for STS-B), mean of 5 seeds, plus held-out-train KL; physical MAC ratio checked.
- Pass (all must hold, fixed now): teacher minus K-prune headroom >= 3 pp at that budget; GRP-RI beats K-prune by >= 1.0 pp (STS-B >= 0.010 Pearson) on ALL three tasks at the same budget, with >= 4 of 5 seed wins per task; no task where GRP-RI loses.
- Fail -> stop the method line. Expected outcome: fail (MRPC already lost at 0.5/0.4).
- Confirmation if it passed: BERT-base SQuAD v1.1 (unspent; K-prune ran SQuAD on a T4 in V15 Phase 1) plus newly fine-tuned teacher seeds; official baselines; physical-shape latency CPU and GPU. Caveat: every GLUE validation split has been used for some hypothesis; fresh confirmatory capacity is the binding constraint for any new claim (the official codes support GLUE and SQuAD loaders only).

### 3.4 Guards
Frozen tags per study; integrity gate before any statistic; analysis code committed before data; physical MAC and shape per
condition; any method reusing baseline masks reports a compression delta of zero; official baselines unmodified (a baseline that
cannot run is a deviation, never a patch); single writer per tree.

### 3.5 Which target is defensible
- High compression: headroom exists, but K-prune is strongest there (VERIFIED: K-prune §4.2, "The accuracy gap between K-prune and
  the competitors grows larger as the compression ratio gets higher"), and STAT also claims its advantage over Kwon at high compression
  (VERIFIED: STAT Fig. 1, §4). Crowded and uncertain.
- Robust multilingual quality: dense prior art; ceiling 1.0-2.5 pp after best repair (V20 descriptive).
- Latency / model size: measured reality is modest (CPU 1.65-2.06x; GPU batch 1 none; multilingual embeddings 52-64% of parameters).
- Most defensible target for this project: an evaluation/analysis paper, not a method target.

## 4. Prior-art matrix and novelty verdict

Search log (dates): 2026-11-12 lit sweep (`LIT_REVIEW_2026-11-12.md`); 2026-11-16 web searches "retraining-free structured pruning
encoder language model outperforms K-prune 2025 2026", "post-training structured pruning mask selection calibration loss after
reconstruction ... 2026 arXiv", "local search swap refinement structured pruning attention heads FFN ..."; G-Pruner PDF read
(2024.conll-1.5.pdf); 2026-11-17/18 multilingual audits (`Q1_GAP_RESEARCH_2026-11-17.md`, `RESEARCH_DIRECTION_AUDIT_2026-11-18.md`);
2026-11-19 searches "Intriguing Properties of Compression on Multilingual Models ...", "pruning multilingual language model
disproportionately hurts low-resource languages ...", "redundant attention heads attention map similarity versus head output
similarity ...", "STAT ... github code; Global-Pruner ... github"; full-text reads of arXiv 2308.03449, 2510.14444v3, 2406.00061,
2503.09657v4 (HTML) and the abstract pages of 2602.23795 and EACL 2023.177.

| Closest work | Overlapping operation | What it does not do | Evidence location | Label | Overlap | Consequence for us |
|---|---|---|---|---|---|---|
| K-prune (ICLR 2024) | Iterative bottom-up sublayer pruning; least-squares re-tuning of output projections (the "drift-corrected" target) | No multilingual; no pre-registered comparison | arXiv 2308.03449 §3.1-3.4, Eq. 10-11, App. B.2; §4.2; Table 2 | VERIFIED | High | Our V16/V17 "mechanism" is K-prune's own Eq. 10-11; claim only the transplant evidence |
| Kwon et al. (NeurIPS 2022) | Fisher masks, rearrangement, LS mask tuning | Weaker repair | Official repo README; paper | PLAUSIBLE | High (baseline) | Beating Kwon is expected, not a contribution |
| STAT (arXiv 2406.00061) | Pivoted-QR head ranking; interpolation matrix folded into the next layer; Llama LS re-run for earlier-layer error | Never compares K-prune; 512 calibration examples | §3.1-3.3, Eq. 2-3; §4 Fig. 1 (20-90% FLOPs, beats Kwon at high compression) | VERIFIED | High | Drift/next-layer correction is published; STAT vs K-prune is an open comparison |
| Global-Pruner (CoNLL 2024) | PPO mask refinement from Fisher masks + CG^2MT LS mask tuning; K-prune as a baseline | No STAT comparison; not reproduced by us | Fig. 2, §5.2, Table 1, §5.1 (2,000 samples, 10 seeds, transformers 4.12.0) | VERIFIED | High | A published claim of lower degradation than K-prune exists; cite in its own units |
| GRAIL (CPAL 2026) | Selector-agnostic post-hoc ridge compensation absorbed into downstream weights | ResNet/ViT/decoder LLMs, not fine-tuned BERT encoders | arXiv 2602.23795 abstract | PLAUSIBLE | High for the selection-vs-compensation framing | Our V14 factorial is an encoder confirmation, not a new insight |
| Free Lunch (arXiv 2510.14444) | Local reconstruction; "mask selection becomes negligible" | Decoder LLMs; structured only in App. F | §1 contributions; §3(iv), Table 4, Fig. 8 | VERIFIED | High | Same direction as V14; cite as corroboration |
| SNOWS (ICLR 2025) | Multi-downstream-operation reconstruction loss | 2:4 masks on ViTs; no BERT heads | §4 Eq. 3 (per `V19_PRIOR_ART.md`, external reviewer) | PLAUSIBLE | High for multi-layer repair | Closes multi-layer repair as a novelty |
| Týr-the-Pruner (arXiv 2503.09657) | Supernet of local pruning at several sparsities; evolutionary sparsity-distribution search; closed-form adjustment; heads + FFN | Decoder LLMs only | §2.2-2.4, Eq. 5-6, Alg. 3-4; §3.1 | VERIFIED | High for allocation search | Closes "repair-aware allocation search" as a novelty |
| CORP (arXiv 2602.05243) | Closed-form ridge compensation, selection vs compensation | Vision | `Q1_GAP_RESEARCH` ([full] read by another session) | PLAUSIBLE | Medium-high | Same as GRAIL |
| OSSCAR (ICML 2024) | Layer-wise combinatorial local search with reconstruction | Layer-wise objective | Abstract | PLAUSIBLE | Medium | Closes local-search mask refinement |
| SoBP (EMNLP 2024) | Global importance + local greedy refinement + module-wise reconstruction, heads + FFN | Not read | Search summary | UNVERIFIED | Medium | Must be read before any allocation claim |
| Li et al. (ACL 2022) | Structured multilingual pruning of XLM-R, English-source transfer; per-language pruning not better | Retraining-based settings | Record + ChatGPT round 2 (VERIFIED by it) | PLAUSIBLE | High for multilingual | H21-1 is a re-examination |
| Ogueji et al. (EMNLP 2022) | Compression effect on 40 languages | Unstructured, during fine-tuning | ChatGPT round 2 (VERIFIED by it) | PLAUSIBLE | Medium | Disparity is not a new phenomenon |
| Shapley Head Pruning (EACL 2023) | Language-specific interfering heads pruned for a target language | Details unread | Abstract | PLAUSIBLE | High for target-language head selection | Closes Direction A novelty |
| M-Wanda (EMNLP 2025), Kurz et al. (TACL 2026), MBS (LREC-COLING 2024) | Multilingual calibration / language-aware pruning statistics | Mostly decoder LLMs (unverified detail) | Records (`RESEARCH_DIRECTION_AUDIT` §10) | PLAUSIBLE | Medium-high | Closes calibration-language novelty |
| NAACL 2021 attention redundancy (BERT) | Pairwise Jensen-Shannon distance between attention maps | Does not compare with head-output similarity (per search summary) | Search summary | UNVERIFIED | Medium | Prop. 1 + measurement may still be a modest contribution; read before claiming |
| CompactifAI (ESANN 2025), OBC (NeurIPS 2022), ZipLM (NeurIPS 2023) | Quantum-inspired MPO compression; joint post-training pruning + quantization | n/a | arXiv 2401.14109 abstract; ChatGPT round 2 | PLAUSIBLE | High | "Prune + quantize" or "add MPO" is not novel by itself |

Novelty verdict (calibrated, not certainty): new pruning algorithm in this space: **low novelty, high confidence**. The analysis
paper: **low-to-medium novelty, medium confidence**. Its defensible pieces are the pre-registered encoder evidence with official
baselines and integrity gates, the attention-blindness proposition plus measurement (pending a read of the NAACL 2021 paper), and
the confirmatory parity results. "Not found in this search" is not "proven novel".

## 5. Candidate method: none passes

Candidates screened and rejected:
1. **Target-language mask selection (Direction A).** Ceiling 1.0-2.5 pp after best repair; confound with calibration language;
   Li et al. report per-language pruning does not beat shared pruning; Shapley Head Pruning occupies target-language head
   selection. Reject.
2. **Repair-aware global allocation search at high compression.** Týr-the-Pruner (supernet + evolutionary allocation +
   closed-form adjustment, §2.2-2.4) and K-prune's global FLOPs-aware search already do this; K-prune is strongest at high
   compression (§4.2). A BERT transfer is a combination of known pieces. Reject as a novelty claim; it survives only as the
   kill-test question in 3.3.
3. **Multi-layer or iterative reconstruction.** SNOWS, Free Lunch (block sizes up to a quarter of the network), STAT's LS re-run.
   Reject.
4. **Head merging before repair.** Our own Theorem says the LS refit error is at most the merge error, so merging cannot beat the
   refit it feeds. Reject.
5. **Quantum-inspired add-ons (MPO, fidelity selection).** CompactifAI; QFC failed empirically. Reject.

Verdict: **no defensible new algorithm emerged.** Best alternative: the analysis paper now; V21 finished as registered; the
kill test in 3.3 only if the owner insists on testing the method line, with "stop" as the expected and accepted outcome.

## 6. Q1 readiness
- Paper type: (a) analysis / negative results on BERT encoders, quantum-information analysis as a diagnostic lens.
- Blockers: Rule 11 unsigned; supervisor proof check; related-work repositioning (GRAIL, Free Lunch, STAT, K-prune Table 2,
  NAACL 2021); compaction of V15-V17; first compile (Overleaf); Elsevier format, generative-AI disclosure, data and code
  availability; V21 placement decision.
- Minimum evidence: what exists (V14, V18, V19, latency, headroom) plus the repositioned literature. No new GPU run is required
  for this paper.
- Journal fit: Neurocomputing lists 2025 SJR 1.465, Q1 (SCImago; provider-specific; reported VERIFIED by ChatGPT round 2). TMLR's
  scope fits analysis/reproducibility but its Q1 status is not established here. A negative/analysis paper faces "limited
  novelty" objections at any venue.
- Realistic readiness: submittable about 1-2 weeks after the supervisor's check. Acceptance is not predictable.

## 7. Evidence gaps (things not in the repository)
1. Supervisor's proof comments (Prop. 1 scope, Prop. 2, Theorem, purified-distance remark).
2. The institution's Q1 definition (SJR or JCR, which category).
3. Whether STAT and G-Pruner release code (not found in this search). Needed only for a future multi-baseline benchmark.
4. Full texts not read: GRAIL, SoBP, Shapley Head Pruning, M-Wanda, Kurz et al., NAACL 2021 attention redundancy, OSSCAR.
5. Kaggle quota reset time (confirm at kaggle.com/settings).
6. ChatGPT round-3 answers T2 (proofs), T4 (literature sweep), T6 (G-Pruner numbers).

## 8. Blunt summary
Stop building selectors, repairs or allocation variants aimed at beating K-prune at 50% MACs: the headroom is too small, the
repair is already the baseline's strength, and every component idea is occupied in the literature. Stop opening new research
questions to find a gap. Finish V21 as frozen and the analysis paper through Rule 11 and the supervisor. The single experiment
that most efficiently decides whether a new-method paper is worth pursuing is the development-only high-compression kill test
(section 3.3, about 8-12 GPU-hours, after V21): unless an allocation that differs from K-prune beats it by at least 1 point on all
three development tasks where headroom is at least 3 points, close the method line for good.
