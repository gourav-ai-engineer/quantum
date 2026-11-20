# Exploration log (mission: docs/CLI_MISSION_BEAT.md; started 2026-10-27, branch research/beat-explore)

Every idea tried is listed here, failures included, so the final paper can disclose all of them.
Entry format: date | idea | code commit | numbers (results path) | outcome.

## Fixed rules (written 2026-10-27 BEFORE any exploration data; do not edit later, add dated amendments below)

**Definition of "beat" (from the mission, unchanged).** Retraining-free pruning of BERT-base (heads + FFN neurons),
same checkpoints, same 2048 calibration rows, same MAC budget, official Kwon et al. and K-prune code as baselines.
Primary metric: validation accuracy at 50% MACs against the better of the two baselines, confirmatory tasks QNLI,
MNLI-m, QQP, paired cluster bootstrap (seed x example, 10,000 resamples), Holm over the 3 tasks, supported if Holm
p < 0.05 and CI lower bound > 0 on >= 2 of 3 tasks with no Holm-significant loss (rule as H17b, PROJECT_STATE
"Pre-registration V17"). Latency of physically pruned models on GPU is reported (rule 8); no speed claim without it.

**Exploration tasks:** MRPC, SST-2, RTE, CoLA, STS-B (dev data, validation accuracy allowed). Confirmatory tasks
(QNLI, MNLI-m, QQP validation splits) are not loaded for anything before a frozen "Pre-registration V18".

**Per-task metric on exploration tasks.** MRPC, SST-2, RTE: validation accuracy and held-out-train logit KL.
CoLA: validation MCC (accuracy secondary); KL as for classification. STS-B (regression, one output): validation
Pearson/Spearman and held-out-train MSE to the teacher output; KL is undefined and not used.

**"Beats both" on one exploration task.** Mean over the same 5 calibration seeds (7, 42, 77, 123, 2024) is strictly
better than BOTH baseline means (Kwon, K-prune), at both 0.5 and 0.4 MACs, on held-out-train KL (MSE for STS-B)
AND dev validation metric. Freeze rule: this on >= 3 of the 5 exploration tasks. RTE (277 val rows) and MRPC (408)
are small: differences of one to two points there are within noise, so a freeze needs the effect to hold on the larger
tasks too (SST-2 has 872 rows, CoLA 1043). A task for which an official baseline cannot be run unmodified does not
count toward the 3/5 (record the reason here before running).

**Eval parity.** All methods on a task are scored through one pipeline. Baseline validation logits come from the
official-code runs (V15 add-on dumps, archive v15_results.tgz); our methods are scored by step22 --mode accuracy.
Before any comparison: unpruned accuracy of the same checkpoint must agree between the two pipelines.

**Novelty.** Every idea family gets a 2024-2026 search before investment; a published idea may be a component or
baseline, not a claim.

## Entries
**E0 | 2026-10-27 | reference points, code 15a6009 (kernel beat-explore-a)** GRP-R and d2i-R (K-prune masks + drift-corrected
refit, the V17 mechanism) scored on validation accuracy for MRPC/SST-2 at c in {0.5, 0.4}, 5 seeds, through
`scripts/beat_run.py`. Bar: official Kwon/K-prune accuracy from `results/beat_explore/baselines_v15.json` (K-prune c0.5:
MRPC .8623, SST-2 .9149; c0.4: .8637, .8986; Kwon c0.5: .8534, .8938; c0.4: .8020, .8603; teachers .8775/.9243).
Eval parity verified (max logit diff 1.2e-5, `scripts/beat_eval_parity.py`).
**Result (kernel A, Tesla T4, head_mask check effective; JSONs in `results/beat_explore/E0_grp_d2i/6aa8e04/`, 40 files):**
accuracy, mean of 5 seeds (K-prune / GRP-R / d2i-R): MRPC c0.5 .8623/.8544/.8627, c0.4 .8637/.8490/.8637;
SST-2 c0.5 .9149/.9174/.9151, c0.4 .8986/.9135/.8972. Held-out KL ratio vs K-prune: GRP-R MRPC 1.19/1.29, SST-2 0.31/0.50;
d2i-R 0.97-1.00 everywhere. Seed wins of GRP-R vs K-prune on accuracy: MRPC 0/5, 0/5; SST-2 3/5, 5/5.
Verdict: GRP-R beats both baselines on SST-2 (acc and KL, both c) and loses on MRPC; d2i-R equals K-prune. The SST-2
KL win does carry into accuracy (+0.0025 at c0.5, +0.0149 at c0.4). Nothing frozen. The kernel cloned the branch head at
push time; the results folder name is the local HEAD at download.

**E1 | 2026-10-27 | input-side alignment ("full-chain" closed-form reconstruction), code `src/qfc/align.py`, methods
`d2i_ri`, `grp_ri`.** Hypothesis (mechanism, not numbers): K-prune's/GRP-R's gain comes from cancelling residual drift in
the OUTPUT projections only; kept units' own input weights (q/k/v rows of kept heads, FFN up rows of kept neurons) still
read drifted inputs. Ridge-LS refit of those rows to the teacher's unit features (q/k/v of the kept heads; pre-activations
of the kept neurons), given student inputs, layer by layer, then the drift-corrected output refit. Unit tests:
`tests/test_align.py` (no-align == drift refit; unpruned student recovered; empty-head layer finite). Novelty search
(2026-10-27, web): no BERT head+FFN retraining-free paper found that refits input-side weights; closest are CORP (ViT,
affine compensation, arXiv 2602.05243), Olica (LLM, SVD calibration), LIAR; layer-wise sequential reconstruction of
every linear layer is standard in GPTQ/OBC-style post-training compression, so this is a combination, not a new
principle. K-prune is already sublayer-wise iterative (knowledge measurement -> mask search -> weight tuning per
sublayer, bottom-up), so "drift-aware iterative selection" is K-prune and is NOT pursued.
**Result (kernel B, Tesla T4, head_mask effective; `results/beat_explore/E1_align/e0063df/`, 40 files):** accuracy mean of
5 seeds, c0.5 / c0.4 (K-prune in brackets). d2i-RI: MRPC .8652/.8647 (.8623/.8637), SST-2 .9158/.9032 (.9149/.8986);
held-out KL ratio vs K-prune MRPC .93/.92, SST-2 .99/.95. GRP-RI: MRPC .8539/.8569, SST-2 .9206/.9112, KL ratio MRPC
.98/1.08, SST-2 .32/.45. Versus E0 (no alignment): d2i-R -> d2i-RI +.0025/+.0010 MRPC, +.0007/+.0060 SST-2 acc; GRP-R ->
GRP-RI KL improves on MRPC (1.19->.98, 1.29->1.08), acc +.0 / +.008 MRPC, +.003 / -.002 SST-2.
Rule check (strict mean win over both baselines, acc and KL, both c): d2i-RI TRUE on MRPC and SST-2; GRP-RI TRUE on
SST-2 only. Caveat: the d2i-RI MRPC margin (+.0029/+.0010 acc, 2/5 and 3/5 seed wins) is far inside the noise of a
408-row dev set; it satisfies the written rule but is not evidence of a real gain by itself. STS-B (third counting
task) pending kernel C (official baselines), then d2i_ri/grp_ri on STS-B.

**C | 2026-10-27 | official baselines on STS-B (kernel C v2, commit 036e44c; v1 died at 17 s on a relative `$0` self-call after
`cd`, fixed in 036e44c).** Unmodified Kwon and K-prune, pinned checkpoint, our calibration rows, c in {0.5, 0.4}, 5 seeds.
Dumps and masks: `results/beat_explore/C_stsb_baselines/036e44c/`; bar in `results/beat_explore/baselines_v15.json` (acc
field = Pearson, heldout_kl field = MSE for stsb). Pearson mean over 5 seeds (teacher .8805): K-prune c0.5 .8804, c0.4
.8783; Kwon c0.5 .8636, c0.4 .8475. K-prune at c0.5 already equals the teacher, so a strict "better than K-prune on Pearson
at c0.5" is a bar of about +0.0001 above the teacher; the MSE half of the rule is the informative one there. The rule
is NOT changed (fixed before data); if STS-B cannot be beaten at c0.5 the freeze does not happen, that is the result.
Parity (CPU, `scripts/beat_heldout_parity.py`, seed 7): our held-out rows train.shuffle(7)[2048:2560] reproduce the official
teacher_heldout dumps (max |dlogit| MRPC 2.5e-5, SST-2 5.0e-6, STS-B 4.8e-6, labels equal); STS-B validation (1500 rows)
max |dlogit| 1.0e-5, labels equal. So the KL/MSE half of the rule compares like with like.

## Amendment 1 (2026-10-27, before any exploration accuracy data): which exploration tasks count
Official Kwon et al. (806ac5a6) and K-prune (9ca629a1) accept only MRPC, SST-2, STS-B, QNLI, MNLI, QQP (+SQuAD) as
`--task_name` (their READMEs). They do NOT support CoLA or RTE, so no unmodified official baseline exists there. Rule:
only MRPC, SST-2, STS-B count toward the freeze rule, and the freeze rule becomes "beats both on all three" (the 3-of-5
rule cannot be met otherwise). RTE and CoLA may still be run with our own pipeline as a secondary, non-counting signal
(against our own K-prune-mask re-implementation is impossible without K-prune masks, so only versus the unpruned
teacher and versus GRP-R/random). STS-B uses held-out-train MSE to the teacher output and validation Pearson/Spearman.
Official baselines are run unmodified except the V15 add-on (calibration-index and logit-dump patch; no change to the
pruning algorithms). STS-B checkpoint pin: textattack/bert-base-uncased-STS-B (revision recorded in step22 SPECS).

**E2 | 2026-10-27 | STS-B with our methods (kernel D, code ed9f196; `results/beat_explore/D_stsb/ed9f196/`, 40 JSONs).**
Pearson mean of 5 seeds, c0.5 / c0.4 (K-prune .8804/.8783, Kwon .8636/.8475, teacher .8805). d2i-RI .8809/.8789
(seed wins vs K-prune 5/5, 4/5); held-out MSE ratio vs K-prune .90/.87. d2i-R .8804/.8786 (MSE ratio 1.00/.99).
GRP-R .8773/.8748 and GRP-RI .8782/.8763 lose (MSE ratio 1.15/1.17 and 1.02/1.05). Rule check: d2i-RI TRUE on STS-B
(all four cells); d2i-R also TRUE but only by ~1e-4 (margin noise); GRP-R/GRP-RI FALSE.
**Freeze rule (Amendment 1: beat both on MRPC, SST-2 and STS-B at 0.5 and 0.4): d2i-RI is the only method meeting it
on all three tasks.** Caveats carried into the pre-registration: the MRPC margin (+.0029/+.0010 acc) and the Pearson
margins (+.0005/+.0006) are tiny; the clear effect is the held-out distance (KL/MSE -5..-13% vs K-prune). d2i-RI uses
K-prune's masks, so its pruned shapes and latency equal K-prune's.

**E3 | 2026-11-16 | portfolio pre-check on EXISTING dev JSONs (no new runs; `scripts/beat_portfolio_check.py`).** Idea: per
(task, seed, c) pick d2i-RI or GRP-RI by lower held-out-TRAIN distance (KL; MSE for STS-B), the only candidate with a
built-in "never worse than K-prune+repair on the calibration objective" property. Validation metric, mean of 5 seeds,
delta vs official K-prune: MRPC c0.5 -0.0010 (seed W/L 2/3), c0.4 -0.0010 (3/2); SST-2 +0.0057 (4/1), +0.0126 (5/0);
STS-B +0.0005 (5/0), +0.0006 (4/1). On MRPC the held-out-KL pick was the accuracy-worse mask in 2/5 seeds at both budgets
(512 held-out rows do not rank these masks reliably). **Verdict: fails** — one real win (SST-2, where GRP selection already
wins) plus a loss and a noise tie; same shape as V18 H18a. Not pursued; no pre-registration written.
Headroom context (teacher minus K-prune accuracy at 0.5 MACs, mean of 5 seeds): MRPC 1.52, SST-2 0.94, STS-B 0.01 pp
(dev, from baselines_v15.json); QNLI 1.27, MNLI-m 2.43, QQP 1.56 pp (from the V18 dumps, descriptive, after the verdict).
At 0.5 MACs any retraining-free method can gain at most ~0-2.4 pp over K-prune, so "beats K-prune everywhere" at this
budget is bounded by headroom, not by method quality.

**E4 | 2026-11-20 | method-direction screen (literature + one arithmetic kill check; no run, no new data).** Seven directions screened for
a new retraining-free structured-pruning method on fine-tuned BERT encoders:
1. Guaranteed / risk-controlled choice of the pruning budget: occupied (Pareto Testing, arXiv 2210.07913, applies Learn-then-Test to
   Transformer head and token pruning; "Confident magnitude-based neural network pruning", arXiv 2408.04759).
2. Training-free depth pruning with a closed-form linear block replacement: occupied for decoder LLMs (ReplaceMe, arXiv 2505.02819);
   dropping layers of fine-tuned BERT without retraining: Sajjad et al. 2020 (arXiv 2004.03844).
3. Latency-aware hybrid depth/width one-shot pruning of BERT: occupied (ZipLM, arXiv 2302.04089, §4.3 one-shot BERT-base, §3.1 removes
   whole attention/FFN modules, §3.2 latency tables).
4-6. Selection-vs-compensation, allocation search, multi-layer repair: occupied (see `docs/STRATEGY_AUDIT_2026-11-19.md` §4-5).
7. Optimism-corrected (GCV / leave-one-sentence-out) refit-aware removal cost, motivated by V14's calibration overfitting (overfitting
   itself is published: arXiv 2410.17711). **Killed by an arithmetic check before any code:** ridge optimism grows as features /
   effective calibration tokens; with 2048 rows, SST-2 (25.16 tokens/example, `AVG_SEQ_LEN`) has about half MRPC's (53.24) tokens, so
   more optimism, yet GRP wins on SST-2 and loses on MRPC. Optimism cannot explain the failure the correction targets.
Conclusion: in this exact setting (retraining-free structured pruning of fine-tuned BERT-family encoders), novelty, not compute, is the
binding constraint. A method paper would need a different setting; that choice is the owner's. No GPU used; V21 t44 keeps queue priority.

**E5 | 2026-11-20 | domain screen for a different setting (literature only; no run).** Protein LMs (ESM-2), genomic LMs, time-series
foundation models, speech SSL and Whisper screened for retraining-free structured head/FFN pruning with closed-form repair.
- Crowded: time-series FMs (structured pruning + fine-tune across seven models, NeurIPS 2025; data-free stable-rank head pruning,
  arXiv 2602.01605); Whisper / speech (arXiv 2609.27980, 2509.26207, data-free clustering 2606.11836; Interspeech 2026 per external
  check); genomic LMs (layer pruning with fine-tuning, Genes 2025, doi 10.3390/genes16111358).
- Least crowded: protein LMs (ESM-2). Known efficiency work: 8/4-bit quantization (bioRxiv 2024.10.22.619563), ESME efficiency study
  (PMC12481099, per external check), co-distillation for accuracy (Nature Methods 2026). External check (ChatGPT) reports a close item:
  Vinod et al. 2026, PLOS Comput. Biol., learned sparse subnetworks across ESM-2 layers (UNVERIFIED here: not found in my search; open
  its link before relying on it). Gap = "not found in targeted searches", not proven.
- Also flagged by the external check and VERIFIED here at search level: AutoDFP (arXiv 2403.08204; Neurocomputing version), data-free
  structured pruning with channel-similarity reconstruction on CNNs. Relevant to the analysis paper's positioning in Neurocomputing.
- Corrections to my earlier message: the "1-2 day CPU pilot" estimate was not supported (ProteinGym scoring cost depends on the assay
  subset and implementation), and the "95% Spearman retention" kill threshold was not derived from evidence. Any ESM-2 pilot needs its
  assay set, comparisons (unpruned, pruning without repair, pruning with repair, plus existing efficiency baselines such as quantization)
  and thresholds fixed before results, with measured CPU latency.
- Joint conclusion (Claude + external check): the fastest route to a submission is the existing BERT analysis paper (Rule 11, supervisor
  proofs, missing raw baseline records, AutoDFP in positioning); ESM-2 is a separate, slower study. Q1 acceptance cannot be guaranteed.
