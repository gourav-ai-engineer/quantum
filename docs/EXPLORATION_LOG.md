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

**E6 | 2026-11-20 | ESM-2 pilot: written BEFORE any code, download or data (development-only kill test; not confirmatory).**
Prior art update: Vinod, Amini, Crawford, Yang, PLOS Comput. Biol. 22(2):e1013925 (2026), VERIFIED (full text read via fetch):
gradient-learned unstructured weight masks (Gumbel-softmax) on ESM-2 650M, <3% of weights, to suppress CATH structure categories;
interpretability framing; no physical compression, no latency, no ProteinGym. Distinct from retraining-free structured pruning for
efficiency; must be cited.
- Question: can retraining-free structured pruning (attention heads + FFN neurons, physically removed) with closed-form LS repair make a
  LARGER ESM-2 at least as good as the dense SMALLER ESM-2 of equal or lower encoder cost, on zero-shot variant-effect prediction?
- Models: facebook/esm2_t30_150M_UR50D (pruned) vs facebook/esm2_t12_35M_UR50D (dense reference). Encoder MACs per token (projections
  + FFN, attention-score term excluded): 35M/150M = 33.2M/147.5M = 0.225. Primary budget: keep <= 0.225 of 150M's encoder MACs.
  Descriptive extra budget: 0.5.
- Arms: (A) structured mask, no repair; (B) same mask + closed-form repair (drift-corrected refit ported to ESM-2's pre-LayerNorm
  blocks); (C) dense 35M; (D) dense 150M (reference); (E) 150M with INT8 dynamic quantization (efficiency baseline). Mask selector:
  fixed before data as GRP's refit-aware global greedy (existing code); a magnitude selector as a second, descriptive selector.
- Scoring: wild-type-marginal log-ratio for single substitutions (one forward pass per assay), ProteinGym v1 DMS substitutions.
- Assays (mechanical rule, fixed now): ProteinGym DMS substitution assays sorted by DMS_id, keep those whose wild-type length <= 400 and
  that contain single mutants; take the first 20. Single mutants only.
- Calibration: 1,024 unlabeled UniRef50 sequences, length 50-400, random sample (seed 7); any sequence whose UniProt ID matches an
  assay protein is dropped. Disclosed: no homology filtering.
- Primary statistic: per-assay Spearman rho; summaries are means over the 20 assays, paired by assay; 95% CIs by bootstrap over assays.
- Kill rules (both must pass to continue; otherwise the ESM-2 direction stops):
  K1 repair matters: at budget 0.225, mean rho(B) - mean rho(A) >= 0.05 and B > A on >= 14 of 20 assays.
  K2 practical value: at budget 0.225, mean rho(B) >= mean rho(C) with the bootstrap 95% CI lower bound of (B - C) > -0.01.
  If K2 fails, report "pruning 150M does not beat dense 35M" and stop.
- Efficiency (descriptive): CPU latency, batch 1 and 8, median of 30 runs after 5 warm-ups, for A/B shapes, C, D, E; parameter
  counts incl. and excl. embeddings.
- Expected outcome stated in advance: K2 is more likely to fail than pass (keeping 22.5% of MACs without retraining is severe).

**E6-A1 | 2026-11-20 | Amendment, written before any data download, forward pass or score. Fills details E6 left open; changes none of E6's models, budgets, arms, scoring, assay rule or K1/K2.**
- Calibration source (E6 allowed a named substitute): HF dataset `zpn/uniref50` (UniRef50 mirror; columns RepID, sequence; revision pinned in
  inputs.json; UniRef release unknown). Two-stage sampling with `random.Random(7)`: pick 8 of the shards, then 128 sequences per shard,
  after the length filter 50-400 and the exclusions below. This is not a uniform sample over UniRef50; it is a shard-level random sample.
- Exclusions: drop a sequence if its RepID equals the UniProt_ID of any of the 20 assay proteins (UniProt_ID from the official ProteinGym
  `reference_files/DMS_substitutions.csv`, pinned by commit and sha256), or if it is identical to an assay wild-type sequence. No homology filter (as E6).
- Assay rule applied mechanically on the parquet `target_seq`: length <= 400, at least one single mutant (no ':' in `mutant`), non-NaN `DMS_score`;
  sort by DMS_id lexicographically; first 20. Mutant position must index `target_seq` at the stated wild-type residue (asserted).
- Scoring: ESM tokenizer, sequence `<cls> wt <eos>`, batch 1, no padding; score = log p(mut) - log p(wt) at token index = mutant position (1-based,
  `<cls>` is index 0). Spearman between score and `DMS_score`, per assay.
- GRP selector for ESM (primary): the existing `grp_select` (refit-aware global greedy, ridge_rel 1e-4, max_frac 0.01) with sensitivity weights, as the
  BERT GRP default. MLM sensitivity: input is the unmasked sequence; for each non-pad token one label is sampled from the teacher's softmax,
  log-probabilities are summed over non-pad tokens, gradients taken w.r.t. each module output, s_m = E||grad||^2 per token. MAC unit costs
  exclude the attention-score term (flag `score_term=False`); per-head 4*s*d*hs, per-neuron 2*s*d. seq_len cancels in the budget ratio.
- Magnitude selector (descriptive): unit score = squared Frobenius norm of the unit's output-projection columns (head: its head_size columns of
  attention.output.dense; neuron: its column of output.dense) divided by the unit's MAC cost; remove globally in ascending order until the MAC budget is
  met; keep >= 1 FFN neuron per layer. Heads may be removed from a layer entirely (both selectors).
- Refit: `sequential_refit(drift_correct=True, ridge_rel=1e-4)`; the existing BERT hook path applies unchanged to ESM (EsmSelfOutput and EsmOutput take
  (hidden, residual) and add the residual with no LayerNorm inside; the student's dense inputs come from its own pre-LayerNorm).
- Arm A = physically pruned, no refit. Arm B = A + refit. Arm E = `torch.ao.quantization.quantize_dynamic` of every nn.Linear (including lm_head) of D, qint8.
- Efficiency (descriptive): CPU, `torch.set_num_threads(8)`, input length = median wild-type length of the 20 assays, batch 1 and 8, median of 30 runs after 5
  warm-ups, forward pass of the full model including lm_head; threads and length recorded.
- Statistics: percentile bootstrap (2.5 / 97.5) over assays, 10,000 resamples, seed 1234. K1 counts strict B > A. K2 lower bound is of mean(B - C) per resample.
- Engineering only: teacher Grams are stored on disk and memory-mapped (15.7 GB machine); no effect on numbers.

**E6-A2 | 2026-11-20 | Amendment, written before any scoring (prepare crashed; no result exists).** The `zpn/uniref50` repo has 108 parquet shards, not 109,
and shards are not length-homogeneous: of the first 8 drawn with `random.Random(7)` one has 0 and one has 40,263 sequences of length 50-400 (the rest
~498k). Rule: shards are taken in the order of `random.Random(7).sample(all_shards, len(all_shards))`; a shard is skipped if it has fewer than 100,000 eligible
sequences (post-filter, post-exclusion); the first 8 kept shards are used, 128 sequences each (same RNG). Skipped shards are recorded in inputs.json.
Calibration is therefore a shard-level sample whose length distribution is whatever those shards hold; no length rebalancing.

**E6-A3 | 2026-11-20 | Amendment, memory only, written before any pruned arm (A or B) was scored.** The first full run (commit c47bda36) scored the dense arms
D, C, E (mean rho 0.4054, 0.3107, 0.4016) and was then killed by the harness for low system memory (15.7 GB machine, ~5.5 GB free) during the teacher-statistics /
selection stage, before any selector output. Fix: `grp.collect_grams(..., store_dir=...)` accumulates each Gram in place into a disk-backed float64 memmap
instead of RAM. Nothing numeric changes: Grams stay float64; the calibration rows, their order and the batch size (8) are the same; additions happen in the same order
(first batch assigned, later batches added) so the Grams are bitwise identical to the in-RAM ones (unit test: max abs diff 0); ridge_rel 1e-4, the sensitivity
sampling (one label per non-pad token, generator seed 0) and every selector and refit setting are unchanged. Numbers are unaffected; only where the Grams live changes.
Resume: arms D, C, E are carried over from the killed run's results.json (not recomputed as results); their per-variant scores, which that run had not yet written, are
re-derived by re-scoring and checked against the carried rho (max abs diff recorded). Only arms A and B are newly run. The tracked-source dirty check now ignores `*.pyc`.

**E6-V | 2026-11-20 | VERDICT of the ESM-2 pilot: STOP (K2 fails). Development-only; not a paper result, not confirmatory.**
Source: `results/esm2_pilot/53c73f70/` (code commit 53c73f70, tracked source clean, `dirty_tracked: false`; 20 assays, 1,024 calibration sequences, 8 threads). Numbers copied from
`results.json`, `verdict.json`, `latency.json`. Arms D, C, E were carried over from the first run (c47bda36); their re-score differed by 0.0 (max abs). Percentile bootstrap
over assays, 10,000 resamples, seed 1234. Budget 0.225, selector GRP (primary), mean Spearman rho over 20 assays:
- A (pruned, no repair) -0.0416; B (pruned + drift-corrected refit) 0.1868; C (dense 35M) 0.3107; D (dense 150M) 0.4054; E (INT8 150M) 0.4016. Measured MAC ratio of A/B and C: 0.225.
- K1 (repair matters): B - A = +0.2284, 95% CI [0.1465, 0.3079], B > A on 16 of 20 assays. Threshold >= 0.05 and >= 14 of 20: PASS.
- K2 (practical value): B - C = -0.1240, 95% CI [-0.1967, -0.0519], B > C on 5 of 20. Needs mean(B) >= mean(C) and CI lower bound > -0.01: FAIL.
- Decision rule: CONTINUE only if both pass; K2 fails, so the ESM-2 direction STOPS. Pruning ESM-2 150M to the encoder MACs of dense 35M, retraining-free, does not beat dense 35M on this protocol.
- Descriptive: the repair is large and real (B - A positive at every setting: GRP 0.5 +0.0761 [0.0299, 0.1213], 17/20; magnitude 0.225 +0.2332 [0.1682, 0.2952], 18/20). GRP beats magnitude at
  0.225 after repair (+0.0545 [0.0077, 0.1125], 14/20). B at 0.225 is below dense 150M by 0.2186 [-0.2953, -0.1452]. INT8 150M vs dense 150M: -0.0038 [-0.0099, 0.0030]. At budget 0.5, B(GRP) = 0.1287, below
  C by 0.1820 [-0.2745, -0.0872] and, unexpectedly, below B(GRP) at 0.225 (0.1868); this non-monotonicity is unexplained and is not tuned or smoothed away here.
- Latency, CPU, 8 threads, 189 tokens, median of 30 after 5 warm-ups (ms, batch 1 / batch 8): dense 150M 501.0 / 2343.5; dense 35M 44.6 / 641.4; INT8 150M 341.5 / 2125.7;
  pruned 0.225 (A/B shape) 157.5 / 665.9; pruned 0.5 shape 245.0 / 1272.2. Parameters (total / excl. embeddings): dense 150M 148,796,794 / 148,775,674; dense 35M 33,993,874 / 33,978,034;
  pruned 0.225 34,417,754 / 34,396,634; pruned 0.5 75,005,434 / 74,984,314 (INT8 not counted). Equal-MAC pruned 150M is ~3.5x slower than dense 35M at batch 1 and about equal at batch 8.
- Incomplete (disclosed): the second run was again killed for low system memory during the refit of the last descriptive arm, so `B|magnitude|0.5` is missing, and `scores.json.gz`
  and `run_meta.json` were not written (per-variant scores were not saved). K1 and K2 use only arms present. Per the instruction not to change the design, no further rerun was made.
- Deviations: E6-A1/A2/A3 above (shard-level calibration sample, shard skipping, memory-only memmap change); sensitivity-weighted GRP as primary; Spearman via ranks (no scipy).
