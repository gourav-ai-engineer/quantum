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

## Amendment 1 (2026-10-27, before any exploration accuracy data): which exploration tasks count
Official Kwon et al. (806ac5a6) and K-prune (9ca629a1) accept only MRPC, SST-2, STS-B, QNLI, MNLI, QQP (+SQuAD) as
`--task_name` (their READMEs). They do NOT support CoLA or RTE, so no unmodified official baseline exists there. Rule:
only MRPC, SST-2, STS-B count toward the freeze rule, and the freeze rule becomes "beats both on all three" (the 3-of-5
rule cannot be met otherwise). RTE and CoLA may still be run with our own pipeline as a secondary, non-counting signal
(against our own K-prune-mask re-implementation is impossible without K-prune masks, so only versus the unpruned
teacher and versus GRP-R/random). STS-B uses held-out-train MSE to the teacher output and validation Pearson/Spearman.
Official baselines are run unmodified except the V15 add-on (calibration-index and logit-dump patch; no change to the
pruning algorithms). STS-B checkpoint pin: textattack/bert-base-uncased-STS-B (revision recorded in step22 SPECS).
