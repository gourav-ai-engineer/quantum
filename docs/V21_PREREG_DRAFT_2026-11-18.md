# V21 pre-registration (DRAFT): confirmatory multilingual analysis of retraining-free structured pruning

**Status: DRAFT, written 2026-11-18, before any V21 code or run. The owner asked for this draft ("draft V21", 2026-11-18). It is NOT frozen.**

The draft is frozen only when all of the following hold:

1. The owner confirms or edits the text before any run.
2. The engineering gates E1–E5 below pass. They read no test split, except E3, which reads labels only.
3. A commit copies the frozen text into `docs/PROJECT_STATE.md` and is tagged `v21-freeze`.

Until then, any part may change, because no V21 number exists. After the freeze, rule 12 applies.

**Branch:** `research/qfc-v21-multilingual-confirm`.

## 1. Why, and what this is not

- **V20 background.** The V20 pilot (validation only, one MuRIL teacher, NO-GO for H20b) produced descriptive observations, and its pre-registration says those numbers are never paper results. Two observations motivate V21:
  - An English-selected K-prune mask collapsed the target languages far more than English. Mask-only accuracy was about 0.41 on as, bn and hi, against 0.685 on en.
  - After drift-corrected repair, the residual losses were small but uneven across languages.
- **What V21 tests.** Whether these observations hold on data and teachers the pilot never touched:
  - the reserved XNLI/IndicXNLI **test** splits;
  - two more MuRIL fine-tunes;
  - a second multilingual encoder, mBERT.
- **What V21 is for.** It is the confirmatory evidence the analysis paper (branch `paper/analysis-v2`) needs before it can include a multilingual section.
- **What V21 is not.**
  - Not a new method.
  - Not a rescue of H20b, which stays dropped.
  - Not target-language mask selection (Direction A), which is out of scope. The reasons are the confound and the ceiling in `docs/RESEARCH_DIRECTION_AUDIT_2026-11-18.md` §10.
- **Disclosure.** The hypotheses below were formed after seeing the V20 validation data, and the paper must say so. They are tested only on data and teachers that V20 never used.

## 2. Fixed design

### Models

| Model | Checkpoint | Pin |
|---|---|---|
| MuRIL | `google/muril-base-cased` | rev `afd9f36c7923d54e97903922ff1b260d091d202f` (as in V20) |
| mBERT | `bert-base-multilingual-cased` | revision pinned at E2, before any run |

IndicBERTv2 and XLM-R are out of scope. IndicBERTv2's architecture and baseline compatibility are unverified. XLM-R is not BERT architecture, and the official codes are untested on it.

### Teachers

- **Recipe.** Every teacher is fine-tuned with the V20 recipe unchanged (`scripts/v20_finetune.py`):
  - data: English GLUE MNLI train, 2 epochs;
  - optimizer: AdamW, lr 2e-5, weight decay 0.01, 10% warmup;
  - batch 32, max_len 128, fp16.
- **Seeds.** Three fine-tuning seeds per model: 42, 43, 44.
  - The MuRIL seed-42 teacher is the V20 teacher (sha256 `e19f3b74…`), reused and disclosed.
  - The other five teachers are new.
- **Teacher gate (validation only).** Each teacher must pass the V20 gate before use: MNLI-m ≥ 0.78 and IndicXNLI-as ≥ 0.55.
  - For mBERT the second condition is IndicXNLI-hi ≥ 0.55 instead. mBERT has very little Assamese pretraining data, and this choice is made now.
  - A teacher that fails the gate is reported. No teacher is replaced after its test numbers are seen.

### Selection

- Official K-prune and official Kwon et al., with the same patched data/index add-on as V15–V20.
- English calibration: 2,048 rows, the content-matched rows of V20 amendment 3.
- Budget: c = 0.5 of MACs.
- Calibration seeds: 7, 42, 77.
- Masks are always selected on English. Target-language selection is out of scope.

### Conditions per (model, teacher, seed)

| # | Condition | What it is |
|---|---|---|
| 1 | mask-only | K-prune masks, no weight change |
| 2 | K-prune (official) | K-prune with its own weight tuning |
| 3 | Kwon et al. (official) | Kwon et al.'s full method |
| 4 | d2i_r/en | drift-corrected refit, English calibration |
| 5 | d2i_r/mixed | drift-corrected refit, mixed calibration (round-robin en/as/bn/hi, 512 rows each, as in V20) |
| 6 | d2i_ri/en | the V18 method, frozen at `v18-freeze-fix1` |

- Repairs never read labels.
- The method code is frozen at the V20 pilot tag `v20-pilot-freeze-fix2` and at `v18-freeze-fix1`. There is no tuning.

### Evaluation

- **Splits.** XNLI-en **test** and IndicXNLI **test** in all 11 Indic languages: as, bn, gu, hi, kn, ml, mr, or, pa, ta, te. No part of the project has read these splits before.
- **Size.** The rows are n-way parallel. The expected size is 5,010 per language, verified at E3.
- **Storage.** Per-example predictions are stored.
- **Validation splits.** None are used for any V21 statistic.

### Language groups (fixed now)

| Group | Languages | Status |
|---|---|---|
| D | as, bn, hi | the V20 development languages |
| N | gu, kn, ml, mr, or, pa, ta, te | never analysed; their validation predictions exist in the V20 files but were never summarised |
| T | D ∪ N | all 11 Indic languages |

## 3. Hypotheses and decision rules

### Notation and statistics

- **Accuracies and residual loss.** U(L) is the unpruned teacher's accuracy on language L, and X(L) is condition X's accuracy. The residual loss is r_X(L) = U(L) − X(L). All quantities are on a 0–1 scale, so 1 point = 0.01.
- **Units.** A unit is one (model, teacher). There are 6 units: 2 models × 3 teachers.
- **Bootstrap.** Paired cluster bootstrap over (calibration seed × example) within each unit, 10,000 resamples, seed 1234 + unit index.
- **Pairing.** Examples are paired across languages and conditions by example id.
- **Pooled statistic.** The mean over units and languages. Its CI comes from resampling seed × example within every unit independently, then averaging, so teacher variance is not inside the CI.
- **Teacher variance.** With only 3 teachers, a teacher-level bootstrap would be degenerate. Instead, every confirmatory claim also requires the **sign to agree in all 3 teachers of a model**.
- **Per model.** Claims are made separately for MuRIL and mBERT. There is no pooling across models.

### H21-1 (primary): selection-side disparity

Under English-selected K-prune masks with no repair, target languages lose more than English.

- **Statistic.** S = mean_{L∈T}[r_M(L)] − r_M(en), where M is the mask-only condition.
- **Supported for a model if all three hold:**
  - the pooled 95% CI lower bound is > 0.02;
  - the point estimate is ≥ 0.05;
  - S > 0 in all 3 teachers of that model.
- **Overall verdict.** H21-1 is supported overall if it is supported for MuRIL. mBERT is reported either way.

### H21-2 (secondary): repair does not equalise languages

After English-calibrated d2i_r repair, the pooled target-language residual loss still exceeds the English one.

- **Statistic.** R = mean_{L∈T}[r_{d2i_r/en}(L)] − r_{d2i_r/en}(en).
- **Supported for a model if both hold:**
  - the 95% CI lower bound is > 0;
  - R > 0 in all 3 teachers.

### H21-3 (secondary): mixed calibration helps targets at a bounded English cost

Both parts must hold:

- **(i) Target gain.** G = mean_{L∈T}[d2i_r/mixed(L) − d2i_r/en(L)] has a 95% CI lower bound > 0, and G > 0 in all 3 teachers.
- **(ii) English non-inferiority.** The difference d2i_r/mixed(en) − d2i_r/en(en) has a 95% CI lower bound > −0.015, and the one-sided p for H0: δ ≤ −0.015 is < 0.05.

### H21-4 (secondary): the V18/V19 picture holds across languages

- **(i) Non-inferiority to K-prune.** d2i-RI is non-inferior to official K-prune with margin 0.015, both on en and on the pooled T mean. This is the same test as H19b.
- **(ii) Superiority over Kwon et al.** d2i-RI beats official Kwon on the pooled T mean, with a CI lower bound > 0.
- **Conditions.** H21-4 is tested only if gate E1 passes. If Kwon cannot run unmodified on a model, (ii) is reported as "not run" for that model. Kwon is never weakened (rule 12).

### Multiplicity and robustness

- **Holm correction.** Per model, over the secondary family: H21-2, H21-3(i), H21-3(ii), H21-4(i), H21-4(ii).
- **Robustness check (reported, not decisive).** Every pooled statistic is recomputed on group N only. If a supported hypothesis has the opposite sign on N to its sign on D, the paper reports it as a limitation.

### Descriptive outputs (rule 8)

- Per-language tables for all 6 conditions.
- Worst-language residual loss.
- CPU latency of the physically pruned shapes: FP32, seq 128, median of 30 runs after 5 warm-up runs (`scripts/v15_latency.py`).
- Parameter counts **including and excluding embeddings**. Embeddings are 64% of MuRIL's parameters and 52% of mBERT's, so the whole-model compression is small, and the paper must say so.

## 4. Engineering gates (before the freeze; no test split is read)

### E1: Kwon loads the multilingual checkpoints unchanged

- **Fix.** Pass both official codes a local directory with the pinned config, tokenizer and weights for each model, instead of a hub id.
  - Under transformers 4.12, the hub id `google/muril-base-cased` failed with an invalid relative-URL redirect.
  - The legacy `bert-base-multilingual-cased` id also redirects (to `google-bert/...`).
- **Tokenizer files.** If transformers 4.12 cannot parse a `tokenizer.json` written by 4.51.3, the directory ships only `vocab.txt`, `tokenizer_config.json` and `special_tokens_map.json`. The tokenization of a fixed sentence set must then be identical under both environments.
- **Parity check (decisive).** A same-kernel A/B on BERT-base MRPC, seed 7, c 0.5, for both official codes: hub-id load versus local-directory load, on the same GPU and in the same environment.
  - Decisive: the token ids of the local directory equal those of our env's pinned hub tokenizer, in both official environments, for BERT-base, MuRIL and mBERT; and the A/B gives identical head/neuron keep patterns and identical validation predictions.
  - Reported, not decisive: raw mask values, and a hub-versus-hub rerun that measures GPU run-to-run determinism.
  - Also reported: K-prune's MuRIL masks under the hub id versus the local directory, and versus the V20 seed-7 masks.
  - Implementation: stage `v21e1` of `scripts/v15_kaggle.sh`; report `scripts/v21_e1_report.py`; kernel `kaggle/v21_e1`.
- **Cross-environment comparison (reported, not decisive).** The masks are also compared against the committed V18-era output. That run used a different environment (the libcublas fix, a different torch), so a mismatch there is not interpretable as a loading bug.
- **Multilingual load.** One Kwon job runs on the V20 MuRIL teacher, seed 7, c 0.5, to show that it loads and produces masks with MAC ratio 0.500.
- **No algorithm change.** Kwon's algorithm is not touched. If parity fails, H21-4(ii) is "not run".

### E2: model compatibility

- Pin the mBERT revision.
- Confirm that both official codes accept each model's vocabulary size.
- Confirm that MAC accounting is identical to V20's: embeddings are excluded from the MAC budget, as before.

### E3: test-split integrity (labels only)

- **Loader.** The V20 loader, extended to the test split behind an explicit `--allow-test` flag. Only the frozen V21 kernels use the flag.
- **Checks, before any prediction is made:**
  - row count per language;
  - label equality of every language with XNLI-en test;
  - for any language with extra rows, the V20 rule: keep the first label-parallel rows and report the count. (Tamil validation had 3,238 rows.)
- **Scope.** E3 reads labels only. No model is run on the test split before the freeze.

### E4: smoke test

- CPU `--smoke --allow-cpu` on tiny validation slices.
- It covers E5's prediction path, the extended fine-tune script and the analysis script.
- This is an engineering check only, not a result (rule 9).

### E5: official tuned students predict on the multilingual sets

Conditions 2 (K-prune official) and 3 (Kwon official) need each official tuned student's per-example predictions on XNLI-en and IndicXNLI.

- **The gap.** Today the add-on dumps predictions only on the task's own dev set (`v15_student_validation.pt`), and V20 ran with `BL_SKIP_CKPT=1`, so no tuned student was saved.
- **Route (chosen 2026-11-18).** The add-on predicts on extra evaluation sets inside the official environment.
  - `V21_EXTRA_EVAL` names a JSON `{name: {premise, hypothesis, label}}`. `v15_dump` evaluates every set through the same model and mask path as validation and writes `v15_<tag>_x_<name>.pt`.
  - With the variable unset, the add-on is unchanged (`tests/test_v15_analysis.py`).
  - The alternative (save students, evaluate in our env) was rejected: Kwon's tuned real-valued masks would have to be re-applied outside the official code.
- **Parity gate.** The chosen route's predictions on MNLI-m validation must equal the existing add-on dump row for row.
- **Scope.** E5 is smoke-tested on validation slices only. No test split is read.

### Gate status (2026-11-18)

- **E2: PASS.** `results/v21_gates/ee832e5a/e2_e3.json` (script `scripts/v21_gates.py`, commit ee832e5a).
  - mBERT is pinned at `google-bert/bert-base-multilingual-cased` rev `3f076fdb1ab68d5b2880cb87a0886f315b8146f8`. The legacy id `bert-base-multilingual-cased` resolves to it.
  - Both pins equal the current hub heads.
  - MuRIL and mBERT have identical dimensions (hidden 768, 12 layers, 12 heads, intermediate 3072, model type bert), so the MAC accounting is identical.
  - Vocabularies: MuRIL 197,285, mBERT 119,547. Whether the official codes accept them is shown by E1's runs.
- **E3: PASS.** Same file. XNLI-en test and IndicXNLI forward/test have 5,010 rows in every one of the 12 languages, no extra rows, and labels equal to XNLI-en everywhere. Only labels, counts and file sha256 were read.
- **E1, E5: running.** Kaggle kernel `gouravcseai/v21-e1` (GPU), code d2d2447f.
- **E4: PASS (engineering only, rule 9).** CPU `--smoke --allow-cpu`, validation, 64 rows per language, seed 7.
  - Inputs: the mBERT random-head checkpoint from `v21_e1_prep.py`, a synthetic 50%-MAC mask npz, and planted official dumps with random logits.
  - `scripts/v21_run.py` ran all four of our conditions, assembled the K-prune/Kwon files and wrote `integrity.json`. `scripts/v21_analysis.py` (bd279c74) ran on that output.
  - Kept physical shapes equal the masks, and the MAC ratio is 0.500.
  - The integrity gate FAILED as intended on the planted faults: 46–47 of 64 teacher rows differed, and the teacher accuracy did not match a fake ft_meta value.
  - No number from the smoke run is a result.

## 5. Compute and execution

- **Kernels.** Six, one per (model, teacher), each under the 12-hour cap.
- **Cost.** The MuRIL fine-tune measured 5,482 s in V20. The estimated total is 20–30 T4 GPU-hours, about 1–1.5 weeks of the 30 GPU-h/week quota.
- **Environment.** torch 2.6.0 and transformers 4.51.3, both asserted. The official codes run in their own pinned environments, as before.
- **Analysis.**
  - Results are published per (unit, seed).
  - The analysis script is committed before any kernel runs.
  - It is run exactly once, after every unit passes the integrity gate.
- **Integrity gate (per unit).**
  - labels and evaluation size are equal across pipelines;
  - unpruned validation accuracy equals the teacher-gate value;
  - the MAC ratio of every mask is 0.500;
  - d2i-RI's kept units equal K-prune's masks.
- **Engineering failures.** Fixed with a dated note and the tag `v21-freeze-fixN`. Only the affected items are rerun. Nothing changes after any statistic has been seen.

## 6. Do-not-claim list

- No claim that pruning causes the disparity in general. The claims cover only English-selected K-prune masks at 50% MACs, MNLI-trained teachers, and these two models.
- No method claim, no SOTA claim, no quantum claim.
- No claim about target-language mask selection, which is not tested.
- No pooling across models. No claim for IndicBERTv2, XLM-R or decoders.
- No speed claim beyond the measured CPU numbers. Whole-model size reduction is reported including embeddings.
- No claim that V20's validation numbers are results. V21 replaces them.

## 7. Owner decisions

Recorded 2026-11-18. The owner's words: "mBERT, 3 seeds, go for pre-freeze checks, use Kaggle CPU".

1. **Second model: mBERT.** IndicBERTv2 is out of scope.
2. **Teachers: three fine-tuning seeds per model** (42, 43, 44).
3. **GPU go for the pre-freeze gates only.** This covers E1–E5, including small GPU kernels where a gate needs them.
   - The 6 confirmatory kernels and the 5 new teacher fine-tunes are NOT yet approved.
   - They need a separate go after the freeze.
4. **"Use Kaggle CPU"** answers the latency question, not a V21 item. The V19/H18b CPU latency is rerun on a Kaggle CPU kernel with shuffled rounds.
