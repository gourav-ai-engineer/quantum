# Prompt for an independent check of V21 (paste into ChatGPT with the attached files)

Attach these files from the repo, at tag `v21-freeze`:

1. `docs/V21_PREREG_DRAFT_2026-11-18.md` (the frozen pre-registration)
2. `scripts/v21_analysis.py`
3. `tests/test_v21_analysis.py`
4. `scripts/v21_run.py`
5. `scripts/v20_pilot.py`
6. `scripts/v20_finetune.py`
7. `src/qfc/metrics.py`
8. `scripts/v15_kaggle.sh`
9. `kaggle/v21_mbert_t43/v21_mbert_t43.ipynb`
10. `kaggle/v21_muril_t42/v21_muril_t42.ipynb`

---

You are an independent, skeptical reviewer of a pre-registered machine-learning experiment. Find errors before any result is read. Do not be encouraging. Do not suggest new methods, new variants or extra experiments. Do not guess results. No result exists yet, and none is given to you.

## Context

The study is about retraining-free structured pruning of BERT-architecture encoders: removing attention heads and FFN neurons at 50% of MACs.

- **Masks.** They are chosen on English data by the official K-prune code. Official Kwon et al. is a second baseline.
- **Repairs.** Our repairs (`d2i_r`, `d2i_ri`) refit the remaining weights on 2,048 unlabeled calibration rows.
- **Models and units.** Two models: MuRIL and mBERT. Three fine-tuned MNLI teachers per model (seeds 42, 43, 44). A unit is one (model, teacher). There are 6 units.
- **Calibration seeds.** 7, 42 and 77. Within a unit they vary the masks and the calibration rows.
- **Evaluation.** The XNLI-en test split and the IndicXNLI test split, 11 languages, 5,010 label-parallel rows each. Per-example predictions are stored.
- **Primary hypothesis H21-1.** English-selected masks without repair lose more accuracy on the target languages than on English.
  - Statistic: S = mean_L r(L) − r(en), where r = unpruned accuracy − pruned accuracy.
  - Decision rule: pooled 95% CI lower bound > 0.02, point estimate ≥ 0.05, and S > 0 in all 3 teachers of the model.
- **Secondary hypotheses H21-2 to H21-4.** Holm-corrected per model.
- **Statistics.** Paired cluster bootstrap over (calibration seed × example) within each unit. The pooled CI is the average of independent unit draws, so teacher variance is NOT in the CI. That gap is why the all-teachers sign rule exists.

## Checks

For each check, answer PASS, FAIL or UNCLEAR, and give the file and line.

1. **Pre-registration vs code.** Does `v21_analysis.py` compute exactly each statistic and decision rule in section 3 of the pre-registration? Check signs, coefficients, margins (0.015, 0.02, 0.05), one-sided versus two-sided p, which hypotheses enter Holm, and the intersection-union max-p for H21-4(i). List every mismatch, however small.
2. **Bootstrap validity.**
   - Is the (seed × example) multinomial resample a valid paired cluster bootstrap for a linear combination of accuracies?
   - Does one shared resample per draw keep the pairing across conditions and languages?
   - Is averaging unit draws a correct CI for the mean of unit statistics?
   - Is `bootstrap_p_le` a correct one-sided bootstrap p-value?
3. **Leakage.**
   - Can any test-split data reach mask selection, calibration, repair, the teacher gate or any choice?
   - Check `--allow-test`, the calibration source (MNLI and IndicXNLI train), the index files, and the order of operations in the kernels.
4. **Pairing integrity.**
   - Are rows identical across languages, conditions and both official environments?
   - Does the code verify this: labels, `parallel_evals`, the label check on the official dumps?
   - Could rows be silently misaligned, truncated or reordered?
5. **Integrity gate.** Does `gate_pass` in `v21_run.py` implement every item in section 5 of the pre-registration? Is there a failure that would pass silently? Check MAC ratio rounding, kept units, the teacher accuracy tolerance of 0.002, and the rule of at most 2 differing official-teacher rows.
6. **Baseline fairness.**
   - Are K-prune and Kwon run unmodified, with the same calibration rows and the same budget?
   - Could the local tokenizer directory, the add-on's extra-set evaluation, or the c = 0.5 versus "remove 1 − c" conversion weaken either baseline?
7. **Decision-rule gaps.** Which outcomes does the text not decide? Examples:
   - a teacher fails its gate;
   - Kwon is missing for one unit but not others;
   - ties, or S exactly 0 in a teacher;
   - a kernel times out after part of the work;
   - the N-versus-D robustness sign flip.

   For each gap, state what the code does.
8. **Tests.** Do the synthetic tests in `tests/test_v21_analysis.py` actually exercise the decision rules (sign rule, Holm, missing Kwon, label mismatch)? Or could the analysis be wrong while they still pass?
9. **Kernel scripts.**
   - Any shell bug in the two notebooks or the `bl` stage that would lose results, mix units, or print test accuracies to stdout? Look at `set -e` interactions, `|| echo`, paths, env vars, and the relative `MODEL_NAME=v21tok_<model>`.
   - Any way a unit could silently produce files for the wrong teacher?
10. **Claims.** Does the do-not-claim list (section 6) match what the design can support? Is anything claimed that the design cannot back? Consider: only 3 teachers, teacher variance outside the CI, one budget, English-only selection, MNLI-only teachers.

## Output format

- A table with columns: check #, verdict, file:line, a concrete failure scenario (inputs or state, then the wrong output), and severity (critical / major / minor).
- After the table, a short list of things you verified as correct.
- Then any rule-12 concern. Rule 12: the method is frozen, so a fix is allowed only for an engineering bug and only before any result is seen. Say which findings are engineering bugs and which would be design changes.
- Quote code exactly. If a file is missing from what you were given, say "not provided" instead of assuming.
