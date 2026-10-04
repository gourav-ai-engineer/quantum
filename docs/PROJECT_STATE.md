# PROJECT_STATE: living state of the QFC project

Update this file after every experiment. CLAUDE.md holds the stable rules; this file holds
what has been done, what is pending, and what to do next. Never put a number here that was
not produced by a run; link the JSON path and commit instead.

## Current status
2026-10-04. `main` contains the V12 baseline fixes (PRs #16/#17) and the V11 objective-alignment
audit with bootstrap CIs (PRs #15/#18). pytest: 47 passed at commit 777e51a (CI-pinned env and
local env). Tiny smoke runs (calibration 16, evaluation 32) of step12/13/14/18/19 completed and
showed every method keeping exactly k heads per layer; those are engineering checks, not results.
**No V5, V6, V10 or V11 experiment has been re-run since the V12 fixes, and no result JSON is
committed to the repo.** All earlier Shannon numbers and all V10 Random numbers are invalid.
The scientific question (does fidelity coverage predict downstream usefulness of a head subset,
and do QFC/IWQFC compete with corrected baselines?) is therefore open.

## Script and version map
| Version | Script | What it tests |
|---|---|---|
| V4 | step12_qfc_stability.py | calibration-seed stability, SST-2 |
| V5 | step13_mrpc_stability.py | calibration-seed stability, MRPC |
| V6 | step14_budget_response.py | budget response (k = 3, 6, 9), SST-2 + MRPC |
| V7 | step15_layer_adaptive_smoke.py | layer-adaptive budget allocation |
| V8 | step16_representation_ablation.py | fidelity vs Hilbert-Schmidt vs cosine |
| V9 | step17_input_conditioned.py | input-conditioned fidelity coverage |
| V10 | step18_v10_confirmatory.py (+ aggregate_v10_confirmatory.py) | confirmatory, full validation |
| V11 | step19_objective_alignment.py | does coverage correlate with local loss (diagnostic) |

Steps 2-7 and 11 are early smoke/stability scripts. step3/4/5/7/15 still use a single Random mask
and keep-high-only entropy; do not cite them.

## Results ledger
All numbers must be copied from the produced JSON. Columns: experiment | script | config |
commit | key numbers | verdict.

| Experiment | Script | Config | Commit | Key numbers | Verdict |
|---|---|---|---|---|---|
| V5 MRPC stability | step13 | PENDING | PENDING | PENDING | **PENDING rerun after V12.** Old numbers INVALID/superseded: Shannon scores were NaN under padding (arbitrary head order); Random was a single mask |
| V6 budget response | step14 | PENDING | PENDING | PENDING | **PENDING rerun after V12.** Old numbers INVALID/superseded: same Shannon bug; Random single mask |
| V10 confirmatory | step18 | PENDING | PENDING | PENDING | **PENDING rerun after V12.** Old numbers INVALID/superseded: the Random row kept all heads (equal to the unpruned model) and the Shannon row was NaN-driven; entropy keep-high only |
| V11 objective alignment | step19 | PENDING | PENDING | PENDING | **PENDING.** The V11 workflow still carries the earlier low-power flags (64 eval examples, 24 subsets); any run of that configuration is superseded. Not affected by the V12 bugs |
| V8 representation ablation | step16 | not re-run | n/a | not recorded in repo | Not affected by the V12 bugs (no Random/Shannon baseline in the script). The owner-reported conclusion (fidelity did not beat cosine/HS) has no committed JSON; re-confirm before citing |
| V9 input-conditioned | step17 | not re-run | n/a | not recorded in repo | Not affected by the V12 bugs. Superseded by V10 for the confirmatory question |

## Pre-registered decision rule for V11 (commit this BEFORE running V11)
Higher coverage should go with LOWER loss, so Spearman(coverage, loss) < 0. Per task (SST-2, MRPC),
per layer, with a bootstrap 95% CI (`tasks.<task>.layers.<l>.spearman_coverage_vs_loss_ci.<fidelity|hilbert_schmidt|cosine>`
in step19's summary.json; `ci_high < 0` means "entirely below 0"; a `degenerate: true` entry counts as
not supported):
- **"Alignment supported"** if the fidelity CI is entirely below 0 in >= 8 of 12 layers on BOTH
  tasks, AND fidelity is not clearly worse than Hilbert-Schmidt and cosine (their CIs are no better
  than fidelity's).
- **"Not supported"** if fidelity CIs include 0 in most layers, or a classical similarity aligns
  clearly better.
- Anything between: report as **inconclusive**; do not interpret as support.

This rule may be changed only before seeing V11 data, with a dated note. Pooled-across-layers rho
is secondary: it mixes between-layer and within-layer variation.

Dated notes:
- 2026-10-04 (added by the assistant when writing this file; OWNER TO CONFIRM BEFORE V11 RUNS):
  "no better than fidelity's" is proposed to be operationalised as: the number of layers (per task)
  whose Hilbert-Schmidt (resp. cosine) CI is entirely below 0 is not greater than fidelity's. If the
  owner prefers a different operationalisation, edit here, dated, before the run.
- 2026-10-04: the V11 workflow was merged with a low-power configuration before this rule was
  written. If output of that configuration was inspected, this rule is not strictly pre-registered
  with respect to that sample; the rerun (300 subsets, >=256 examples, fresh bootstrap) is the one
  that counts.

## Decision tree after V10/V6/V11 reruns
A. Alignment supported AND QFC/IWQFC competitive with corrected baselines (judge with paired
   bootstrap on the full validation set, Random as a distribution): write a method paper; add
   RoBERTa-base and one more GLUE task, HIES as a baseline, physical pruning cost table,
   post-pruning fine-tuning.
B. Alignment not supported, or classical similarity matches fidelity: write the honest
   negative-result/analysis paper ("does quantum-state geometry of attention predict head
   redundancy?") with the corrected baselines and the objective audit.
C. Never: resurrect QIS claims, hide negative results, or tune until QFC wins.

## Next steps (ordered checklist)
- [ ] Run the pending experiments on a GPU with `bash scripts/run_gpu.sh --commit <hash> --results-root <dir>`
      (or colab/run_experiments.ipynb). It pins the environment, runs the pre-flight (versions, CUDA,
      head_mask effective) and executes the four items below in order. Results land in
      `results/<experiment>/<commit>/`; generate ledger rows with
      `python scripts/make_ledger_row.py <dir>`. Do not run step19 before the V11 decision rule is merged.
      (The runner and notebook have only been smoke-tested on CPU, never on a GPU.)
- [ ] V10: step18 on GPU: tasks sst2+mrpc, seeds 7,42,77, full validation, 30 random masks,
      heads-to-keep 6. Aggregation (scripts/aggregate_v10_confirmatory.py <dir>) is run by the runner.
- [ ] Run step14 (V6) on GPU: budgets 3,6,9, full validation, both tasks.
- [ ] Re-run step13 (V5) for MRPC stability with the fixed Shannon.
- [ ] Commit the V11 decision rule (this file; done once the docs PR is merged), then run step19 on
      GPU with 300 subsets and >=256 evaluation examples (full validation if feasible).
- [ ] Update the Results ledger and apply the decision tree; write the verdict.
- [ ] Upgrade steps 3,4,5,7,15 or delete them (they still use a single Random mask / keep-high
      only); do not cite them.
- [ ] If branch A: add HIES baseline, RoBERTa-base, physical-pruning FLOPs/params/latency,
      post-pruning fine-tuning, one more GLUE task.
- [ ] Pre-submission: independent code review, regenerate every table from JSON by script, no
      hand-typed numbers.

Repo hygiene items found while writing this file (added by the assistant, not by the owner):
- [x] (done in infra/gpu-runner: requirements-ci.txt, pinned pyproject extras, tests/test_env_pins.py)
      Make the environment reproducible from the repo: a pinned requirements file or tightened
      `pyproject.toml` extras (currently `transformers>=4.45` and `datasets>=2.20` are unpinned, so
      `pip install -e ".[dev,transformers]"` can install transformers 5.x, which ignores head_mask).
- [~] (runner done: scripts/run_gpu.sh + colab/run_experiments.ipynb, untested on a GPU; the GitHub
      workflows are still CPU-only) Add a GPU path for the V10/V6/V11 workflows (all 22 `runs-on` entries across the workflows are
      CPU `ubuntu-latest`; step18's matrix has `timeout-minutes: 60`, which full-validation runs with
      30 random masks are unlikely to meet on CPU; not measured).
- [~] (CLAUDE_WORKFLOW.md done; experiment_protocol.md still open) Reconcile docs/experiment_protocol.md and docs/CLAUDE_WORKFLOW.md with CLAUDE.md (see
      "Known inconsistencies").
- [x] (results/README.md + scripts/make_ledger_row.py) Add a results-recording convention (per-experiment summary with environment and commit), since
      no result JSON is currently in git.

## Known limitations / open questions
- The mean density state averages position-aligned token-by-token matrices across different
  sentences; its meaning is questionable (a hypothesis to test, e.g. by comparing against per-sample
  conditional states).
- Coverage selects representatives, not important heads.
- Only BERT-base encoders on 2 GLUE tasks so far.
- The V11 pooled Spearman mixes between- and within-layer variation.
- step19 perturbs one layer at a time (local mask); local alignment need not transfer to global
  pruning of all layers at once.
- Selections are stored sorted (since V12); masks are unaffected.

## Known inconsistencies in the repo (as of 2026-10-04)
- docs/experiment_protocol.md says calibrate on the SST-2 validation set (final: all 872 examples)
  and describes physical pruning; CLAUDE.md rule 4 requires calibration from train. The older
  scripts step2/3/4/6 calibrate and evaluate on disjoint slices of the validation split.
- RESOLVED in infra/gpu-runner: docs/CLAUDE_WORKFLOW.md told a new session to `pip install -e ".[dev,transformers]"` in `.venv`
  (can install transformers 5.x) and to record work in `docs/research_log.md`, which does not exist.
- README.md "Verified so far" quotes 32-example smoke numbers (QFC 0.96875 vs unpruned 0.9375). They
  are engineering evidence only and came from pre-V12 code; do not cite them.
- Standing rule 1 (print/assert heads kept per layer) is implemented in step18 (prints and asserts);
  steps 12/13/14 assert via validate_all but do not print; older scripts do neither.
- Standing rules 2 and 3 (Random >=30 masks; entropy both directions) are implemented only in steps
  12, 13, 14, 18.
- The history in CLAUDE.md (QIS Spearman/p-value, "V4-V10 showed...") is owner-reported; neither the
  old notebook nor any result JSON is in this repo.

- Legacy scripts (steps 2-7, 11, 15-17) now carry status banners saying which standing rules they
  satisfy and that they must not be cited (tests/test_script_banners.py keeps them in place).

## Paper outline (fill only with produced evidence)
Method, theory (objective-level only), corrected baselines, objective-alignment audit, budget
curves, ablations (fidelity vs HS vs cosine), limitations.
