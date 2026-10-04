# Project: Attention-head pruning with Quantum-Fidelity Coverage (QFC)

Owner: Gourav, M.Tech, NIT Silchar (supervisor: Dr. Aparajita Dutta).
Target: a conference paper first (mid-tier IEEE), a journal extension later.
The paper must report only what this repo actually produces.
Scientific correctness matters more than a favorable number.

Living state, results ledger and next steps live in **docs/PROJECT_STATE.md**. Read it first.

## Goal
Test whether representing each attention head as a density operator
rho = A A^T / Tr(A A^T) and selecting a subset of heads by fidelity coverage
(C(S) = sum_i max_{j in S} F(rho_i, rho_j), F = squared Uhlmann-Jozsa fidelity)
gives a useful way to prune BERT heads, compared with proper baselines.
Weighted variant (IWQFC): nonnegative Michel gate-sensitivity weights, fixed before greedy
selection, normalised per layer. No tuned mixing coefficients.

## Why we are here (history)
- The original paper used a hand-weighted score QIS = 0.5*VNE + 0.3*QJSD - 0.2*F.
  It was dropped: the weights are arbitrary, and the notebook's own check showed no
  correlation with head importance (Spearman about -0.05, p=0.54). Its "multi-seed"
  results were identical across seeds and it used ttest_ind labeled as paired.
  Do NOT revive QIS or cite its numbers. (Owner-reported; the notebook is not in this repo.)
- Replacement: QFC, a subset-selection objective. Proven (docs/theory.md): monotone
  submodular, so greedy has a (1-1/e) guarantee for the OBJECTIVE only. Never claim this
  guarantees accuracy.
- V4-V10 showed QFC was not consistently better than VNE, Michel or Random, and fidelity
  did not beat cosine/Hilbert-Schmidt (V8). Input-conditioned QFC did not hold up on full
  validation (V10). (Owner's reading of CI artifacts; no result JSON is committed.)
- V12 (PRs #16/#17, merged) fixed two baseline bugs: the V10 "Random" kept ALL heads (equal
  to the unpruned model), and the Shannon baseline was NaN for every head under padding
  (so it always kept heads 0..k-1). All earlier Shannon numbers and all V10 Random numbers
  are INVALID. QFC/IWQFC/Michel/VNE(keep-high) rows are not invalidated by these bugs, but
  must be re-read against corrected baselines.

## Prior work we must position against (novelty is thin; be honest)
HIES (Choi et al., arXiv 2510.13832: gradient importance + attention entropy), CAHP
(arXiv 2606.19150: graph/complementary head selection), Differentiable Subset Pruning
(Li et al., TACL 2021: head pruning as subset selection), BHPVAS, AMAP, Michel et al.
NeurIPS 2019. "Pruning as subset selection" and "combine information measures" are NOT
novel. Do not claim "quantum advantage"; all computation is classical. The quantum
formalism is a representation only. Do not call zeroed projections structural compression.
(Citations are as supplied by the owner; verify each before it goes in the paper.)

## Environment (critical)
- Use `.venv-ci` (Python 3.12, torch 2.6.0, transformers 4.51.3, datasets 3.6.0,
  pyarrow 24.0.0). transformers 5.x silently IGNORES head_mask: results from it are
  meaningless. Never run experiments in any other environment.
- `.venv-ci` is untracked and local. Recreate it with:
  `py -3.12 -m venv .venv-ci`, then in it `pip install torch==2.6.0 transformers==4.51.3
  "datasets>=2.20,<4" "accelerate>=0.34" "numpy>=1.26,<3" pyarrow==24.0.0 pytest` and
  `pip install -e . --no-deps`. (On the owner's Windows machine, newer pyarrow was blocked
  by Smart App Control; 24.0.0 loads. Do not disable Windows security features.)
- CPU is too slow for full runs. Run V10/V6/V11 on GPU (Colab or a GPU runner); CPU is for
  smoke tests only. The existing GitHub workflows use CPU `ubuntu-latest` + Python 3.11.
- Pinned models: SST-2 `textattack/bert-base-uncased-SST-2` rev 205ffbd1...;
  MRPC `textattack/bert-base-uncased-MRPC` rev ddeddf4a... (full hashes in scripts/SPECS).
- On Windows use Git Bash/PowerShell as available; GateGuard-style hooks may ask for facts
  before edits; state them and retry.

## Standing rules (never break)
1. Every selector must keep exactly k distinct heads per layer
   (`qfc.baselines.validate_selection`). Print/assert heads kept per layer in every experiment.
2. Random is always a distribution (>=30 masks), never a single mask.
3. Entropy baselines (VNE, Shannon) are reported keep-high AND keep-low.
4. Calibration data comes from train; evaluation uses the held-out validation split (full
   split for any paper number). Never choose heads on eval data.
5. Every results JSON includes the unpruned baseline (accuracy, loss, F1 for MRPC) and the
   evaluation size.
6. Statistics: paired bootstrap / permutation on the same examples; never an independent
   t-test; never present identical re-runs as independent seeds. Report mean and std over
   calibration seeds; fixed evaluation data when testing calibration stability.
7. Do not add new QFC variants to chase a win. A new variant needs a written hypothesis in
   docs/PROJECT_STATE.md BEFORE it is implemented, plus math definition, unit tests, a
   reproducible experiment and appropriate baselines.
8. Do not claim speedup from masking. Report parameter count, attention FLOPs and measured
   latency separately, from physically pruned models, on GPU. Distinguish functional
   masking from structural pruning.
9. Smoke-test outputs (tiny sizes) are engineering checks, never results.
10. Workflow: one branch + PR per experiment (`research/qfc-vN-...`), tests green
    (`pytest` in .venv-ci, plus `python -m compileall -q src scripts`), do not merge without
    the user's go-ahead. Never work directly on main; `git status`, `git branch --show-current`,
    `git pull --ff-only` before changing code. A green CI is not a scientific verdict.
11. Before anything is called "paper-ready", an independent check of the code and numbers is
    required (checklist in docs/PROJECT_STATE.md).
12. Never fabricate, infer or smooth results; never change a method after seeing test results
    unless it is an explicit ablation or new experiment; report negative results honestly;
    baselines are implemented per their published definitions, not weakened approximations.

## Repository map
- `src/qfc/`: states, fidelity, coverage (greedy / weighted), conditional (per-sample),
  baselines (shared selectors, Random distribution, validation), alignment (Spearman +
  bootstrap CI), hf_experiments (model/data loading, masks, Michel importance), metrics.
- `scripts/stepN_*.py`: experiment entry points (V4=step12, V5=step13, V6=step14, V7=step15,
  V8=step16, V9=step17, V10=step18 + aggregate_v10_confirmatory.py, V11=step19).
  Steps 2-7 and 11 are early smoke/stability scripts; do not cite them (see PROJECT_STATE).
- `tests/`: unit and regression tests (47 passed at commit 777e51a).
- `docs/`: theory.md (claims boundary), methodology.md, experiment_protocol.md (partly
  outdated, see PROJECT_STATE), PROJECT_STATE.md (living state).
- `.github/workflows/`: one workflow per experiment.
Do not put one-off notebook logic into the core library.

## Working style
Skeptical research engineer and reviewer: inspect code and git state first, state the exact
scientific question, make the smallest change, add tests, run the smallest smoke test, inspect,
and only then propose the larger run. Look for leakage, bugs and unfair baselines. Reproduce
before trusting. When evidence is insufficient, stop and report instead of guessing. Do not run
several large experiments at once. Do not call a heuristic "theoretically proven" without
checking the proof.

## How to continue autonomously
Open docs/PROJECT_STATE.md. Do the first unchecked item in "Next steps". After each run,
record the outcome in the Results ledger (exact numbers from the JSON, file path, commit
hash, environment) and update the status and next steps. If the decision rule applies, apply
it exactly as written and record the verdict.
