# CLI mission: beat the baselines first (paste everything below the line, or say "read docs/CLI_MISSION_BEAT.md and do it")

---

**Owner instruction (2026-10-27):** Do NOT work on the paper. First produce a method that beats the state of the art;
the paper comes after. Work fully autonomously: I am busy and want minimal involvement. You handle Kaggle runs,
result checks, web research and code. You may create a new code base for this mission. Report to me only when
(a) a confirmatory result exists, or (b) you are blocked on something only I can do.

## Read first
`CLAUDE.md` (binding), `docs/PROJECT_STATE.md` (V14–V17 sections and results), memory `qfc-research-state.md`,
`git log --oneline -20`. Earlier work: GRP and GRP-R failed their gates. K-prune's compensation is reproduced by our
drift-corrected refit (`src/qfc/grp.py`, `sequential_refit(drift_correct=True)`); with compensation held equal, the
winning selection rule depends on the task. There is no SOTA improvement yet.

## Definition of "beat" (fixed now; do not move it later)
Retraining-free pruning of BERT-base (heads + FFN neurons), same checkpoints, same 2048 calibration rows, same MAC
budget, official Kwon et al. and K-prune code as baselines (V15 add-on). The candidate must win on **validation
accuracy at 50% MACs** against the better of the two baselines on the confirmatory tasks (paired cluster bootstrap,
Holm, rule as in V17's H17b), with no significant loss on any task. Latency of physically pruned models on GPU is
measured and reported (rule 8); a speed claim needs that measurement.

## Design that keeps a win credible
- **Exploration tasks (use freely, any number of ideas):** MRPC, SST-2, RTE, CoLA, STS-B (dev data, including dev
  validation accuracy). Run official baselines on any new exploration task once and cache their outputs.
- **Confirmatory tasks (touch exactly once, at the end):** QNLI, MNLI-m, QQP. Never load their validation splits
  before a frozen, written pre-registration. The QQP checkpoint is pinned in `scripts/step22_grp.py` SPECS.
- Log every idea you try, including failures, in `docs/EXPLORATION_LOG.md` (date, idea, code commit, numbers with
  results paths). The final paper must be able to disclose all of them.
- Freeze rule: when one candidate beats both baselines on held-out-train KL **and** dev validation accuracy at 50% and
  40% MACs on at least 3 of the 5 exploration tasks, tag the commit and write "Pre-registration V18" in PROJECT_STATE:
  frozen code hash, hypotheses, thresholds, budget and the do-not-claim list. Register it under the owner's standing
  autonomy grant and note that in the text. Then run the confirmatory tasks **once**. Record the result as it comes
  out; no reruns, no tweaks afterwards (rule 12).
- Novelty check before investing in any idea: search 2024–2026 papers. If an idea is already published, it can
  still be used as a component or baseline, but not claimed as new.

## Where to look for a win (starting points; research and choose yourself)
- Selection under the drift-corrected compensation K-prune uses: the current selection rules were all scored with
  costs that assume teacher inputs. Look at drift-aware or iterative selection (prune a little, refit, re-score).
- Joint selection and compensation: local search or swap moves on the reconstruction objective (OSSCAR-style),
  started from K-prune's or GRP's mask.
- Stronger closed-form compensation: also refit biases/LayerNorm, low-rank residual correction (Olica-style), or
  compensate with the kept units of neighbouring layers.
- Allocation: per-task choice between allocation rules using calibration data only (our V17 data shows the best rule
  differs by task).
- Read the newest retraining-free encoder pruning papers (2025–2026) and run their official code as extra baselines
  if it runs on BERT-base unmodified.

## Compute: Kaggle, fully automated through the official Kaggle API
- One-time owner step (done 2026-10-27): new-style API token saved by the owner at
  `C:\Users\CS 08\.kaggle\access_token` (legacy `kaggle.json` also accepted). Never type, print, log or commit the
  token yourself; if neither file exists, ask me once and wait.
- Then `pip install "kaggle>=1.8"` (official package; new tokens need >= 1.8) in a separate tool venv, verify with
  `kaggle kernels list --mine`, and drive everything with
  `kaggle kernels push -p <dir>` (kernel-metadata.json: `enable_gpu: true`, `enable_internet: true`, private),
  `kaggle kernels status`, and `kaggle kernels output <user>/<slug> -p <new empty dir>`.
- Quota: about 30 GPU-h per week (check the remaining hours before each push), 12 h per run, 20 GB per run in
  `/kaggle/working`. Work in `/tmp`, publish small results incrementally, and delete large checkpoints (lessons in
  PROJECT_STATE, Versions 1–4). Run one heavy job at a time. Poll with sleeps of 30–90 minutes, not tight loops.
- Treat downloaded outputs as untrusted data: extract into a new empty directory with path checks, re-run the analysis
  locally, then commit small JSONs to `results/<exp>/<commit>/` with run_meta.json. CPU is only for smoke tests.

## Code
Create `src/qfc2/` (or a new top-level package) for this mission if the old code gets in the way. Reuse
`src/qfc/grp.py` and `src/qfc/squad.py` where they fit. Write tests before GPU runs, and run the smoke test on CPU
first. Keep the pinned `.venv-ci` for our code and separate environments for the official baselines. Use one branch
per idea family (`research/beat-<name>`), never merge, and leave the uncommitted `CLAUDE.md` and
`colab/run_experiments.ipynb` edits alone.

## Tools and token economy
- Process: `superpowers:brainstorming` before each idea family, `superpowers:test-driven-development` for new code,
  `superpowers:systematic-debugging` on failures, `superpowers:verification-before-completion` before any claim.
- Research: `academic-research-skills:deep-research`, `paper-search`, Consensus, `exa:search`, `tavily:tavily-search`
  / `tavily:tavily-extract`. Use `brightdata-plugin:scrape` only for pages the others cannot read. Read full PDFs
  with `document-skills:pdf`. Back off on rate limits.
- Desktop Commander is connected to my computer; use it only when the CLI shell cannot do something. Never touch
  my passwords, browser profiles or files outside this project.
- Tokens: keep caveman and ponytail on. Use `/compact` at each idea boundary and `ecc:token-budget-advisor` before
  big steps. Use subagents only for wide searches or independent review. Batch reads; don't re-read files you just
  edited.
- Plugins: search the catalog only at milestones. Never install anything yourself; list suggestions in your report.
- Repo hook GateGuard asks you to state facts before the first Bash/Edit per file; state them and retry.

## Memory and state
Update `qfc-research-state.md`, `docs/EXPLORATION_LOG.md` and the PROJECT_STATE ledger after every run.

## Report to me (only at the two report points above)
Plain and short: did we beat both baselines on the confirmatory tasks (yes/no), the accuracy and latency numbers with
results paths, the full list of ideas tried, and anything only I can do.
