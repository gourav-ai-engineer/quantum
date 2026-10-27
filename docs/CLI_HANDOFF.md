# Prompt for the Claude Code CLI (paste everything below the line)

---

**Owner decision (answered by this prompt):** Phase 1 = write the analysis paper now. Phase 2 (a new method study,
V18) = DRAFT ONLY, then stop and wait for my explicit "confirm V18" in chat. No standing-grant registration this time.

## Who you are and the goal
You are continuing my M.Tech research on retraining-free structured pruning of BERT (attention heads + FFN neurons).
Goal: a peer-reviewed journal paper (Q1 if the evidence earns it) and a PhD-grade thesis chapter. Scientific
correctness beats a favourable number. CLAUDE.md in this repo is binding; read it fully.

## Read first (do not re-derive anything that is written down)
1. `CLAUDE.md` (rules), then `docs/PROJECT_STATE.md`: the sections "Pre-registration V15", "V16" and "V17" with their
   result records. These are the source of truth.
2. `paper/main.tex` and `paper/refs.bib`: the current draft. It already contains the V14 table, the H10 table (V15), the
   V16 transplant text and the selection × compensation table (V17).
3. Memory index (MEMORY.md) and `qfc-research-state.md`.
4. `git status`, `git branch --show-current`, `git log --oneline -15`. Branches `research/qfc-v15-grp`, `qfc-v16-diag`
   and `qfc-v17-drift` are stacked and unmerged. Never merge, and never commit the uncommitted edits to `CLAUDE.md` or
   `colab/run_experiments.ipynb` without asking me.

## Where we honestly are (verify against the files, never against this summary)
- V14: after least-squares refit, the choice of head selector is statistically indistinguishable; compensation dominates.
- V15 (GRP): both official baselines reproduced (K-prune, Kwon et al. via amendment A1). Calibration gate H10 FAILED
  (wins on SST-2, loses on MRPC). So no accuracy comparison was run.
- V16: transplant decomposition; branch A-3 (no transplant beats K-prune).
- V17 (GRP-R, drift-corrected refit): the mechanism was CONFIRMED (our refit reproduces K-prune's compensation) but the
  gate FAILED. With compensation held equal, which selection rule wins depends on the task.
- **There is NO accuracy or speed improvement over the state of the art.** Never claim one. Latency was never measured.
- V17 was registered under my standing grant after V15/V16 data had been seen. The paper must disclose that.

## Hard rules this prompt might tempt you to break
- MRPC and SST-2 are burned development tasks (used in V15–V17). A new method may not be tuned or confirmed on them.
- QNLI, MNLI-m and QQP validation splits are the only clean confirmatory data left. Do not load them for anything until
  a confirmed pre-registration says so. (The QQP checkpoint is already pinned in step22 SPECS; its label order was
  checked on train only.)
- Every number in the paper comes from a JSON under `results/` with its path in a LaTeX comment. Never type, infer or
  smooth numbers.
- Nothing is "paper-ready" before rule 11: an independent check of code and numbers, plus my supervisor
  (Dr. Aparajita Dutta) checking the three proofs. You prepare it; you never sign it off.
- No new method variant without a written, dated pre-registration in PROJECT_STATE before any code (rule 7), and
  my explicit confirmation in chat.

## Phase 1 — analysis paper (no GPU needed). Do this now.
Working title is in main.tex. Target: TMLR first. Check each journal's current SJR quartile (scimagojr.com) before
naming it to me as Q1/Q2; do not quote tiers from memory. Steps, one commit per step on a new branch
`paper/analysis-v1` (branched from `research/qfc-v17-drift`):
1. Novelty and related-work refresh. Search 2025–2026 work on retraining-free / post-training structured pruning of
   encoders, layer-wise sparsity allocation, and compensation/reconstruction. The known close papers are recorded in the
   PROJECT_STATE V16 "Novelty check" (arXiv 2606.15161, 2605.25508, DP-Prune). Verify every new citation exists (DOI or
   arXiv id resolves) before adding it to refs.bib.
2. Restructure the story around the evidence: (i) attention similarity is the wrong redundancy signal, (ii) after
   refit, selection barely matters at uniform sparsity, (iii) under global allocation, compensation explains the
   baselines' advantage exactly (the V17 mechanism), and the remaining selection effect depends on the task. Put the
   pre-registration and negative-results framing up front.
3. Figures from committed JSONs only: a V15 H10 KL plot and the V16/V17 factorial plot. Use the `dataviz` skill
   before writing plotting code. Render with the existing scripts' style and save under `paper/figs/`.
4. Fill every `\todo{}`: abstract last, contributions, limitations (one model family, KL on held-out train as the
   gate metric, no latency, MRPC has 408 validation examples), and a reproducibility appendix.
5. Prepare rule 11: write `docs/RULE11_CHECKLIST.md` (each number → JSON path → script; each proof → the step a
   reviewer must check), and run an independent read-only review with a subagent. Report its findings; do not
   self-approve.
6. Compile the PDF (local LaTeX, or tell me what to install), then give me a 10-line summary and the open items that
   only I or my supervisor can close.

## Phase 2 — draft V18 only if a principled idea survives the literature
- Brainstorm with `superpowers:brainstorming`, then do a novelty check (searches below) on each candidate. Keep only
  an idea that (a) follows from a mechanism, not from MRPC/SST-2 numbers, (b) is not already published, and (c) can
  be tested on QNLI/MNLI-m/QQP with official Kwon and K-prune baselines on the same checkpoints and calibration rows.
- Write the draft as a dated "Pre-registration V18 (DRAFT)" section in PROJECT_STATE: hypothesis, math, unit tests,
  gates with numeric thresholds, budget, venue by outcome, and a do-not-claim list. Then STOP and ask me. If no idea
  survives, say so; that is an acceptable result.

## Compute (later phases only)
Kaggle T4, about 30 GPU-h per week. Driver: `scripts/v15_kaggle.sh <stage> [work_dir] [out_dir]`. Notebooks are imported
from GitHub (`https://www.kaggle.com/kernels/welcome?src=<github blob url>`). The CLI may not have a browser, so either
I launch runs in the Kaggle UI, or I install the kaggle CLI and place my own API token. You never enter credentials.
Download outputs, extract with path checks into a new empty directory, re-run the analysis locally, then commit to
`results/<exp>/<commit>/` with run_meta.json.

## Tools: use these and keep token use low
- Process: `superpowers:using-superpowers` at the start; `superpowers:brainstorming` before new ideas;
  `superpowers:writing-plans` / `superpowers:executing-plans` for Phase 1; `superpowers:verification-before-completion`
  before saying anything is done; `superpowers:requesting-code-review` for code changes.
- Writing and research: `academic-research-skills:academic-paper` (writing, revision, citation check, abstract),
  `academic-research-skills:academic-paper-reviewer` (a simulated referee before I see a draft),
  `academic-research-skills:deep-research` / `anthropic-skills:deep-research` for literature, `paper-search`,
  the Consensus search tool, `zetesis` for claim checks, and `document-skills:pdf` for reading PDFs.
- Online search and scraping (in this order, cheapest first): WebSearch/WebFetch → `exa:search` / `tavily:tavily-search`
  / `tavily:tavily-extract` → `brightdata-plugin:scrape` only for pages the others cannot read. arXiv/Semantic Scholar
  APIs rate-limit; back off instead of hammering them.
- Token economy: caveman and ponytail are already active through hooks; keep them. Use `ecc:token-budget-advisor`
  / `ecc:context-budget` when a step will be large, and `/compact` at every phase boundary. Use subagents only for
  independent reviews or wide searches, not for work you can do inline. Batch file reads and don't re-read files
  you just edited. Space out polling of long jobs (1–3 h sleeps).
- Memory: update `qfc-research-state.md` and the PROJECT_STATE ledger after every milestone.
- Plugin hygiene: search the plugin catalog only at phase boundaries (keywords: latex, citation, arxiv, token usage).
  Suggest at most two plugins and never install anything yourself; I review and install.
- Repo hooks: GateGuard asks you to state facts before the first Bash/Edit per file. State them briefly and retry;
  do not try to disable it.

## Reporting to me
Short and in plain terms: what was done, the numbers with file paths, what failed, and what needs my decision. Return
when Phase 1 is complete (PDF plus the rule-11 checklist), when a V18 draft is ready for my confirmation, or when you
are blocked.
