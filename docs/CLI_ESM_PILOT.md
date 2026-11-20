# CLI task: run the ESM-2 pilot (E6) exactly as pre-registered

Paste into Claude CLI: "Read docs/CLI_ESM_PILOT.md and do it."

Read first: `CLAUDE.md`, `docs/EXPLORATION_LOG.md` entry **E6** (the design, fixed 2026-11-20 before any code or data; commit 21eb8687).
Do not change anything E6 fixes (models, budgets, arms, scoring, assay rule, calibration rule, kill rules K1/K2). A deviation found
necessary before any result is seen is written as a dated amendment under E6 first; nothing changes after a result is seen.

## Scope
Development-only pilot. CPU is enough (wild-type-marginal scoring is one forward pass per assay). Do not use Kaggle GPU (the weekly
quota is reserved for V21 muril_t44 / mbert_t44). Do not touch V21 files or read V21 condition results. Do not edit the paper.
New branch from the current one: `research/esm2-pilot`. `.venv-ci` only (transformers 4.51.3 has `EsmForMaskedLM`).

## Steps
1. **Data, recorded with hashes in `results/esm2_pilot/<commit>/inputs.json`:**
   - ESM-2 checkpoints `facebook/esm2_t30_150M_UR50D` and `facebook/esm2_t12_35M_UR50D` (pin the hub revisions).
   - ProteinGym v1 DMS substitutions (HF `OATML-Markslab/ProteinGym_v1`, `DMS_substitutions.csv`, stream/filter; or the official
     reference file + per-assay CSVs). Apply the E6 assay rule mechanically (sort by DMS_id; wild-type length <= 400; has single
     mutants; first 20). Write the chosen list before scoring anything.
   - Calibration: 1,024 UniRef50 sequences, length 50-400, seed 7; drop exact UniProt IDs of the 20 assay proteins. If no clean UniRef50
     source is available, write an E6 amendment naming the substitute BEFORE using it.
2. **Port the repair to ESM-2 (pre-LayerNorm).** In `EsmLayer`, attention runs on `LayerNorm(h)` and `EsmSelfOutput` adds the residual;
   the FFN likewise (`EsmIntermediate` on the layer's LayerNorm, `EsmOutput` adds the residual). Adapt the drift-corrected refit
   (`src/qfc/grp.py` `sequential_refit`, ridge_rel 1e-4) so the LS targets use the teacher's sublayer output plus the residual drift
   (teacher residual input minus student residual input), with student inputs taken after the student's own LayerNorm. Add ESM support
   to the GRP selector (heads + FFN neurons, MAC budget per E6's formula) and physical pruning. Unit tests (CPU, tiny random ESM config):
   physical pruning equals masking; unpruned student recovered by the refit; refit lowers hidden-state error; kept units equal the
   selected masks; MAC ratio equals the budget to 3 decimals. `pytest` green + `python -m compileall -q src scripts`.
3. **Smoke test** on 2 assays and 64 calibration sequences (`--smoke`; rule 9: engineering check, not a result).
4. **Run the pilot once:** arms A-E of E6 at budget 0.225 (primary) and 0.5 (descriptive), selector GRP (primary) and magnitude
   (descriptive). Save per-assay Spearman and per-variant scores as JSON under `results/esm2_pilot/<commit>/`.
5. **Analysis script committed before step 4**, computing exactly K1 and K2 (bootstrap over assays, 10,000 resamples, seed 1234).
   Then apply the rules and write the verdict under E6 with numbers copied from the JSON: CONTINUE only if both K1 and K2 pass;
   otherwise STOP the ESM-2 direction and say so plainly.
6. **Descriptive efficiency:** CPU latency batch 1 and 8 (median of 30 after 5 warm-ups) for the pruned shapes, dense 35M, dense 150M,
   INT8 150M; parameter counts including and excluding embeddings.

## Report back
The verdict (K1, K2 with numbers), the latency table, the commit, and any deviation. No claims beyond the pilot; it is not a paper
result and not confirmatory.
