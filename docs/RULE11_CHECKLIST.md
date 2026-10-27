# Rule 11 checklist: paper/main.tex (draft analysis-paper v1)

Status: PREPARED, NOT SIGNED OFF. Reviewer (independent of the author of the code) ticks each row. The three proofs
need the supervisor (Dr. Aparajita Dutta). Re-derive every number from the JSON with the named script; do not trust
the paper text or PROJECT_STATE.

Environment: `.venv-ci` (Python 3.12, torch 2.6.0, transformers 4.51.3). Analysis scripts are CPU-only and read committed JSON.

## A. Numbers -> JSON -> script

| Paper item | Source JSON (under `results/`) | Script that derives it | Check |
|---|---|---|---|
| Sec 5.1: 54/36 pairs >0.99, 39/35 with output fid <0.5, Spearman 0.437/0.263, bound Spearman 0.929/0.959, ratio 3.26/3.12, layer-level errors | `v14_bcm_diag/4bd649f/diag_summary.json`, `v14_pairs/71cb52d/pairs_{sst2,mrpc}_seed7.json` | `scripts/v14_diag_summary.py`, `scripts/v14_pairs.py` | [ ] |
| Sec 5.1 all-task shares (72/97/92/82/20 %, counts), Spearman 0.915-0.955, ratio 3.12-3.26, BCM gap 0.008-0.022 | `v14_bcm_diag/full_be03d75/diag_summary.json` (+ per-seed `v14_<task>_seed<s>_diag.json`) | `scripts/v14_diag_summary.py` | [ ] |
| Fig 1 (pairs) | `v14_pairs/71cb52d/*.json` | `paper/figs/fig_pairs.py` | [ ] |
| Table 1 (V14 means, Random mean+-sd) | `v14_bcm/9c4b999/v14_<task>_seed<s>.json`, `v14_<task>_random.json` | `scripts/step21_bcm.py` (producer), `scripts/v14_analysis.py` | [ ] |
| Sec 5.2 paired gains and CIs (+4.03 QNLI, +7.15 RTE), BCM-Fisher range (-0.30..+1.30), H9b/H9c/H9d verdicts | `v14_bcm/9c4b999/v14_verdicts.json` | `scripts/v14_analysis.py` (10,000 resamples, Holm) | [ ] |
| Fig 2 (V14) | `v14_bcm/9c4b999/` | `paper/figs/fig_v14.py` | [ ] |
| H9a calibration check (refit lowers error and KL everywhere; H9a failure on SST-2/MRPC = ceiling) | `v14_h9a_check/fdd808d/h9a_<task>_seed<s>.json` | `scripts/check_h9a.py` | [ ] |
| Table 2 (H10 KL by method/budget), "40-75 % below K-prune", SST-2 wins at all four budgets, MRPC none | `v15_h10/c43dd3e/h10/v15_h10_verdict.json` (+ `h10/grp/*.json`, `h10/logs/*.log`) | `scripts/v15_h10_analysis.py` | [ ] |
| Fig 3 (H10 KL) | same verdict JSON | `paper/figs/fig_kl.py` | [ ] |
| Sec 5.3 V16 text (34-60 % worse, 0.0445 vs 0.0527, 0.0480 vs 0.0205, branch A-3) | `v16_stageA/b2b444e/v16_stageA_verdict.json` (+ per-run JSONs) | `scripts/v16_stageA_analysis.py` | [ ] |
| V17 text (1.19/1.29 MRPC, 0.31/0.50 SST-2, within 3 %, gate failed) | `v17_rdev/195c18f/v17_rdev_verdict.json` | `scripts/v17_rdev_analysis.py` | [ ] |
| Table 3 + Fig 4 (factorial ratios) | `v16_stageA` (K-prune/module-output), `v17_rdev` (drift rows, GRP-R), `v15_h10` (GRP/module-output) | `paper/figs/fig_kl.py` (Fig 4 is generated; Table 3 is typed from the same JSON: re-check each cell) | [ ] |
| Appendix: K-prune reproduction (MRPC 0.8431..., mean 0.8485; SQuAD F1 74.73...; 84.80 / 74.16 published) | **No raw JSON committed.** Only `v15_h10/c43dd3e/phase1/kwon_unpruned_{mrpc,sst2}.log` exist. Numbers come from PROJECT_STATE (Kaggle Versions 2-4) | none | [ ] **OPEN: obtain Kaggle output, commit under `results/v15_phase1/<commit>/`, or soften the claim** |
| Appendix: Kwon drops vs reported drops; unpruned 0.8701 / 0.9312 | Same gap (only unpruned logs committed) | none | [ ] **OPEN, same as above** |
| Appendix: pinned checkpoints | revisions in the `SPECS` of `scripts/step22_grp.py` (verify full hashes) | n/a | [ ] |
| Published numbers (Kwon, K-prune tables) | external papers | n/a | [ ] verify against the PDFs |
| Intro/abstract claims (e.g. "four of five GLUE tasks") | derived from rows above | n/a | [ ] |

## B. Proofs -> what the reviewer must check

| Statement | Check | Code evidence (test) |
|---|---|---|
| Prop 1 (attention fidelity blind to value mixing) | Is `rho_h = A A^T / tr(A A^T)` invariant to `A -> A Q` with `Q Q^T = I`? Is `Q` square orthogonal over the key axis (so `A Q` is a valid, if not row-stochastic, pattern)? The paper says "arbitrary outputs"; check whether `A_j = A_i Q` can be row-stochastic, or restate as a permutation-of-keys case only. | `tests/test_output_coverage.py` (indirect) |
| Prop 2 (scaled-orthogonal Procrustes residual = `||Z_j|| sqrt(1-F)`) | Trace step: max of `tr(U^T M)` over orthogonal `U` is `||M||_*`; the scale `s` is unconstrained in sign; `F = (||M||_*/(||Z_i|| ||Z_j||))^2`. | `test_scaled_procrustes_residual_identity` |
| Thm 1 (coverage bound) | (a) `Y - Y_merge = sum_{j notin S} (Z_j - Z_pi(j) R_j) W_j`: shapes consistent, Procrustes scale `s` absorbed into `R_j`? (b) `||M W_j||_F <= ||M||_F ||W_j||_2` (true). (c) merged weights are feasible for the LS problem over `Z_S`, so LS error <= merge error. (d) `w_j = ||W_j||_2 ||Z_j||_F`; min over `i` uses `sqrt(1-F_ji)`. | `test_bound_merge_ls_ordering`, `test_bcm_select_objective_and_guarantee`, `test_merge_on_real_bert_layer_matches_gram_prediction` |
| Remark `B(S) = sum w_j - f(S)` and greedy `(1-1/e)` | `f(S) = sum_j w_j (1 - min_{i in S} sqrt(1-F_ji))` is a weighted facility-location function (monotone submodular, nonneg weights, `f(emptyset)=0` convention); guarantee is on `f` only. | `tests/test_submodularity.py` |
| GRP removal cost `Delta E_R = tr(V_R^T P_RR^{-1} V_R)` (Sec 4) | Schur-complement block downdate for the ridge refit. | `test_removal_cost_and_downdate_match_brute_force`, `test_block_cost_matches_brute_force_joint_removal` |
| Drift-corrected target (`y_t + h_t - h_s`) | Equivalent to K-prune's LS target; confirm against the K-prune source at commit 9ca629a1. | `test_drift_correct_equals_plain_where_no_drift_exists`, `test_drift_correct_lowers_downstream_preln_error` |

## C. Code and protocol checks for the reviewer

- [ ] Calibration rows come from train; held-out KL rows are train rows disjoint from calibration (`scripts/step22_grp.py`).
- [ ] No QNLI / MNLI-m / QQP validation data loaded by any V15-V17 path (grep `scripts/step22_grp.py`, `scripts/step23_v16_stageA.py`).
- [ ] Official K-prune / Kwon patches change exactly four lines each (`results/v15_h10/c43dd3e/patch_*.diff`) and are inactive without the index file.
- [ ] Budget conversion (`c` vs `1-c`) and MAC formula identical across the three methods (`test_kwon_mac_formula`).
- [ ] Standing rule 1: exactly k distinct heads per layer in V14 selectors (`qfc.baselines.validate_selection`); GRP allows empty layers by design (state this in review).
- [ ] V14 Kaggle run lacks exact package versions (disclosed in Limitations).
- [ ] `pytest` green and `python -m compileall -q src scripts` in `.venv-ci` at the reviewed commit.
- [ ] Paper claims no speed, FLOP, latency or accuracy-vs-SOTA improvement (grep main.tex for "speed", "faster", "outperform", "state of the art").
- [ ] Every cited paper resolves (DOI/arXiv). New in v1: 2407.13331, 2506.08436, 2606.15161, 2605.25508 (arXiv abstract pages read 2026-10-27); Global-Pruner (ACL Anthology 2024.conll-1.5). "DP-Prune" (IPCCC 2024) could NOT be found and is NOT cited.

## D. Sign-off (not filled by the assistant)

- Independent code/number reviewer: ______  date: ______
- Supervisor proof check (Prop 1, Prop 2, Thm 1): ______  date: ______
