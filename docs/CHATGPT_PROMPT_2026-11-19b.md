# Prompt for ChatGPT, round 3 (2026-11-19): pre-data decisions, proof and code audit, literature sweep

How to use: this continues the earlier two prompts (`docs/CHATGPT_CONTEXT_2026-11-16.md`, `docs/CHATGPT_PROMPT_2026-11-19.md`). If ChatGPT has lost that context, paste those first. Then paste everything below the line.

---

Thank you for round 2. I followed your verdict A (finish V21 as frozen). Your two gate "defects" were checked against the code and are the same items our freeze audit had already dispositioned: kept units are counted per layer, but `repair()` prunes with the K-prune npz masks of that seed, so identity holds by construction; `gate_pass` is strict on purpose and the frozen reading rule handles a unit whose only failure is a missing Kwon dump. No code changed. Same rules as before: label every claim VERIFIED / PLAUSIBLE / UNVERIFIED, never invent a number or citation, separate know / infer / recommend.

## 1. Status since round 2 (no pruned-condition V21 number has been read)
- Units muril_t42 and mbert_t42: integrity gate PASS. Frozen code (tag v21-freeze, commit 746330d), torch 2.6.0, transformers 4.51.3, test split, 5010 examples per language, 12 languages (en + as, bn, gu, hi, kn, ml, mr, or, pa, ta, te). Teacher val-en equals the fine-tune record; MAC ratio 0.4999-0.5000; kept units equal K-prune in 12/12 conditions; the official K-prune and Kwon teachers differ from ours in 0 rows in every language and seed.
- By design the integrity file also stores the UNPRUNED teacher's test accuracy, so I have seen these (unpruned only): MuRIL t42 en 0.843, targets 0.693-0.764. mBERT t42 en 0.822, targets 0.511-0.647 except Odia (or) 0.341, which is at chance for 3-class XNLI (chance about 0.333).
- Units muril_t43 and mbert_t43 are running; t44 units are queued (Kaggle allows 2 GPU sessions). The analysis runs exactly once after all six units pass.

## 2. Tasks, in priority order

### T1 (time-critical: must be settled BEFORE any V21 statistic is read). Floor effect and near-chance languages
The frozen primary statistic is S = mean over target languages of r_M(L) minus r_M(en), with r = unpruned accuracy minus mask-only accuracy; H21-1 needs pooled CI lower bound > 0.02, point >= 0.05, and S > 0 in all 3 teachers of a model. It cannot change. mBERT's unpruned Odia accuracy is at chance, so r_M(or) is bounded near 0 by the floor and pulls S toward 0 for mBERT (it cannot be pushed further below chance in expectation); for MuRIL no language is near chance in t42.
I want to declare now, in writing, the descriptive sensitivity analyses the paper will report, so they are fixed before data. Please give:
(a) an exact, mechanical rule for flagging a "floor language" per model, using only unpruned teacher accuracy (for example: unpruned accuracy < chance + delta in any teacher of that model), with a justified delta, and say whether the rule should use one teacher, all three, or the mean;
(b) the exact list of descriptive statistics to report alongside the frozen S: S recomputed excluding flagged floor languages; the headroom-scaled version (U - X)/(U - chance), with how to handle U close to chance (instability) and values above 1; a per-language table; and anything else you consider necessary;
(c) one paragraph of paper wording that states these are post-hoc-labelled descriptive checks declared before the V21 data were read, that they cannot rescue or overturn H21-1, and how a reader should interpret a disagreement between the frozen S and the floor-excluded S;
(d) whether the same flagging should apply to the secondaries H21-2 (repair residual) and H21-3 (mixed calibration target gain), and how.
Answer T1 first and keep it short enough to paste into our ledger.

### T2. Independent proof check (we may not self-sign this; your review is advisory, a human still signs)
Check these statements (in `paper/main.tex`, section "Theory: a quantum-information view of head redundancy"; checklist in `docs/RULE11_CHECKLIST.md`, section B). If you cannot read the files, ask me to paste them.
- Proposition 1: attention-pattern density operators rho_h = A_h A_h^T / tr(.) are blind to value mixing: if A_j = A_i Q with Q Q^T = I, then rho_j = rho_i while outputs A_i V_i and A_i Q V_j differ. Open issue: A_i Q is row-stochastic only if Q is a permutation (or A_i rank-deficient); is the claim correctly scoped?
- Proposition 2: min over s in R and orthogonal U of ||Z_j - s Z_i U||_F = ||Z_j||_F sqrt(1 - F_ij), with F_ij = (||Z_i^T Z_j||_* / (||Z_i||_F ||Z_j||_F))^2 (von Neumann trace inequality; s unconstrained in sign).
- Theorem: merging each pruned head j into its nearest kept head with the scaled-orthogonal Procrustes map gives ||Y - Y_merge||_F <= B(S) = sum_{j not in S} w_j min_{i in S} sqrt(1 - F_ji), w_j = ||W_j||_2 ||Z_j||_F, and the unregularized LS-refit error is at most the merge error; B(S) = sum_j w_j - f(S) with f a weighted facility-location function, so greedy has the (1-1/e) guarantee on f only.
- Remark: with squared fidelity on trace-normalized operators, sqrt(1 - F_ij) is the purified distance (Tomamichel, Colbeck, Renner 2010, Definition 4: P = sqrt(1 - Fbar^2), Fbar unsquared); and F_ij equals the Uhlmann fidelity of Z_i Z_i^T / tr and Z_j Z_j^T / tr by Uhlmann's theorem.
For each: the exact statement checked, the assumptions needed, holds / does not hold, and a counterexample if not.

### T3. Audit of the frozen analysis script against the frozen text (engineering bugs only)
Compare `scripts/v21_analysis.py` with `docs/PROJECT_STATE.md` section "Pre-registration V21", subsection 3 (hypotheses and decision rules) and subsection 5 (integrity, Kwon-missing rule). Check: statistic definitions (S, R, G, the non-inferiority margin 0.015 and its one-sided p), the bootstrap (paired cluster over calibration seed x example inside each unit, 10,000 resamples, seed 1234 + unit index, pooled by averaging over units and languages), the all-teachers sign rule (over teachers that passed; fewer than 2 means "not run"), Holm over the five secondaries per model, the N/D robustness recomputation, and H21-4(ii) reported "not run" when any unit of a model lacks Kwon files. Report ONLY discrepancies between code and frozen text, with file and line. A real discrepancy can still be fixed as an engineering fix, because no statistic has been seen; a design preference is not a discrepancy.

### T4. Systematic literature sweep, 2023-2026
Topic: post-training or retraining-free structured pruning (heads, FFN neurons, layers) of multilingual encoders or multilingual LLMs, with attention to: calibration language effects (English vs target vs mixed calibration), cross-lingual disparity after compression, Indic languages, and closed-form/least-squares repair. Include work on calibration-data language for one-shot pruning or quantization of multilingual LLMs if it exists (for example studies of SparseGPT/Wanda/GPTQ calibration language). Give a table: title, venue/year, URL, setting (model, languages, method, retraining or not), finding, label, and how close it is to V21 (same / adjacent / distant). End with an honest statement of what V21 adds that none of them covers, or say that nothing remains.

### T5. Related-work and framing text for the multilingual section
Draft about 200 words of related work that positions V21 against Ogueji et al. (EMNLP 2022), Li et al. (ACL 2022), Ma et al. (ACL 2021) and anything VERIFIED from T4, framing H21-1 as a confirmatory re-examination in a specific setting, not a discovery. Only use claims you labelled VERIFIED.

### T6. Global-Pruner exact numbers
From Yao et al., CoNLL 2024 (Global-Pruner), read Figure 2 and Table 1. For BERT-base at the budgets closest to 50% of FLOPs, report each task's accuracy (or accuracy drop) for G-Pruner and for K-prune exactly as printed, say which are read off a plot (approximate) and which are table values, and state every protocol difference from ours (FLOPs vs MACs, calibration rows 2,000 vs 2,048, seeds 10 vs 5, software versions, evaluation split). I will cite it as a published, un-reproduced claim in its own units.

## 3. Answer format
- T1 first, as ledger-ready text.
- T2 and T3 as findings lists: item, verdict, evidence (file and line or equation), recommended action.
- T4 as the table described, then the novelty statement.
- T5 as paste-ready LaTeX-free prose with citations as (Author, Year).
- T6 as a table with a column "source: table / figure-read / not found".
- If you cannot access a file or paper, say so and tell me exactly what to paste.

## 4. Constraints (unchanged)
Pre-registration first; confirmatory data single-use; no change to the frozen V21 design after any statistic is seen (engineering bugs only, before data). Official baselines unmodified. Kaggle T4, 2 concurrent GPU sessions, about 30 GPU-hours per week. No quantum-advantage, state-of-the-art or speed claims. Personal and institutional details are omitted on purpose.
