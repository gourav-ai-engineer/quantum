# Full project context for an outside research assistant (written 2026-11-16)

Paste this whole file into ChatGPT, then ask your question. It is self-contained. Personal details were left out on purpose.

## 0. Rules for the assistant (please follow)
- Scientific honesty beats a favourable number. Never invent a citation, a number, a result or a venue fact. If you are not sure, say "unverified".
- Label every factual claim you add as VERIFIED (you read the primary source), PLAUSIBLE (secondary source) or UNVERIFIED.
- This project makes NO quantum-advantage claim. All computation is classical; quantum information is used only as a mathematical representation.
- Do not suggest tuning a method after seeing test results. The project uses pre-registration; a new method variant needs a written hypothesis first.
- When you suggest an experiment, state the hypothesis, the baseline, the statistic, and what outcome would count as failure.

## 1. One-paragraph summary
We study retraining-free (post-training) structured pruning of fine-tuned BERT-base: physically removing whole attention heads and whole feed-forward (FFN) neurons using only a small calibration set and a closed-form weight repair, with no fine-tuning, so the model runs on a weaker machine. The project began from a quantum-inspired head-scoring idea (heads as density operators, fidelity-based selection). Through a long chain of pre-registered experiments the quantum selector gave no measurable benefit over classical similarity, and every attempt to beat the strongest official baseline (K-prune, ICLR 2024) failed its pre-registered gate. The final frozen method (d2i-RI) ties K-prune and beats the older Kwon et al. (NeurIPS 2022) baseline on three never-developed-on GLUE tasks. The honest deliverable is a rigorous analysis paper, "When Does Head Selection Matter? A Quantum-Information View of Attention-Head Redundancy in Retraining-Free Transformer Pruning", aimed at a Q1 journal (Neurocomputing is the target; TMLR is a Q2 fallback). A second-model extension (DistilBERT-weights, up to 7 datasets) is pre-registered and in progress.

## 2. Background (for readers new to the field)
- A BERT-base layer has 12 attention heads and an FFN with 3072 neurons; there are 12 layers. "Structured pruning" removes whole heads/neurons so the matrices physically shrink (fewer parameters and multiply-accumulates, MACs).
- "Retraining-free": after removing units we may not fine-tune. We use ~2048 unlabeled training examples ("calibration") and repair the remaining weights by least squares (LS) so each layer's output still matches the original model's.
- Budget: we keep 50% of the MACs (a "0.5 constraint"). Both official baseline codes define the budget the same way (K-prune reports FLOPs = 2 x MACs).
- The two official baselines: Kwon et al. 2022 ("A Fast Post-Training Pruning Framework for Transformers", Fisher-information masks + mask rearrangement + LS mask tuning) and K-prune (Park et al., ICLR 2024, "Accurate Retraining-free Pruning for Pretrained Encoder-based Language Models": knowledge-preserving mask search plus per-sublayer LS weight tuning, bottom-up).

## 3. Origin and what was dropped
- Original paper: a hand-weighted score QIS = 0.5*VNE + 0.3*QJSD - 0.2*F (von Neumann entropy, quantum Jensen-Shannon divergence, fidelity of per-head density operators rho = A A^T / Tr(A A^T), A = attention matrix). It was dropped: weights arbitrary, no correlation with head importance (Spearman about -0.05, p = 0.54), "multi-seed" results were identical across seeds, and an independent t-test was labelled paired. Do not revive it.
- Replacement, QFC (quantum-fidelity coverage): choose a subset S of heads maximizing C(S) = sum_i max_{j in S} F(rho_i, rho_j), F = squared Uhlmann-Jozsa fidelity. Proven monotone submodular, so greedy has a (1-1/e) guarantee on the OBJECTIVE only, never on accuracy.
- Later experiments (V4-V13) showed QFC was not consistently better than the Michel gate-sensitivity or von Neumann-entropy baselines, and fidelity did not beat cosine or Hilbert-Schmidt similarity. Two baseline bugs were fixed in V12 (the "Random" baseline kept all heads; the Shannon baseline was NaN under padding); all earlier Shannon and Random numbers are invalid.

## 4. The quantum-information formalism now used (this is the paper's lens)
- For heads i, j let Z_h be the head's context vectors over N calibration tokens and Z = [Z_1..Z_H]. Output Gram operators Z_i Z_i^T / tr(.) are density operators. Squared fidelity F_ij = (||Z_i^T Z_j||_* / (||Z_i||_F ||Z_j||_F))^2, the squared Uhlmann fidelity. sqrt(1 - F_ij) is the purified distance (Tomamichel-Colbeck-Renner 2010, Definition 4).
- Proposition 1 (proved): attention-pattern density operators rho_h = A_h A_h^T / tr are blind to value mixing. If A_j = A_i Q with Q orthogonal then fidelity is 1 although outputs differ.
- Proposition 2 (proved): the best scaled-orthogonal Procrustes map of head i onto head j leaves residual ||Z_j||_F * sqrt(1 - F_ij), i.e. size times purified distance (uses von Neumann's trace inequality).
- Theorem (proved): merging a pruned head into its nearest kept head gives output error at most B(S) = sum_{j not in S} w_j min_{i in S} sqrt(1 - F_ji), w_j = ||W_j||_2 ||Z_j||_F; the LS-refit error is at most the merge error. B(S) = const - f(S) with f a weighted facility-location function, so greedy has a (1-1/e) guarantee on f (not on B, not on accuracy).
- Measured: on 5 GLUE tasks x 5 seeds, the share of attention-near-identical head pairs (attention fidelity above 0.99) whose output fidelity is below 0.5 is 72% (SST-2), 97% (MRPC), 92% (RTE), 82% (QNLI) and only 20% (CoLA). The coverage bound ranks random subsets like the true merge error (median per-layer Spearman 0.915-0.955) but overestimates it about 3x.
- These proofs have NOT been independently checked yet (supervisor check pending). The purified-distance remark is definitional (checked against the fidelity convention in the repo and the cited paper).

## 5. Experimental protocol (fixed rules)
- Environment is pinned: Python 3.12, torch 2.6.0, transformers 4.51.3, datasets 3.6.0 (transformers 5.x silently ignores head_mask, so results from it are meaningless). Compute is a Kaggle T4 GPU, about 30 GPU-hours per week; CPU is for smoke tests only.
- Calibration data come from the training split; evaluation uses the full validation split; heads/neurons are never chosen on evaluation data. Five calibration seeds (7, 42, 77, 123, 2024). Every selector keeps exactly the budgeted number of units. Random is always a distribution (30-100 masks). Entropy baselines are reported keep-high and keep-low.
- Statistics: paired cluster bootstrap over (seed x example), 10,000 resamples, Holm correction across tasks. Never an independent t-test; never present identical reruns as independent seeds.
- Hypotheses and thresholds are written in docs/PROJECT_STATE.md BEFORE the data. Engineering failures (crash, platform change) may be fixed without touching method, hyper-parameters, seeds, tasks or thresholds; any change after a statistic is seen is forbidden.
- A "paper-ready" label needs an independent audit (docs/RULE11_CHECKLIST.md) plus the supervisor's proof check. Not yet signed off.

## 6. Experiment chain and results (chronological)
- V14 (output-space coverage + LS compensation; heads only, keep 6 of 12 per layer; 5 GLUE tasks, 5 seeds, 512 calibration rows): pre-registered selector x compensation factorial. After a least-squares refit, four selectors (output coverage, Fisher, magnitude, greedy LS) are statistically indistinguishable; attention-pattern coverage stays clearly worse. Compensation, not selection, carries the recovery.
- V15 (GRP, global refit-aware allocation over heads and FFN neurons, Fisher-weighted, jointly refit, physically pruned): calibration gate on the logit KL to the teacher over 512 held-out training rows against the official baselines. Result: SST-2 yes (GRP KL 40-75% lower than K-prune at all four budgets), MRPC no (0 of 4). Gate fails, so no accuracy comparison was run. Baseline reproduction passed (K-prune MRPC 0.8485 vs published 84.80; SQuAD F1 74.20 vs 74.16).
- V16 (diagnosis): transplanting K-prune masks with the GRP refit is 34-60% WORSE than K-prune's own tuning. Reading the code showed K-prune's least-squares target is the teacher sublayer output plus (teacher residual input minus student residual input), i.e. a drift-corrected target.
- V17: GRP + drift-corrected refit (GRP-R) fails its gate (MRPC KL 1.19-1.29x K-prune; SST-2 0.31-0.50x). K-prune masks + drift-corrected refit reproduce K-prune to 0.97-1.00x. The factorial shows that with compensation fixed, the winning selection rule is task-dependent. Caveat: V17 hypothesis was registered after V15/V16 development data had been seen.
- Exploration (MRPC, SST-2, STS-B only, development tasks): added input-side alignment (ridge LS refit of the kept q/k/v rows and FFN up-projection rows to the teacher's unit features given student inputs, layer by layer). d2i-RI = K-prune masks + input-side alignment + drift-corrected output refits met the written freeze rule (beat both baselines on all three at both budgets) but margins were inside noise: K-prune vs d2i-RI accuracy at 0.5 budget MRPC .8623 vs .8652, SST-2 .9149 vs .9158, STS-B Pearson .8804 vs .8809 (teachers .8775/.9243/.8805). The clearest effect was lower held-out distance to the teacher (KL/MSE 5-13% below K-prune).
- V18 (pre-registered confirmatory test, frozen code, run once): d2i-RI vs the better of the two official baselines (K-prune, fixed from development data) at 50% MACs on QNLI, MNLI-m, QQP (tasks never used in development), 5 seeds, full validation sets. Rule: Holm one-sided p < 0.05 and CI lower bound > 0 on at least 2 of 3 tasks, and no significant loss. RESULT: H18a NOT SUPPORTED. d2i-RI minus K-prune accuracy: QNLI -0.0002 [-0.0031, +0.0026]; MNLI-m +0.0016 [-0.0010, +0.0040]; QQP +0.0013 [+0.0002, +0.0024] (only QQP significant). d2i-RI minus Kwon et al.: +0.0276, +0.0805, +0.0558, all Holm-significant. No losses. Context (accuracy, mean of 5 seeds): teacher / K-prune QNLI .9154/.9027, MNLI-m .8457/.8214, QQP .9091/.8935. The paper reports the d2i-RI-minus-K-prune CIs as descriptive (within about [-0.31, +0.40] pp), not as an equivalence test (none was pre-registered).
- Post-hoc (unregistered) mechanism check on the V18 tasks: held-out-train KL ratio d2i-RI/K-prune 0.958 (QNLI), 0.994 (MNLI-m), 0.952 (QQP). The development-set effect mostly does not transfer.
- E3 (portfolio pre-check, existing JSONs only): choosing d2i-RI vs GRP-RI per seed by held-out KL gives MRPC -0.10 pp (and picks the accuracy-worse mask in 2/5 seeds), SST-2 +0.57/+1.26 pp, STS-B +0.05 pp. Fails; not pursued.
- Headroom: unpruned minus K-prune accuracy at 0.5 MACs is 0.01-2.43 pp across six tasks (MRPC 1.52, SST-2 0.94, STS-B 0.01, QNLI 1.27, MNLI-m 2.43, QQP 1.56). At this budget no method can win by much, so "beat everything at 50% MACs" is bounded by the task, not by method quality.
- Latency (descriptive, physically pruned random-weight BERT-base shapes at 50% MACs, seq len 128): encoder parameters 85.1M -> about 42.2-42.5M. CPU (one i7-13700, FP32, seed-7 shapes): K-prune 1.65x faster at batch 1, 2.05x at batch 32; Kwon et al. 1.4-1.5x and 1.7-1.9x. T4 GPU (FP16, 100 runs, all 15 shapes): no speed-up at batch 1 (8.96 ms vs 9.2-9.8 ms), about 1.2-1.3x at batch 32 (45.7 ms vs 34-40 ms). d2i-RI keeps K-prune's masks, so its shapes and latency equal K-prune's by construction; no speed claim for d2i-RI.

## 7. In progress: V19 (second-model extension)
- Pre-registered 2026-11-16 before any V19 code or data. Frozen d2i-RI is applied unchanged to the DistilBERT weights released with the official codes. Finding: those folders are 6-layer BERT-format weights (config model_type "bert", no tokenizer), so the frozen BERT code runs unchanged; the paper will call them "DistilBERT" and state this.
- Tasks: MRPC, SST-2, STS-B, QNLI, MNLI-m, QQP, and SQuAD v1.1 (SQuAD support is being added to the runner; if not ready the study is six GLUE tasks). RoBERTa and BERT-large are out of scope (no official checkpoints; a ported K-prune would be a weakened baseline).
- Hypotheses: H19a superiority over K-prune on at least two thirds of tasks (Holm, no significant loss); H19b non-inferiority within 1.5 pp of K-prune on EVERY task (CI lower bound > -1.5 pp, Holm); H19c descriptive latency. The expected and acceptable outcome is H19a not supported and H19b supported.
- Status: step 1 (port/engineering checks) passed; step 2 (driver, analysis script, kernels) underway; no GPU result yet. Cost about 10-15 GPU-hours, roughly one to two weeks.

## 8. Literature map (statuses: VERIFIED means the repo author read the primary source)
- Head pruning: Michel et al. 2019 (NeurIPS), Voita et al. 2019 (ACL), Differentiable Subset Pruning (Li et al., TACL 2021). Training-based: Movement pruning, CoFi.
- Retraining-free encoder pruning: Kwon et al. 2022 (NeurIPS), K-prune (ICLR 2024), KCM (Nova et al., ICML 2023), LIAR (arXiv 2407.13331; BERT-base GLUE at 50% PARAMETERS vs KCM/Mask-Tuning, not K-prune), STAT (arXiv 2406.00061; compensation folded into the next layer; BERT compared with Kwon et al.), OSSCAR (arXiv 2403.12983), CORP (arXiv 2602.05243), Olica (ICML 2025), Neuron Merging, SNOWS (arXiv 2411.18376; vision, 2:4 masks, multi-layer reconstruction).
- Global-Pruner (G-Pruner, CoNLL 2024): VERIFIED by reading the PDF. It names K-prune as a baseline (its Figure 2 plots accuracy degradation vs relative FLOPs, about 0.50-0.75, BERT-base on QQP/MNLI/SST-2/QNLI) and claims the least degradation in most cases; headline +6.02 F1 on SQuAD v2.0 at 60% FLOPs; 2000 calibration rows, 10 seeds, transformers 4.12.0. It is NOT reproduced by us and not directly comparable (relative FLOPs vs our MACs, different seeds/rows/software). The project therefore makes no claim to be at or above the best published method.
- Other 2026 papers read in full by the CLI session: ASF-S (2609.24401; AG News, FFN of 3 blocks, classifier-head fine-tuning), CAHP (2606.19150; heads only, brief fine-tuning), Greedy-Gnorm (2602.04491; heads only), Magnitude Profile (2609.26177; OPT/RoBERTa-large). None reports a comparison with K-prune on BERT-base GLUE under our protocol.
- Quantum references: Uhlmann 1976, Jozsa 1994, Tomamichel-Colbeck-Renner 2010, CompactifAI (arXiv 2401.14109, ESANN 2025; quantum-inspired tensor-network/MPO compression of LLaMA-2 7B, needs brief retraining, no BERT results). Genuine quantum transformers (Quixer, Quantum Adaptive Self-Attention) run on quantum hardware or simulators and are a different project.
- A 2025-26 search found "A Free Lunch in LLM Compression" (arXiv 2510.14444) reporting that, with a good reconstruction step, mask-selection criteria matter less (decoder LLMs); consistent with our V14 finding on encoders. Not yet cited.

## 9. Gaps identified and the claimed contribution
1. Selection and compensation are usually evaluated together; we separate them in pre-registered factorials.
2. Attention-pattern similarity is the natural redundancy signal but was never shown to fail; we prove and measure that it is blind to value mixing (task-dependent, 20-97%).
3. K-prune's compensation mechanism was not isolated; we reproduce it to within 3% with a drift-corrected LS target (a post-hoc finding on development tasks; STAT already publishes a related next-layer correction, so novelty is limited).
4. Few retraining-free papers run a confirmatory test on tasks never used in development with the analysis code frozen; we do, with a negative result.
5. One model family only (BERT-base); V19 adds a second.
Honest novelty assessment: thin. No new pruning principle is claimed; layer-wise input-side reconstruction is standard in one-shot pruning (GPTQ/OBC-style). The contribution is analysis, a quantum-information formulation of redundancy, pre-registration discipline and a clean negative confirmatory result.

## 10. Why we are not claiming a win
Four pre-registered gates failed (V15 H10, V16, V17, V18 H18a). A multi-method portfolio fails (E3). Within this project the quantum-fidelity selector added nothing over classical similarity, and d2i-RI contains no quantum quantity. The only direction with real headroom is high compression (keep 20-30% of MACs); a dev-only kill-test was proposed, not run.

## 11. Paper status
- File: paper/main.tex (about 450 lines, pdfLaTeX, natbib, refs.bib; four figure PDFs). Sections: introduction, related work, setup, theory, methods compared, pre-registered protocol and disclosures, results (diagnostics, V14, V15, V16, V18 incl. latency), limitations, appendix with baseline reproduction. Written in an anonymous-author style.
- Open before submission: first real compile (done on Overleaf by the owner), independent audit (Rule 11), supervisor proof check, journal formatting (Elsevier template, highlights, generative-AI-use declaration, data/code availability), venue check (Neurocomputing reported as Q1 by third-party aggregators, SJR about 1.47; scimagojr.com itself was not reachable, so UNVERIFIED), V19 results and a DistilBERT section.

## 12. Repository map
- src/qfc/: states, fidelity, coverage, grp (global reconstruction pruning, refit, physical pruning), align (input-side alignment), metrics (cluster bootstrap, Holm), squad, hf_experiments.
- scripts/: beat_run.py (methods d2i_ri etc.), beat_confirm_analysis.py (V18 analysis), v15_kaggle.sh (Kaggle driver), v15_latency.py, make_ledger_row.py. tests/: about 154 passing tests.
- docs/: PROJECT_STATE.md (ledger and pre-registrations), EXPLORATION_LOG.md, theory.md, RULE11_CHECKLIST.md, LIT_REVIEW_2026-11-12.md.
- results/<experiment>/<commit>/: small JSONs; key confirmatory file results/beat_confirm/797af30/beat_confirm_verdict.json.

## 13. Where an outside assistant can help most (pick any)
1. Simulate a hostile peer review of the paper for Neurocomputing and for TMLR; list the top rejection reasons and the cheapest fixes.
2. Independently verify the proofs in section 4 and the purified-distance identity; look for hidden assumptions (for example row-stochasticity in Proposition 1).
3. Check whether any 2025-2026 retraining-free encoder pruning work beats K-prune on BERT-base GLUE at about 50% MACs under a comparable protocol (read result tables, not abstracts; label VERIFIED or UNVERIFIED).
4. Critique the V19 design, especially the non-inferiority test (margin 1.5 pp, one-sided, Holm) and the 2/3-of-tasks superiority rule, for multiplicity and power problems.
5. Judge whether a high-compression regime (keep 20-30% MACs) has literature precedent and a fair chance of a real win; propose the dev-only kill-test rule.
6. Suggest honest ways to add quantum content that do real work (for example quantum-inspired tensor-network compression after pruning, healed by the closed-form refit) without claiming advantage; assess novelty against CompactifAI and joint pruning-plus-quantization literature (OBC, ZipLM).
7. Review the abstract and contribution wording for overclaiming.

## 14. Known weak points (be skeptical of these)
- d2i-RI was developed on MRPC, SST-2 and STS-B development data after V15-V17, and its main selling point (lower held-out KL) mostly does not transfer.
- The V17 hypothesis was registered after development data were seen.
- Latency uses random-weight models of the pruned shapes, one CPU and one GPU type.
- Whether the SQuAD "squad" checkpoint folder is v1.1 is a judgement from train-row behaviour.
- Rule 11 (independent audit) and the supervisor's proof check are not done.
