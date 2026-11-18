# Research-direction audit: post-training structured pruning of multilingual encoders

**Date:** 2026-11-18  
**Purpose:** Skeptical review of candidate research directions against the current project state and close primary literature.  
**Scope:** Retraining-free physical pruning of attention heads and FFN neurons in BERT-family/multilingual encoders, with calibration-only least-squares repair.  
**Status:** This is a bounded literature audit, not a guarantee of novelty, a Q1 acceptance prediction, or an instruction to run a confirmatory experiment.

## 1. Decision summary

The V20 repair-disparity hypothesis is **NO-GO**. The V20 result says the large cross-language collapse occurred in the English-selected masks before repair; repair recovered the target languages more than English. Do not keep the old H20b explanation or describe V20 as evidence that English calibration caused the disparity.

The remaining experimental opening is target-language or multilingual **mask selection** for physical structured heads/FFN in a fine-tuned multilingual encoder, evaluated after a fixed repair. This is a narrow transfer/application gap, not a proven novel method. Prior work already covers multilingual structured pruning, target-language head pruning, multilingual one-shot pruning, multilingual calibration, and encoder vocabulary/structure compression. If the proposed selection experiment fails, the strongest honest result is a carefully controlled negative/benchmark study, not a claimed algorithmic win.

My recommendation for next week is a small, pre-registered **mask-selection pilot** using the existing MuRIL teacher and frozen V20 masks/repair code, restricted to development data. It tests the one untested factor indicated by V20. It is a go/no-go experiment, not a Q1 paper result. Require the independent rule-11 review of V19 and fix the Kwon/MuRIL loading issue before any confirmatory comparison.

## 2. Project facts and corrections

The local source of truth is `docs/PROJECT_STATE.md`, including the V18, V19, and V20 verdicts.

- **V18:** H18a superiority of d2i-RI over K-prune was not supported. d2i-RI did significantly outperform Kwon on the confirmatory tasks, but that is not a win over the stronger K-prune baseline. See `docs/PROJECT_STATE.md` around the V18 verdict and `results/beat_confirm/797af30/`.
- **V19:** K-prune superiority was not supported (one of six tasks; QQP +0.16 percentage point); 1.5 pp non-inferiority was supported on all six. d2i-RI keeps K-prune's masks and showed no speed advantage. The project ledger says the independent rule-11 human review remains outstanding.
- **V20:** H20b is NO-GO. P1 = −0.1357, 95% CI [−0.1634, −0.1072], the opposite sign from the registered prediction. After English-calibrated d2i-r, target-language recovery was greater in absolute terms than English recovery. The current residual post-repair Bengali gap to the teacher is about 7.1 percentage points; Assamese about 4.1 pp; Hindi about 1.4 pp. Mixed calibration is descriptive only and is not proof of a method win. The pilot uses one fine-tuned teacher; bootstrap intervals do not include teacher fine-tuning variance.
- **Untried factor:** V20 held mask-selection language fixed to English. Target-language-specific mask selection was reserved. It is a new hypothesis, not a rescue of the failed repair hypothesis.
- **Baseline blocker:** the ledger reports that official Kwon could not load MuRIL under its transformers 4.12 environment. That must be repaired through an allowed local config/tokenizer setup and checked for parity before any confirmatory claim involving Kwon. Do not weaken or rewrite the baseline.

The user-provided V18/V19/V20 results above match the 2026-11-18 ledger. These are **verified against the repository ledger and result pointers**, not independently recomputed in this audit.

## 3. Prior-art audit: the broad gap claims are closed

The claim that multilingual structured pruning has not been studied is **false**. Li et al. study structured pruning of multilingual XLM-R on nine cross-lingual tasks; they report that per-language pruning does not consistently outperform a shared model, and that a fast model is not necessarily small. [ACL 2022 paper](https://aclanthology.org/2022.acl-long.130/)

The following claims must not be made:

- “First multilingual pruning” / “first language-aware pruning.” M-Wanda uses multilingual activations and cross-language importance for layerwise one-shot pruning across 15 languages. [EMNLP 2025](https://aclanthology.org/2025.emnlp-main.1369/)
- “First target-language-specific attention-head pruning.” Shapley Head Pruning removes language-specific interfering heads and reports target-language improvements. [EACL 2023](https://aclanthology.org/2023.eacl-main.177/)
- “Target-language calibration fixes multilingual compression.” Multilingual Brain Surgeon samples multilingual calibration data; Kurz et al. (TACL 2026) find target-language calibration can help perplexity but does not consistently improve downstream tasks. [LREC-COLING 2024](https://aclanthology.org/2024.lrec-main.1030/), [TACL 2026](https://aclanthology.org/2026.tacl-1.9/)
- “Vocabulary trimming plus structured encoder pruning is new.” Vocabulary trimming is established, and a 2025 low-resource multilingual encoder study combines structured pruning, depth/width changes, distillation, and vocabulary trimming. [EMNLP Findings 2023](https://aclanthology.org/2023.findings-emnlp.981/), [IJCNLP-AACL Student Research Workshop 2025](https://aclanthology.org/2025.ijcnlp-srw.5/)
- “Choosing calibration examples for pruning is new.” Williams and Aletras study calibration-data effects for post-training pruning/quantization; M-Wanda, Kurz et al., and the 2026 OASIS data-selection paper further occupy this axis. [ACL 2024](https://aclanthology.org/2024.acl-long.544/), [OASIS, EMNLP Findings 2026](https://aclanthology.org/2026.findings-emnlp.510/)
- “Combining pruning with token/head selection is new.” Token-and-head adaptive transformers predate this project. [COLING 2022](https://aclanthology.org/2022.coling-1.404/)

### Narrowest plausible delta

The possible delta is: **task-finetuned multilingual BERT-family encoders; physical structured head + FFN removal at matched MACs; official K-prune/Kwon-style retraining-free LS compensation; language-aware mask selection; per-language quality and measured CPU latency.** The exact combined setting is not established by the papers linked above. But this is only a plausible application/method-transfer gap: it may be judged incremental, especially given the existing structured multilingual XLM-R study and Shapley head pruning. A targeted forward-citation search, full methods/code comparison, and baseline parity checks are still needed. Never say “completely novel” or “not discovered anywhere.”

## 4. Candidate directions (all hypotheses; estimates are rough planning values)

### Direction A — Post-repair multilingual risk-controlled mask selection

**Name and idea.** Language-Robust Post-Repair Structured Selection (LR-PRSS): choose one shared physical mask by minimizing worst-tail teacher–student distortion across languages **after** the same frozen LS repair.

**Objective.** Let \(S\) be the set of removed heads/FFN groups, \(c(S)\) its measured MAC or target-CPU latency cost, \(R(S;C)\) the preregistered layerwise repair using unlabeled calibration sets \(C=(C_1,\ldots,C_L)\), and
\[
D_\ell(S)=\frac{1}{|C_\ell|}\sum_{x\in C_\ell}
  D_{\rm KL}\!\left(p_T(\cdot\mid x)\,\|\,p_{R(S;C)}(\cdot\mid x)\right).
\]
For fixed removal budget \(B\), use
\[
S^*=\arg\min_{S:\,c(S)\le B}
  \operatorname{CVaR}_{\alpha,\ell\sim q}
  \left[ D_\ell(S)/\tau_\ell\right],
\]
where \(q\), tail level \(\alpha\), and scale \(\tau_\ell\) are fixed using only development/calibration data. A practical greedy solver adds the feasible unit with the smallest increase in CVaR per saved measured microsecond (or MAC), recomputing/factoring the repair at each step or using a verified surrogate. **No general approximation guarantee** is claimed: the repair-conditioned loss and CVaR set function need not be submodular. The algorithm must be compared to official methods, not described as a guaranteed optimizer of downstream accuracy.

**Why this is motivated by our numbers.** English-selected masks reduce Assamese/Bengali/Hindi validation accuracy to roughly 0.41 before repair, while d2i-r repair recovers them to 0.645/0.670/0.739. That indicates a potentially large mask-side effect; however, post-repair remaining loss is much smaller (especially Assamese/Hindi), limiting realistic margin. This comparison is pilot evidence, not confirmatory evidence.

**Three closest works and exact delta.** (1) ACL 2022 XLM-R structured pruning: multilingual structured pruning, shared vs per-language masks, but not the project’s fixed retraining-free LS repair question; (2) EACL 2023 Shapley Head Pruning: language-specific head interference, but no same physical MAC-budget + fixed LS-repair protocol; (3) EMNLP 2025 M-Wanda / TACL 2026 Kurz: multilingual calibration/language-aware pruning, but unstructured decoder LLM studies rather than physical BERT encoder head+FFN removal with LS compensation. These overlaps make novelty **low-to-moderate**, not high.

**Pre-registerable test.** Primary endpoint: minimum per-language accuracy across a frozen IndicXNLI language subset, at exactly 50% MACs and after repair, paired against K-prune’s English-selected mask. Also require English accuracy non-inferiority. An illustrative meaningful threshold is +2.0 percentage points in worst-language accuracy, with a paired 95% CI lower bound above 0 and no Holm-significant >1.5 pp loss in any language; freeze this after a power analysis, before looking at the new masks. Kill if improvement is <2 pp, its interval includes zero, English loses >1.5 pp, the surrogate ranking fails to predict actual repaired validation quality, or latency does not improve at matched quality.

**Cheapest kill test (≤3 T4 GPU-hours; estimate).** Reuse the verified MuRIL teacher and fixed V20 code. On training-only unlabeled data, form K-prune masks from English vs one preselected target-language calibration (Bengali) for the same three pruning seeds; apply the same d2i-r repair; evaluate only on the already-authorized development validation split. This tests the largest observed residual gap (Bengali) at low cost. If this cannot clear +2 pp worst-language with no English harm, stop before implementing the full CVaR selector. Do not select on validation labels. The 3-hour cap is an estimate; terminate and record a compute overrun if official baseline execution exceeds it.

**Confirmatory design and cost.** If pilot passes, freeze one model (MuRIL) and masks on development languages, then evaluate untouched XNLI/IndicXNLI test examples and languages not used in development. At minimum compare K-prune, Kwon (after compatibility parity), shared vs target-language mask selection, random/magnitude, and Shapley-style head-only selection where official code can be run fairly. Report physical slicing and CPU latency, multiple budgets, per-language metrics, teacher seed variation, and paired uncertainty. Rough estimate 12–25 T4 GPU-hours for one model/primary task family; full Kwon compatibility or several fine-tuning seeds may exceed the 30 GPU-hour weekly limit.

**Risk / odds.** Medium-high technical risk and high novelty risk. Subjective chance of beating the strongest matched baseline by a reviewer-meaningful margin: **low (about 10–20%)**. Reason: prior multilingual mask work is close, after-repair headroom is bounded, and V19/V20 show gains disappear or are small after compensation.

### Direction B — Calibration-design study for the LS repair

**Name and idea.** Repair-Targeted Multilingual Experimental Design (RT-MED): select a small unlabeled calibration subset that makes the layerwise LS repair informative across language families, rather than adding a new head score.

**Exact math.** For layer repair inputs \(X_S\) and desired outputs \(Y_S\), fit \(W_S=(X_S^\top X_S+\lambda I)^{-1}X_S^\top Y_S\). A D-optimal subset is
\[
S^*=\arg\max_{|S|=m}\log\det\!\left(\lambda I+\sum_{x\in S}x x^\top\right).
\]
A language-stratified variant can maximize a prespecified weighted sum of per-language log determinants subject to quotas. Greedy log-det maximization for a fixed cardinality has the standard monotone-submodular \((1-1/e)\) guarantee **for this log-det design objective only**, not for repaired task accuracy. A worst-language minimum of log determinants does not automatically inherit that guarantee.

**Closest prior work / delta.** (1) ACL 2024 calibration-data impact study, (2) EMNLP Findings 2026 OASIS, which directly optimizes calibration sample selection for pruning, and (3) LREC-COLING 2024 Multilingual Brain Surgeon, which selects a multilingual calibration mixture. Exact delta would be selecting examples for a closed-form post-pruning LS repair in multilingual encoders using a design criterion, but OASIS makes generic sample-selection novelty unlikely. **Novelty: low.**

**Hypothesis and rule.** At a fixed 2,048-row calibration budget, RT-MED lowers worst-language post-repair teacher KL by ≥5% vs random and proportional language-stratified sampling, with a 95% CI excluding zero, and does not lose >1.5 pp accuracy on any language. Kill if lower confidence bound is ≤0, if KL gains do not transfer to task accuracy, or if gain disappears with a fresh calibration/evaluation split.

**Cheapest kill / confirmation.** Use cached V20 activations and repair targets if available; otherwise recompute one layer on 512 calibration rows. Compare D-opt, random, proportional-mix and OASIS-like baseline only if the official implementation is available and compatible. ≤2 T4 hours estimated for the first-layer/single-seed proxy, then up to 8–15 GPU-hours for all layers × 3 seeds on one model. Confirm on untouched language and test splits. Main risks are singular/ill-conditioned activations, expensive subset selection, and proxy mismatch.

**Odds.** Low (10–15%) for a new algorithm paper; moderate for a narrow empirical/analysis contribution if transfer is strong. Do not choose this as the next primary route because direct calibration-selection prior art is current.

### Direction C — Joint vocabulary + structured pruning for an Indic deployment package

**Name and idea.** Compact Indic Encoder Package (CIEP): trim tokenizer/embedding vocabulary for a declared set of Indic languages, then apply structured head/FFN pruning and LS repair, optimizing total footprint and CPU latency.

**Optimization.** Let \(V\) be retained token IDs and \(S\) removed structural units. Minimize
\[
\max_{\ell\in L}\;\Delta_\ell(V,S)
\quad\text{s.t.}\quad
  \operatorname{bytes}(V,S)\le M,\quad
  \operatorname{latency}_{CPU}(V,S)\le T,
\]
while requiring complete coverage for reserved special tokens and a prespecified corpus token-coverage threshold. Use a finite Pareto search over vocabulary quotas and structural budgets; do not claim a global optimum or accuracy guarantee. Tokenizer remapping must preserve every reserved ID and be checked for exact encode/decode behavior.

**Closest prior work / delta.** (1) Ushio et al. EMNLP Findings 2023 vocabulary trimming; (2) Gurgurov et al. IJCNLP-AACL SRW 2025, already combining vocabulary trimming and structured encoder compression; (3) XLM-R/ACL 2022 structured multilingual pruning. Delta: training-free LS repair, physical attention heads, and a shared multi-Indic rather than monolingual model. Since the broad combination already exists, **novelty: very low** unless a new multi-language vocabulary-sharing formulation and measurable deployment contract are developed.

**Hypothesis / kill.** At ≤75% of original bytes, retain all languages within 1.5 pp of the full teacher and improve CPU latency ≥1.2× at batch 1, with paired CI excluding no gain. Kill if token-remapping changes teacher outputs on retained-language inputs, any language has >1.5 pp significant loss, or memory drops without actual latency improvement.

**Cheapest kill / confirmation.** On CPU first, prune vocab on 1k training-only sentences per language and verify identical tokenization on covered text; this requires no GPU. Then evaluate one model and one pruning point on validation, ≤3 GPU hours estimated. Full confirmation requires at least MuRIL and mBERT or IndicBERTv2, all language test sets, full quality–size–latency curves, plus compression baselines; estimate 15–30+ GPU-hours and substantial tokenizer engineering.

**Odds.** Very low (<10%) as an algorithm novelty claim. It could be a useful systems/application paper if the goal is a genuinely smaller multi-Indic model, but the project's retraining-free constraint and the existing 2025 encoder-compression paper make this a poor first choice.

### Direction D — Preregistered multilingual encoder pruning benchmark and mechanism study

**Name and idea.** Indic-Prune: a reproducible controlled study separating selection, repair, language transfer, and physical CPU efficiency for retraining-free structured multilingual encoder pruning.

**Math / analysis.** For each model, language \(\ell\), selector \(s\), repair \(r\), budget \(b\), and seed \(k\), record \(A_{\ell,s,r,b,k}\). Primary contrasts are paired differences in (a) worst-language task accuracy and (b) area under the accuracy-vs-measured-latency frontier. Fit a prespecified mixed-effects model with model/task/language fixed effects and fine-tuning seed as a cluster, or use the project's paired cluster bootstrap + Holm protocol. No new solver or optimization guarantee; the contribution is the controlled evidence and benchmark artifact.

**Closest prior work / delta.** (1) Li et al. ACL 2022 structured multilingual XLM-R pruning; (2) Kurz et al. TACL 2026 calibration-language pruning study; (3) M-Wanda EMNLP 2025. Exact delta: encoder task fine-tuning + physical heads/FFN + K-prune/Kwon-style LS repair + an explicit selector × repair factorial + actual CPU latency, focused on Indic languages. This is defensible as a **bounded benchmark/analysis delta**, not a claim that multilingual pruning itself is new.

**Preregistered test.** Primary hypothesis is descriptive/falsifiable: the selector×repair interaction contributes less to worst-language loss than the repair main effect, while language-specific mask selection may change the worst-group frontier. Set effect thresholds after a power analysis from untouched test sizes; preregister all tasks, masks, exclusions and baselines before test. Kill the strong mechanism claim if the interaction is statistically unresolved or inconsistent across model/task; retain the benchmark with a null result if integrity and power are adequate.

**Cheapest kill / confirmation.** Cheap check ≤3 hours: reanalyze stored V20 per-example predictions for mask-only vs repair in the three prespecified dev languages, but do not reuse them as independent confirmation of a new selection algorithm. Full study: MuRIL plus one additional encoder, all IndicXNLI test languages, perhaps SIB-200 only if power permits, 5 calibration seeds and ≥3 teacher fine-tuning seeds, matched physical latency. Rough cost 20–30+ T4 GPU-hours across two models; schedule over two weeks and preserve the 12-hour job limit.

**Odds.** Highest of the four for a publishable study, but still only **low-to-medium (25–40%)** for a Q1 journal without a clear method or unusually strong evidence. Reviewers may view the ACL 2022 and TACL 2026 studies as already covering the benchmark question. Value depends on genuinely different model/task/protocol coverage, robust baseline parity, and release quality.

## 5. Ranking by (expected publication value × success probability) / cost

These are subjective triage scores, not statistical probabilities. Publication value is 1–5; probability is a rough probability of producing a defensible paper-level result (not acceptance); cost is approximate T4 GPU-hours for a minimum credible study.

| Rank | Direction | Value | Probability | Cost | Heuristic score | Reason |
|---:|---|---:|---:|---:|---:|---|
| 1 | A. Post-repair multilingual risk-controlled mask | 5 | 0.20 | 12 | 0.083 | Best chance of a distinct method effect; cheap kill exists; novelty and remaining headroom are serious risks. |
| 2 | D. Preregistered benchmark/mechanism study | 3 | 0.35 | 20 | 0.053 | Best evidence-per-compute and most honest fallback; can reveal why selection and repair behave differently, but novelty is limited by existing studies. |
| 3 | B. LS calibration experimental design | 3 | 0.12 | 12 | 0.030 | Calibration selection is already crowded by OASIS, TACL, MBS and calibration-data studies. |
| 4 | C. Vocabulary + structured pruning system | 3 | 0.08 | 25 | 0.010 | Strong deployment motivation but close 2025 encoder compression precedent and high tokenizer risk. |

Scores are close for A and D; this is a tie-breaker only. The next week should test A's cheap mask-selection question, while preparing D as the predeclared fallback. Do not begin B or C before deciding whether the primary research question is algorithmic novelty or deployment science.

## 6. Why a paper using current results alone will likely be rejected

1. **No established win over K-prune.** V18/V19 show parity/non-inferiority, not superiority. Cheapest fix: stop claiming a new SOTA method; present the negative evidence honestly and test the distinct target-language mask-selection factor.
2. **The main causal story was falsified.** V20 H20b failed with the wrong sign. Cheapest fix: remove claims that repair causes the language disparity; relabel the mask-side question as untested and preregister it.
3. **Close prior work is missing from the novelty story.** ACL 2022, Shapley heads, M-Wanda, TACL 2026, MBS, and 2025 multilingual encoder compression must be included. Cheapest fix: replace “first” claims with exact protocol-level comparison and run code where compatible.
4. **Insufficient model/training variability.** One fine-tuning teacher does not capture model-training variance; one backbone cannot establish generality. Cheapest fix: multiple fine-tuning seeds for the primary model and a second encoder family, within a realistic budget.
5. **Efficiency evidence is not yet aligned with the real claim.** Same masks mean no method-specific speed win; FLOPs do not establish actual device speed. Cheapest fix: physical slice, CPU batch-1/32 timings, memory, warm-up/repeats, sequence lengths and full quality–latency curves.
6. **Baseline parity gap.** Kwon does not yet load the MuRIL checkpoint in its old environment; current official baseline coverage is incomplete. Cheapest fix: a CPU engineering parity gate before GPU metrics; abort rather than weaken the official code.
7. **Overclaiming Q1/novelty.** A new application plus familiar parts can be judged incremental. Cheapest fix: frame the paper as a narrowly scoped empirical mechanism/deployment study unless the estimator is mathematically and experimentally distinct.

## 7. Strongest honest paper and venue fit

On **current results alone**, I would not submit a Q1 method paper. The strongest honest paper is a **negative/benchmark study**: “What survives retraining-free structured pruning of multilingual encoders? A preregistered selection–repair–latency analysis on Indic NLI.” It should include the V18–V20 negative results only after a human rule-11 audit, corrected literature positioning, full per-language uncertainty, model-seed limitations, official baseline parity, and a reproducible CPU deployment harness. It is not assured Q1; a narrowly scoped version may be more realistic in Q2 unless the new mask-selection result materially changes the deployment frontier.

If a Q1 target is required, **Neurocomputing** is a plausible fit for a method + empirical compression/deployment paper; its 2025 SCImago entries show Q1 in Artificial Intelligence, Cognitive Neuroscience, and Computer Science Applications. [SCImago AI ranking](https://www.scimagojr.com/journalrank.php?category=1702), [journal rank table](https://www.scimagojr.com/journalrank.php?category=2805). Quartiles depend on ranking system, category, and year; verify the institution's required JCR/SJR criterion at submission. TNNLS is a poor target for the current evidence; the supplied October 2026 external review also judged it unrealistic. [Internal review record](../docs/REVIEW_2026-10.md)

## 8. Next-week protocol for the single recommended experiment

1. Complete the independent rule-11 human review of V19 and leave the V19/V20 verdicts unchanged.
2. Write and timestamp a new *exploration* preregistration for language-specific mask selection. Keep the V20 test split untouched; declare which validation languages are used for development and which languages/tests remain confirmatory.
3. First run CPU-only integrity checks for K-prune/Kwon with the local MuRIL config/tokenizer and official code. Any baseline behavior change beyond a documented loading/data adapter stops the study.
4. If parity passes, run the ≤3-GPU-hour Bengali mask-selection kill test from Direction A with existing teacher/repair artifacts. Log all attempts and outputs; never inspect confirmatory test metrics.
5. Proceed only if the prespecified +2 pp worst-language criterion and English guard are met. Otherwise stop the new-method claim and draft Direction D as a descriptive analysis with no Q1 promise.

## 9. Primary sources consulted

- [Li et al., “Probing Structured Pruning on Multilingual Pre-trained Models,” ACL 2022](https://aclanthology.org/2022.acl-long.130/)
- [Kwon et al., “A Fast Post-Training Pruning Framework for Transformers,” arXiv 2204.09656](https://arxiv.org/abs/2204.09656)
- [Park et al., “K-prune,” ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/file/45fc8e1c55df116b23488f3cdb2fc642-Paper-Conference.pdf)
- [Held & Yang, “Shapley Head Pruning,” EACL 2023](https://aclanthology.org/2023.eacl-main.177/)
- [Ushio et al., “Efficient Multilingual Language Model Compression through Vocabulary Trimming,” EMNLP Findings 2023](https://aclanthology.org/2023.findings-emnlp.981/)
- [Zeng et al., “Multilingual Brain Surgeon,” LREC-COLING 2024](https://aclanthology.org/2024.lrec-main.1030/)
- [Williams & Aletras, “On the Impact of Calibration Data in Post-training Quantization and Pruning,” ACL 2024](https://aclanthology.org/2024.acl-long.544/)
- [Choenni & Titov, “M-Wanda,” EMNLP 2025](https://aclanthology.org/2025.emnlp-main.1369/)
- [Gurgurov et al., “On Multilingual Encoder Language Model Compression for Low-Resource Languages,” IJCNLP-AACL SRW 2025](https://aclanthology.org/2025.ijcnlp-srw.5/)
- [Kurz et al., “On the Limitations of Language-targeted Pruning,” TACL 2026](https://aclanthology.org/2026.tacl-1.9/)
- [Ai et al., “Learning What Matters: End-to-End Adaptive Calibration Data Selection for LLM Pruning,” EMNLP Findings 2026](https://aclanthology.org/2026.findings-emnlp.510/)
- [Tucker et al., “Token and Head Adaptive Transformers,” COLING 2022](https://aclanthology.org/2022.coling-1.404/)

## 10. Reviewer check of this audit (Claude, 2026-11-18)

**Citations.** The ACL Anthology record pages for six of the papers above were opened and match the titles and authors given here. The check was abstract/record level only; no full text was read.

| Paper | Authors | What the record shows |
|---|---|---|
| ACL 2022 acl-long.130 | Li et al. | Abstract: “individually pruning for each language does not induce a better result” |
| EACL 2023 main.177 | Held & Yang | Language-specific interfering heads, pruned from a fixed model |
| EMNLP 2025 main.1369 | Choenni & Titov (M-Wanda) | Multilingual LLMs, language-aware activation statistics |
| TACL 2026 tacl-1.9 | Kurz, Chen, Flek, Zhao | Multilingual LLMs. Target-language calibration keeps perplexity but does not consistently help downstream tasks |
| Findings EMNLP 2026 .510 | Ai et al. (OASIS) | Llama/Qwen calibration selection |
| IJCNLP-AACL SRW 2025 .5 | Gurgurov et al. | Compression of multilingual encoders: distillation, structured pruning, vocabulary trimming. Produces monolingual students |

The structured/unstructured and encoder/decoder labels used in sections 3–4 for M-Wanda and Kurz et al. are not stated on the record pages. They are UNVERIFIED until the PDFs are read.

**Correction to Direction A's kill test (confound and ceiling).** The kill test compares an English-selected mask with a Bengali-selected mask under “the same d2i-r repair”. The test only isolates mask selection if the repair calibration language is held at the best known setting. That means own-language or mixed calibration, not English.

V20 already shows that changing the repair calibration language alone moves Bengali from 0.670 to 0.724 (+5.4 pp). If English-calibrated repair is the comparator, a +2 pp “win” would partly re-measure that known calibration effect.

With the correct control, the room left for mask selection is small. Gaps to the teacher after repair, by repair calibration:

| Repair calibration | as | bn | hi | en |
|---|---|---|---|---|
| Own-language | −1.0 pp | −1.7 pp | −1.6 pp | – |
| Mixed | −1.8 pp | −2.5 pp | −2.5 pp | −2.2 pp |

A +2 pp worst-language threshold is therefore at or above the ceiling. Any gain under 2.5 pp is also close to the paired noise on 2,490 examples. In addition, ACL 2022 (Li et al.) reports that per-language pruning does not beat shared pruning on XLM-R, which is prior evidence against Direction A.

**Consequences:**
- If Direction A is run, the comparator is the English-selected mask plus own-language and mixed repair.
- The success criterion must be written relative to the remaining gap, with a power analysis, before any new mask is evaluated.
- The pre-run expected odds are lower than the 10–20% stated in section 4.
- Direction D (the analysis/benchmark paper) is the more likely outcome.
