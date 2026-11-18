# Prompt for an outside research assistant (ChatGPT), written 2026-11-18

Copy everything below the line into ChatGPT (a reasoning/"deep research" model with web access works best). It is self-contained. Personal details are left out on purpose.

---

You are a senior ML researcher and a hostile, honest reviewer. Help me find a research direction or a NEW algorithm that can support a Q1-journal paper on **retraining-free (post-training) structured pruning of transformer encoders**. Below is the full state of the project: what we tried, the exact numbers, and what failed. Read all of it before answering.

## 0. Rules you must follow
1. **Honesty over favourable numbers.** Never invent a paper, a citation, a number, a benchmark result or a venue fact. Label every factual claim **VERIFIED** (you read the primary source; give arXiv id or DOI), **PLAUSIBLE** (secondary source) or **UNVERIFIED**.
2. **No "quantum advantage".** All computation is classical. Quantum information was used only as a mathematical representation.
3. **Pre-registration discipline.** We never tune a method after seeing test results. Every idea you propose must come with a falsifiable hypothesis, a decision rule, a failure condition and fair baselines. The decision rule must be written before any data.
4. **Fair baselines.** Official code of the strongest published method, never a weakened re-implementation.
5. **Feasible compute only.** One free Kaggle T4 GPU (16 GB), about 30 GPU-hours per week, 12-hour job cap. The software is pinned: Python 3.12, torch 2.6.0, transformers 4.51.3. The official baseline codes use transformers 4.12.
6. **Novelty check every idea.** For each idea, name the 3 closest existing papers and state the exact delta. If the idea already exists, say so plainly. "Pruning as subset selection", "combining importance scores", "layer-wise least-squares reconstruction" and "calibration-data choice matters" are NOT novel by themselves.

## 1. The problem in one paragraph
We remove whole attention heads and whole FFN neurons from a fine-tuned encoder (BERT-base, DistilBERT, MuRIL), so the matrices physically shrink. We use only about 2,048 unlabeled calibration rows from the training split and a closed-form least-squares (LS) weight repair, with no fine-tuning. The budget is usually 50% of the MACs. The strongest official baseline is **K-prune** (Park et al., ICLR 2024). It does a knowledge-preserving mask search, then a per-sublayer LS weight tuning whose target is drift-corrected. The older baseline is **Kwon et al.** (NeurIPS 2022): Fisher-based masks, mask rearrangement and LS mask tuning.

## 2. History: what we tried, and the exact outcome
- **QIS (dropped).** A hand-weighted score 0.5*VNE + 0.3*QJSD − 0.2*Fidelity over per-head density operators rho = A A^T / tr(A A^T), where A is the attention matrix. The weights were arbitrary. It did not correlate with head importance (Spearman about −0.05, p = 0.54). Do not revive it.
- **QFC.** Choose a head subset S that maximises C(S) = sum_i max_{j in S} F(rho_i, rho_j), with F the squared Uhlmann fidelity. C is monotone submodular, so greedy has a (1−1/e) guarantee on the objective only. It was not consistently better than Michel gate sensitivity, von Neumann entropy or Random. Fidelity did not beat cosine or Hilbert–Schmidt similarity.
- **Theory we proved** (not yet independently checked):
  - Attention-pattern density operators are blind to value mixing.
  - The best scaled-orthogonal Procrustes residual between two heads is ||Z_j||_F * sqrt(1 − F_ij), using output-space fidelity of the head context vectors Z.
  - A merge-error bound B(S) = sum_{j not in S} w_j min_{i in S} sqrt(1 − F_ji) is a weighted facility-location function.
  - Measured: 20–97% of head pairs with near-identical attention (fidelity > 0.99) have output fidelity < 0.5, depending on the task.
- **V14.** After an LS refit, four selectors (output coverage, Fisher, magnitude, greedy LS) are statistically indistinguishable. Compensation, not selection, carries the recovery.
- **V15–V17.**
  - Our global refit-aware allocation (GRP) failed its gate against K-prune.
  - We found that K-prune's LS target is drift-corrected: teacher sublayer output plus (teacher residual input − student residual input).
  - K-prune masks plus that refit reproduce K-prune to within 0.97–1.00x.
- **d2i-RI** = K-prune masks + input-side alignment (ridge LS of the kept q/k/v and FFN-up rows) + drift-corrected output refits. It was developed on MRPC, SST-2 and STS-B.
- **V18 (pre-registered confirmatory, BERT-base, 50% MACs, QNLI/MNLI-m/QQP, 5 seeds):**
  - Superiority over K-prune was NOT supported.
  - d2i-RI minus K-prune: QNLI −0.02 pp, MNLI-m +0.16 pp, QQP +0.13 pp. Only QQP is significant.
  - d2i-RI beats Kwon et al. by +2.8 / +8.1 / +5.6 pp, all Holm-significant.
- **Headroom.** Unpruned minus K-prune at 50% MACs is only 0.01–2.43 pp across six GLUE tasks. At this budget the task bounds the possible gain, not the method.
- **V19 (pre-registered, frozen d2i-RI on 6-layer DistilBERT-format weights, 6 GLUE tasks, 5 seeds):**
  - Superiority over K-prune NOT supported: 1 of 6 tasks won, QQP +0.16 pp.
  - Non-inferiority within 1.5 pp of K-prune SUPPORTED on 6 of 6.
  - d2i-RI beats Kwon et al. on 4 of 6 (Holm-significant).
  - Conclusion: a parity result.
- **Latency** (physically pruned shapes, 50% MACs): CPU 1.4–2.05x faster, T4 GPU about 1.0x at batch 1 and 1.2–1.3x at batch 32. d2i-RI keeps K-prune's masks, so it has no speed advantage of its own.

## 3. The newest direction (V20) and its pilot result
- **Idea.** Retraining-free structured pruning of MULTILINGUAL encoders for low-resource Indian languages: MuRIL, IndicBERTv2, mBERT; Assamese, Bengali, Hindi and others; IndicXNLI.
- **Literature gap.**
  - Multilingual pruning work is mostly on decoders, unstructured, or perplexity only. Examples: Kurz et al. TACL 2024 (arXiv 2408.14398), M-Wanda, depth pruning (arXiv 2610.21383).
  - Head-pruning work on XLM-R uses no weight update (Shapley head pruning, arXiv 2210.05709).
  - We found no paper applying K-prune/Kwon-family retraining-free structured pruning with weight compensation to multilingual encoders on Indic languages. (UNVERIFIED as exhaustive.)
- **Model facts.** Embeddings are 52–69% of the parameters in these models (MuRIL 64%, IndicBERTv2 69%, mBERT 52%). Head and FFN pruning at 50% MACs therefore shrinks only about 15–24% of the model.
- **Hypothesis tested (H20b).** English-calibrated LS repair creates the cross-lingual accuracy disparity, so target-language or mixed calibration would close it.
- **Pilot setup.**
  - Model: MuRIL fine-tuned once on English MNLI (2 epochs).
  - Masks: official K-prune, selected on ENGLISH calibration, 50% MACs, seeds 7/42/77.
  - Factors: repair in {none, d2i_r (drift-corrected refit), d2i_ri} × calibration language in {en, as, bn, hi, mixed}.
  - Calibration rows are content-matched (the same MNLI rows, translated).
  - Evaluation: XNLI-en plus IndicXNLI validation, 2,490 parallel examples.
  - Statistics: paired cluster bootstrap.
- **Verdict: NO-GO.** P1 = [R_en(en) − M(en)] − [R_en(T) − M(T)], pooled over as/bn/hi, was −0.136, 95% CI [−0.163, −0.107]. The sign is wrong: repair helps the target languages MORE than English.

Mean accuracy over 3 seeds:

| setting | en | as | bn | hi |
|---|---|---|---|---|
| unpruned teacher | 0.835 | 0.686 | 0.741 | 0.753 |
| mask only (English-selected K-prune masks) | 0.685 | 0.409 | 0.410 | 0.410 |
| d2i_r, English calibration | 0.824 | 0.645 | 0.670 | 0.739 |
| d2i_r, own-language calibration | – | 0.676 | 0.724 | 0.737 |
| d2i_r, mixed (en/as/bn/hi) calibration | 0.813 | 0.668 | 0.716 | 0.728 |
| d2i_ri, English calibration | 0.829 | 0.616 | 0.717 | 0.734 |

**Reading.**
- The English-selected mask, before repair, collapses the target languages to near chance (0.41) while English keeps 0.685. The disparity enters through MASK SELECTION, not through repair.
- After repair the remaining gap is small: bn −7.1 pp, as −4.1 pp, hi −1.4 pp, en −1.1 pp.
- Mixed calibration cuts the target gaps to about 2–3 pp, at a cost of about 1 pp on English.
- Never tested: target-language or mixed-language mask SELECTION.
- Reserved and untouched: IndicXNLI and XNLI test splits, SIB-200, IndicBERTv2, mBERT.

## 4. What I need from you
Give me 3 to 5 research directions or NEW algorithms. Each should be able to carry a Q1 journal paper (for example Neurocomputing, Neural Networks, IEEE TNNLS, Information Sciences, Expert Systems with Applications, or IEEE/ACM TASLP for the multilingual angle). Think beyond what we tried. Borrow mechanisms from other fields if they fit, for example:
- optimal transport
- optimal experimental design / D-optimality for calibration selection
- low-rank and tensor-network compression
- vocabulary/embedding trimming
- Fisher/Hessian geometry
- multi-objective / minimax-fairness subset selection
- distributionally robust least squares
- token pruning combined with head pruning
- quantization combined with pruning
- model merging
- language-family structure.

Possible angles (critique them, don't just accept them):
- (a) Language-robust mask selection: a minimax or worst-language objective over calibration languages, solved retraining-free.
- (b) Multilingual calibration-set design for LS repair, with theory (e.g. which rows minimise worst-language reconstruction error).
- (c) Joint vocabulary trimming + head/FFN pruning, so the 52–69% embedding share is finally addressed, with real CPU latency and memory numbers.
- (d) High-compression regime (keep 20–30% of MACs), where headroom is large.
- (e) Anything better you can find.

For EACH proposal, give:
1. **Name and one-sentence idea.**
2. **Exact math:** objective, constraints and the closed-form or greedy solver. Say if any guarantee exists (and on what: the objective only, never accuracy).
3. **Why it should work, given OUR numbers above.** Point to the specific table cells it would change and by how much would be needed for a publishable effect.
4. **Novelty check:** the 3 closest papers (VERIFIED/UNVERIFIED) and the exact delta.
5. **Pre-registrable hypothesis:** primary statistic, baseline(s), decision rule (CI threshold, minimum effect), and the result that would KILL the idea.
6. **Cheapest kill-test** first (≤ 3 GPU-hours on a T4, development/validation data only), then the confirmatory design (held-out tasks/languages/models, seeds, test splits).
7. **Compute estimate** in T4 GPU-hours, and the main engineering risk.
8. **Honest odds** (low/medium/high) that it beats the official baseline by a margin a reviewer would accept, and why.

Then:
- Rank the proposals by (expected publishable value × probability of success) ÷ cost.
- Name the single one you would run next week.
- List the top 5 reasons a Q1 reviewer would reject a paper built from our CURRENT results alone, with the cheapest fix for each.
- If you think no Q1-grade win is realistic within these constraints, say so directly and propose the strongest honest paper we can write instead (for example an analysis, benchmark or negative-result paper), with the venue.

Keep the answer structured with headings. No filler.
