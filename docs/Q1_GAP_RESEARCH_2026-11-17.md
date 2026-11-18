# Q1 gap research 2026-11-17: retraining-free structured compression of multilingual encoders for low-resource Indian languages

> **SUPERSEDED STATUS UPDATE (2026-11-18):** This earlier note's claim that no multilingual structured pruning exists and its H20b repair-disparity hypothesis must not be used as current conclusions. The project ledger records V20 H20b as **NO-GO** (wrong sign); ACL 2022 already studies structured pruning on multilingual XLM-R; Shapley Head Pruning already studies target-language-specific heads; and subsequent papers cover multilingual pruning, calibration, and encoder vocabulary/structure compression. See [RESEARCH_DIRECTION_AUDIT_2026-11-18.md](RESEARCH_DIRECTION_AUDIT_2026-11-18.md) for the updated primary-source audit and narrower candidate.

**Status: CANDIDATE, not a confirmed gap.** It is written after V19 closed (parity, `results/v19/final`). Nothing below is a result. Two blocking checks are still open (section 7): the remaining keyword citation searches and the pilot. No code has been written and no GPU has been used for this.

Source grades: `[full]` means primary text read (HTML/PDF) by me; `[partial]` means part of the primary text was read; `[abstract]` means an abstract or record only; `[summary]` means a search-engine summary only (not trusted; the 2026-11-12 sweep showed summaries can be wrong).

## 1. Executive judgment

The monolingual BERT track is exhausted for a Q1 method claim. V18 and V19 show d2i-RI ties K-prune. Selection-vs-compensation is already claimed for vision (CORP 2602.05243, GRAIL 2602.23795, both `[full]`). Multi-layer reconstruction is occupied (SNOWS, ESWA two-stage). Global task-loss selection is occupied (GISP, ACL 2026).

The angle that survives the checks so far is the **multilingual low-resource encoder**, deep in one domain: Indian and Northeast-Indian languages.
- **The gap.** No study found applies *retraining-free* structured pruning (whole heads + FFN neurons, Kwon/K-prune family, with weight compensation) to *multilingual encoders*. No study separates whether the cross-lingual cost comes from **selection** (which units are removed) or from **compensation** (the least-squares repair fitted on calibration-language activations).
- **The parallel decoder work exists but stops short.** It is unstructured or depth-only, and it never isolates repair: Kurz et al., TACL; M-Wanda; the Indian-language block-influence preprint.
- **What we reuse.** It reuses all V14–V19 machinery: official baselines, integrity gate, paired cluster bootstrap and Holm, held-out-only selection, once-only confirmatory run.

Why this can be Q1 rather than an application note:
- It asks a mechanism question with a falsifiable prediction (section 4, H20b).
- The deployment angle is specific and measurable: in multilingual encoders the embeddings are 52–69% of parameters (section 6), so head/FFN pruning alone barely shrinks the model.
- It serves real users with no labeled data (Assamese, Manipuri, Bodo-region deployment), where retraining is impossible.

This is a bet, not a promise. If H20b fails, the paper becomes a descriptive benchmark, which is weaker (section 8).

## 2. Angles considered and why they were not chosen

| Angle | Closest prior work (grade) | Verdict |
|---|---|---|
| Selection vs compensation factorial, monolingual BERT | CORP `[full]`, GRAIL `[full]` (vision) | Transfer claim only; reviewers will cite CORP/GRAIL. Kept as the *engine* of the new study, not as its novelty. |
| Multi-layer reconstruction for structured BERT | SNOWS `[full]`, ESWA two-stage `[abstract]`, STAT `[full]` | Occupied; needs non-closed-form optimization; invites a "why not fine-tune" reviewer. |
| Global task-loss head/FFN scores | GISP, ACL 2026 `[full]` | Occupied (decoders); a BERT port is a baseline, not a contribution. |
| Training-free token merging + head pruning | PiToMe, NeurIPS 2024 `[summary]` (training-free BERT merging exists) | Combination possibly open, but it is an engineering composition with a weak scientific question. Possible later latency extension. |
| Clinical/biomedical retraining-free pruning | Chalmers MSc thesis (BioBERT prune+INT8) `[summary]`; no peer-reviewed retraining-free clinical study found | Real gap, but main tasks are NER (official baselines support only sequence classification/QA) and clinical data (n2c2, MIMIC) need credentialing. Feasibility poor. |
| Beyond-accuracy effects (loyalty, ECE) | Xu et al. EMNLP 2021 "Beyond preserved accuracy" `[summary]` | Incremental alone. Folded in as secondary metrics (we already store per-example predictions). |
| **Multilingual low-resource encoders (chosen)** | see section 3 | Gap survives citation and keyword checks so far. |

## 3. What exists (closest work) and the exact delta

1. **Kurz et al., "On the Limitations of Language-targeted Pruning", TACL (arXiv 2408.14398)** `[partial]`. Llama-3 8B/70B, Aya-23 8B/35B (decoders). Wanda and SparseGPT, *unstructured* 50%. Seven languages incl. Swahili; no Indic. Findings:
   - Target-language calibration is best only 58.3% (Llama-3) / 45.8% (Aya-23) of the time.
   - Multilingual calibration sometimes helps.
   - Low-activation language-specific FFN neurons are hit hardest.

   They describe SparseGPT's weight update but never compare the calibration sensitivity of SparseGPT and Wanda, nor relate it to reconstruction. Table 2, Belebele-es column only `[partial]`: the min–max spread across seven calibration languages is model-dependent, so the evidence is **mixed** and only motivates the question.
- Llama-3 8B: SparseGPT 23.6 points vs Wanda 9.5.
- Aya-23 8B: SparseGPT 4.3 vs Wanda 5.3. **Delta:** encoders, structured heads+FFN, task-fine-tuned models with zero-shot cross-lingual evaluation, explicit selection × repair factorial, Indic/Northeast languages.
2. **Choenni & Titov, M-Wanda (arXiv 2505.21171)** `[abstract]`. Language-aware activation statistics and layerwise sparsity for multilingual LLMs; one-shot. **Delta:** as in 1. Wanda has no compensation.
3. **Pandhare & Dhoka, "Calibration Language Shapes Block-Influence Pruning" (arXiv 2610.21383, 2026)** `[full]`.
   - Setup: decoders 1.2–2.6B; *depth* pruning; no weight update; en/hi/mr/ta.
   - Measures: perplexity only; no downstream accuracy, no latency.
   - Finding: balanced calibration helps Qwen/Gemma, but Llama is a counterexample.

   **Delta:** structured width pruning with compensation, encoders, downstream task accuracy, Northeast languages, latency.
4. **Shapley Head Pruning (arXiv 2210.05709)** `[full]`.
   - Setup: XLM-R base fine-tuned on English; heads gated off with *no weight update*.
   - Removes 4–9 heads to cut cross-lingual *interference*; +1.59 XNLI average.
   - Shapley cost is days on one GPU; one model, no Indic.

   **Delta:** compression at fixed MAC budgets with compensation, not interference removal.
5. **Gurgurov et al., multilingual encoder compression for low-resource languages (arXiv 2505.16956, IJCNLP SRW 2025)** `[full]`.
   - Setup: mBERT/XLM-R for Maltese, Slovak, Swahili.
   - Pipeline: distillation + FFN-only gradient pruning + hidden truncation + vocabulary trimming. **Training-based.**
   - Attention heads are not pruned: they report that removing three heads drops performance by over 50%. This conflicts with Budhraja 2021 and Shapley pruning; a point our study can settle under retraining-free compensation.

   **Delta:** retraining-free; heads included; calibration language; Indic.
6. **Li et al., "Probing Structured Pruning on Multilingual Pre-trained Models" (ACL 2022, arXiv 2204.02601)** `[partial, extracted PDF]`. XLM-R; training-based (fine-tuning, L0 regularization, dynamic sparsification). **Delta:** retraining-free.
7. **Budhraja et al. 2021 (arXiv 2109.12683)** `[summary]`: mBERT head pruning beyond ~25% hurts XNLI cross-lingual transfer, worse for low-resource and non-SVO languages. **Foroutan et al. EMNLP 2022** `[summary]`: language-neutral subnetworks shrink at high sparsity. Both are training-based or mask-only, with no compensation.
8. **DRPruning (arXiv 2411.14055)** `[full, first 100k chars]`. DRO over domain/language mixture *during training-based* structured pruning + continued pretraining. Its multilingual experiments are on Qwen2 with eight CulturaX languages. **Delta:** retraining-free, closed-form repair, encoders.
9. **Compression hurts low-resource/non-Latin languages more:**
   - Marchisio et al., Findings EMNLP 2024, quantization `[abstract]`.
   - Ogueji et al., EMNLP 2022, sparsity; moderate sparsity can *help* low-resource languages `[summary]`.
   - What do compressed MT models forget, 2205.10828 `[summary]`.

   These are motivation and must be cited, but none is retraining-free structured encoder pruning.
10. **Vocabulary trimming (Ushio et al., Findings EMNLP 2023)** `[summary]`: retraining-free, about 50% vocabulary often enough. Known component; we use it as a deployment step, not as a claim.
11. **Other near work, not covering the gap:**
   - TextPruner (ACL 2022 demo) `[summary]`: training-free *vocabulary* pruning of XLM-R with XNLI results.
   - Ma et al. ACL 2021 (2108.08375) `[summary]`: gradient head ranking in mBERT/XLM-R for POS/NER, interference not compression.
   - "Towards Robust Pruning", EMNLP 2023 `[summary]`: post-training, monolingual BERT only.

**Forward-citation check (Semantic Scholar API, 2026-11-17).**
- **Scope:** all citing papers of Kwon 2022 (261 returned), K-prune 2308.03449 (17), OSSCAR 2403.12983 (22), STAT 2406.00061 (4) and LIAR 2407.13331 (4).
- **Filter:** multilingual / cross-lingual / XLM / mBERT / low-resource / Indic.
- **Result:** **no multilingual follow-up of any of them.** The only keyword hits are generic ("languages").
- **Coverage caveat:** S2 citation coverage is incomplete. Google Scholar was not queried.

**Keyword search (S2, first query only, 106 results):**
- **Only near hit:** "Effects of Attention Head Pruning on Encoder-only Language Models for Multilingual Recipe Classification", SpeD 2025, 5 pages `[record only]`. It is a small application note; read it before any novelty claim.
- **Query "pruning multilingual BERT low-resource languages disparity"** (837 results, top 40 read by title): one pruning hit, "Pruning for Performance" (Konkani, mBERT+BiLSTM, IJCNLP-AACL 2025, arXiv 2506.02005) `[summary]`. It prunes 12 of 144 heads by gradient importance for a single fine-tuned low-resource task. That is not retraining-free compression at a MAC budget, so it is not the gap.
- **Query "head pruning cross-lingual transfer without fine-tuning"** (199 results, titles filtered for prun/compress/sparsi): GRASP LoRA (adapter sparsity) and CLOVER (pruning with fine-tuning). Neither is relevant.
- **Still unrun:** three queries stayed rate-limited after retries (XLM-R calibration language, Indian-language encoder compression, LS reconstruction multilingual). The ACL Anthology/web pass covered the same terms.

## 4. Research questions and candidate hypotheses (NOT frozen; to be pre-registered as V20 before code)

Setting: a multilingual BERT-architecture encoder fine-tuned on English NLI; evaluated zero-shot on Indic languages (IndicXNLI), plus translate-train variants.
- **Pruning:** retraining-free at fixed MAC budget c.
- **Calibration:** unlabeled text only. K-prune's KD objective needs no labels; this is the realistic low-resource case.

- **RQ1 (who pays).** At matched MACs, how is the accuracy loss distributed across languages? Candidate predictors, registered beforehand:
  - pretraining-data share of the language;
  - script (shared with English? with Bengali?);
  - tokenizer fertility.
- **RQ2 (where the cost enters), the mechanism.** Factorial:
  - selection {Kwon, K-prune, ≥30 random at K-prune's per-layer counts};
  - repair {none, layer-local LS, drift-corrected LS (d2i-RI)};
  - calibration language for selection and for repair, set independently, each {en, target, mixed}.
- **H20b (falsifiable).** With English calibration, LS repair raises English accuracy relative to mask-only, but raises target-language accuracy less, or lowers it. Equivalently, the en-minus-target gap after repair exceeds the gap with mask-only. Prediction: repaired with target or mixed calibration, the gap closes, even with selection fixed.
  - If this holds, the bias is introduced by *compensation*, not selection, which is new.
  - If it fails (repair helps all languages equally), the mechanism story dies (section 8).
- **RQ3 (deployment).** One shared compressed model vs language-specialized ones (target-language calibration + retraining-free vocabulary trimming). Report separately:
  - parameter count and memory;
  - attention/FFN MACs;
  - measured CPU latency on physically pruned models (rule 8).
- **Held for later (rule 7):** a worst-language (minimax) least-squares repair. That is a new method variant. It is proposed only if H20b is supported, with written hypothesis, math, tests and its own confirmatory set. Prior art to cite for it: Diana et al. minimax group fairness `[summary]`, DRPruning `[full]`, FairPrune `[summary]`.

## 5. Design sketch (to be frozen in PROJECT_STATE as V20 before any code)

- **Models.** All `model_type: bert`, verified from HF `config.json` 2026-11-17, so the official Kwon/K-prune code runs; only its GLUE data loader needs an IndicXNLI patch, disclosed as V15 patches were.
  - MuRIL-base (vocab 197,285);
  - IndicBERTv2-MLM-only (vocab 250,000);
  - mBERT-cased (vocab 119,547).
  - XLM-R is `xlm-roberta`, so it is dropped unless the official code runs unpatched, mirroring the SQuAD decision.
- **Fine-tuning.** No released fine-tuned checkpoints exist. One pinned recipe per model, checkpoint sha256 recorded, disclosed. Fine-tune re-runs are never called seeds (rule 6). Calibration seeds are the seeds.
- **Tasks.** Sequence classification only. Naamapadam NER and multiple-choice tasks need baseline patches beyond data loading, so they are excluded.
  - IndicXNLI (`Divyanshu/indicxnli`): 11 Indic languages incl. as, bn; validation 2,490 and test 5,010 per language; n-way parallel, so pairing is across languages too.
  - Optional SIB-200 (`Davlan/sib200`): has `asm_Beng`, `mni_Beng`, `lus_Latn`, `sat_Olck`, `ben_Beng`, `hin_Deva`, `eng_Latn`. Only 204 test rows per language; compute the minimum detectable effect before registering, or pool.
- **Calibration text.** Unlabeled, from train splits only (rule 4): IndicXNLI train premises/hypotheses per language, or IndicCorp v2.
- **Splits.** Development: IndicXNLI validation on a declared language subset. Confirmatory: IndicXNLI test, plus languages held out from development (e.g. as, or), untouched until freeze.
- **Statistics.** As V18/V19: paired cluster bootstrap over (calibration seed × example), 10,000 resamples, Holm; per-language and pooled.

## 6. Feasibility facts checked (CPU, 2026-11-17)

- **Embedding share of parameters** (vocab × 768 / total, approximate totals from configs):
  - MuRIL 151.5M of ~237M = 64%;
  - IndicBERTv2 192M of ~278M = 69%;
  - mBERT 91.8M of ~178M = 52%.

  Head/FFN pruning at c = 0.5 removes ~42M encoder parameters, only ~15–24% of the model. This motivates RQ3 and is a point most pruning papers on BERT-English never face.
- **GPU budget.** Fine-tuning a BERT-base on 392k NLI pairs takes about 1–2 h on a T4 (estimate, not measured). Each pruning run is minutes (V19 experience). Fits Kaggle's ~30 GPU-h/week, one large run at a time.

## 7. Blocking checks before telling anyone this is a gap

1. Finish the remaining S2 keyword queries (rate-limited; running) and add a Google Scholar / ACL Anthology pass for "post-training pruning" + XLM-R / mBERT / MuRIL / IndicBERT.
2. Read the SpeD 2025 recipe-classification head-pruning paper.
3. Kurz et al.: Table 2 Belebele-es re-checked (mixed, see section 3.1). Limitations appendix still unread.
- ACL Anthology/web pass done 2026-11-17: no post-training structured (head/neuron) pruning of mBERT/XLM-R/MuRIL/IndicBERT evaluated on XNLI was found. Search engines are not exhaustive.
4. **Pilot (cheap, discriminating; needs owner OK for GPU).**
   - Setup: MuRIL fine-tuned on English MNLI/XNLI; K-prune masks at c 0.5.
   - Contrast: calibration {en, as, mixed} × repair {none, LS}; per-language IndicXNLI *validation* accuracy only.
   - Before it runs: reserve the confirmatory languages/test split in PROJECT_STATE and open branch `research/qfc-v20-multilingual`.
   - How to read it: if English-calibrated repair does not hurt as/bn more than mask-only does, H20b is dead before any confirmatory data is touched.

## 8. Kill criteria and fallback

- **Pilot shows no repair-induced gap.** Drop the mechanism claim. What remains is a descriptive benchmark (who pays, plus the embedding-dominance deployment result). That is publishable but likely Q2.
- **The official baselines cannot run on MuRIL/IndicBERTv2 with a data-loader patch only.** Stop and report; do not rewrite baselines (rule 12).
- **Fine-tuned teachers are much weaker than published numbers** (e.g. MuRIL IndicXNLI in the IndicXNLI paper). Fix the recipe *before* any pruning, never after.

## 9. Venue fit (to verify at submission time)

These are journals where a mechanism + deployment study on low-resource language compression fits: Information Processing & Management; Knowledge-Based Systems; Neurocomputing; Engineering Applications of AI; ACM TALLIP (quartile to verify). Q1 labels are not re-verified here; check SCImago before choosing.
