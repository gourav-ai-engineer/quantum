# Thesis narrative: from a quantum-inspired head score to a pre-registered analysis of retraining-free pruning (2026-11-20)

Source of every number: `docs/PROJECT_STATE.md` (ledger), `docs/EXPLORATION_LOG.md`, `docs/STRATEGY_AUDIT_2026-11-19.md`.
Items marked "owner-reported" have no committed result file. V21 is unfinished; no V21 pruned-condition number has been read.

## Chapter outline (suggested)
1. Motivation: making BERT-type encoders smaller without retraining.
2. Phase I, the quantum-inspired selector (QIS -> QFC), and why it failed.
3. Phase II, selection versus compensation (V14) and the theory of redundancy.
4. Phase III, trying to beat the official baselines (V15-V17, exploration), and what the failures revealed.
5. Phase IV, pre-registered confirmation (V18, V19) and real latency.
6. Phase V, the multilingual extension (V20 pilot, V21 confirmatory).
7. Phase VI, method and domain screens (E4-E6) and the ESM-2 pilot.
8. Lessons, limitations and future work.

## Version-by-version record

| Step | Question / hypothesis | What we did | Result | Mistake or weakness found | What we changed next |
|---|---|---|---|---|---|
| Origin | A hand-weighted score QIS = 0.5 VNE + 0.3 QJSD - 0.2 F ranks head importance | Notebook study | Spearman with importance about -0.05 (p = 0.54); "multi-seed" runs identical; independent t-test labelled paired (owner-reported) | Arbitrary weights, no validation of the score, seeds varied nothing, wrong test | Dropped QIS; built a principled objective with a proof |
| QFC, V4-V10 | Fidelity-coverage subset selection (heads as density operators) beats VNE, Michel, Random | Monotone-submodular objective (proved), greedy selection | Not consistently better; fidelity no better than cosine or Hilbert-Schmidt (V8); input-conditioned QFC failed on full validation (V10) (owner-reported). V10 confirmatory MRPC: unpruned 0.8775, IWQFC 0.8235, Michel 0.8015, Random 0.7019 (mean of 30 masks) | Guarantee covers the objective only, not accuracy | Fixed the baselines (V12) and asked whether diversity adds anything (V13) |
| V12 | (bug fix) | Audit of baselines | "Random" kept every head; Shannon was NaN under padding | Unvalidated baselines | All earlier Random/Shannon numbers declared invalid; selectors now assert k distinct heads |
| V13 | Diversity (IWQFC) beats importance (Michel) | Pre-registered paired test | INCONCLUSIVE: MRPC 2/3 seeds CI > 0, SST-2 0/3 | Small, task-dependent effects | Moved from attention patterns to output space |
| V14 | Output-space coverage (BCM) + compensation beats classical selectors | Diagnostics + pre-registered selector x compensation factorial, 5 tasks x 5 seeds | Attention-pattern fidelity is blind to value mixing (proved; 72-97% of near-identical pairs differ in output, 20% on CoLA). Coverage bound ranks subsets (Spearman 0.915-0.955) but about 3x loose. After LS refit no significant difference among BCM, Fisher, Greedy-LS (H9b/c not supported; e.g. MRPC 0.8564 / 0.8583 / 0.8623 vs unpruned 0.8775); attention coverage clearly worse (0.7328) | Selection was a weak lever once a repair is applied (null result at the available power, not equivalence) | Turned to global allocation and repair, compared with the official baselines |
| V15 | Global refit-aware allocation (GRP, heads + FFN) beats Kwon and K-prune on held-out KL (H10) | Reproduced official codes (K-prune MRPC 0.8485 vs published 84.80; SQuAD F1 74.20 vs 74.16), then the gate | Gate failed: SST-2 4/4 budgets (KL 40-75% below K-prune), MRPC 0/4. Accuracy test not run | Engineering losses (disk full, quota, missing packages); task-dependent allocation | Diagnosed why (V16) |
| V16 | Why is GRP task-dependent? | Transplants: K-prune masks with our refit | Our refit was 34-60% worse than K-prune's own tuning; K-prune's strength is a drift-corrected LS target | Our repair was weaker than the baseline's | Gave GRP the drift correction (V17) |
| V17 | GRP + drift-corrected refit (GRP-R) beats K-prune | Development gate | Failed: MRPC 1.19/1.29x K-prune KL, SST-2 0.31/0.50x; K-prune masks + drift refit = 0.97-1.00x K-prune | Hypothesis registered after development data were seen (disclosed) | Exploration of better repairs on dev tasks |
| E0-E2 | Input-side alignment + drift refit (d2i-RI) beats both baselines | Exploration on MRPC, SST-2, STS-B | Met the freeze rule with noise-level margins (MRPC 0.8652 vs 0.8623; SST-2 0.9158 vs 0.9149; STS-B 0.8809 vs 0.8804); held-out KL 5-13% lower | Freeze rule had no minimum effect size | Froze d2i-RI and tested it once on untouched tasks (V18) |
| V18 | d2i-RI beats the better baseline (K-prune) on >= 2 of 3 new tasks | Pre-registered confirmatory: QNLI, MNLI-m, QQP, 5 seeds, 50% MACs, integrity gates | NOT supported. vs K-prune: QNLI -0.0002, MNLI-m +0.0016, QQP +0.0013 (only QQP significant). vs Kwon: +2.8 / +8.1 / +5.6 points. Post-hoc held-out KL ratio 0.958 / 0.994 / 0.952 (development effect did not transfer) | d2i-RI reuses K-prune's masks, so it cannot be smaller or faster | Measured real latency (H18b); computed headroom (E3) |
| H18b | Does 50% MAC pruning make BERT faster? | Physically pruned shapes, CPU and T4 GPU | CPU 1.65x (batch 1), 2.05x (batch 32); GPU no speedup at batch 1, 1.2-1.3x at batch 32 | FLOPs are not latency | Reported as is |
| E3 | Can a per-seed choice between two methods win everywhere? | Existing JSONs only | Fails (MRPC -0.10 points). Headroom over K-prune at 50% MACs is only 0.01-2.43 points on six tasks | The 50% budget leaves little room for any method | Second architecture instead of more tuning (V19) |
| V19 | V18's picture holds on the 6-layer DistilBERT weights | Pre-registered, six GLUE tasks | Superiority NOT supported (1/6, QQP +0.16 points); non-inferiority (margin 1.5 points) SUPPORTED on 6/6; beats Kwon on 6/6 (4 significant). SQuAD dropped before analysis (official K-prune out of memory on a T4) | Same masks again; no speed gain | Paper reframed as an analysis with a quantum-information lens; multilingual question opened |
| V20 | English-calibrated repair hurts target languages more than English (H20b) | MuRIL pilot, XNLI-en + IndicXNLI validation | NO-GO, wrong sign: pooled -0.1357 [-0.1634, -0.1072]. The English-selected mask, not the repair, collapsed target languages (about 0.41 vs 0.685 for English) | Wrong causal story; one teacher | Confirmatory test of the mask-side observation on fresh data (V21) |
| V21 | English-selected masks hurt target languages more than English (H21-1) + secondaries | Frozen; MuRIL and mBERT x 3 teachers, test split | Pending: 4 of 6 units pass integrity; 2 wait for GPU quota | Prior work (Ogueji 2022, Li 2022) means this is a re-examination | Run the single registered analysis when all units pass |
| E4 | Is there a new pruning method left in this setting? | Seven directions screened against primary sources | Six occupied (e.g. ZipLM one-shot BERT, ReplaceMe, Pareto Testing, Free Lunch, Tyr-the-Pruner); one killed by an arithmetic check | Novelty, not compute, is the limit in this setting | Search other domains (E5) |
| E5 | Which domain lacks retraining-free structured pruning with repair? | Protein, genomic, time-series, speech screens | Protein LMs (ESM-2) least crowded; closest work (Vinod et al. 2026) uses learned unstructured masks for interpretability | "Not found" is not "proven new"; earlier time estimates were optimistic | Pre-registered ESM-2 pilot (E6) |
| E6 | Does a pruned ESM-2 150M at the cost of ESM-2 35M beat the dense 35M on zero-shot variant effects? | Design fixed before code (K1 repair matters, K2 practical value) | Not run yet (CLI handoff `docs/CLI_ESM_PILOT.md`) | Expected to fail K2 (severe 77.5% cut without retraining) | Run; stop the direction if K1 or K2 fails |

## What the thesis can claim (supported by the record)
- Attention-pattern similarity is a provably blind redundancy signal; output-space fidelity (with the purified-distance identity) ranks merge subsets well.
- After closed-form repair, no selector difference was detected at 50% heads per layer; repair carries the recovery.
- K-prune's advantage comes from its drift-corrected least-squares target (identified by transplants; already in K-prune's own equations).
- A strong combination (d2i-RI) is non-inferior to K-prune on two architectures and better than Kwon et al., but not superior to K-prune.
- At 50% MACs the remaining headroom over K-prune is 0.01-2.43 points; head/FFN pruning gives real CPU speedups but no GPU batch-1 speedup.
- Multilingual (pending V21): the disparity enters through mask selection, not repair (V20 pilot, descriptive).

## Lessons (the methodological contribution of the thesis)
1. Validate a score before using it (QIS).
2. Validate baselines (V12).
3. Measure headroom before designing a method (E3, too late in this project).
4. A tie or non-inferiority is not a win; reusing a baseline's masks cannot be a compression win.
5. Check prior art before building, not after (STAT, SNOWS, ZipLM).
6. Pre-registration, integrity gates and single-use test data made every negative result trustworthy.

## Open before the thesis is complete
V21 analysis; ESM-2 pilot verdict; Rule 11 independent audit; supervisor proof check.
