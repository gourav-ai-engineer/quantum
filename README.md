# Quantum-Fidelity Coverage for Transformer Attention Pruning

Research repository for the revised attention-head pruning study.

## Current hypothesis

Instead of assigning each attention head an arbitrary weighted importance score, we represent each head through a calibration-derived density operator and formulate pruning as a representative-subset problem.

For head h and input x:

rho_h(x) = A_h(x) A_h(x)^T / Tr(A_h(x) A_h(x)^T)

For a retained set S, the implemented coverage objective is

C(S) = sum_i max_{j in S} F(rho_bar_i, rho_bar_j)

where rho_bar_h is the calibration mean state for head h and F is squared Uhlmann-Jozsa fidelity.

The current greedy selector is a facility-location-style coverage optimizer. For fixed similarities in [0, 1], the objective has the standard monotone submodular diminishing-returns structure.

## Verified so far

- Core mathematical unit tests pass in GitHub Actions.
- Source compilation passes in GitHub Actions.
- A real BERT-base SST-2 smoke run completed successfully after fixing package discovery.
- At 25% head pruning (9/12 heads kept per layer) on a tiny 32-example engineering evaluation split, QFC achieved 0.96875 accuracy vs 0.9375 for the unpruned model (+3.125 percentage points) and lower evaluation loss (0.14176 vs 0.20824). Von Neumann entropy also reached 0.96875 on this same tiny split.
- A separate 50% head-pruning smoke run on 32 examples produced 0.90625 vs 0.9375 for the full model (-3.125 points).

These smoke results are engineering evidence only. They are not paper claims and do not establish superiority.

## Repository plan

1. Mathematical core and unit tests.
2. Real BERT/RoBERTa attention extraction and calibration.
3. Functional comparison: QFC, Shannon, VNE, Michel-style gate sensitivity, Random.
4. Structured head surgery with exact parameter/FLOP accounting.
5. Multi-split stability on the full validation sets.
6. Contemporary baselines including HIES and recent complementary/global pruning methods.
7. Statistical analysis and paper-ready tables/figures.

## Research constraints

- No arbitrary 0.5/0.3/0.2 weights.
- No claim of quantum hardware speedup.
- Do not call zeroed projections structural compression.
- Accuracy preservation/improvement must be demonstrated independently on full validation data.
- Use calibration/evaluation separation to avoid selection leakage.
- Correctly reproduce baselines before comparing against them.
- Report parameter/FLOP reduction and measured latency separately.
