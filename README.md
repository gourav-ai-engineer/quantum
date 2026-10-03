# Quantum-Fidelity Coverage for Transformer Attention Pruning

Research repository for the revised attention-head pruning study.

## Current hypothesis

Instead of assigning each attention head an arbitrary weighted importance score, we represent each head as an input-conditioned density operator and formulate pruning as a representative-subset problem.

For head h and input x:

rho_h(x) = A_h(x) A_h(x)^T / Tr(A_h(x) A_h(x)^T)

For retained set S, the current coverage objective is

C(S) = sum_x sum_i max_{j in S} F(rho_i(x), rho_j(x))

where F is squared Uhlmann-Jozsa fidelity.

The greedy selector is implemented in src/qfc/coverage.py. The mathematical claim to be tested is that this fixed-similarity facility-location objective is monotone submodular; the downstream accuracy/loss preservation claim remains empirical and must not be assumed.

## Repository plan

1. Mathematical core and unit tests — implemented in this first commit.
2. Attention extraction from BERT/RoBERTa on a calibration set.
3. Exact causal head-masking baseline and corrected Michel-style gate sensitivity.
4. Structured head surgery with real parameter/FLOP accounting.
5. Experimental runner: SST-2, MRPC, then additional tasks.
6. Statistical analysis and paper-ready tables/figures.

## Research constraints

- No arbitrary 0.5/0.3/0.2 weights.
- No claim of quantum hardware speedup.
- Do not call zeroed projections structural compression.
- Validate selected subsets by downstream loss/accuracy and model-computation metrics.
- Compare against contemporary pruning baselines before making novelty claims.
