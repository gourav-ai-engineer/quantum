# Methodology: QFC v0.1

## 1. State construction

Given a nonzero real attention matrix A in R^(N x N), define

rho(A) = A A^T / Tr(A A^T).

This matrix is symmetric positive semidefinite with unit trace. The construction itself is a classical normalized Gram/spectral representation; the manuscript must not claim a quantum-computational advantage from this step.

## 2. Fidelity

We use squared Uhlmann-Jozsa fidelity:

F(rho,sigma) = [Tr sqrt(sqrt(rho) sigma sqrt(rho))]^2.

## 3. Input-conditioned representation

A head is observed on a calibration set x_1,...,x_M, yielding states rho_h(x_m). We retain this collection rather than collapsing a head to one arbitrary scalar.

## 4. Coverage objective

For a selected set S,

C(S) = sum_(m=1)^M sum_(i=1)^H max_(j in S) F(rho_i(x_m),rho_j(x_m)).

The objective is a facility-location/coverage form. For fixed similarities in [0,1], its marginal gain has diminishing returns, so it is monotone submodular. The greedy algorithm is implemented directly from marginal gains.

## 5. What remains to prove/validate

The submodularity property is a statement about the mathematical selection objective. It does not prove that QFC-selected heads preserve task accuracy. That must be established empirically through causal masking, structured pruning, downstream metrics, and confidence intervals.

## 6. Planned theory extension

A later version may derive an explicit bound linking the attention-state distance to the layer-output change. This is not claimed by v0.1 until the full Transformer computation (V projection, output projection, residual connection, and normalization) is handled carefully.
