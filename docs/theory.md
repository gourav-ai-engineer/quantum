# QFC theory and claim boundaries

## Definition

For heads h = 1,...,H let rho_h be valid density operators and let

s(i,j) = F(rho_i, rho_j) in [0,1]

be squared Uhlmann-Jozsa fidelity.

For a retained subset S, define

f(S) = sum_i max_(j in S) s(i,j),

with f(empty) = 0.

## Proposition 1 — monotonicity

For S and S union {h},

f(S union {h}) - f(S)
= sum_i max(0, s(i,h) - max_(j in S) s(i,j)) >= 0.

Therefore f is monotone.

## Proposition 2 — submodularity

Let S subseteq T and h notin T. For every i,

max_(j in S) s(i,j) <= max_(j in T) s(i,j).

Hence

max(0, s(i,h)-max_(j in S)s(i,j))
>=
max(0, s(i,h)-max_(j in T)s(i,j)).

After summing over i,

f(S union {h}) - f(S)
>=
f(T union {h}) - f(T).

Therefore f is submodular.

## Greedy guarantee

For a cardinality constraint |S| <= K, the standard greedy algorithm for monotone submodular maximization has the classical (1 - 1/e) approximation guarantee relative to the optimal subset, assuming the objective and constraints match the theorem.

## Claim boundary

This theorem is about the mathematical coverage objective. It does NOT prove that downstream accuracy or task loss is preserved. Accuracy/loss preservation must be measured empirically on held-out evaluation data.

## Fidelity convention

The implementation uses squared fidelity:

F(rho,sigma) = [Tr sqrt(sqrt(rho) sigma sqrt(rho))]^2.

Any distance inequality used later must use this convention consistently.
