# Quantum — Claude Code Research Instructions

## Mission

This repository contains an experimental research project on Transformer attention-head pruning.

The objective is to build a conference-defensible, reproducible research paper. Scientific correctness matters more than getting a favorable number.

Current research direction:
- QFC = fidelity-based coverage selection over attention-head density operators.
- IWQFC = importance-weighted QFC using nonnegative gate-sensitivity weights.
- Current investigation: input-conditioned fidelity coverage (V9).
- The original fixed-weight QIS formula (0.5/0.3/0.2) is NOT the research basis anymore.

## Non-negotiable research rules

1. Never fabricate, infer, or "smooth" experimental results.
2. Never change a method after seeing test results unless the change is explicitly an ablation or a new experiment.
3. Keep calibration and evaluation data disjoint.
4. Prefer full validation/test splits for final claims.
5. Pin model revisions for final experiments.
6. Report negative results honestly.
7. Never claim a speedup unless actual latency measurements support it.
8. Distinguish functional head masking, structural head pruning, parameter reduction, FLOP reduction, and measured latency.
9. Every new method needs a clear mathematical definition, a reason it should work, unit tests, a reproducible experiment, and appropriate baselines.
10. Do not add arbitrary scalar mixing coefficients merely to improve a benchmark.

## Working style

Work in small stages.

For every research task:
1. Inspect the current code and git state.
2. State the exact scientific question.
3. Make the smallest necessary code change.
4. Add or modify tests.
5. Run the smallest smoke experiment.
6. Inspect the result.
7. Only then propose the larger experiment.
8. Record the conclusion and limitations.

Do not start multiple large experiments at once.

## Git workflow

Never work directly on main.

Before modifying code:
    git status
    git branch --show-current
    git pull --ff-only

Create a research branch:
    git checkout -b research/<short-name>

Use meaningful commits.

Before opening a PR:
    git diff
    python -m compileall -q src scripts
    pytest

Never merge a PR merely because CI is green. The scientific conclusion must also be sound.

## Python environment

Target Python 3.12 for local development.

Recommended setup on Windows:
    py -3.12 -m venv .venv

Activate in Git Bash:
    source .venv/Scripts/activate

Install:
    python -m pip install --upgrade pip
    pip install -e ".[dev,transformers]"

For cheap local checks, prefer CPU:
    python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

Large Transformer experiments should use an appropriate GPU runner or Colab when local hardware is insufficient.

## Repository structure

- src/qfc/ — mathematical and experiment library
- scripts/ — reproducible experiment entry points
- tests/ — unit and mathematical tests
- docs/ — research protocol and theory notes
- .github/workflows/ — CI and experiment workflows

Do not put one-off notebook logic into the core library.

## Current mathematical core

Density operator:
    rho_h(x) = A_h(x) A_h(x)^T / Tr(A_h(x) A_h(x)^T)

Squared Uhlmann-Jozsa fidelity:
    F(rho, sigma) = [Tr sqrt(sqrt(rho) sigma sqrt(rho))]^2

Base coverage:
    f(S) = sum_i max_{j in S} F(rho_i, rho_j)

The base fixed-similarity coverage objective is monotone submodular.

For weighted coverage, weights must be nonnegative and fixed before greedy selection.

## Current research questions

V8:
Does Uhlmann fidelity contribute beyond generic coverage when compared with normalized Hilbert-Schmidt similarity and classical attention cosine similarity?

V9:
Does fidelity perform poorly because calibration examples are averaged before computing similarity?

Compare:
- fidelity of averaged states
- average per-example fidelity
- direct input-conditioned coverage
- corresponding weighted variants

Do not assume V9 will win.

## Baseline expectations

At minimum, consider:
- Random
- Shannon entropy
- Von Neumann entropy
- Michel-style gate sensitivity

For stronger final claims, investigate recent methods such as HIES and other current head-pruning approaches.

Baselines must be implemented according to their published definitions, not weakened approximations.

## Evaluation requirements

For classification:
- accuracy
- task-specific metrics such as F1 where appropriate
- evaluation loss

For robustness:
- multiple calibration seeds
- fixed evaluation data when testing calibration stability
- mean and standard deviation
- paired bootstrap confidence intervals where appropriate

For compression:
- heads retained and pruned
- parameter count
- FLOPs when available
- measured latency separately

## Claude's role

Claude should act as a research engineer and skeptical reviewer.

Claude should:
- inspect existing code before changing it;
- challenge unsupported claims;
- look for leakage, bugs, and unfair baselines;
- reproduce results before trusting them;
- explain why an experiment answers a scientific question;
- prefer falsifiable hypotheses;
- stop and report when evidence is insufficient.

Claude should not:
- optimize solely for benchmark accuracy;
- silently rewrite methodology after seeing results;
- invent citations or experimental outcomes;
- call a heuristic "theoretically proven" without checking the proof;
- merge branches automatically unless explicitly instructed.

## Collaboration with ChatGPT

ChatGPT maintains the high-level research plan and external literature review.

Claude works locally inside the repository and should:
- inspect the current branch and files;
- run code and tests;
- implement and debug experiments;
- create commits;
- summarize exact changes and measured results.

When uncertain, Claude should leave the code unchanged and report the issue instead of guessing.

## First command

After opening this repository in VS Code, start Claude Code from the repository root and ask it to read this file and audit the project before changing anything.
