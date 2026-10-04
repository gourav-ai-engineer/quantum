# Using Claude Code with the Quantum research repository

## 1. Open the repository

Open the cloned quantum folder itself in VS Code.

The integrated terminal should start in the repository root.

Verify:
    git status
    git remote -v
    git branch --show-current

## 2. Use Git Bash on Windows

Claude Code is supported on Windows through Git for Windows or WSL. Git Bash is a straightforward option.

In VS Code:
Terminal -> New Terminal -> choose Git Bash.

Then:
    cd /path/to/quantum

## 3. Create the Python environment

Use Python 3.12 and the pinned environment (transformers 4.51.3; transformers 5.x silently
ignores head_mask and invalidates every result):
    py -3.12 -m venv .venv-ci

Activate in Git Bash:
    source .venv-ci/Scripts/activate

Install (versions are pinned in requirements-ci.txt; `.[dev,transformers]` pins the same ones):
    python -m pip install --upgrade pip
    pip install -r requirements-ci.txt
    pip install -e . --no-deps

Verify (versions, head_mask actually changes the model; --allow-cpu is for local smoke tests only):
    python --version
    python scripts/preflight.py check --allow-cpu

Full experiments run on a GPU: `bash scripts/run_gpu.sh --commit <hash>` or colab/run_experiments.ipynb.

## 4. Start Claude Code

From the repository root:
    claude

Also run:
    claude doctor
    /memory
    /terminal-setup

The memory command should show the project instructions loaded from CLAUDE.md.

## 5. First Claude prompt

Read CLAUDE.md, README.md, docs/theory.md, pyproject.toml, all files under src/qfc/, scripts/, and tests/.

Do not modify anything yet.

First produce:
1. current architecture,
2. current mathematical formulation,
3. every experiment implemented so far,
4. which claims are actually supported by saved results,
5. known bugs or methodological weaknesses,
6. the exact next scientific experiment you recommend.

Challenge the methodology rather than assuming it is correct.

## 6. Skeptical reviewer audit

Then ask:

Review the entire repository as a skeptical ML conference reviewer. Do not edit files yet. Find correctness bugs, data leakage, unfair baselines, numerical issues, and unsupported claims. Rank findings by severity.

## 7. One task at a time

Use this pattern:

Implement only the smallest change needed for the approved experiment.
First explain the plan.
Then edit the code.
Then add tests.
Then run the smallest smoke test.
Do not start the full experiment until I approve it.

## 8. Safe Git pattern

For every experiment:
    git checkout main
    git pull --ff-only
    git checkout -b research/<name>

Before creating or merging a PR:
    git diff main...HEAD
    pytest
    python -m compileall -q src scripts

## 9. Keep research state in files

Important conclusions should go into version-controlled files such as:
    docs/PROJECT_STATE.md      (results ledger, decision rule, next steps)
    results/<experiment>/<commit>/   (the JSONs the ledger rows are generated from)
    docs/experiment_protocol.md
    docs/theory.md

Every experiment entry should record:
- hypothesis
- model revision
- dataset
- calibration split
- evaluation split
- seed
- pruning budget
- method
- result
- conclusion
- limitations

This prevents future sessions from treating an old smoke result as a final paper result.
