#!/usr/bin/env bash
# Run the pending GPU experiments in order (V10, V6, V5, V11) and record the results.
#
# Usage:
#   scripts/run_gpu.sh [--repo URL --commit REF] [--workdir DIR] [--results-root DIR]
#                      [--venv DIR] [--only step18,step14,step13,step19]
#                      [--step19-eval-size N] [--tag TAG] [--force]
#                      [--smoke [--allow-cpu]]
#
#   --repo URL --commit REF   clone URL into --workdir (default ./qfc_repo) and check out REF.
#                             Without --repo the current checkout is used (HEAD must match
#                             --commit if given; the tree is never switched).
#   --results-root DIR        default <repo>/results; use the mounted Drive path in Colab.
#                             Output: DIR/<experiment>/<commit>[-TAG]/ with the JSONs,
#                             run_meta.json (commit, date, args, environment) and a log named
#                             log_<experiment>_<commit>_<YYYYMMDD>.txt. Existing non-empty
#                             run directories are never overwritten unless --force.
#   --venv DIR                reuse an existing environment (no install); default creates
#                             <repo>/.venv-gpu from requirements-ci.txt.
#   --step19-eval-size N      evaluation examples for step19 (default 256; -1 = full
#                             validation). Values below 256 are refused outside --smoke.
#   --smoke                   tiny sizes for an engineering check; writes under
#                             <results-root>/_smoke and is never a result. --allow-cpu is only
#                             accepted together with --smoke.
#
# Experiments (fixed protocol, see docs/PROJECT_STATE.md):
#   step18  V10 confirmatory: sst2+mrpc, seeds 7,42,77, full validation, 30 random masks, k=6
#   step14  V6 budget response: budgets 3,6,9, full validation, both tasks, 30 random masks
#   step13  V5 MRPC stability: seeds 7,42,77, full validation, 30 random masks, k=6
#   step19  V11 objective alignment: 300 subsets per layer, >=256 evaluation examples
set -euo pipefail

usage() { sed -n '2,32p' "$0"; exit "${1:-0}"; }

REPO_URL=""; COMMIT=""; WORKDIR=""; RESULTS_ROOT=""; VENV=""; TAG=""
ONLY="step18,step14,step13,step19"; STEP19_EVAL=256
SMOKE=0; ALLOW_CPU=0; FORCE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo) REPO_URL="$2"; shift 2;;
    --commit) COMMIT="$2"; shift 2;;
    --workdir) WORKDIR="$2"; shift 2;;
    --results-root) RESULTS_ROOT="$2"; shift 2;;
    --venv) VENV="$2"; shift 2;;
    --only) ONLY="$2"; shift 2;;
    --step19-eval-size) STEP19_EVAL="$2"; shift 2;;
    --tag) TAG="$2"; shift 2;;
    --smoke) SMOKE=1; shift;;
    --allow-cpu) ALLOW_CPU=1; shift;;
    --force) FORCE=1; shift;;
    -h|--help) usage 0;;
    *) echo "unknown argument: $1" >&2; usage 2;;
  esac
done

if [[ $ALLOW_CPU -eq 1 && $SMOKE -eq 0 ]]; then
  echo "--allow-cpu is only valid together with --smoke (full runs require a GPU)" >&2; exit 2
fi
if [[ $SMOKE -eq 0 && $STEP19_EVAL -ne -1 && $STEP19_EVAL -lt 256 ]]; then
  echo "--step19-eval-size must be >= 256 (or -1 for full validation)" >&2; exit 2
fi

# ---- 1. repository at the requested commit ---------------------------------------------
if [[ -n "$REPO_URL" ]]; then
  [[ -n "$COMMIT" ]] || { echo "--commit is required with --repo" >&2; exit 2; }
  WORKDIR="${WORKDIR:-qfc_repo}"
  [[ -d "$WORKDIR/.git" ]] || git clone "$REPO_URL" "$WORKDIR"
  cd "$WORKDIR"
  git fetch --all --tags --prune
  git checkout --detach "$COMMIT"
else
  cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  if [[ -n "$COMMIT" && "$(git rev-parse HEAD)" != "$(git rev-parse "${COMMIT}^{commit}")" ]]; then
    echo "checked-out HEAD is not $COMMIT; refusing to switch your working tree" >&2; exit 2
  fi
fi
REPO_ROOT="$(pwd)"
COMMIT_SHORT="$(git rev-parse --short=12 HEAD)"
DATE_UTC="$(date -u +%Y%m%d)"
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "WARNING: working tree has uncommitted changes; run_meta.json will record dirty=true" >&2
fi

# ---- 2. pinned environment -------------------------------------------------------------
if [[ -n "$VENV" ]]; then
  VENV_DIR="$VENV"
else
  VENV_DIR="$REPO_ROOT/.venv-gpu"
  PYBOOT="$(command -v python3 || command -v python)"
  [[ -d "$VENV_DIR" ]] || "$PYBOOT" -m venv "$VENV_DIR"
fi
if [[ -x "$VENV_DIR/bin/python" ]]; then PY="$VENV_DIR/bin/python"; else PY="$VENV_DIR/Scripts/python.exe"; fi
if [[ -z "$VENV" ]]; then
  "$PY" -m pip install --quiet --upgrade pip
  "$PY" -m pip install --quiet -r requirements-ci.txt
  "$PY" -m pip install --quiet -e . --no-deps
fi

# ---- 3. pre-flight: versions, CUDA, head_mask effective --------------------------------
PREFLIGHT_FLAGS=""; [[ $ALLOW_CPU -eq 1 ]] && PREFLIGHT_FLAGS="--allow-cpu"
"$PY" scripts/preflight.py check $PREFLIGHT_FLAGS

# ---- 4. result directories -------------------------------------------------------------
ROOT="${RESULTS_ROOT:-$REPO_ROOT/results}"
[[ $SMOKE -eq 1 ]] && ROOT="$ROOT/_smoke"
SMOKE_FLAG=""; [[ $SMOKE -eq 1 ]] && SMOKE_FLAG="--smoke"

exp_name() {
  case "$1" in
    step18) echo v10_confirmatory;;
    step14) echo v6_budget_response;;
    step13) echo v5_mrpc_stability;;
    step19) echo v11_objective_alignment;;
    *) echo "unknown step: $1" >&2; exit 2;;
  esac
}
run_dir() { echo "$ROOT/$(exp_name "$1")/${COMMIT_SHORT}${TAG:+-$TAG}"; }

IFS=',' read -r -a STEPS <<< "$ONLY"
for s in "${STEPS[@]}"; do
  d="$(run_dir "$s")"
  if [[ -d "$d" && -n "$(ls -A "$d" 2>/dev/null)" && $FORCE -eq 0 ]]; then
    echo "refusing to overwrite non-empty $d (use --tag or --force)" >&2; exit 2
  fi
done

# ---- 5. protocol (full) or tiny sizes (--smoke) ----------------------------------------
if [[ $SMOKE -eq 1 ]]; then
  S18_SEEDS="7"; S18_FLAGS="--calibration-size 16 --evaluation-size 32 --heads-to-keep 6 --batch-size 8 --max-length 32 --bootstrap 20 --random-masks 3"
  S14_FLAGS="--calibration-size 16 --calibration-seed 42 --budgets 6 --batch-size 16 --max-length 32 --random-masks 2"
  S13_FLAGS="--calibration-size 16 --evaluation-size 32 --heads-to-keep 6 --seeds 7 --batch-size 8 --max-length 32 --bootstrap 20 --random-masks 2"
  S19_FLAGS="--calibration-size 16 --calibration-seed 42 --evaluation-size 32 --heads-to-keep 6 --random-subsets 4 --bootstrap 20 --batch-size 8 --max-length 32"
else
  S18_SEEDS="7 42 77"; S18_FLAGS="--calibration-size 128 --heads-to-keep 6 --batch-size 16 --max-length 128 --bootstrap 1000 --random-masks 30"
  S14_FLAGS="--calibration-size 256 --calibration-seed 42 --budgets 3,6,9 --batch-size 16 --max-length 128 --random-masks 30"
  S13_FLAGS="--calibration-size 128 --evaluation-size -1 --heads-to-keep 6 --seeds 7,42,77 --batch-size 16 --max-length 128 --bootstrap 1000 --random-masks 30"
  S19_FLAGS="--calibration-size 128 --calibration-seed 42 --evaluation-size $STEP19_EVAL --heads-to-keep 6 --random-subsets 300 --bootstrap 1000 --batch-size 16 --max-length 64"
fi

write_meta() {  # step, flags
  "$PY" scripts/preflight.py meta --out "$(run_dir "$1")/run_meta.json" --experiment "$(exp_name "$1")" \
    $SMOKE_FLAG --args-json "{\"script\": \"$1\", \"flags\": \"$2\"}"
}

run_step() {
  local step="$1" d name
  d="$(run_dir "$step")"; name="$(exp_name "$step")"
  mkdir -p "$d"
  echo "=== $step ($name) -> $d"
  {
    case "$step" in
      step18)
        for task in sst2 mrpc; do
          for seed in $S18_SEEDS; do
            PYTHONPATH=scripts "$PY" scripts/step18_v10_confirmatory.py --task "$task" --seed "$seed" $S18_FLAGS --output-dir "$d"
          done
        done
        "$PY" scripts/aggregate_v10_confirmatory.py "$d"
        write_meta "$step" "tasks=sst2,mrpc seeds=${S18_SEEDS// /,} $S18_FLAGS";;
      step14)
        "$PY" scripts/step14_budget_response.py $S14_FLAGS --output-dir "$d"
        write_meta "$step" "$S14_FLAGS";;
      step13)
        "$PY" scripts/step13_mrpc_stability.py $S13_FLAGS --output-dir "$d"
        write_meta "$step" "$S13_FLAGS";;
      step19)
        "$PY" scripts/step19_objective_alignment.py $S19_FLAGS --output-dir "$d"
        write_meta "$step" "$S19_FLAGS";;
    esac
  } 2>&1 | tee "$d/log_${name}_${COMMIT_SHORT}_${DATE_UTC}.txt"
  # Results are committed to git: keep them small.
  find "$d" -name '*.json' -size +5M -print | sed 's/^/WARNING: large JSON (keep results small): /' >&2 || true
}

for s in "${STEPS[@]}"; do run_step "$s"; done

echo
echo "Done. Ledger rows (copy into docs/PROJECT_STATE.md; no hand-typed numbers):"
for s in "${STEPS[@]}"; do echo "  $PY scripts/make_ledger_row.py $(run_dir "$s")"; done
