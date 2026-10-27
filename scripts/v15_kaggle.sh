#!/usr/bin/env bash
# V15 driver for Kaggle (GPU). Stages (docs/PROJECT_STATE.md, "Pre-registration V15"):
#   setup  : our pinned env (system python) + official Kwon/K-prune codes in their own uv envs, patched (v15 add-on),
#            K-prune's BERT-base checkpoints (amendment A1).
#   phase1 : reproduction gates. Kwon (no V15_INDEX_FILE => identical to unpatched) on K-prune's MRPC/SST-2
#            checkpoints at 0.7/0.6/0.5; K-prune at --constraint 0.8 (= remove 80%) on MRPC and SQuAD1.1. Seeds 0-4.
#   h10    : calibration-side gate. Our 2048 calibration rows and pinned textattack checkpoints for every method:
#            Kwon (keep c), K-prune (--constraint 1-c = remove 1-c), GRP (step22 --mode kl). MRPC/SST-2, 5 seeds,
#            c in 0.7/0.6/0.5/0.4. Logits on held-out TRAIN rows (and validation, unused by H10) are dumped.
# Usage: bash scripts/v15_kaggle.sh <stage> [work_dir] [out_dir]
# Work happens in /tmp (envs, checkpoints and K-prune's per-run model.ckpt overflowed Kaggle's 20 GB /kaggle/working
# in Version 3); each stage ends by copying its small results (logs, masks, logit dumps, JSONs) to out_dir.
set -euo pipefail
STAGE=$1
W=${2:-/tmp/v15}
OUT=${3:-/kaggle/working/v15}
REPO=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$W" && cd "$W"
KWON_SHA=806ac5a6ff53b4978e5330d45a3de692493e4d0b
KPRUNE_SHA=9ca629a1db1ebd835234ade03a4c0ce8fa522d7b
KP_OUT=$W/kprune/src/outputs/bert-base-uncased
SEEDS="7 42 77 123 2024"
CONS="0.7 0.6 0.5 0.4"

kwon()   { (cd "$W/kwon" && "$W/env_kwon/bin/python" main.py --model_name bert-base-uncased "$@"); }
publish() { mkdir -p "$OUT"; for d in "$@"; do [ -e "$W/$d" ] && { mkdir -p "$OUT/$(dirname "$d")"; cp -r "$W/$d" "$OUT/$(dirname "$d")/"; }; done; du -sh "$OUT"; }
kprune() { (cd "$W/kprune/src" && "$W/env_kprune/bin/python" main.py --model_name bert-base-uncased --gpu 0 \
            --lam_pred 1. --lam_rep 2.5e-4 --T 2 --mu 64 --sublayerwise_tuning "$@"); }

case "$STAGE" in
setup)
  pip install -q -r "$REPO/requirements-ci.txt" && pip install -q -e "$REPO" --no-deps
  pip uninstall -y -q torchvision torchaudio || true
  python "$REPO/scripts/preflight.py" check
  pip install -q -U uv gdown
  [ -d kwon ] || { git clone -q https://github.com/WoosukKwon/retraining-free-pruning kwon && git -C kwon checkout -q $KWON_SHA; }
  [ -d kprune ] || { git clone -q https://github.com/snudm-starlab/K-prune kprune && git -C kprune checkout -q $KPRUNE_SHA; }
  cp "$REPO/scripts/v15_kwon_unpruned.py" kwon/
  cp "$REPO/scripts/v15_official_addon.py" kwon/ && cp "$REPO/scripts/v15_official_addon.py" kprune/src/
  cp "$REPO/src/qfc/squad.py" kwon/squad_v15.py && cp "$REPO/src/qfc/squad.py" kprune/src/squad_v15.py
  python "$REPO/scripts/v15_official_addon.py" patch kwon/main.py kwon
  python "$REPO/scripts/v15_official_addon.py" patch kprune/src/main.py kprune
  git -C kwon diff > patch_kwon.diff; git -C kprune diff > patch_kprune.diff
  [ -d env_kwon ] || { uv venv -q -p 3.9 env_kwon && uv pip install -q -p env_kwon/bin/python torch==1.13.1 \
      transformers==4.12.0 datasets==2.14.7 numpy==1.23.5 scipy scikit-learn==1.2.2 tqdm cupy-cuda12x; }
  "$W/env_kwon/bin/python" -c "import sklearn" 2>/dev/null || uv pip install -q -p env_kwon/bin/python scikit-learn==1.2.2  # load_metric('glue') needs it
  [ -d env_kprune ] || { uv venv -q -p 3.10 env_kprune && uv pip install -q -p env_kprune/bin/python torch==1.12.1 \
      transformers==4.25.1 datasets==2.14.7 numpy==1.23.4 scipy==1.10.0 scikit-learn==1.2.1 tqdm colorama; }
  # (newer gdown dropped --remaining-ok; the folder has < 50 files, so it is not needed)
  if [ "${SKIP_KP_CKPT:-0}" != 1 ]; then  # the 'bl' stage uses only our pinned textattack checkpoints
  [ -d ck_kprune ] || gdown -q --folder https://drive.google.com/drive/folders/1G4EB76sd2Pq-aIPnUj6JTWXBEaDMLFPt -O ck_kprune
  ls ck_kprune/bert-base-uncased
  for t in mrpc sst2; do [ -f ck_kprune/bert-base-uncased/$t/config.json ] || { echo "missing K-prune ckpt $t"; exit 1; }; done
  fi
  publish patch_kwon.diff patch_kprune.diff
  df -h /tmp /kaggle/working || true
  ;;
phase1)
  mkdir -p phase1
  # A1: unpruned accuracy of K-prune's checkpoints under Kwon's own evaluation (the drop is pruned minus this).
  for t in mrpc sst2; do (cd "$W/kwon" && "$W/env_kwon/bin/python" v15_kwon_unpruned.py $t "$W/ck_kprune/bert-base-uncased/$t")     > "$W/phase1/kwon_unpruned_$t.log" 2>&1 || echo "FAIL unpruned $t"; grep -H "Unpruned" "$W/phase1/kwon_unpruned_$t.log" || true; done
  # Kwon Phase 1 pruned runs done in Version 3 (2026-10-16, all 30 read): skipped by default; SKIP_KWON_PHASE1=0 reruns.
  [ "${SKIP_KWON_PHASE1:-1}" = 1 ] || for t in mrpc sst2; do for c in 0.7 0.6 0.5; do for s in 0 1 2 3 4; do
    o=$W/phase1/kwon_${t}_c${c}_s${s}; [ -f $o/head_mask.pt ] && continue
    kwon --task_name $t --ckpt_dir "$W/ck_kprune/bert-base-uncased/$t" --constraint $c --seed $s --output_dir $o \
      > $o.log 2>&1 || echo "FAIL $o"
  done; done; done
  # K-prune Phase 1 reproduced on 2026-10-16 (Version 2): skipped by default; SKIP_KPRUNE_PHASE1=0 reruns it.
  [ "${SKIP_KPRUNE_PHASE1:-1}" = 1 ] || for t in mrpc squad; do for s in 0 1 2 3 4; do
    e=phase1/kprune_${t}_r0.8_s${s}; [ -f "$KP_OUT/$e/model.ckpt" ] && continue
    kprune --task_name $t --ckpt_dir "$W/ck_kprune/bert-base-uncased/$t" --exp_name $e --seed $s \
      --num_tokens 100000 --constraint 0.8 > "$W/phase1/kprune_${t}_r0.8_s${s}.log" 2>&1 || echo "FAIL $e"
  done; done
  grep -H "Test accuracy" "$W"/phase1/kwon_*.log || true
  grep -H "FLOPs:" "$W"/phase1/kprune_*.log || true
  publish phase1
  ;;
h10)
  (cd "$REPO" && PYTHONPATH=scripts python scripts/v15_prepare_baselines.py --tasks mrpc,sst2 --out "$W/inputs")
  mkdir -p h10
  for t in mrpc sst2; do for s in $SEEDS; do
    export V15_INDEX_FILE=$W/inputs/indices/${t}_seed${s}.json
    for c in $CONS; do
      o=$W/h10/kwon_${t}_seed${s}_c${c}
      [ -f $o/v15_student_heldout.pt ] || { mkdir -p $o; kwon --task_name $t --ckpt_dir "$W/inputs/ckpt/$t" \
        --constraint $c --seed 0 --output_dir $o > $o.log 2>&1 || echo "FAIL $o"; }
      r=$(python -c "print(round(1-$c,4))")
      e=h10/kprune_${t}_seed${s}_c${c}
      [ -f "$KP_OUT/$e/v15_student_heldout.pt" ] || { kprune --task_name $t --ckpt_dir "$W/inputs/ckpt/$t" \
        --exp_name $e --seed 0 --constraint $r > "$W/h10/kprune_${t}_seed${s}_c${c}.log" 2>&1 || echo "FAIL $e"; }
      rm -f "$KP_OUT/$e/model.ckpt"
    done
    unset V15_INDEX_FILE
    # publish after every (task, seed) so a run stopped by Kaggle's 12 h limit keeps what it finished
    mkdir -p "$W/h10/kprune_out" && cp -r "$KP_OUT/h10/." "$W/h10/kprune_out/" 2>/dev/null || true
    find "$W/h10/kprune_out" -name model.ckpt -delete || true
    publish h10 inputs/indices
  done; done
  # GRP writes its JSONs (one per run, resumable) straight into the published directory
  (cd "$REPO" && PYTHONPATH=scripts python scripts/step22_grp.py --tasks mrpc,sst2 --constraints 0.7,0.6,0.5,0.4 \
    --variants "${GRP_VARIANTS:-grp}" --mode kl --output-dir "$OUT/h10/grp" 2>&1 | grep -v -i warn | tail -45)
  ;;
h11)
  # Primary test (run only if Phase 1 and H10 pass). Constraint 0.5; dev tasks pick the better baseline once.
  TASKS=${H11_TASKS:-"mrpc sst2 qnli mnli squad"}
  (cd "$REPO" && PYTHONPATH=scripts python scripts/v15_prepare_baselines.py --tasks "$(echo $TASKS | tr ' ' ,)" --out "$W/inputs")
  mkdir -p h11
  for t in $TASKS; do for s in $SEEDS; do
    export V15_INDEX_FILE=$W/inputs/indices/${t}_seed${s}.json
    o=$W/h11/kwon_${t}_seed${s}_c0.5
    [ -f $o/v15_student_validation.pt ] || { mkdir -p $o; kwon --task_name $t --ckpt_dir "$W/inputs/ckpt/$t" \
      --constraint 0.5 --seed 0 --output_dir $o > $o.log 2>&1 || echo "FAIL $o"; }
    e=h11/kprune_${t}_seed${s}_c0.5
    [ -f "$KP_OUT/$e/v15_student_validation.pt" ] || { kprune --task_name $t --ckpt_dir "$W/inputs/ckpt/$t" \
      --exp_name $e --seed 0 --constraint 0.5 > "$W/h11/kprune_${t}_seed${s}.log" 2>&1 || echo "FAIL $e"; }
    rm -f "$KP_OUT/$e/model.ckpt"
    unset V15_INDEX_FILE
  done; done
  (cd "$REPO" && PYTHONPATH=scripts python scripts/step22_grp.py --tasks "$(echo $TASKS | tr ' ' ,)" \
    --constraints 0.5 --variants grp --mode accuracy --output-dir "$W/h11/grp" 2>&1 | grep -v -i warn | tail -40)
  mkdir -p "$W/h11/kprune_out" && cp -r "$KP_OUT/h11/." "$W/h11/kprune_out/" 2>/dev/null || true
  find "$W/h11/kprune_out" -name model.ckpt -delete || true
  (cd "$REPO" && python scripts/v15_h11_analysis.py "$W/h11") || true
  (cd "$REPO" && python scripts/v15_latency.py --grp "$W/h11/grp/v15_grp_*_c0.5.json" \
    --kwon $(ls -d $W/h11/kwon_*_seed7_c0.5) --kprune $(ls -d $W/h11/kprune_out/kprune_*_seed7_c0.5) \
    --out "$W/h11/latency.json") || true
  find "$W/h11/kprune_out" -name model.ckpt -delete || true
  publish h11 inputs/indices
  ;;
v16a)
  # V16 Stage A (docs/PROJECT_STATE.md, "Pre-registration V16"): our env only; K-prune masks and GRP solutions come
  # from results/v15_h10/c43dd3e in the repo. JSONs are written straight to the published dir (resumable).
  pip install -q -r "$REPO/requirements-ci.txt" && pip install -q -e "$REPO" --no-deps
  pip uninstall -y -q torchvision torchaudio || true
  python "$REPO/scripts/preflight.py" check
  mkdir -p "$OUT/v16a"
  (cd "$REPO" && PYTHONPATH=scripts python scripts/step23_v16_stageA.py --output-dir "$OUT/v16a" 2>&1 \
    | grep -v -i warn | tail -120)
  (cd "$REPO" && python scripts/v16_stageA_analysis.py "$OUT/v16a" results/v15_h10/c43dd3e/h10/v15_h10_verdict.json) || true
  ;;
v17r)
  # V17 Stage R-dev (docs/PROJECT_STATE.md, "Pre-registration V17"): GRP-R and K-prune masks + drift-corrected refit.
  pip install -q -r "$REPO/requirements-ci.txt" && pip install -q -e "$REPO" --no-deps
  pip uninstall -y -q torchvision torchaudio || true
  python "$REPO/scripts/preflight.py" check
  mkdir -p "$OUT/v17r"
  (cd "$REPO" && PYTHONPATH=scripts python scripts/step23_v16_stageA.py --parts grp_r,d2i_r --output-dir "$OUT/v17r" 2>&1     | grep -v -i warn | tail -60)
  (cd "$REPO" && python scripts/v17_rdev_analysis.py "$OUT/v17r" results/v15_h10/c43dd3e/h10/v15_h10_verdict.json) || true
  ;;
bl)
  # Official Kwon / K-prune baselines on a NEW task with our pinned checkpoint and calibration rows (EXPLORATION_LOG,
  # Amendment 1: STS-B). BL_TASKS (default stsb), BL_CONS (default "0.5 0.4"). Same patched unmodified algorithms as h10.
  # Results are published after every (task, seed) so a Kaggle time-out keeps what finished.
  SKIP_KP_CKPT=1 bash "$0" setup "$W" "$OUT"
  BT=${BL_TASKS:-stsb}; BC=${BL_CONS:-"0.5 0.4"}
  (cd "$REPO" && PYTHONPATH=scripts python scripts/v15_prepare_baselines.py --tasks "$(echo $BT | tr ' ' ,)" --out "$W/inputs")
  mkdir -p bl
  for t in $BT; do for s in $SEEDS; do
    export V15_INDEX_FILE=$W/inputs/indices/${t}_seed${s}.json
    for c in $BC; do
      o=$W/bl/kwon_${t}_seed${s}_c${c}
      [ -f $o/v15_student_heldout.pt ] || { mkdir -p $o; kwon --task_name $t --ckpt_dir "$W/inputs/ckpt/$t" \
        --constraint $c --seed 0 --output_dir $o > $o.log 2>&1 || echo "FAIL $o"; }
      r=$(python -c "print(round(1-$c,4))")
      e=bl/kprune_${t}_seed${s}_c${c}
      [ -f "$KP_OUT/$e/v15_student_heldout.pt" ] || { kprune --task_name $t --ckpt_dir "$W/inputs/ckpt/$t" \
        --exp_name $e --seed 0 --constraint $r > "$W/bl/kprune_${t}_seed${s}_c${c}.log" 2>&1 || echo "FAIL $e"; }
      rm -f "$KP_OUT/$e/model.ckpt"
    done
    unset V15_INDEX_FILE
    mkdir -p "$W/bl/kprune_out" && cp -r "$KP_OUT/bl/." "$W/bl/kprune_out/" 2>/dev/null || true
    find "$W/bl/kprune_out" -name model.ckpt -delete || true
    publish bl inputs/indices
  done; done
  python "$REPO/scripts/beat_masks.py" "$W/bl/kprune_out" "$OUT/bl/kprune_masks.npz" || true
  ;;
beat)
  # Beat-the-baselines exploration (docs/EXPLORATION_LOG.md): our env only. BEAT_ARGS = scripts/beat_run.py arguments
  # (kernel cell sets it); preflight failure aborts (set -e) so a wrong environment never produces numbers.
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
  pip install -q -r "$REPO/requirements-ci.txt" && pip install -q -e "$REPO" --no-deps
  pip uninstall -y -q torchvision torchaudio || true
  python "$REPO/scripts/preflight.py" check
  mkdir -p "$OUT/beat"
  (cd "$REPO" && PYTHONPATH=scripts python scripts/beat_run.py $BEAT_ARGS --output-dir "$OUT/beat" 2>&1 | grep -v -i warn | tail -150)
  ;;
*) echo "unknown stage $STAGE"; exit 1 ;;
esac
