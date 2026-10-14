#!/usr/bin/env bash
# V15 driver for Kaggle (GPU). Stages (docs/PROJECT_STATE.md, "Pre-registration V15"):
#   setup  : our pinned env (system python) + official Kwon/K-prune codes in their own uv envs, patched (v15 add-on),
#            K-prune's BERT-base checkpoints (amendment A1).
#   phase1 : reproduction gates. Kwon (no V15_INDEX_FILE => identical to unpatched) on K-prune's MRPC/SST-2
#            checkpoints at 0.7/0.6/0.5; K-prune at --constraint 0.8 (= remove 80%) on MRPC and SQuAD1.1. Seeds 0-4.
#   h10    : calibration-side gate. Our 2048 calibration rows and pinned textattack checkpoints for every method:
#            Kwon (keep c), K-prune (--constraint 1-c = remove 1-c), GRP (step22 --mode kl). MRPC/SST-2, 5 seeds,
#            c in 0.7/0.6/0.5/0.4. Logits on held-out TRAIN rows (and validation, unused by H10) are dumped.
# Usage: bash scripts/v15_kaggle.sh <stage> [work_dir]
set -euo pipefail
STAGE=$1
W=${2:-/kaggle/working/v15}
REPO=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$W" && cd "$W"
KWON_SHA=806ac5a6ff53b4978e5330d45a3de692493e4d0b
KPRUNE_SHA=9ca629a1db1ebd835234ade03a4c0ce8fa522d7b
KP_OUT=$W/kprune/src/outputs/bert-base-uncased
SEEDS="7 42 77 123 2024"
CONS="0.7 0.6 0.5 0.4"

kwon()   { (cd "$W/kwon" && "$W/env_kwon/bin/python" main.py --model_name bert-base-uncased "$@"); }
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
  cp "$REPO/scripts/v15_official_addon.py" kwon/ && cp "$REPO/scripts/v15_official_addon.py" kprune/src/
  python "$REPO/scripts/v15_official_addon.py" patch kwon/main.py kwon
  python "$REPO/scripts/v15_official_addon.py" patch kprune/src/main.py kprune
  git -C kwon diff > patch_kwon.diff; git -C kprune diff > patch_kprune.diff
  [ -d env_kwon ] || { uv venv -q -p 3.9 env_kwon && uv pip install -q -p env_kwon/bin/python torch==1.13.1 \
      transformers==4.12.0 datasets==2.14.7 numpy==1.23.5 scipy tqdm cupy-cuda12x; }
  [ -d env_kprune ] || { uv venv -q -p 3.10 env_kprune && uv pip install -q -p env_kprune/bin/python torch==1.12.1 \
      transformers==4.25.1 datasets==2.14.7 numpy==1.23.4 scipy==1.10.0 scikit-learn==1.2.1 tqdm colorama; }
  [ -d ck_kprune ] || gdown -q --folder --remaining-ok https://drive.google.com/drive/folders/1G4EB76sd2Pq-aIPnUj6JTWXBEaDMLFPt -O ck_kprune
  ls ck_kprune/bert-base-uncased
  ;;
phase1)
  mkdir -p phase1
  for t in mrpc sst2; do for c in 0.7 0.6 0.5; do for s in 0 1 2 3 4; do
    o=$W/phase1/kwon_${t}_c${c}_s${s}; [ -f $o/head_mask.pt ] && continue
    kwon --task_name $t --ckpt_dir "$W/ck_kprune/bert-base-uncased/$t" --constraint $c --seed $s --output_dir $o \
      > $o.log 2>&1 || echo "FAIL $o"
  done; done; done
  for t in mrpc squad; do for s in 0 1 2 3 4; do
    e=phase1/kprune_${t}_r0.8_s${s}; [ -f "$KP_OUT/$e/model.ckpt" ] && continue
    kprune --task_name $t --ckpt_dir "$W/ck_kprune/bert-base-uncased/$t" --exp_name $e --seed $s \
      --num_tokens 100000 --constraint 0.8 > "$W/phase1/kprune_${t}_r0.8_s${s}.log" 2>&1 || echo "FAIL $e"
  done; done
  grep -H "Test accuracy" "$W"/phase1/kwon_*.log || true
  grep -H "FLOPs:" "$W"/phase1/kprune_*.log || true
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
    done
    unset V15_INDEX_FILE
  done; done
  (cd "$REPO" && PYTHONPATH=scripts python scripts/step22_grp.py --tasks mrpc,sst2 --constraints 0.7,0.6,0.5,0.4 \
    --variants grp,no_sens --mode kl --output-dir "$W/h10/grp" 2>&1 | grep -v -i warn | tail -45)
  mkdir -p "$W/h10/kprune_out" && cp -r "$KP_OUT/h10/." "$W/h10/kprune_out/" 2>/dev/null || true
  find "$W/h10/kprune_out" -name model.ckpt -delete || true
  ;;
*) echo "unknown stage $STAGE"; exit 1 ;;
esac
