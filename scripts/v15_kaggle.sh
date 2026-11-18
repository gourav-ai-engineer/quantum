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
MODEL_NAME=${MODEL_NAME:-bert-base-uncased}   # V19: MODEL_NAME=distilbert-base-uncased (+ CKPT_DIR_ROOT, CK_REQUIRE)
KP_OUT=$W/kprune/src/outputs/$MODEL_NAME
SEEDS="7 42 77 123 2024"
CONS="0.7 0.6 0.5 0.4"

# cupy-cuda12x needs libcublas.so.12 etc.; the Kaggle image no longer ships CUDA 12 system libs, so use the pip nvidia-*-cu12 wheels
kwon_ld() { for d in "$W"/env_kwon/lib/python3.9/site-packages/nvidia/*/lib; do printf '%s:' "$d"; done; }
kwon()   { (cd "$W/kwon" && LD_LIBRARY_PATH="$(kwon_ld)${LD_LIBRARY_PATH:-}" "$W/env_kwon/bin/python" main.py --model_name "$MODEL_NAME" "$@"); }
# V19 fix1 (engineering only): Kaggle's shared IP gets HTTP 429 from the HF Hub, which killed official runs at tokenizer/dataset
# load. Retry a run ONLY when its log shows a 429 (a real failure is never retried); sleep grows 90 s per attempt.
hf_retry() { local log=$1 n=0; shift
  until "$@" > "$log" 2>&1; do n=$((n+1))
    if grep -q "429 Client Error\|Too Many Requests" "$log" && [ "$n" -lt "${HF_RETRIES:-8}" ]; then echo "HF 429, retry $n: $log"; sleep $((90*n)); else return 1; fi
  done; }
inlist() { case " $2 " in *" $1 "*) return 0;; *) return 1;; esac; }   # inlist <item> "<space-separated list>"
publish() { mkdir -p "$OUT"; for d in "$@"; do [ -e "$W/$d" ] && { mkdir -p "$OUT/$(dirname "$d")"; cp -r "$W/$d" "$OUT/$(dirname "$d")/"; }; done; du -sh "$OUT"; }
kprune() { (cd "$W/kprune/src" && "$W/env_kprune/bin/python" main.py --model_name "$MODEL_NAME" --gpu 0 \
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
  uv pip install -q -p env_kwon/bin/python nvidia-cublas-cu12 nvidia-cuda-runtime-cu12 nvidia-cusparse-cu12 nvidia-cusolver-cu12 nvidia-curand-cu12 nvidia-cufft-cu12 nvidia-cuda-nvrtc-cu12 nvidia-nvjitlink-cu12
  LD_LIBRARY_PATH="$(kwon_ld)${LD_LIBRARY_PATH:-}" "$W/env_kwon/bin/python" -c "from cupyx.scipy.sparse.linalg import lsmr" || { echo "CUPY_IMPORT_FAILED"; exit 1; }
  "$W/env_kwon/bin/python" -c "import sklearn" 2>/dev/null || uv pip install -q -p env_kwon/bin/python scikit-learn==1.2.2  # load_metric('glue') needs it
  [ -d env_kprune ] || { uv venv -q -p 3.10 env_kprune && uv pip install -q -p env_kprune/bin/python torch==1.12.1 \
      transformers==4.25.1 datasets==2.14.7 numpy==1.23.4 scipy==1.10.0 scikit-learn==1.2.1 tqdm colorama; }
  # (newer gdown dropped --remaining-ok; the folder has < 50 files, so it is not needed)
  if [ "${SKIP_KP_CKPT:-0}" != 1 ]; then  # the 'bl' stage uses only our pinned textattack checkpoints
  # V19 fix2: a partial Drive download (small files present, large pytorch_model.bin missing: Drive quota) passed the
  # config.json check below and then killed the run. Retry until every required checkpoint HAS its weights file.
  for a in 1 2 3 4 5 6; do
    gdown -q --folder https://drive.google.com/drive/folders/1G4EB76sd2Pq-aIPnUj6JTWXBEaDMLFPt -O ck_kprune || echo "gdown exit $?"
    ok=1; for t in ${CK_REQUIRE:-mrpc sst2}; do [ -f ck_kprune/$MODEL_NAME/$t/pytorch_model.bin ] || ok=0; done
    [ $ok = 1 ] && break; echo "ckpt incomplete, retry $a"; sleep $((300*a))
  done
  [ $ok = 1 ] || { echo "CKPT_DOWNLOAD_FAILED"; exit 1; }
  ls ck_kprune/$MODEL_NAME
  for t in ${CK_REQUIRE:-mrpc sst2}; do [ -f ck_kprune/$MODEL_NAME/$t/config.json ] || { echo "missing K-prune ckpt $t"; exit 1; }; done
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
  SKIP_KP_CKPT=${BL_SKIP_CKPT:-1} bash "$REPO/scripts/v15_kaggle.sh" setup "$W" "$OUT"  # $0 is relative and we cd "$W" above
  BT=${BL_TASKS:-stsb}; BC=${BL_CONS:-"0.5 0.4"}
  (cd "$REPO" && PYTHONPATH=scripts python scripts/v15_prepare_baselines.py --tasks "$(echo $BT | tr ' ' ,)" --out "$W/inputs" ${PREP_ARGS:-})
  CKR=${CKPT_DIR_ROOT:-$W/inputs/ckpt}   # V19: the original official checkpoint folders (sha256-verified), not re-saved copies
  [ -z "${CKPT_SHA_FILE:-}" ] || python "$REPO/scripts/v19_verify_ckpt.py" "$CKR" "$CKPT_SHA_FILE" $BT
  mkdir -p bl
  for t in $BT; do for s in $SEEDS; do
    export V15_INDEX_FILE=$W/inputs/indices/${t}_seed${s}.json
    for c in $BC; do
      o=$W/bl/kwon_${t}_seed${s}_c${c}
      # V19 fix1: BL_KWON_SEEDS / BL_KPRUNE_SEEDS (default: all seeds) rerun ONLY the items that failed
      inlist $s "${BL_KWON_SEEDS:-$SEEDS}" && { [ -f $o/v15_student_heldout.pt ] || { mkdir -p $o; hf_retry $o.log kwon --task_name $t --ckpt_dir "$CKR/$t" \
        --constraint $c --seed 0 --output_dir $o || echo "FAIL $o"; }; }
      r=$(python -c "print(round(1-$c,4))")
      e=bl/kprune_${t}_seed${s}_c${c}
      # BL_SKIP_KPRUNE=1: Kwon-only rerun (V18 fix1); the finished K-prune items must not be rerun
      [ -n "${BL_SKIP_KPRUNE:-}" ] || ! inlist $s "${BL_KPRUNE_SEEDS:-$SEEDS}" || [ -f "$KP_OUT/$e/v15_student_heldout.pt" ] || { hf_retry "$W/bl/kprune_${t}_seed${s}_c${c}.log" kprune --task_name $t --ckpt_dir "$CKR/$t" \
        --exp_name $e --seed 0 --constraint $r || echo "FAIL $e"; }
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
  # V19 fix1: full log to a published file (the old `| tail -150` left an empty log when beat crashed); exit status is kept
  # beat_run.py is resumable (skips finished JSONs), so a 429 retry only redoes unfinished items
  (cd "$REPO" && PYTHONPATH=scripts hf_retry "$OUT/beat/beat.log" python scripts/beat_run.py $BEAT_ARGS --output-dir "$OUT/beat") || { tail -40 "$OUT/beat/beat.log"; exit 1; }
  grep -v -i warn "$OUT/beat/beat.log" | tail -60
  ;;
v21e1)
  # V21 pre-freeze gates E1/E2/E5 (docs/V21_PREREG_DRAFT_2026-11-18.md). VALIDATION data only; engineering checks, never
  # results. V21_TEACHER_DIR = the V20 MuRIL teacher (sha256-checked by the kernel), V21_V20_IN = V20 kernel output root.
  SKIP_KP_CKPT=1 bash "$REPO/scripts/v15_kaggle.sh" setup "$W" "$OUT"
  IN21=$W/v21e1in; E=$W/v21e1; mkdir -p "$E"
  (cd "$REPO" && PYTHONPATH=scripts python scripts/v21_e1_prep.py --out "$IN21")
  (cd "$REPO" && PYTHONPATH=scripts python scripts/v15_prepare_baselines.py --tasks mrpc --out "$W/inputs")
  # The official codes use --model_name only for AutoTokenizer.from_pretrained and their output path, so the local
  # tokenizer dirs are linked under a RELATIVE name into both working dirs (an absolute path would hijack os.path.join).
  for d in "$IN21"/tok/*; do k=$(basename "$d"); ln -sfn "$d" "$W/kwon/v21tok_$k"; ln -sfn "$d" "$W/kprune/src/v21tok_$k"; done
  for d in "$IN21"/tok/*; do k=$(basename "$d")
    (cd "$W/kwon" && "$W/env_kwon/bin/python" "$REPO/scripts/v21_tok_ids.py" "v21tok_$k" "$IN21/pairs.json" "$E/tok_kwon_$k.json")
    (cd "$W/kprune/src" && "$W/env_kprune/bin/python" "$REPO/scripts/v21_tok_ids.py" "v21tok_$k" "$IN21/pairs.json" "$E/tok_kprune_$k.json")
  done
  for k in bert-base-uncased google/muril-base-cased bert-base-multilingual-cased; do n=$(echo $k | tr / _)
    (cd "$W/kwon" && "$W/env_kwon/bin/python" "$REPO/scripts/v21_tok_ids.py" "$k" "$IN21/pairs.json" "$E/tokhub_kwon_$n.json") || true
    (cd "$W/kprune/src" && "$W/env_kprune/bin/python" "$REPO/scripts/v21_tok_ids.py" "$k" "$IN21/pairs.json" "$E/tokhub_kprune_$n.json") || true
  done
  pick() { (cd "$REPO" && python scripts/v21_e1_report.py choose "$E" "$IN21" "$1" "$2") || true; }  # prints a dir name or NONE
  # run_pair <name> <model_name for kwon> <model_name for kprune> <task> <ckpt dir>: one Kwon and one K-prune run, c 0.5
  run_pair() { local name=$1 mk=$2 mp=$3 t=$4 ck=$5
    if [ "$mk" != NONE ]; then MODEL_NAME=$mk; o=$E/kwon_$name; mkdir -p "$o"
      hf_retry "$o.log" kwon --task_name $t --ckpt_dir "$ck" --constraint 0.5 --seed 0 --output_dir "$o" || echo "FAIL $o"; fi
    if [ "$mp" != NONE ]; then MODEL_NAME=$mp; e=v21e1/kprune_$name; kp=$W/kprune/src/outputs/$mp/$e
      hf_retry "$E/kprune_$name.log" kprune --task_name $t --ckpt_dir "$ck" --exp_name $e --seed 0 --constraint 0.5 || echo "FAIL $e"
      rm -f "$kp/model.ckpt"; mkdir -p "$E/kprune_$name"; cp -r "$kp/." "$E/kprune_$name/" 2>/dev/null || true; fi
    MODEL_NAME=bert-base-uncased; }
  # 1. same-kernel A/B on BERT-base MRPC, seed 7: hub id vs local dir (+ a hub rerun for GPU determinism)
  export V15_INDEX_FILE=$W/inputs/indices/mrpc_seed7.json
  run_pair mrpc_hub bert-base-uncased bert-base-uncased mrpc "$W/inputs/ckpt/mrpc"
  run_pair mrpc_hub2 bert-base-uncased bert-base-uncased mrpc "$W/inputs/ckpt/mrpc"
  run_pair mrpc_local "$(pick kwon bert)" "$(pick kprune bert)" mrpc "$W/inputs/ckpt/mrpc"
  # 2. V20 MuRIL teacher, MNLI seed 7 (the V20 calibration index), with the E5 extra validation sets
  export V15_INDEX_FILE=$V21_V20_IN/inputs/indices/mnli_seed7.json V21_EXTRA_EVAL=$IN21/extra_eval.json
  run_pair muril_local "$(pick kwon muril)" "$(pick kprune muril)" mnli "$V21_TEACHER_DIR"
  run_pair muril_hub NONE google/muril-base-cased mnli "$V21_TEACHER_DIR"
  unset V21_EXTRA_EVAL
  # 3. mBERT with a seeded random head: vocabulary/loading/MAC check only
  run_pair mbert_local "$(pick kwon mbert)" "$(pick kprune mbert)" mnli "$IN21/ckpt_mbert_rand"
  unset V15_INDEX_FILE
  mkdir -p "$OUT/v21e1"
  (cd "$REPO" && PYTHONPATH=scripts python scripts/v21_e1_report.py report "$E" "$IN21" --v20-npz "$V21_V20_IN/bl/kprune_masks.npz" \
    --v15-dir "$REPO/results/v15_h10/c43dd3e/h10_logits" --out "$OUT/v21e1/e1_e5.json") || echo REPORT_FAILED
  cp "$IN21/prep_meta.json" "$OUT/v21e1/"; cp -r "$E" "$OUT/v21e1/runs"
  ;;
*) echo "unknown stage $STAGE"; exit 1 ;;
esac
