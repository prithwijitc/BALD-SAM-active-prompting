#!/usr/bin/env bash
# ============================================================================
# Train Laplace heads for all fine-tuned SAM checkpoints on PointPrompt data.
# Stages 1-2 (manifests, prompt sets) already exist — reused via symlinks.
# Stages 3-6 (precompute, train, laplace) run PER SAM CHECKPOINT.
#
# Usage: bash run_finetuned_sam_pipeline.sh
# ============================================================================
set -euo pipefail

PYTHON=/home/prithwijit/miniconda3/envs/dino-vit-feats-env/bin/python
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export CUDA_VISIBLE_DEVICES=1
DEVICE="cuda:0"  # maps to physical GPU 1

# ---- Paths ----
POINTPROMPT_ROOT=/mnt/HD_18T_pt2/prithwijit/bald
SAM_CKPT_DIR=/mnt/HD_18T_pt2/prithwijit/bald/SAM_ckpt_full
RUNS_DIR="${POINTPROMPT_ROOT}/sam_runs"

# Fine-tuned SAM checkpoints (baseline v3 already exists — skip it)
SAM_NAMES=(busi chexmask combined drive isic2018 kvasir_seg)

echo "============================================"
echo "  Fine-tuned SAM Pipeline — PointPrompt Data"
echo "  GPU: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "  SAM models: ${SAM_NAMES[*]}"
echo "  Data: PointPrompt (6,400 images, splits ready)"
echo "  Start: $(date)"
echo "============================================"
echo ""

# ============================================================================
# STAGE 0: Generate per-checkpoint configs
# ============================================================================
echo "[Stage 0] Generating configs..."
mkdir -p "${RUNS_DIR}"

for NAME in "${SAM_NAMES[@]}"; do
    RUN_DIR="${RUNS_DIR}/${NAME}"
    CONFIG="${RUN_DIR}/config.yaml"
    SAM_CKPT="${SAM_CKPT_DIR}/${NAME}/sam_vit_h_4b8939.pth"
    mkdir -p "$RUN_DIR"

    cat > "$CONFIG" <<YAML
data:
  images_root: /mnt/HD_18T_pt2/prithwijit/pace_backup/pointprompt/Images
  masks_root: /mnt/HD_18T_pt2/prithwijit/pace_backup/pointprompt/Masks
  project_root: ${RUN_DIR}
  split_seed: 123
  train_frac: 0.80
  val_frac: 0.10
  test_frac: 0.10

prompt_generation:
  num_promptsets_train: 10
  num_promptsets_val: 4
  num_promptsets_test: 4
  min_pairs: 1
  max_pairs: 10
  prompt_seed: 777
  gaussian_sigma: 8

sam:
  checkpoint: ${SAM_CKPT}
  model_type: vit_h
  device: ${DEVICE}
  multimask_output: true
  select_by: sam_score

head:
  variant: no_image
  channels_in: 3
  hidden_channels: 16

training:
  batch_size: 64
  num_epochs: 50
  lr: 0.001
  weight_decay: 0.0001
  dice_lambda: 1.0
  num_workers: 8
  amp: true
  early_stop_patience: 10
  logit_clip: 20.0
  target_size: [256, 256]
  scheduler: cosine
  scheduler_min_lr: 1.0e-6
  scheduler_plateau_patience: 3
  prefetch_factor: 2

hdf5:
  use_hdf5: true
  chunk_size: 64

laplace:
  enabled: true
  target_layer: final_conv
  approx: diagonal
  prior_precision_grid: [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]
  bald_prior_precision: 0.001

wandb:
  enabled: true
  project: laplace-head-sam
  entity: null
  group: ${NAME}
YAML

    echo "  ${NAME}: ${CONFIG}"
done
echo ""

# ============================================================================
# STAGES 3-6: Per SAM checkpoint
# ============================================================================
for NAME in "${SAM_NAMES[@]}"; do
    RUN_DIR="${RUNS_DIR}/${NAME}"
    CONFIG="${RUN_DIR}/config.yaml"
    RUN_ID="${NAME}"

    echo "============================================"
    echo "  SAM checkpoint: ${NAME}"
    echo "  Project root: ${RUN_DIR}"
    echo "  $(date)"
    echo "============================================"

    # Symlink shared manifests and prompt_sets from existing PointPrompt pipeline
    if [ ! -e "${RUN_DIR}/manifests" ]; then
        ln -sf "${POINTPROMPT_ROOT}/manifests" "${RUN_DIR}/manifests"
        echo "  Linked manifests"
    fi
    if [ ! -e "${RUN_DIR}/prompt_sets" ]; then
        ln -sf "${POINTPROMPT_ROOT}/prompt_sets" "${RUN_DIR}/prompt_sets"
        echo "  Linked prompt_sets"
    fi

    # Stage 3: SAM precompute (most expensive — ~1-2h per checkpoint)
    echo "[Stage 3] Precomputing SAM outputs for ${NAME}..."
    $PYTHON precompute_sam_outputs.py --config "$CONFIG"
    echo ""

    # Stage 4: Training indices
    echo "[Stage 4] Building training indices for ${NAME}..."
    $PYTHON build_training_indices.py --config "$CONFIG"
    echo ""

    # Stage 4.5: HDF5 packing
    echo "[Stage 4.5] Packing to HDF5 for ${NAME}..."
    $PYTHON pack_to_hdf5.py --config "$CONFIG"
    echo ""

    # Stage 5: Train head
    echo "[Stage 5] Training head for ${NAME}..."
    $PYTHON train_no_image_head.py --config "$CONFIG" --run_id "$RUN_ID"
    echo ""

    # Stage 6: Fit Laplace
    echo "[Stage 6] Fitting Laplace for ${NAME}..."
    $PYTHON fit_laplace_no_image.py --config "$CONFIG" --run_id "$RUN_ID"
    echo ""

    echo "  DONE: ${NAME} ($(date))"
    echo ""
done

# ============================================================================
# Summary
# ============================================================================
echo "============================================"
echo "  All SAM checkpoints processed!"
echo "  Finished: $(date)"
echo "============================================"
echo ""
echo "=== Summary ==="
echo ""
printf "%-15s  %-12s  %-12s  %-12s\n" "SAM" "best_val_iou" "prior_prec" "val_nll"
printf "%-15s  %-12s  %-12s  %-12s\n" "---" "---" "---" "---"

# Include baseline v3 for comparison
EPOCH_V3="${POINTPROMPT_ROOT}/checkpoints/no_image/run_v3/epoch_state.json"
if [ -f "$EPOCH_V3" ]; then
    IOU=$($PYTHON -c "import json; print(f'{json.load(open(\"$EPOCH_V3\"))[\"best_val_iou\"]:.4f}')" 2>/dev/null || echo "?")
    printf "%-15s  %-12s  %-12s  %-12s\n" "baseline (v3)" "$IOU" "-" "-"
fi

for NAME in "${SAM_NAMES[@]}"; do
    RUN_DIR="${RUNS_DIR}/${NAME}"
    EPOCH_STATE="${RUN_DIR}/checkpoints/no_image/run_${NAME}/epoch_state.json"
    LAPLACE_META="${RUN_DIR}/laplace/no_image/run_${NAME}/posterior_meta.json"

    IOU="?"
    PRIOR="?"
    NLL="?"

    if [ -f "$EPOCH_STATE" ]; then
        IOU=$($PYTHON -c "import json; print(f'{json.load(open(\"$EPOCH_STATE\"))[\"best_val_iou\"]:.4f}')" 2>/dev/null || echo "?")
    fi
    if [ -f "$LAPLACE_META" ]; then
        PRIOR=$($PYTHON -c "import json; print(f'{json.load(open(\"$LAPLACE_META\"))[\"chosen_prior_precision\"]:.4f}')" 2>/dev/null || echo "?")
        NLL=$($PYTHON -c "import json; print(f'{json.load(open(\"$LAPLACE_META\"))[\"best_val_nll\"]:.4f}')" 2>/dev/null || echo "?")
    fi

    printf "%-15s  %-12s  %-12s  %-12s\n" "$NAME" "$IOU" "$PRIOR" "$NLL"
done

echo ""
echo "Results in: ${RUNS_DIR}/"
echo "Each run: {sam_runs}/<name>/checkpoints/no_image/run_<name>/best.pt"
echo "          {sam_runs}/<name>/laplace/no_image/run_<name>/laplace_state.pt"
