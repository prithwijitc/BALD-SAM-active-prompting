#!/usr/bin/env bash
# ============================================================================
# Full pipeline for all medical SAM checkpoints.
# Stages 1-2 (manifests, prompt sets) run ONCE (shared).
# Stages 3-6 (precompute, train, laplace) run PER SAM CHECKPOINT.
#
# Usage: bash run_medical_pipeline.sh
# ============================================================================
set -euo pipefail

PYTHON=/home/prithwijit/miniconda3/envs/dino-vit-feats-env/bin/python
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export CUDA_VISIBLE_DEVICES=1
DEVICE="cuda:0"  # maps to physical GPU 1

# ---- Paths ----
MEDICAL_ROOT=/mnt/HD_18T_pt2/medical_data
SAM_CKPT_DIR=/mnt/HD_18T_pt2/prithwijit/bald/SAM_ckpt_full
BASELINE_SAM=/home/prithwijit/BALD/sam_vit_h_4b8939.pth
RUNS_DIR="${MEDICAL_ROOT}/sam_runs"

# All SAM checkpoints to evaluate (baseline + 6 fine-tuned)
SAM_NAMES=(baseline busi chexmask combined drive isic2018 kvasir_seg)

echo "============================================"
echo "  Medical SAM Pipeline — Full Run"
echo "  GPU: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "  SAM models: ${SAM_NAMES[*]}"
echo "  Start: $(date)"
echo "============================================"
echo ""

# ============================================================================
# STAGE 0: Generate per-checkpoint configs
# ============================================================================
echo "[Stage 0] Generating configs..."

SHARED_CONFIG="${MEDICAL_ROOT}/config_shared.yaml"

# Write the shared base config (stages 1-2 only)
cat > "$SHARED_CONFIG" <<'YAML'
data:
  images_root: /mnt/HD_18T_pt2/medical_data/images
  masks_root: /mnt/HD_18T_pt2/medical_data/masks
  project_root: /mnt/HD_18T_pt2/medical_data
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
  checkpoint: /home/prithwijit/BALD/sam_vit_h_4b8939.pth
  model_type: vit_h
  device: cuda:0
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
  enabled: false
  project: laplace-head-sam-medical
  entity: null
  group: null
YAML

# Generate per-checkpoint configs
mkdir -p "${RUNS_DIR}"
for NAME in "${SAM_NAMES[@]}"; do
    RUN_DIR="${RUNS_DIR}/${NAME}"
    CONFIG="${RUN_DIR}/config.yaml"
    mkdir -p "$RUN_DIR"

    # Determine SAM checkpoint path
    if [ "$NAME" = "baseline" ]; then
        SAM_CKPT="$BASELINE_SAM"
    else
        SAM_CKPT="${SAM_CKPT_DIR}/${NAME}/sam_vit_h_4b8939.pth"
    fi

    cat > "$CONFIG" <<YAML
data:
  images_root: /mnt/HD_18T_pt2/medical_data/images
  masks_root: /mnt/HD_18T_pt2/medical_data/masks
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
  enabled: false
  project: laplace-head-sam-medical
  entity: null
  group: ${NAME}
YAML

    echo "  Config: $CONFIG (SAM: $SAM_CKPT)"
done
echo ""

# ============================================================================
# STAGES 1-2: Manifests + Prompt Sets (shared, run once)
# ============================================================================
echo "[Stage 1] Building manifests (shared)..."
$PYTHON build_manifests.py --config "$SHARED_CONFIG"
echo ""

echo "[Stage 2] Generating prompt sets (shared)..."
$PYTHON generate_promptsets.py --config "$SHARED_CONFIG"
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

    # Symlink shared manifests and prompt_sets into per-run project root
    if [ ! -e "${RUN_DIR}/manifests" ]; then
        ln -sf "${MEDICAL_ROOT}/manifests" "${RUN_DIR}/manifests"
        echo "  Linked manifests"
    fi
    if [ ! -e "${RUN_DIR}/prompt_sets" ]; then
        ln -sf "${MEDICAL_ROOT}/prompt_sets" "${RUN_DIR}/prompt_sets"
        echo "  Linked prompt_sets"
    fi

    # Stage 3: SAM precompute
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

echo "============================================"
echo "  All SAM checkpoints processed!"
echo "  Results in: ${RUNS_DIR}/"
echo "  Finished: $(date)"
echo "============================================"

# Print summary
echo ""
echo "=== Summary ==="
for NAME in "${SAM_NAMES[@]}"; do
    RUN_DIR="${RUNS_DIR}/${NAME}"
    EPOCH_STATE="${RUN_DIR}/checkpoints/no_image/run_${NAME}/epoch_state.json"
    LAPLACE_META="${RUN_DIR}/laplace/no_image/run_${NAME}/posterior_meta.json"
    if [ -f "$EPOCH_STATE" ]; then
        BEST_IOU=$(python3 -c "import json; print(json.load(open('$EPOCH_STATE'))['best_val_iou'])" 2>/dev/null || echo "?")
        echo "  ${NAME}: best_val_iou=${BEST_IOU}"
    else
        echo "  ${NAME}: no checkpoint found"
    fi
    if [ -f "$LAPLACE_META" ]; then
        PRIOR=$(python3 -c "import json; print(json.load(open('$LAPLACE_META'))['chosen_prior_precision'])" 2>/dev/null || echo "?")
        NLL=$(python3 -c "import json; print(json.load(open('$LAPLACE_META'))['best_val_nll'])" 2>/dev/null || echo "?")
        echo "           prior=${PRIOR}  val_nll=${NLL}"
    fi
done
