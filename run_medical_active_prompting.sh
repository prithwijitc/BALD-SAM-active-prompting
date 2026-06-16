#!/usr/bin/env bash
# ============================================================================
# Run active prompting on medical datasets with all SAM heads.
#
# 7 runs:
#   baseline_v3  — v3 head + original SAM on ALL 1297 medical images
#   ft_busi      — busi head + busi SAM on BUSI (630)
#   ft_chexmask  — chexmask head + chexmask SAM on CheXmask (247)
#   ft_drive     — drive head + drive SAM on DRIVE (20)
#   ft_isic2018  — isic2018 head + isic2018 SAM on ISIC2018 (200)
#   ft_kvasir_seg— kvasir_seg head + kvasir_seg SAM on KvasirSEG (200)
#   ft_combined  — combined head + combined SAM on ALL 1297 medical images
#
# Usage: bash run_medical_active_prompting.sh
# ============================================================================
set -euo pipefail

PYTHON=/home/prithwijit/miniconda3/envs/dino-vit-feats-env/bin/python
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export CUDA_VISIBLE_DEVICES=1
DEVICE="cuda:0"

# ---- Paths ----
MEDICAL_ROOT=/mnt/HD_18T_pt2/medical_data
BALD_ROOT=/mnt/HD_18T_pt2/prithwijit/bald
SAM_CKPT_DIR="${BALD_ROOT}/SAM_ckpt_full"
BASELINE_SAM=/home/prithwijit/BALD/sam_vit_h_4b8939.pth
MANIFEST_DIR="${MEDICAL_ROOT}/manifests"
CONFIGS_DIR="${MEDICAL_ROOT}/active_configs"

BUDGET=20
MC_SAMPLES=20

LOG_FILE="${MEDICAL_ROOT}/active_prompting_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Logging to: $LOG_FILE"

echo "============================================"
echo "  Medical Active Prompting — All SAM Heads"
echo "  GPU: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "  Budget: $BUDGET  MC Samples: $MC_SAMPLES"
echo "  Start: $(date)"
echo "============================================"
echo ""

# ============================================================================
# STEP 1: Generate filtered manifests per dataset
# ============================================================================
echo "[Step 1] Generating filtered manifests..."
$PYTHON -c "
import pandas as pd
df = pd.read_csv('${MANIFEST_DIR}/test.csv')
name_map = {'BUSI': 'busi', 'CheXmask': 'chexmask', 'DRIVE': 'drive',
            'ISIC2018': 'isic2018', 'KvasirSEG': 'kvasir_seg'}
for group, out_name in name_map.items():
    sub = df[df['group'] == group]
    out_path = '${MANIFEST_DIR}/test_' + out_name + '.csv'
    sub.to_csv(out_path, index=False)
    print(f'  {out_path}: {len(sub)} samples')
"
echo ""

# ============================================================================
# STEP 2: Generate per-run configs and run active prompting
# ============================================================================
mkdir -p "$CONFIGS_DIR"

# --- Helper function ---
run_active_prompting() {
    local RUN_ID="$1"
    local SAM_CKPT="$2"
    local HEAD_CKPT="$3"
    local LAPLACE_DIR="$4"
    local MANIFEST="$5"

    local CONFIG="${CONFIGS_DIR}/config_${RUN_ID}.yaml"

    cat > "$CONFIG" <<YAML
data:
  images_root: ${MEDICAL_ROOT}/images
  masks_root: ${MEDICAL_ROOT}/masks
  project_root: ${MEDICAL_ROOT}
  head_checkpoint: ${HEAD_CKPT}
  laplace_dir: ${LAPLACE_DIR}
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
  group: null
YAML

    echo "============================================"
    echo "  Run: ${RUN_ID}"
    echo "  SAM: ${SAM_CKPT}"
    echo "  Head: ${HEAD_CKPT}"
    echo "  Laplace: ${LAPLACE_DIR}"
    echo "  Manifest: ${MANIFEST}"
    echo "  $(date)"
    echo "============================================"

    $PYTHON active_prompting.py \
        --config "$CONFIG" \
        --run_id "$RUN_ID" \
        --budget $BUDGET \
        --mc_samples $MC_SAMPLES \
        --manifest "$MANIFEST" \
        --device "$DEVICE"

    echo "  DONE: ${RUN_ID} ($(date))"
    echo ""
}

# ============================================================================
# RUN 1: baseline_v3 — original SAM on ALL medical data
# ============================================================================
run_active_prompting \
    "baseline_v3" \
    "$BASELINE_SAM" \
    "${BALD_ROOT}/checkpoints/no_image/run_v3/best.pt" \
    "${BALD_ROOT}/laplace/no_image/run_v3" \
    "${MANIFEST_DIR}/test.csv"

# ============================================================================
# RUNS 2-6: Fine-tuned heads on their matched datasets
# ============================================================================
declare -A FT_MANIFEST_MAP=(
    [busi]="${MANIFEST_DIR}/test_busi.csv"
    [chexmask]="${MANIFEST_DIR}/test_chexmask.csv"
    [drive]="${MANIFEST_DIR}/test_drive.csv"
    [isic2018]="${MANIFEST_DIR}/test_isic2018.csv"
    [kvasir_seg]="${MANIFEST_DIR}/test_kvasir_seg.csv"
)

for NAME in busi chexmask drive isic2018 kvasir_seg; do
    run_active_prompting \
        "ft_${NAME}" \
        "${SAM_CKPT_DIR}/${NAME}/sam_vit_h_4b8939.pth" \
        "${BALD_ROOT}/sam_runs/${NAME}/checkpoints/no_image/run_${NAME}/best.pt" \
        "${BALD_ROOT}/sam_runs/${NAME}/laplace/no_image/run_${NAME}" \
        "${FT_MANIFEST_MAP[$NAME]}"
done

# ============================================================================
# RUN 7: ft_combined — combined SAM on ALL medical data
# ============================================================================
run_active_prompting \
    "ft_combined" \
    "${SAM_CKPT_DIR}/combined/sam_vit_h_4b8939.pth" \
    "${BALD_ROOT}/sam_runs/combined/checkpoints/no_image/run_combined/best.pt" \
    "${BALD_ROOT}/sam_runs/combined/laplace/no_image/run_combined" \
    "${MANIFEST_DIR}/test.csv"

# ============================================================================
# Summary
# ============================================================================
echo "============================================"
echo "  All runs complete!"
echo "  Finished: $(date)"
echo "============================================"
echo ""
echo "=== Output directories ==="
for RUN_ID in baseline_v3 ft_busi ft_chexmask ft_drive ft_isic2018 ft_kvasir_seg ft_combined; do
    DIR="${MEDICAL_ROOT}/active_prompting/run_${RUN_ID}"
    if [ -f "${DIR}/step_results.csv" ]; then
        ROWS=$(wc -l < "${DIR}/step_results.csv")
        echo "  ${RUN_ID}: ${DIR}/ (${ROWS} rows)"
    else
        echo "  ${RUN_ID}: ${DIR}/ (no results)"
    fi
done
