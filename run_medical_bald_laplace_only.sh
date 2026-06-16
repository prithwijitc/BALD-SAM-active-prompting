#!/usr/bin/env bash
# ============================================================================
# Run only BALD (Laplace) active prompting on medical data.
#
# Output run_ids are intentionally new because active_prompting resume state is
# tracked per (sample_id, seed), not per strategy. Reusing old multi-strategy
# run directories would skip work incorrectly.
#
# Runs:
#   baseline_v3_baldlaplace      - baseline v3 on ALL medical data
#   ft_busi_baldlaplace          - BUSI finetune on BUSI only
#   ft_chexmask_baldlaplace      - CheXmask finetune on CheXmask only
#   ft_drive_baldlaplace         - DRIVE finetune on DRIVE only
#   ft_isic2018_baldlaplace      - ISIC2018 finetune on ISIC2018 only
#   ft_kvasir_seg_baldlaplace    - KvasirSEG finetune on KvasirSEG only
#   ft_combined_baldlaplace      - combined finetune on ALL medical data
#   sam2_v1_baldlaplace          - SAM2 head + Laplace on ALL medical data
#
# Usage:
#   bash run_medical_bald_laplace_only.sh
#   bash run_medical_bald_laplace_only.sh --max_images 50
#   bash run_medical_bald_laplace_only.sh --force
# ============================================================================
set -euo pipefail

GENERIC_PYTHON=/home/prithwijit/miniconda3/envs/dino-vit-feats-env/bin/python
SAM2_PYTHON=/home/prithwijit/miniconda3/envs/hippo/bin/python

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Restrict the whole run to physical GPU 1 only.
export CUDA_VISIBLE_DEVICES=1
DEVICE="cuda:0"

MEDICAL_ROOT=/mnt/HD_18T_pt2/medical_data
BALD_ROOT=/mnt/HD_18T_pt2/prithwijit/bald
SAM_CKPT_DIR="${BALD_ROOT}/SAM_ckpt_full"
BASELINE_SAM=/home/prithwijit/BALD/sam_vit_h_4b8939.pth
MANIFEST_DIR="${MEDICAL_ROOT}/manifests"
CONFIGS_DIR="${MEDICAL_ROOT}/active_configs_bald_laplace"

BUDGET="${BUDGET:-20}"
MC_SAMPLES="${MC_SAMPLES:-50}"
STRATEGY=bald_laplace

EXTRA_ARGS=("$@")

LOG_FILE="${MEDICAL_ROOT}/active_prompting_bald_laplace_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "============================================"
echo "  Medical Active Prompting - BALD Laplace Only"
echo "  GPU: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "  Device: ${DEVICE}"
echo "  Strategy: ${STRATEGY}"
echo "  Budget: ${BUDGET}"
echo "  MC Samples: ${MC_SAMPLES}"
echo "  Generic Python: ${GENERIC_PYTHON}"
echo "  SAM2 Python: ${SAM2_PYTHON}"
echo "  Extra args: ${EXTRA_ARGS[*]:-(none)}"
echo "  Start: $(date)"
echo "  Log: ${LOG_FILE}"
echo "============================================"
echo ""

mkdir -p "$CONFIGS_DIR"

echo "[Step 1] Generating filtered manifests..."
"$GENERIC_PYTHON" -c "
import pandas as pd
df = pd.read_csv('${MANIFEST_DIR}/test.csv')
name_map = {
    'BUSI': 'busi',
    'CheXmask': 'chexmask',
    'DRIVE': 'drive',
    'ISIC2018': 'isic2018',
    'KvasirSEG': 'kvasir_seg',
}
for group, out_name in name_map.items():
    sub = df[df['group'] == group]
    out_path = '${MANIFEST_DIR}/test_' + out_name + '.csv'
    sub.to_csv(out_path, index=False)
    print(f'  {out_path}: {len(sub)} samples')
"
echo ""

write_generic_config() {
    local config_path="$1"
    local sam_ckpt="$2"
    local head_ckpt="$3"
    local laplace_dir="$4"

    cat > "$config_path" <<YAML
data:
  images_root: ${MEDICAL_ROOT}/images
  masks_root: ${MEDICAL_ROOT}/masks
  project_root: ${MEDICAL_ROOT}
  head_checkpoint: ${head_ckpt}
  laplace_dir: ${laplace_dir}
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
  checkpoint: ${sam_ckpt}
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
}

run_generic_bald() {
    local out_run_id="$1"
    local sam_ckpt="$2"
    local head_ckpt="$3"
    local laplace_dir="$4"
    local manifest="$5"

    local config_path="${CONFIGS_DIR}/config_${out_run_id}.yaml"
    write_generic_config "$config_path" "$sam_ckpt" "$head_ckpt" "$laplace_dir"

    echo "============================================"
    echo "  Run: ${out_run_id}"
    echo "  Script: active_prompting.py"
    echo "  SAM: ${sam_ckpt}"
    echo "  Head: ${head_ckpt}"
    echo "  Laplace: ${laplace_dir}"
    echo "  Manifest: ${manifest}"
    echo "  $(date)"
    echo "============================================"

    "$GENERIC_PYTHON" active_prompting.py \
        --config "$config_path" \
        --run_id "$out_run_id" \
        --budget "$BUDGET" \
        --mc_samples "$MC_SAMPLES" \
        --strategies "$STRATEGY" \
        --manifest "$manifest" \
        --device "$DEVICE" \
        "${EXTRA_ARGS[@]}"

    echo "  DONE: ${out_run_id} ($(date))"
    echo ""
}

run_sam2_bald() {
    local out_run_id="sam2_v1_baldlaplace"
    local manifest="${MANIFEST_DIR}/test.csv"

    echo "============================================"
    echo "  Run: ${out_run_id}"
    echo "  Script: active_prompting_sam2.py"
    echo "  Config: config_medical_sam2.yaml"
    echo "  Manifest: ${manifest}"
    echo "  $(date)"
    echo "============================================"

    "$SAM2_PYTHON" active_prompting_sam2.py \
        --config config_medical_sam2.yaml \
        --run_id "$out_run_id" \
        --budget "$BUDGET" \
        --mc_samples "$MC_SAMPLES" \
        --strategies "$STRATEGY" \
        --manifest "$manifest" \
        --device "$DEVICE" \
        "${EXTRA_ARGS[@]}"

    echo "  DONE: ${out_run_id} ($(date))"
    echo ""
}

# baseline on all medical data
run_generic_bald \
    "baseline_v3_baldlaplace" \
    "$BASELINE_SAM" \
    "${BALD_ROOT}/checkpoints/no_image/run_v3/best.pt" \
    "${BALD_ROOT}/laplace/no_image/run_v3" \
    "${MANIFEST_DIR}/test.csv"

# dataset-specific finetunes on matched datasets only
declare -A FT_MANIFEST_MAP=(
    [busi]="${MANIFEST_DIR}/test_busi.csv"
    [chexmask]="${MANIFEST_DIR}/test_chexmask.csv"
    [drive]="${MANIFEST_DIR}/test_drive.csv"
    [isic2018]="${MANIFEST_DIR}/test_isic2018.csv"
    [kvasir_seg]="${MANIFEST_DIR}/test_kvasir_seg.csv"
)

for name in busi chexmask drive isic2018 kvasir_seg; do
    run_generic_bald \
        "ft_${name}_baldlaplace" \
        "${SAM_CKPT_DIR}/${name}/sam_vit_h_4b8939.pth" \
        "${BALD_ROOT}/sam_runs/${name}/checkpoints/no_image/run_${name}/best.pt" \
        "${BALD_ROOT}/sam_runs/${name}/laplace/no_image/run_${name}" \
        "${FT_MANIFEST_MAP[$name]}"
done

# combined finetune on all medical data
run_generic_bald \
    "ft_combined_baldlaplace" \
    "${SAM_CKPT_DIR}/combined/sam_vit_h_4b8939.pth" \
    "${BALD_ROOT}/sam_runs/combined/checkpoints/no_image/run_combined/best.pt" \
    "${BALD_ROOT}/sam_runs/combined/laplace/no_image/run_combined" \
    "${MANIFEST_DIR}/test.csv"

# SAM2 on all medical data
run_sam2_bald

echo "============================================"
echo "  BALD Laplace runs complete"
echo "  Finished: $(date)"
echo "============================================"
echo ""

echo "=== Output directories ==="
for run_id in \
    baseline_v3_baldlaplace \
    ft_busi_baldlaplace \
    ft_chexmask_baldlaplace \
    ft_drive_baldlaplace \
    ft_isic2018_baldlaplace \
    ft_kvasir_seg_baldlaplace \
    ft_combined_baldlaplace \
    sam2_v1_baldlaplace
do
    dir="${MEDICAL_ROOT}/active_prompting/run_${run_id}"
    if [ -f "${dir}/step_results.csv" ]; then
        rows=$(wc -l < "${dir}/step_results.csv")
        echo "  ${run_id}: ${dir}/ (${rows} rows)"
    else
        echo "  ${run_id}: ${dir}/ (no results)"
    fi
done
