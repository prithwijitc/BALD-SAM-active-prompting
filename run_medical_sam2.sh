#!/bin/bash
# Run active prompting on 5 medical datasets using SAM2 head + Laplace
# Datasets: ISIC2018 (200), KvasirSEG (200), BUSI (630), DRIVE (20), CheXmask (247)
# Total: 1297 samples
#
# Prerequisites:
#   - run_sam2_pipeline.sh must have completed (produces sam2_v1 head + laplace)
#   - Medical datasets available at /mnt/HD_18T_pt2/medical_data/
#
# Usage:
#   bash run_medical_sam2.sh
#   bash run_medical_sam2.sh --max_images 50   # quick test

set -e
cd "$(dirname "$0")"

PYTHON=/home/prithwijit/miniconda3/envs/hippo/bin/python
export CUDA_VISIBLE_DEVICES=1

EXTRA_ARGS="${@}"

LOG_FILE="/mnt/HD_18T_pt2/medical_data/active_prompting_sam2_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=== Active Prompting on Medical Datasets (SAM2) ==="
echo "Config: config_medical_sam2.yaml"
echo "Run ID: sam2_v1"
echo "Python: $PYTHON"
echo "GPU: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES (device=cuda:0)"
echo "Log: $LOG_FILE"
echo "Extra args: ${EXTRA_ARGS}"
echo ""

$PYTHON active_prompting_sam2.py \
    --config config_medical_sam2.yaml \
    --run_id sam2_v1 \
    --budget 20 \
    --mc_samples 20 \
    --device cuda:0 \
    ${EXTRA_ARGS}

echo ""
echo "=== Active Prompting Complete ==="
echo "Results: /mnt/HD_18T_pt2/medical_data/active_prompting/run_sam2_v1/"

echo ""
echo "=== Generating Analysis & Plots ==="
$PYTHON analyze_medical_runs.py

echo "=== Done ==="
