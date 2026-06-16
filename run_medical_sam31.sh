#!/bin/bash
# Run active prompting on 5 medical datasets using SAM3.1 head + Laplace
# Datasets: ISIC2018 (200), KvasirSEG (200), BUSI (630), DRIVE (20), CheXmask (247)
# Total: 1297 samples
#
# This script is intentionally disabled. The official facebook/sam3.1 release is
# a video Object Multiplex checkpoint, and this repo does not provide a
# validated single-image SAM3.1 active-prompting backend.
#
# Usage:
#   bash run_medical_sam31.sh
#   bash run_medical_sam31.sh --max_images 50   # quick test

set -e
cd "$(dirname "$0")"

PYTHON=/home/prithwijit/miniconda3/envs/sam3-env/bin/python
export CUDA_VISIBLE_DEVICES=0

EXTRA_ARGS="${@}"

LOG_FILE="/mnt/HD_18T_pt2/medical_data/active_prompting_sam31_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=== Active Prompting on Medical Datasets (SAM3.1) ==="
echo "Config: config_medical_sam31.yaml"
echo "Run ID: sam31_v1"
echo "Python: $PYTHON"
echo "GPU: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES (device=cuda:0)"
echo "Log: $LOG_FILE"
echo "Extra args: ${EXTRA_ARGS}"
echo ""

echo "ERROR: SAM3.1 medical active prompting is disabled in this repo."
echo "Use run_medical_sam3.sh for the supported image-point pipeline, or add a"
echo "dedicated SAM3.1 backend before re-enabling this script."
exit 2
