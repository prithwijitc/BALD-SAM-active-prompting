#!/bin/bash
# Run active prompting on 5 medical datasets using SAM3 head + Laplace
# Datasets: ISIC2018 (200), KvasirSEG (200), BUSI (630), DRIVE (20), CheXmask (247)
# Total: 1297 samples
#
# Prerequisites:
#   - run_sam3_pipeline.sh must have completed (produces sam3_v1 head + laplace)
#   - Medical datasets available at /mnt/HD_18T_pt2/medical_data/
#
# Usage:
#   bash run_medical_sam3.sh
#   bash run_medical_sam3.sh --max_images 50   # quick test

set -e
cd "$(dirname "$0")"

PYTHON=/home/prithwijit/miniconda3/envs/sam3-env/bin/python
export CUDA_VISIBLE_DEVICES=0

EXTRA_ARGS="${@}"

LOG_FILE="/mnt/HD_18T_pt2/medical_data/active_prompting_sam3_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=== Active Prompting on Medical Datasets (SAM3) ==="
echo "Config: config_medical_sam3.yaml"
echo "Run ID: sam3_v1"
echo "Python: $PYTHON"
echo "GPU: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES (device=cuda:0)"
echo "Log: $LOG_FILE"
echo "Extra args: ${EXTRA_ARGS}"
echo ""

if ! "$PYTHON" - <<'PY'
import importlib.util
import sys

required = ("sam3", "pycocotools")
sys.exit(0 if all(importlib.util.find_spec(name) for name in required) else 1)
PY
then
    echo "ERROR: sam3-env is missing the official SAM3 runtime dependencies."
    echo "Install them with:"
    echo "  /home/prithwijit/miniconda3/envs/sam3-env/bin/pip install \\"
    echo "    'git+https://github.com/facebookresearch/sam3.git@bfbed072a07a6a52c8d5fdc75a7a186251a835b1'"
    echo "  /home/prithwijit/miniconda3/envs/sam3-env/bin/pip install pycocotools"
    exit 2
fi

$PYTHON active_prompting_sam3.py \
    --config config_medical_sam3.yaml \
    --run_id sam3_v1 \
    --budget 20 \
    --mc_samples 20 \
    --device cuda:0 \
    ${EXTRA_ARGS}

echo ""
echo "=== Active Prompting Complete ==="
echo "Results: /mnt/HD_18T_pt2/medical_data/active_prompting/run_sam3_v1/"

echo ""
echo "=== Generating Analysis & Plots ==="
$PYTHON analyze_medical_runs.py

echo "=== Done ==="
