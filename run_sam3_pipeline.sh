#!/usr/bin/env bash
# ============================================================================
# SAM3 Pipeline: stages 3 → 6 using SAM3 backbone (sam3-env, Python 3.12)
#
# Stages 1-2 (manifests + prompt sets) are shared with SAM1 — not re-run.
# Stage 3 must be re-run: SAM3 logits differ from SAM1/SAM2 logits.
# Stages 4-6 are SAM-agnostic; they run against the new project_root.
#
# Prerequisites:
#   - sam3-env conda environment with the official facebookresearch/sam3 package
#     installed and Python 3.12
#   - SAM1 pipeline must have run stages 1-2 (manifests + prompt_sets)
#
# Usage: bash run_sam3_pipeline.sh [--run_id <id>]
# Default run_id: sam3_v1
# ============================================================================
set -euo pipefail

PYTHON=/home/prithwijit/miniconda3/envs/sam3-env/bin/python
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export CUDA_VISIBLE_DEVICES=0
CONFIG=config_sam3.yaml
RUN_ID="${1:-sam3_v1}"

LOG_FILE="/mnt/HD_18T_pt2/prithwijit/bald_sam3/sam3_pipeline_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "$LOG_FILE")"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "============================================"
echo "  SAM3 Pipeline"
echo "  Config: $CONFIG"
echo "  Run ID: $RUN_ID"
echo "  GPU: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "  Python: $PYTHON"
echo "  Log: $LOG_FILE"
echo "  Start: $(date)"
echo "============================================"
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

# ----------------------------------------------------------------------------
# NOTE: Stages 1-2 share SAM1 manifests and prompt sets.
# The SAM3 project_root symlinks them from SAM1.
# ----------------------------------------------------------------------------
SAM3_ROOT=/mnt/HD_18T_pt2/prithwijit/bald_sam3
SAM1_ROOT=/mnt/HD_18T_pt2/prithwijit/bald

mkdir -p "$SAM3_ROOT"

echo "[Setup] Symlinking manifests and prompt_sets from SAM1 project root..."
for dir in manifests prompt_sets; do
    if [ ! -e "$SAM3_ROOT/$dir" ]; then
        ln -s "$SAM1_ROOT/$dir" "$SAM3_ROOT/$dir"
        echo "  Linked: $SAM3_ROOT/$dir -> $SAM1_ROOT/$dir"
    else
        echo "  Already exists: $SAM3_ROOT/$dir"
    fi
done
echo ""

# ----------------------------------------------------------------------------
# Stage 3: Precompute SAM3 outputs
# ----------------------------------------------------------------------------
echo "============================================"
echo "  Stage 3: Precompute SAM3 Outputs"
echo "  $(date)"
echo "============================================"
$PYTHON precompute_sam3_outputs.py --config "$CONFIG"
echo "  Stage 3 DONE ($(date))"
echo ""

# ----------------------------------------------------------------------------
# Stage 4: Build training indices (SAM-agnostic)
# ----------------------------------------------------------------------------
echo "============================================"
echo "  Stage 4: Build Training Indices"
echo "  $(date)"
echo "============================================"
$PYTHON build_training_indices.py --config "$CONFIG"
echo "  Stage 4 DONE ($(date))"
echo ""

# ----------------------------------------------------------------------------
# Stage 4.5: Pack to HDF5 (SAM-agnostic)
# ----------------------------------------------------------------------------
echo "============================================"
echo "  Stage 4.5: Pack to HDF5"
echo "  $(date)"
echo "============================================"
$PYTHON pack_to_hdf5.py --config "$CONFIG"
echo "  Stage 4.5 DONE ($(date))"
echo ""

# ----------------------------------------------------------------------------
# Stage 5: Train deterministic head (SAM-agnostic)
# ----------------------------------------------------------------------------
echo "============================================"
echo "  Stage 5: Train NoImageHead"
echo "  Run ID: $RUN_ID"
echo "  $(date)"
echo "============================================"
$PYTHON train_no_image_head.py --config "$CONFIG" --run_id "$RUN_ID"
echo "  Stage 5 DONE ($(date))"
echo ""

# ----------------------------------------------------------------------------
# Stage 6: Fit Laplace (SAM-agnostic)
# ----------------------------------------------------------------------------
echo "============================================"
echo "  Stage 6: Fit Laplace Posterior"
echo "  Run ID: $RUN_ID"
echo "  $(date)"
echo "============================================"
$PYTHON fit_laplace_no_image.py --config "$CONFIG" --run_id "$RUN_ID"
echo "  Stage 6 DONE ($(date))"
echo ""

echo "============================================"
echo "  SAM3 Pipeline Complete!"
echo "  Finished: $(date)"
echo "============================================"
echo ""
echo "Artifacts:"
echo "  Head:    $SAM3_ROOT/checkpoints/no_image/run_${RUN_ID}/best.pt"
echo "  Laplace: $SAM3_ROOT/laplace/no_image/run_${RUN_ID}/laplace_state.pt"
echo "  Log:     $LOG_FILE"
