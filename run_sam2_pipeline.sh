#!/usr/bin/env bash
# ============================================================================
# SAM2 Pipeline: stages 3 → 6 using SAM2 backbone (hippo env, Python 3.10)
#
# Stages 1-2 (manifests + prompt sets) are shared with SAM1 — not re-run.
# Stage 3 must be re-run: SAM2 logits differ from SAM1 logits.
# Stages 4-6 are SAM-agnostic; they run against the new project_root.
#
# Usage: bash run_sam2_pipeline.sh [--run_id <id>]
# Default run_id: sam2_v1
# ============================================================================
set -euo pipefail

PYTHON=/home/prithwijit/miniconda3/envs/hippo/bin/python
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export CUDA_VISIBLE_DEVICES=1
CONFIG=config_sam2.yaml
RUN_ID="${1:-sam2_v1}"

LOG_FILE="/mnt/HD_18T_pt2/prithwijit/bald_sam2/sam2_pipeline_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "$LOG_FILE")"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "============================================"
echo "  SAM2 Pipeline"
echo "  Config: $CONFIG"
echo "  Run ID: $RUN_ID"
echo "  GPU: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "  Python: $PYTHON"
echo "  Log: $LOG_FILE"
echo "  Start: $(date)"
echo "============================================"
echo ""

# ----------------------------------------------------------------------------
# NOTE: Stages 1-2 share SAM1 manifests and prompt sets.
# The SAM2 project_root must symlink or copy them first if not already present.
# ----------------------------------------------------------------------------
SAM2_ROOT=/mnt/HD_18T_pt2/prithwijit/bald_sam2
SAM1_ROOT=/mnt/HD_18T_pt2/prithwijit/bald

mkdir -p "$SAM2_ROOT"

echo "[Setup] Symlinking manifests and prompt_sets from SAM1 project root..."
for dir in manifests prompt_sets; do
    if [ ! -e "$SAM2_ROOT/$dir" ]; then
        ln -s "$SAM1_ROOT/$dir" "$SAM2_ROOT/$dir"
        echo "  Linked: $SAM2_ROOT/$dir -> $SAM1_ROOT/$dir"
    else
        echo "  Already exists: $SAM2_ROOT/$dir"
    fi
done
echo ""

# ----------------------------------------------------------------------------
# Stage 3: Precompute SAM2 outputs
# ----------------------------------------------------------------------------
echo "============================================"
echo "  Stage 3: Precompute SAM2 Outputs"
echo "  $(date)"
echo "============================================"
$PYTHON precompute_sam2_outputs.py --config "$CONFIG"
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
echo "  SAM2 Pipeline Complete!"
echo "  Finished: $(date)"
echo "============================================"
echo ""
echo "Artifacts:"
echo "  Head:   $SAM2_ROOT/checkpoints/no_image/run_${RUN_ID}/best.pt"
echo "  Laplace: $SAM2_ROOT/laplace/no_image/run_${RUN_ID}/laplace_state.pt"
echo "  Log:    $LOG_FILE"
