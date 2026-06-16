#!/usr/bin/env bash
# ============================================================================
# SAM3.1 Pipeline: disabled pending a dedicated SAM3.1 backend
#
# The official facebook/sam3.1 release is an Object Multiplex video-tracking
# checkpoint. This repo does not implement a validated single-image SAM3.1
# point-prompt path, so this script exits immediately instead of running an
# unsafe checkpoint/model combination.
#
# Usage: bash run_sam31_pipeline.sh [--run_id <id>]
# Default run_id: sam31_v1
# ============================================================================
set -euo pipefail

PYTHON=/home/prithwijit/miniconda3/envs/sam3-env/bin/python
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export CUDA_VISIBLE_DEVICES=0
CONFIG=config_sam31.yaml
RUN_ID="${1:-sam31_v1}"

LOG_FILE="/mnt/HD_18T_pt2/prithwijit/bald_sam31/sam31_pipeline_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "$LOG_FILE")"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "============================================"
echo "  SAM3.1 Pipeline"
echo "  Config: $CONFIG"
echo "  Run ID: $RUN_ID"
echo "  GPU: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "  Python: $PYTHON"
echo "  Log: $LOG_FILE"
echo "  Start: $(date)"
echo "============================================"
echo ""

echo "ERROR: SAM3.1 is not supported by this image point-prompt pipeline."
echo "The official facebook/sam3.1 release is a video Object Multiplex checkpoint."
echo "Use run_sam3_pipeline.sh for the supported SAM3 image path, or add a"
echo "dedicated SAM3.1 backend before re-enabling this script."
exit 2
