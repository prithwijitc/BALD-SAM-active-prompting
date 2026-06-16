#!/bin/bash
# Full pipeline: manifest → prompts → SAM precompute → index → HDF5 → train → Laplace
#
# Usage:
#   bash run_pipeline.sh                        # full run, run_id=v1
#   bash run_pipeline.sh --run_id v2            # custom run id
#   bash run_pipeline.sh --resume               # resume training from last checkpoint
#   bash run_pipeline.sh --skip_precompute      # skip stages 1-4.5 (already done)
#   bash run_pipeline.sh --from_hdf5            # skip stages 1-4, start from stage 4.5
#   bash run_pipeline.sh --sanity_check         # stop after sanity check, do not train
#   bash run_pipeline.sh --config my.yaml       # custom config

set -euo pipefail

# ── defaults ────────────────────────────────────────────────────────────────
CONFIG="config.yaml"
RUN_ID="v1"
RESUME=""
SKIP_PRECOMPUTE=""
FROM_HDF5=""
SANITY_CHECK=""

# ── argument parsing ─────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)          CONFIG="$2";        shift 2 ;;
        --run_id)          RUN_ID="$2";        shift 2 ;;
        --resume)          RESUME="--resume";  shift   ;;
        --skip_precompute) SKIP_PRECOMPUTE=1;  shift   ;;
        --from_hdf5)       FROM_HDF5=1;        shift   ;;
        --sanity_check)    SANITY_CHECK=1;     shift   ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ── helpers ──────────────────────────────────────────────────────────────────
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# # ── pipeline ─────────────────────────────────────────────────────────────────
# if [[ -z "$SKIP_PRECOMPUTE" && -z "$FROM_HDF5" ]]; then

#     log "=== Stage 1: build manifests ==="
#     python build_manifests.py --config "$CONFIG"

#     log "=== Stage 2: generate prompt sets ==="
#     python generate_promptsets.py --config "$CONFIG"

#     log "=== Stage 3: SAM precompute (GPU) ==="
#     python precompute_sam_outputs.py --config "$CONFIG"

#     log "=== Stage 4: build training indices ==="
#     python build_training_indices.py --config "$CONFIG"

# fi

# if [[ -z "$SKIP_PRECOMPUTE" ]]; then

#     log "=== Stage 4.5: pack to HDF5 ==="
#     python pack_to_hdf5.py --config "$CONFIG"

# fi

# log "=== Sanity visualisation ==="
# python visualize_sanity.py --config "$CONFIG" --n 8 --split train

# log "=== Sanity check: overfit tiny batch ==="
# python train_no_image_head.py --config "$CONFIG" --run_id "$RUN_ID" --sanity_check

# if [[ -n "$SANITY_CHECK" ]]; then
#     log "--sanity_check flag set — stopping before full training."
#     exit 0
# fi

# log "=== Stage 5: train no-image head ==="
# python train_no_image_head.py --config "$CONFIG" --run_id "$RUN_ID" $RESUME

log "=== Stage 6: fit Laplace ==="
python fit_laplace_no_image.py --config "$CONFIG" --run_id "$RUN_ID"
