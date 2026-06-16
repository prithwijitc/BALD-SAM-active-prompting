#!/bin/bash
# Stages 5+6 only (HDF5 already built) — bigger head, v2 config
#
# Usage:
#   bash run_pipeline_bigger.sh                   # train + Laplace, run_id=v2
#   bash run_pipeline_bigger.sh --run_id v3       # custom run id
#   bash run_pipeline_bigger.sh --resume          # resume training from last.pt
#   bash run_pipeline_bigger.sh --sanity_check    # overfit check only, no full train
#   bash run_pipeline_bigger.sh --skip_train      # skip training, refit Laplace only
#   bash run_pipeline_bigger.sh --force_laplace   # refit Laplace even if artifacts exist

set -euo pipefail

# ── defaults ─────────────────────────────────────────────────────────────────
CONFIG="config.yaml"
RUN_ID="v2"
RESUME=""
SANITY_CHECK=""
SKIP_TRAIN=""
FORCE_LAPLACE=""

# ── argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)        CONFIG="$2";          shift 2 ;;
        --run_id)        RUN_ID="$2";          shift 2 ;;
        --resume)        RESUME="--resume";    shift   ;;
        --sanity_check)  SANITY_CHECK=1;       shift   ;;
        --skip_train)    SKIP_TRAIN=1;         shift   ;;
        --force_laplace) FORCE_LAPLACE="--force"; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ── helpers ───────────────────────────────────────────────────────────────────
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "=== Config:  $CONFIG ==="
log "=== Run ID:  $RUN_ID ==="

# ── Stage 5: sanity check ─────────────────────────────────────────────────────
log "=== Stage 5a: sanity check (overfit 2 batches) ==="
python train_no_image_head.py --config "$CONFIG" --run_id "$RUN_ID" --sanity_check

if [[ -n "$SANITY_CHECK" ]]; then
    log "=== --sanity_check flag set — stopping. ==="
    exit 0
fi

# ── Stage 5: full training ────────────────────────────────────────────────────
if [[ -z "$SKIP_TRAIN" ]]; then
    log "=== Stage 5b: train no-image head ==="
    python train_no_image_head.py --config "$CONFIG" --run_id "$RUN_ID" $RESUME
else
    log "=== --skip_train set — skipping training. ==="
fi

# ── Stage 6: Laplace ──────────────────────────────────────────────────────────
log "=== Stage 6: fit Laplace ==="
python fit_laplace_no_image.py --config "$CONFIG" --run_id "$RUN_ID" $FORCE_LAPLACE

log "=== Pipeline complete. Run ID: $RUN_ID ==="
