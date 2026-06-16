#!/usr/bin/env bash
# Run active prompting evaluation on GPU 1.
# Usage: bash run_active_prompting.sh

set -euo pipefail

PYTHON=/home/prithwijit/miniconda3/envs/dino-vit-feats-env/bin/python
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export CUDA_VISIBLE_DEVICES=1

echo "=== Active Prompting Evaluation ==="
echo "GPU: cuda:1 (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
echo "Start: $(date)"
echo ""

$PYTHON active_prompting.py \
    --config config.yaml \
    --run_id v2 \
    --budget 20 \
    --mc_samples 20 \
    --seeds 0 1 2 \
    --strategies bald_laplace bald_mcdropout entropy random maxdist oracle \
    --exclusion_radius 5 \
    --dropout_rate 0.1 \
    --device cuda:0 \
    --mc_ablation \
    --mc_ablation_images 20

echo ""
echo "Done: $(date)"
echo "Results: /mnt/HD_18T_pt2/prithwijit/bald/active_prompting/run_v2/"
