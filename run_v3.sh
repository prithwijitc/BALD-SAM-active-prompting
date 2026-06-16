#!/bin/bash
# Run pipeline stages 5, 6, and active prompting for run_id v3
# Stages 1-4.5 (data prep / SAM precompute) are reused from previous runs.

set -e
cd "$(dirname "$0")"

echo "=== Stage 5: Train no-image head (v3) ==="
python train_no_image_head.py --config config.yaml --run_id v3

echo "=== Stage 6: Fit Laplace (v3) ==="
python fit_laplace_no_image.py --config config.yaml --run_id v3

echo "=== Active Prompting (v3) ==="
python active_prompting.py --config config.yaml --run_id v3

echo "=== Done ==="
