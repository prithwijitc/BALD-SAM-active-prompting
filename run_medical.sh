#!/usr/bin/env bash
# Run active prompting on 5 medical datasets using the SAM3 head + Laplace.
# Datasets: ISIC2018 (200), KvasirSEG (200), BUSI (630), DRIVE (20), CheXmask/SCR (247)
# Total: 1297 samples.
#
# Prerequisites:
#   - Datasets available at /mnt/HD_18T_pt2/medical_data/
#   - run_sam3_pipeline.sh completed with RUN_ID=sam3_v1
#   - config_medical_sam3.yaml points to:
#       /mnt/HD_18T_pt2/prithwijit/bald_sam3/checkpoints/no_image/run_sam3_v1/best.pt
#       /mnt/HD_18T_pt2/prithwijit/bald_sam3/laplace/no_image/run_sam3_v1/
#
# Usage:
#   bash run_medical.sh
#   bash run_medical.sh --max_images 50
#   BUDGET=10 MC_SAMPLES=50 bash run_medical.sh --force

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${PYTHON:-/home/prithwijit/miniconda3/envs/sam3-env/bin/python}"
MEDICAL_ROOT="${MEDICAL_ROOT:-/mnt/HD_18T_pt2/medical_data}"
CONFIG="${CONFIG:-config_medical_sam3.yaml}"
RUN_ID="${RUN_ID:-sam3_v1}"
BUDGET="${BUDGET:-20}"
MC_SAMPLES="${MC_SAMPLES:-50}"
DEVICE="${DEVICE:-cuda:0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

EXTRA_ARGS=("$@")

LOG_FILE="${MEDICAL_ROOT}/active_prompting_sam3_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=== Active Prompting on Medical Datasets ==="
echo "Backbone: SAM3"
echo "Config: ${CONFIG}"
echo "Run ID: ${RUN_ID}"
echo "Budget: ${BUDGET}"
echo "MC samples: ${MC_SAMPLES}"
echo "Python: ${PYTHON}"
echo "GPU: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} (device=${DEVICE})"
echo "Log: ${LOG_FILE}"
echo "Extra args: ${EXTRA_ARGS[*]:-(none)}"
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

"$PYTHON" - "$CONFIG" <<'PY'
import sys
from pathlib import Path

import yaml

config_path = Path(sys.argv[1])
cfg = yaml.safe_load(config_path.read_text())
head = Path(cfg["data"]["head_checkpoint"])
laplace_dir = Path(cfg["data"]["laplace_dir"])
required = [
    ("config", config_path),
    ("head checkpoint", head),
    ("laplace state", laplace_dir / "laplace_state.pt"),
]
missing = [f"{label}: {path}" for label, path in required if not path.exists()]
if missing:
    raise SystemExit("Missing required SAM3 artifacts:\n  " + "\n  ".join(missing))

print(f"Head checkpoint: {head}")
print(f"Laplace dir: {laplace_dir}")
PY
echo ""

"$PYTHON" active_prompting_sam3.py \
    --config "$CONFIG" \
    --run_id "$RUN_ID" \
    --budget "$BUDGET" \
    --mc_samples "$MC_SAMPLES" \
    --device "$DEVICE" \
    "${EXTRA_ARGS[@]}"

echo "=== Active Prompting Complete ==="
echo "Results: ${MEDICAL_ROOT}/active_prompting/run_${RUN_ID}/"

echo "=== Generating Analysis & Plots ==="
"$PYTHON" analyze_medical_runs.py

echo "=== Done ==="
