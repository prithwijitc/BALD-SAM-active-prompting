#!/bin/bash
# ============================================================
# Run active prompting on new datasets (camouflage, cracks,
# salient objects, dichotomous segmentation)
#
# Datasets: CAMO, CHAMELEON, COD10K, CRACK500, CrackForest,
#           CrackTree200, DIS5K, DeepCrack, ECSSD, PASCAL-S,
#           USOD, USOD10K
#
# SETUP (run once before first use):
#   1. Edit config_newdata.yaml — set datasets_root, project_root,
#      head_checkpoint, laplace_dir, and sam checkpoint paths
#   2. Ensure each dataset has Images/ and Masks/ (with .npy binary masks)
#   3. Build manifest:
#        python build_newdata_manifest.py --config config_newdata.yaml
#
# USAGE:
#   bash run_newdata.sh                         # run all samples (default caps per dataset)
#   bash run_newdata.sh --max_images 500        # cap total images to 500
#   bash run_newdata.sh --max_images 100        # quick test with 100 images
#   bash run_newdata.sh --budget 10             # only 10 prompting steps
#   bash run_newdata.sh --device cuda:1         # specific GPU
#
# The --max_images flag caps the TOTAL number of images from the manifest.
# To cap PER-DATASET, use build_newdata_manifest.py --max_per_dataset N
# ============================================================

set -e
cd "$(dirname "$0")"

CONFIG="config_newdata.yaml"
RUN_ID="v3"
BUDGET=20
MC_SAMPLES=20

# Parse our own flags, pass the rest through
EXTRA_ARGS=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)    CONFIG="$2"; shift 2 ;;
        --run_id)    RUN_ID="$2"; shift 2 ;;
        --budget)    BUDGET="$2"; shift 2 ;;
        *)           EXTRA_ARGS="${EXTRA_ARGS} $1"; shift ;;
    esac
done

echo "=============================================="
echo " Active Prompting — New Datasets"
echo "=============================================="
echo " Config:     ${CONFIG}"
echo " Run ID:     ${RUN_ID}"
echo " Budget:     ${BUDGET}"
echo " MC samples: ${MC_SAMPLES}"
echo " Extra args: ${EXTRA_ARGS}"
echo "=============================================="

# Step 1: Build manifest if it doesn't exist
PROJECT_ROOT=$(python3 -c "import yaml; print(yaml.safe_load(open('${CONFIG}'))['data']['project_root'])")
MANIFEST="${PROJECT_ROOT}/manifests/test.csv"

if [ ! -f "${MANIFEST}" ]; then
    echo ""
    echo ">>> Building manifest..."
    python build_newdata_manifest.py --config "${CONFIG}"
    echo ""
fi

echo "Manifest: ${MANIFEST}"
echo "Samples:  $(tail -n +2 "${MANIFEST}" | wc -l)"
echo ""

# Step 2: Run active prompting
echo ">>> Running active prompting..."
python active_prompting.py \
    --config "${CONFIG}" \
    --run_id "${RUN_ID}" \
    --budget "${BUDGET}" \
    --mc_samples "${MC_SAMPLES}" \
    ${EXTRA_ARGS}

echo ""
echo "=== Active Prompting Complete ==="
echo "Results: ${PROJECT_ROOT}/active_prompting/run_${RUN_ID}/"
echo ""
echo "To generate plots, run:"
echo "  python plot_newdata_results.py --config ${CONFIG} --run_id ${RUN_ID}"
