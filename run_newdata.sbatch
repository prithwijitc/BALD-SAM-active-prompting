#!/bin/bash
#SBATCH -J bald-newdata
#SBATCH -A gts-ga42
#SBATCH -q inferno
#SBATCH -N 1 --ntasks-per-node=1
#SBATCH --gres=gpu:H200:1
#SBATCH --mem=32G
#SBATCH -t 36:00:00
#SBATCH -o slurm-%j-newdata.out
#SBATCH -e slurm-%j-newdata.err

# ============================================================
# PACE SLURM job: Build manifest + run active prompting
# on 12 new datasets (camouflage, cracks, salient, DIS)
#
# Usage:
#   sbatch run_newdata.sbatch
#   sbatch --gres=gpu:H100:1 run_newdata.sbatch   # different GPU
#   sbatch -t 12:00:00 run_newdata.sbatch          # shorter wall time
# ============================================================

cd "$SLURM_SUBMIT_DIR"

# Load conda
module load anaconda3
conda activate vit-attn

CONFIG="config_newdata.yaml"
RUN_ID="v3"
BUDGET=20
MC_SAMPLES=20

echo "=============================================="
echo " PACE Job: Active Prompting — New Datasets"
echo "=============================================="
echo " Job ID:     ${SLURM_JOB_ID}"
echo " Node:       $(hostname)"
echo " GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo " Config:     ${CONFIG}"
echo " Run ID:     ${RUN_ID}"
echo " Budget:     ${BUDGET}"
echo " MC samples: ${MC_SAMPLES}"
echo "=============================================="

# Step 1: Build manifest if it doesn't exist
PROJECT_ROOT=$(python3 -c "import yaml; print(yaml.safe_load(open('${CONFIG}'))['data']['project_root'])")
MANIFEST="${PROJECT_ROOT}/manifests/test.csv"

if [ ! -f "${MANIFEST}" ]; then
    echo ""
    echo ">>> Building manifest..."
    srun python -u build_newdata_manifest.py --config "${CONFIG}"
    echo ""
fi

echo "Manifest: ${MANIFEST}"
echo "Samples:  $(tail -n +2 "${MANIFEST}" | wc -l)"
echo ""

# Step 2: Run active prompting
echo ">>> Running active prompting..."
srun python -u active_prompting.py \
    --config "${CONFIG}" \
    --run_id "${RUN_ID}" \
    --budget "${BUDGET}" \
    --mc_samples "${MC_SAMPLES}"

echo ""
echo "=== Active Prompting Complete ==="
echo "Results: ${PROJECT_ROOT}/active_prompting/run_${RUN_ID}/"
echo ""
echo "To generate plots, run:"
echo "  python plot_newdata_results.py --config ${CONFIG} --run_id ${RUN_ID}"
