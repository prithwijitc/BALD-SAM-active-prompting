# Reproducibility Notes

This repository is staged for the PointPrompt and medical active-prompting results only. F3 seismic code/results are intentionally excluded from this first GitHub repo.

## Scope

Included:

- Frozen SAM/SAM2/SAM3 active prompting code.
- Prompt-aware Bayesian head training and diagonal Laplace fitting.
- Medical and PointPrompt analysis scripts.
- `BALD_SAM/all_tables.tex`, the paper table artifact to reproduce.
- Root medical archives currently available in this workspace: `KvasirSEG.zip` and `ISIC2018_masks.zip`.

Excluded:

- `f3_sam_uncertainty_v3/`
- `Story_TGRS/`
- `Story/`
- W&B run logs, checkpoints, SAM precompute arrays, trained head checkpoints, Laplace posterior artifacts, and other generated heavy outputs.

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install git+https://github.com/facebookresearch/segment-anything.git
```

Install any SAM2/SAM3 dependencies required by your local checkpoints separately; those model packages are not vendored here.

## Main Training Pipeline

Edit one of the config files so `data.images_root`, `data.masks_root`, `data.project_root`, and the SAM checkpoint paths point to your machine.

```bash
python build_manifests.py --config config.yaml
python generate_promptsets.py --config config.yaml
python precompute_sam_outputs.py --config config.yaml
python build_training_indices.py --config config.yaml
python pack_to_hdf5.py --config config.yaml
python train_no_image_head.py --config config.yaml --run_id v3
python fit_laplace_no_image.py --config config.yaml --run_id v3
```

The active prompting evaluation is then run with:

```bash
python active_prompting.py --config config.yaml --run_id v3 --budget 20 --mc_samples 20
```

Use the SAM2/SAM3 variants for those checkpoints:

```bash
python active_prompting_sam2.py --config config_medical_sam2.yaml --run_id sam2_v1
python active_prompting_sam3.py --config config_medical_sam3.yaml --run_id sam3_v1
```

## Inputs Expected by Table Scripts

The table scripts consume completed active-prompting run directories containing files such as `step_results.csv`, `aggregate.csv`, `per_group.csv`, and `run_meta.json`.

Set these environment variables if your run directories differ from the original workstation paths:

```bash
export BALD_ACTIVE_ROOT=/path/to/medical_data/active_prompting
export BALD_RESULTS_ROOT=$PWD/Medical_Results
export POINTPROMPT_RESULTS_ROOT="/path/to/pointprompt/Prompting results"
export POINTPROMPT_MANIFEST=/path/to/bald/manifests/test.csv
export POINTPROMPT_BALD_RUN=/path/to/bald/active_prompting/run_v3/point_prompt
export POINTPROMPT_OUT_DIR=/path/to/point_prompt_human_compare
```

## Rebuilding `BALD_SAM/all_tables.tex`

The checked-in `BALD_SAM/all_tables.tex` contains eight PointPrompt/medical tables:

1. `tab:BALD Laplace_vs_human_iou_dice`
2. `tab:miou_with_d2q`
3. `tab:overall_summary`
4. `tab:overall_summary_fixed_rivals_stats`
5. `tab:overall_scores`
6. `tab:strata_dominance`
7. `tab:shared_seed_next_prompt_trio`
8. `tab:sam123_iou_comparison`

Run the analysis scripts after completing the evaluation runs:

```bash
python build_pointprompt_human_trajectory_and_compare.py
python build_medical_bald_across_runs.py
python build_medical_sam_story.py
python build_active_prompting_experiment_report.py
python plot_sam123_bald_laplace_comparison.py
python plot_sam123_all_strategies_comparison.py
```

The scripts write summary CSVs and figures under `Medical_Results/` or the output directories above. `BALD_SAM/all_tables.tex` is the publication TeX assembled from these summaries.

## GitHub Push

After reviewing staged files:

```bash
git init
git add .
git status --short
git commit -m "Initial BALD-SAM active prompting release"
git branch -M main
git remote add origin git@github.com:<user>/<repo>.git
git push -u origin main
```
