# Prompt-Aware Laplace Head on Frozen SAM

A lightweight Bayesian segmentation refinement system built on top of frozen [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything). A small CNN head learns to refine SAM's predictions conditioned on prompt locations, and a diagonal Laplace approximation is fitted on its final layer for calibrated uncertainty estimation.

This GitHub staging copy contains the PointPrompt and medical active-prompting code/results needed for `BALD_SAM/all_tables.tex`. F3 seismic code/results are intentionally not included in this initial release. See `REPRODUCIBILITY.md` for table-level reproduction notes and expected input artifacts.

---

## Overview

Standard SAM predictions are prompt-conditioned but deterministic and uncalibrated. This project adds a **prompt-aware corrective head** that:

1. Takes SAM's logits (downsampled to 256×256) and Gaussian prompt maps (positive + negative) as input
2. Outputs a refined segmentation logit map
3. Has a **Laplace posterior** fitted on its final layer, enabling stochastic sampling for downstream uncertainty quantification (BALD, mutual information, prompt scoring)

SAM remains **fully frozen** throughout. All SAM inference is precomputed once and cached.

---

## Architecture

```
Input: [SAM logits, pos_map, neg_map]  →  (3, H, W)
         │
   Conv(3→16, 3×3) + ReLU
         │
   Conv(16→16, 3×3) + ReLU
         │
   Conv(16→1, 1×1)   ←── Laplace approximation fitted here
         │
Output: refined logits  →  (1, H, W)
```

Prompt maps are per-point Gaussians (σ=8 px) summed and clamped to [0, 1]. Input logits from SAM are stored as float16, clipped to [−20, 20] at training time.

---

## Pipeline

The system runs in six sequential stages:

```
Stage 1  build_manifests.py          Scan dataset, validate samples, create splits
Stage 2  generate_promptsets.py      Sample pos/neg point prompts per image, save JSON
Stage 3  precompute_sam_outputs.py   Run frozen SAM, cache logits + prompt maps
Stage 4    build_training_indices.py   Build flat CSV index pointing to precomputed files
Stage 4.5  pack_to_hdf5.py            Pack all arrays into per-split HDF5 for fast I/O
Stage 5    train_no_image_head.py     Train deterministic CNN head (MAP)
Stage 6    fit_laplace_no_image.py    Fit diagonal Laplace on final layer
```

Each stage is independently resumable. Completed work is never recomputed.

---

## Installation

```bash
# Clone and enter the project
git clone <repo-url>
cd Laplace_Head

# Install dependencies
pip install -r requirements.txt

# Install SAM (not on PyPI)
pip install git+https://github.com/facebookresearch/segment-anything.git
```

**Dependencies:** `torch>=2.0`, `numpy`, `pandas`, `Pillow`, `PyYAML`, `matplotlib`, `laplace-torch`, `wandb`

Download a SAM checkpoint from the [official repo](https://github.com/facebookresearch/segment-anything#model-checkpoints) (ViT-H recommended).

---

## Configuration

All stages share a single `config.yaml`. Edit the paths before running:

```yaml
data:
  images_root: /path/to/images_root   # nested: images_root/<class>/<stem>.jpg
  masks_root: /path/to/masks_root     # aligned: masks_root/<class>/<stem>.npy
  project_root: /path/to/project_root # all outputs written here
  split_seed: 123

prompt_generation:
  num_promptsets_train: 10
  num_promptsets_val: 4
  num_promptsets_test: 4
  min_pairs: 1
  max_pairs: 10
  prompt_seed: 777
  gaussian_sigma: 8

sam:
  checkpoint: /path/to/sam_vit_h.pth
  model_type: vit_h                   # vit_h | vit_l | vit_b
  device: cuda

head:
  channels_in: 3
  hidden_channels: 16

training:
  batch_size: 8
  num_epochs: 30
  lr: 0.001
  weight_decay: 0.0001
  dice_lambda: 0.5
  num_workers: 8
  amp: true
  early_stop_patience: 5
  logit_clip: 20.0

laplace:
  target_layer: final_conv
  approx: diagonal
  prior_precision_grid: [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]

wandb:
  enabled: true
  project: laplace-head-sam
  entity: null   # your wandb username or team
  group: null
```

---

## Usage

### Step 1 — Build manifests

```bash
python build_manifests.py --config config.yaml
```

Scans `images_root/` and `masks_root/`, validates every image/mask pair, and writes:
- `manifests/all_samples.csv` — full validated manifest
- `manifests/train.csv`, `val.csv`, `test.csv` — 80/10/10 stratified splits (fixed after first run)

Re-run with `--force` to recompute.

### Step 2 — Generate prompt sets

```bash
python generate_promptsets.py --config config.yaml
```

For each sample, samples point prompt pairs (k ∈ [1, 10]) from foreground/background pixels and writes `prompt_sets/<split>/<safe_id>.json`. Skips samples whose JSON already exists.

### Step 3 — Precompute SAM outputs

```bash
python precompute_sam_outputs.py --config config.yaml

# Process a single split only
python precompute_sam_outputs.py --config config.yaml --split train
```

Runs frozen SAM once per image (with embedding caching), evaluates all prompt sets, and saves per prompt set:

| File | dtype | Description |
|---|---|---|
| `logits_256.npy` | float16 | SAM logits postprocessed and downsampled to 256×256 |
| `pos_map.npy` | float16 | Gaussian map at positive points, built at 256×256 with rescaled coords |
| `neg_map.npy` | float16 | Same for negative points |
| `meta.json` | — | Points (original + rescaled), branch index, sigma, dtypes, logits_size |

SAM mask is not stored — derived as `logits_256 > 0` wherever needed.
Skips any prompt set whose required files already exist.

### Step 4 — Build training indices

```bash
python build_training_indices.py --config config.yaml
```

Writes flat CSV files with one row per (sample, prompt set). Key columns: `sample_id`, `safe_id`, `split`, `promptset_id`, `image_path`, `image_h`, `image_w`, `mask_path`, `logits_256_path`, `pos_map_path`, `neg_map_path`, `k_pairs`.

- `training_data/no_image/train_index.csv`
- `training_data/no_image/val_index.csv`
- `training_data/no_image/test_index.csv`

### Step 5 — Train the head

```bash
# Fresh run
python train_no_image_head.py --config config.yaml --run_id v1

# Resume from checkpoint
python train_no_image_head.py --config config.yaml --run_id v1 --resume

# Sanity check: overfit a tiny batch before full training
python train_no_image_head.py --config config.yaml --run_id v1 --sanity_check
```

Trains with AdamW + AMP + combined BCE/Dice loss. Saves checkpoints after every epoch; best checkpoint selected by validation IoU. Early stopping with configurable patience.

Checkpoint layout:
```
checkpoints/no_image/run_v1/
  config.yaml
  last.pt
  best.pt
  optimizer_last.pt
  scaler_last.pt
  epoch_state.json
```

### Step 6 — Fit Laplace

```bash
python fit_laplace_no_image.py --config config.yaml --run_id v1
```

Loads `best.pt`, fits diagonal last-layer Laplace via `laplace-torch`, tunes prior precision on validation NLL over the configured grid, and saves:

```
laplace/no_image/run_v1/
  laplace_state.pt
  laplace_config.yaml
  posterior_meta.json
```

Re-run with `--force` to refit even if artifacts exist.

### Sanity visualisation

Before training, inspect precomputed outputs for a few samples:

```bash
python visualize_sanity.py --config config.yaml --n 8 --split train
```

Saves 6-panel figures (image / GT mask / SAM mask / SAM logits / pos map / neg map) to `reports/debug_panels/`.

---

## Output Directory Structure

```
project_root/
  manifests/
  prompt_sets/        train/ val/ test/
  sam_precompute/     train/ val/ test/ <safe_id>/promptset_NNNN/
  training_data/      no_image/  train_index.csv  val_index.csv  test_index.csv
  checkpoints/        no_image/run_<id>/
  laplace/            no_image/run_<id>/
  logs/
  reports/
```

---

## Weights & Biases Logging

Set `wandb.enabled: true` in `config.yaml`. Each pipeline stage creates its own run under the configured project.

| Stage | Logged metrics |
|---|---|
| Manifest | valid/invalid counts, split sizes, per-group breakdown table, invalid sample table |
| Prompt generation | generated/skipped/failed counts per split |
| SAM precompute | done/skipped/failed promptset counts, logged every 100 processed |
| Training indices | rows, unique samples, mean k_pairs per split |
| Training | per-epoch train loss, val loss, val IoU, val Dice, best IoU, LR, patience; gradient/weight histograms via `wandb.watch`; best checkpoint as versioned model artifact |
| Laplace | per-grid-step prior precision vs val NLL; chosen precision; posterior artifacts |

To disable wandb without removing it: set `wandb.enabled: false`.

---

## Dataset Format

```
images_root/
  <class>/
    <stem>.jpg   (or .jpeg / .png)

masks_root/
  <class>/
    <stem>.npy   # 2D array, binarised as mask > 0
```

Masks must be 2D, have at least one foreground pixel, and not be entirely foreground. If mask and image sizes differ, the mask is resized with nearest-neighbour interpolation at prompt-generation time.

**Sample IDs:**
- Canonical: `<class>/<stem>` (used in CSVs and metadata)
- Safe: `<class>__<stem>` (used for directory names)

---

## Resumability

| Stage | Resume behaviour |
|---|---|
| Manifests | Skipped if all four CSVs exist (unless `--force`) |
| Prompt sets | Skipped per sample if JSON exists (unless `--force`) |
| SAM precompute | Skipped per prompt set if all required `.npy` + `meta.json` exist |
| Training | `--resume` loads `last.pt` + optimizer/scaler states; fails loudly if any file missing |
| Laplace | Skipped if artifacts exist and MD5 matches source checkpoint (unless `--force`) |

---

## Project Structure

```
.
├── config.yaml
├── build_manifests.py
├── generate_promptsets.py
├── precompute_sam_outputs.py
├── build_training_indices.py
├── train_no_image_head.py
├── fit_laplace_no_image.py
├── visualize_sanity.py
├── requirements.txt
└── modules/
    ├── scanner.py          Dataset scanning and image/mask pairing
    ├── validator.py        Per-sample integrity checks
    ├── promptset.py        Point sampling and Gaussian map construction
    ├── sam_wrapper.py      Frozen SAM with image-level embedding caching
    ├── head.py             NoImageHead model definition
    ├── dataset.py          PyTorch Dataset reading from index CSVs
    ├── losses.py           CombinedLoss (BCE + Dice)
    ├── metrics.py          IoU and Dice from logits
    ├── checkpoint.py       Save/load full training state
    ├── laplace_wrapper.py  Laplace fit, prior tuning, artifact I/O
    └── wandb_utils.py      Shared wandb init helper and no-op stub
```

---

## Acceptance Criteria

- [x] Correctly indexes nested folder data and creates manifests
- [x] Generates reproducible prompt sets (seeded per sample)
- [x] Precomputes and saves logits, mask, prompt maps, and metadata for every prompt set
- [x] Trains deterministic no-image head with BCE + Dice loss
- [x] Training is fully resumable from any checkpoint
- [x] Fits diagonal last-layer Laplace with prior precision tuning
- [x] Laplace artifacts are reloadable and fully documented
- [x] Completed preprocessing is never recomputed unnecessarily
