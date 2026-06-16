#!/usr/bin/env python3
"""
Build a test manifest CSV for the new datasets (camouflage, cracks,
salient objects, dichotomous segmentation).

Each dataset folder must contain:
    <dataset>/Images/*.jpg (or .png)
    <dataset>/Masks/*.npy  (2D binary uint8)

Usage:
    # Default: subsample large datasets to 500, keep small ones as-is
    python build_newdata_manifest.py --config config_newdata.yaml

    # Custom cap per dataset
    python build_newdata_manifest.py --config config_newdata.yaml --max_per_dataset 300

    # No cap (use all images)
    python build_newdata_manifest.py --config config_newdata.yaml --max_per_dataset 0
"""

import argparse
import csv
import numpy as np
from pathlib import Path
from PIL import Image
from collections import Counter
import yaml

# Datasets and their default subsample caps (0 = use all)
DATASET_CAPS = {
    "CAMO":         500,   # 1250 total
    "CHAMELEON":    0,     # 76 total — use all
    "COD10K":       500,   # 10000 total
    "CRACK500":     500,   # 3618 total
    "CrackForest-dataset": 0,  # 118 total — use all
    "CrackTree200": 0,    # 206 total — use all
    "DIS5K":        500,   # 5470 total
    "DeepCrack":    0,     # 537 total — use all
    "ECSSD":        500,   # 1000 total
    "PASCAL-S":     500,   # 850 total
    "USOD":         0,     # 300 total — use all
    "USOD10K":      500,   # 10255 total
}

FIELDS = [
    "sample_id", "safe_id", "group", "image_path", "mask_path",
    "image_h", "image_w", "mask_h", "mask_w", "fg_pixels", "bg_pixels",
    "valid", "note",
]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def find_image_mask_pairs(img_dir, msk_dir):
    """Find all (image, mask) pairs. Images can be any format, masks are .npy."""
    pairs = []
    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue
        stem = img_path.stem
        msk_path = msk_dir / f"{stem}.npy"
        if msk_path.exists():
            pairs.append((img_path, msk_path, stem))
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Build test manifest for new datasets")
    parser.add_argument("--config", required=True, help="Path to config_newdata.yaml")
    parser.add_argument("--max_per_dataset", type=int, default=-1,
                        help="Override max samples per dataset. "
                             "-1=use defaults, 0=all, N=cap at N")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for subsampling")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    datasets_root = Path(cfg["data"]["datasets_root"])
    project_root = Path(cfg["data"]["project_root"])
    out_dir = project_root / "manifests"
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    rows = []

    # Discover datasets: use DATASET_CAPS if defined, else auto-discover
    dataset_dirs = []
    for ds_name in DATASET_CAPS:
        ds_path = datasets_root / ds_name
        if ds_path.exists():
            dataset_dirs.append((ds_name, ds_path))
        else:
            print(f"  SKIP {ds_name}: not found at {ds_path}")

    # Also auto-discover any additional dataset folders not in DATASET_CAPS
    for d in sorted(datasets_root.iterdir()):
        if d.is_dir() and d.name not in DATASET_CAPS:
            img_sub = d / "Images"
            msk_sub = d / "Masks"
            if img_sub.exists() and msk_sub.exists():
                dataset_dirs.append((d.name, d))
                print(f"  AUTO-DISCOVERED: {d.name}")

    print(f"Found {len(dataset_dirs)} datasets\n")

    for ds_name, ds_path in dataset_dirs:
        img_dir = ds_path / "Images"
        msk_dir = ds_path / "Masks"

        if not img_dir.exists() or not msk_dir.exists():
            print(f"  SKIP {ds_name}: missing Images/ or Masks/ subdir")
            continue

        pairs = find_image_mask_pairs(img_dir, msk_dir)
        if not pairs:
            print(f"  SKIP {ds_name}: no valid image-mask pairs found")
            continue

        # Determine cap
        if args.max_per_dataset >= 0:
            cap = args.max_per_dataset  # CLI override
        else:
            cap = DATASET_CAPS.get(ds_name, 500)  # default cap

        # Subsample if needed (cap=0 means use all)
        if cap > 0 and len(pairs) > cap:
            indices = rng.choice(len(pairs), cap, replace=False)
            indices.sort()
            pairs = [pairs[i] for i in indices]

        n_ok = 0
        n_skip = 0
        for img_path, msk_path, stem in pairs:
            try:
                with Image.open(img_path) as img:
                    w, h = img.size
                mask = np.load(msk_path)
                fg = int(mask.sum())
                bg = int(mask.size - fg)
                mh, mw = mask.shape

                # Basic validation
                if fg == 0 or bg == 0:
                    n_skip += 1
                    continue
                if mask.ndim != 2:
                    n_skip += 1
                    continue

                sample_id = f"{ds_name}/{stem}"
                safe_id = f"{ds_name}__{stem}"

                rows.append({
                    "sample_id": sample_id,
                    "safe_id": safe_id,
                    "group": ds_name,
                    "image_path": str(img_path),
                    "mask_path": str(msk_path),
                    "image_h": h,
                    "image_w": w,
                    "mask_h": mh,
                    "mask_w": mw,
                    "fg_pixels": fg,
                    "bg_pixels": bg,
                    "valid": 1,
                    "note": "ok",
                })
                n_ok += 1
            except Exception as e:
                n_skip += 1
                continue

        print(f"  {ds_name}: {n_ok} samples (skipped {n_skip}, from {n_ok + n_skip} pairs)")

    # Write manifest
    out_path = out_dir / "test.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nTotal: {len(rows)} samples written to {out_path}")
    print("\nPer group:")
    counts = Counter(r["group"] for r in rows)
    for g, c in sorted(counts.items()):
        print(f"  {g}: {c}")


if __name__ == "__main__":
    main()
