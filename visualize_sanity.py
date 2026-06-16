"""
Sanity visualization: save debug panels for a small number of samples before training.

Usage:
    python visualize_sanity.py --config config.yaml --n 8 --split train
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from PIL import Image


def _resize_nearest(arr: np.ndarray, th: int, tw: int) -> np.ndarray:
    if arr.shape == (th, tw):
        return arr
    t = torch.from_numpy(arr.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    return F.interpolate(t, size=(th, tw), mode="nearest").squeeze().numpy()


def visualize_sample(row: dict, out_dir: Path, idx: int, target_size: tuple):
    th, tw = target_size
    image_h, image_w = int(row["image_h"]), int(row["image_w"])

    # original image (shown at native res)
    img = np.array(Image.open(row["image_path"]).convert("RGB"))

    # GT mask — two-step alignment matching the training pipeline:
    #   raw mask → image space → target space
    mask_raw = (np.load(row["mask_path"]) > 0).astype(np.float32)
    if mask_raw.shape != (image_h, image_w):
        mask_raw = _resize_nearest(mask_raw, image_h, image_w)
    gt = _resize_nearest(mask_raw, th, tw)

    # SAM logits at target size
    logits = np.load(row["logits_256_path"]).astype(np.float32)
    logits = _resize_nearest(logits.astype(np.float32), th, tw)  # bilinear would be better but nearest is fine for viz
    sam_mask = (logits > 0).astype(np.float32)  # derived — no separate file

    pos_map = np.load(row["pos_map_path"]).astype(np.float32)
    neg_map = np.load(row["neg_map_path"]).astype(np.float32)

    fig, axes = plt.subplots(1, 6, figsize=(24, 4))
    titles = ["Image", "GT Mask (aligned)", "SAM Mask", "SAM Logits", "Pos Map", "Neg Map"]
    arrays = [img,      gt,                  sam_mask,   logits,       pos_map,   neg_map]
    cmaps  = [None,     "gray",              "gray",     "RdBu_r",     "hot",     "hot"]

    for ax, title, arr, cmap in zip(axes, titles, arrays, cmaps):
        if cmap is None:
            ax.imshow(arr)
        else:
            ax.imshow(arr, cmap=cmap)
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    sid = str(row["sample_id"]).replace("/", "__")
    fig.suptitle(f"{sid}  ps={row['promptset_id']}", fontsize=9)
    plt.tight_layout()
    out_path = out_dir / f"debug_{idx:04d}_{sid}_ps{row['promptset_id']}.png"
    plt.savefig(out_path, dpi=80, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    project_root = Path(cfg["data"]["project_root"])
    target_size  = tuple(cfg["training"].get("target_size", [256, 256]))
    index_csv    = project_root / "training_data" / "no_image" / f"{args.split}_index.csv"

    if not index_csv.exists():
        print(f"Index CSV not found: {index_csv}. Run build_training_indices.py first.")
        return

    df = pd.read_csv(index_csv)
    df = df.sample(n=min(args.n, len(df)), random_state=0).reset_index(drop=True)

    out_dir = project_root / "reports" / "debug_panels"
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, (_, row) in enumerate(df.iterrows()):
        try:
            p = visualize_sample(row.to_dict(), out_dir, i, target_size)
            print(f"Saved: {p}")
        except Exception as e:
            print(f"Failed for {row['sample_id']}: {e}")

    print(f"\nDone. {min(args.n, len(df))} debug panels saved to {out_dir}")


if __name__ == "__main__":
    main()
