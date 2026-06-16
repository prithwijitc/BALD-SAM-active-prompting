"""
Stage 4.5: Pack precomputed .npy files into per-split HDF5 files for fast training.

Replaces thousands of small file opens per epoch with a single mmap'd file.
GT masks are resized and stored here so the training loop does zero preprocessing.

Usage:
    python pack_to_hdf5.py --config config.yaml
    python pack_to_hdf5.py --config config.yaml --split train
    python pack_to_hdf5.py --config config.yaml --force
"""

import argparse
import logging
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

from modules.wandb_utils import init_wandb


def setup_logging(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout),
        ],
    )


def resize_bilinear(arr: np.ndarray, th: int, tw: int) -> np.ndarray:
    if arr.shape == (th, tw):
        return arr
    t = torch.from_numpy(arr.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    t = F.interpolate(t, size=(th, tw), mode="bilinear", align_corners=False)
    return t.squeeze().numpy()


def resize_nearest(arr: np.ndarray, th: int, tw: int) -> np.ndarray:
    if arr.shape == (th, tw):
        return arr
    t = torch.from_numpy(arr.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    t = F.interpolate(t, size=(th, tw), mode="nearest")
    return t.squeeze().numpy()


def pack_split(
    split: str,
    index_csv: Path,
    out_h5: Path,
    target_size: tuple,
    chunk_size: int,
    wandb_run=None,
):
    df = pd.read_csv(index_csv)
    N = len(df)
    th, tw = target_size

    logging.info("[%s] Packing %d rows → %s", split, N, out_h5)

    if N == 0:
        logging.warning("[%s] Index CSV is empty — writing empty HDF5 file.", split)

    # string dtype for sample_id
    str_dt = h5py.special_dtype(vlen=str)

    with h5py.File(str(out_h5), "w") as hf:
        # chunk size must be ≥ 1; for an empty split chunks are irrelevant
        cs = max(1, min(chunk_size, N)) if N > 0 else 1
        logits_ds  = hf.create_dataset("logits",        (N, th, tw), dtype="float16", chunks=(cs, th, tw))
        pos_ds     = hf.create_dataset("pos_map",       (N, th, tw), dtype="float16", chunks=(cs, th, tw))
        neg_ds     = hf.create_dataset("neg_map",       (N, th, tw), dtype="float16", chunks=(cs, th, tw))
        gt_ds      = hf.create_dataset("gt",            (N, th, tw), dtype="uint8",   chunks=(cs, th, tw))
        sid_ds     = hf.create_dataset("sample_id",     (N,),        dtype=str_dt)
        psid_ds    = hf.create_dataset("promptset_id",  (N,),        dtype="int32")
        kpairs_ds  = hf.create_dataset("k_pairs",       (N,),        dtype="int32")
        # valid flag: False for any row that failed to pack — dataset loader will exclude these
        valid_ds   = hf.create_dataset("valid",         (N,),        dtype="bool")

        # store metadata as HDF5 attributes
        hf.attrs["split"]       = split
        hf.attrs["target_h"]    = th
        hf.attrs["target_w"]    = tw
        hf.attrs["n_samples"]   = N

        n_failed = 0
        for i, (_, row) in enumerate(df.iterrows()):
            try:
                logits  = np.load(row["logits_256_path"]).astype(np.float32)
                pos_map = np.load(row["pos_map_path"]).astype(np.float32)
                neg_map = np.load(row["neg_map_path"]).astype(np.float32)

                logits  = resize_bilinear(logits,  th, tw)
                pos_map = resize_bilinear(pos_map, th, tw)
                neg_map = resize_bilinear(neg_map, th, tw)

                mask_raw = np.load(row["mask_path"])
                gt = (mask_raw > 0).astype(np.float32)
                if gt.ndim != 2:
                    raise ValueError(f"GT mask not 2D: shape={gt.shape}")
                # two-step resize: raw mask → image space → target space
                # (mirrors generate_promptsets.py which samples prompts in image space)
                image_h, image_w = int(row["image_h"]), int(row["image_w"])
                if gt.shape != (image_h, image_w):
                    gt = resize_nearest(gt, image_h, image_w)
                gt = resize_nearest(gt, th, tw)

                logits_ds[i]  = logits.astype(np.float16)
                pos_ds[i]     = pos_map.astype(np.float16)
                neg_ds[i]     = neg_map.astype(np.float16)
                gt_ds[i]      = (gt > 0.5).astype(np.uint8)
                sid_ds[i]     = str(row["sample_id"])
                psid_ds[i]    = int(row["promptset_id"])
                kpairs_ds[i]  = int(row["k_pairs"])
                valid_ds[i]   = True

            except Exception as e:
                logging.error("Failed row %d (%s): %s", i, row.get("sample_id", "?"), e)
                n_failed += 1
                # mark as invalid — HDF5HeadDataset will exclude this index entirely
                valid_ds[i] = False

            if (i + 1) % 1000 == 0:
                logging.info("[%s] %d/%d packed  failed=%d", split, i + 1, N, n_failed)
                if wandb_run:
                    wandb_run.log({
                        f"pack_hdf5/{split}/packed": i + 1,
                        f"pack_hdf5/{split}/failed": n_failed,
                    })

    size_mb = out_h5.stat().st_size / 1e6
    logging.info(
        "[%s] Done. %d rows packed, %d failed. File size: %.1f MB",
        split, N - n_failed, n_failed, size_mb,
    )
    if wandb_run:
        wandb_run.log({
            f"pack_hdf5/{split}/total_rows": N,
            f"pack_hdf5/{split}/failed": n_failed,
            f"pack_hdf5/{split}/file_size_mb": size_mb,
        })
    return N - n_failed, n_failed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="all")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    project_root = Path(cfg["data"]["project_root"])
    log_path = project_root / "logs" / "pack_hdf5.log"
    setup_logging(log_path)

    target_size = tuple(cfg["training"].get("target_size", [256, 256]))
    chunk_size  = cfg.get("hdf5", {}).get("chunk_size", 64)
    index_dir   = project_root / "training_data" / "no_image"
    out_dir     = project_root / "training_data" / "no_image"
    out_dir.mkdir(parents=True, exist_ok=True)

    run = init_wandb(cfg, job_type="pack_hdf5", run_name="pack_to_hdf5")

    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    for split in splits:
        out_h5   = out_dir / f"{split}.h5"
        index_csv = index_dir / f"{split}_index.csv"

        if not index_csv.exists():
            logging.warning("Index CSV not found for %s — run build_training_indices.py first.", split)
            continue

        if not args.force and out_h5.exists():
            logging.info("%s already exists. Skipping. Use --force to repack.", out_h5)
            continue

        pack_split(
            split=split,
            index_csv=index_csv,
            out_h5=out_h5,
            target_size=target_size,
            chunk_size=chunk_size,
            wandb_run=run,
        )

    run.finish()
    logging.info("HDF5 packing complete.")


if __name__ == "__main__":
    main()
