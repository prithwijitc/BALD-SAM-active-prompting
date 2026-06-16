"""
PyTorch Datasets for the no-image head.

HDF5HeadDataset  — primary, fast: single mmap'd file, one open per worker
NoImageHeadDataset — fallback: reads individual .npy files per sample
"""

import logging
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Primary dataset: HDF5-backed
# ---------------------------------------------------------------------------

class HDF5HeadDataset(Dataset):
    """
    Reads from a packed HDF5 file produced by pack_to_hdf5.py.

    Each sample is a direct index into pre-resized, pre-stacked arrays — no
    per-sample file opens, no resize at load time.

    HDF5 layout expected:
        /logits     (N, H, W)  float16
        /pos_map    (N, H, W)  float16
        /neg_map    (N, H, W)  float16
        /gt         (N, H, W)  uint8
        /sample_id  (N,)       str
        /promptset_id (N,)     int32

    h5py files are NOT fork-safe, so we open lazily per worker process.
    """

    def __init__(self, h5_path: str, logit_clip: float = 20.0):
        self.h5_path = h5_path
        self.logit_clip = logit_clip
        self._file = None  # opened lazily per worker

        import h5py
        with h5py.File(h5_path, "r") as f:
            total = f["logits"].shape[0]
            if "valid" in f:
                self._indices = np.where(f["valid"][:])[0]
                n_invalid = total - len(self._indices)
                if n_invalid > 0:
                    log.warning(
                        "HDF5HeadDataset: excluding %d invalid rows from %s", n_invalid, h5_path
                    )
            else:
                self._indices = np.arange(total)
        self._len = len(self._indices)

        log.info("HDF5HeadDataset: %d valid samples from %s", self._len, h5_path)

    def _get_file(self):
        if self._file is None:
            import h5py
            # swmr=True allows multiple concurrent readers
            self._file = h5py.File(self.h5_path, "r", swmr=True)
        return self._file

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> dict:
        f = self._get_file()
        actual = int(self._indices[idx])
        logits  = f["logits"][actual].astype(np.float32)
        pos_map = f["pos_map"][actual].astype(np.float32)
        neg_map = f["neg_map"][actual].astype(np.float32)
        gt      = f["gt"][actual].astype(np.float32)

        logits = np.clip(logits, -self.logit_clip, self.logit_clip)
        inp = np.stack([logits, pos_map, neg_map], axis=0)  # (3, H, W)
        gt  = gt[np.newaxis]                                 # (1, H, W)

        return {
            "input":        torch.from_numpy(inp),
            "target":       torch.from_numpy(gt),
            "sample_id":    str(f["sample_id"][actual]),
            "promptset_id": int(f["promptset_id"][actual]),
        }

    def __del__(self):
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Fallback dataset: individual .npy files
# ---------------------------------------------------------------------------

class NoImageHeadDataset(Dataset):
    """
    Reads from a training index CSV, loading individual .npy files per sample.
    Resizes all spatial arrays to target_size at load time.

    Use this only when HDF5 packing has not been run yet.
    """

    def __init__(
        self,
        index_csv: str,
        logit_clip: float = 20.0,
        target_size: Tuple[int, int] = (256, 256),
    ):
        self.df = pd.read_csv(index_csv)
        self.logit_clip = logit_clip
        self.target_size = target_size
        log.info(
            "NoImageHeadDataset: %d rows from %s  target_size=%s",
            len(self.df), index_csv, target_size,
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]

        logits  = np.load(row["logits_256_path"]).astype(np.float32)
        pos_map = np.load(row["pos_map_path"]).astype(np.float32)
        neg_map = np.load(row["neg_map_path"]).astype(np.float32)

        logits = np.clip(logits, -self.logit_clip, self.logit_clip)

        logits  = self._resize_bilinear(logits)
        pos_map = self._resize_bilinear(pos_map)
        neg_map = self._resize_bilinear(neg_map)

        inp = np.stack([logits, pos_map, neg_map], axis=0)  # (3, H, W)

        mask_raw = np.load(row["mask_path"])
        if mask_raw.ndim != 2:
            raise ValueError(f"GT mask not 2D: {row['mask_path']}")
        gt = (mask_raw > 0).astype(np.float32)
        # two-step resize: raw mask → image space → target space
        # (mirrors generate_promptsets.py which samples prompts in image space)
        image_h, image_w = int(row["image_h"]), int(row["image_w"])
        if gt.shape != (image_h, image_w):
            gt = self._resize_nearest(gt, image_h, image_w)
        gt = self._resize_nearest(gt, *self.target_size)
        gt = gt[np.newaxis]  # (1, H, W)

        return {
            "input":        torch.from_numpy(inp),
            "target":       torch.from_numpy(gt),
            "sample_id":    str(row["sample_id"]),
            "promptset_id": int(row["promptset_id"]),
        }

    def _resize_bilinear(self, arr: np.ndarray, th: int = None, tw: int = None) -> np.ndarray:
        th = th or self.target_size[0]
        tw = tw or self.target_size[1]
        if arr.shape == (th, tw):
            return arr
        t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
        t = F.interpolate(t, size=(th, tw), mode="bilinear", align_corners=False)
        return t.squeeze().numpy()

    def _resize_nearest(self, arr: np.ndarray, th: int = None, tw: int = None) -> np.ndarray:
        th = th or self.target_size[0]
        tw = tw or self.target_size[1]
        if arr.shape == (th, tw):
            return arr
        t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
        t = F.interpolate(t, size=(th, tw), mode="nearest")
        return t.squeeze().numpy()
