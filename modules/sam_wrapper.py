"""Frozen SAM wrapper with per-image embedding caching."""

import logging
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

log = logging.getLogger(__name__)


def load_sam_predictor(checkpoint: str, model_type: str, device: str):
    """Load SAM and return a SamPredictor."""
    try:
        from segment_anything import SamPredictor, sam_model_registry
    except ImportError:
        raise ImportError(
            "segment_anything not installed. "
            "Install with: pip install git+https://github.com/facebookresearch/segment-anything.git"
        )

    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.eval()
    sam.to(device)
    for p in sam.parameters():
        p.requires_grad_(False)

    return SamPredictor(sam)


def run_sam_for_promptset(
    predictor,
    positive_points: List[List[int]],
    negative_points: List[List[int]],
    image_hw: Tuple[int, int],
    multimask_output: bool = True,
    target_size: Tuple[int, int] = (256, 256),
    select_by: str = "sam_score",
) -> dict:
    """
    Run SAM for one prompt set. Assumes predictor.set_image() has already been called.

    Returns logits downsampled to target_size (default 256×256) via postprocess_masks
    then bilinear resize. This avoids storing native-resolution arrays.

    Returns dict with keys:
        logits_256         (target_h, target_w) float32
        sam_scores         (n_masks,) float32
        selected_branch_index int
    """
    all_points = np.array(positive_points + negative_points, dtype=np.float32)
    labels = np.array(
        [1] * len(positive_points) + [0] * len(negative_points), dtype=np.int32
    )

    with torch.no_grad():
        _, scores, logits_lowres = predictor.predict(
            point_coords=all_points,
            point_labels=labels,
            multimask_output=multimask_output,
            return_logits=True,
        )
    # logits_lowres: (N_masks, 256, 256) — SAM's raw decoder output in padded space

    if select_by != "sam_score":
        raise ValueError(
            f"select_by='{select_by}' is not supported. Only 'sam_score' is implemented."
        )
    selected = int(np.argmax(scores))
    H, W = image_hw

    # postprocess: remove SAM's padding, restore to original image resolution
    # postprocess_masks calls F.interpolate with a 2D output size, requiring 4D input (N,C,H,W).
    # logits_lowres[[selected]] is (1,256,256); unsqueeze/squeeze to add/remove the batch dim.
    logits_full = predictor.model.postprocess_masks(
        torch.from_numpy(logits_lowres[[selected]]).float().unsqueeze(0),  # (1,1,256,256)
        input_size=predictor.input_size,
        original_size=(H, W),
    ).squeeze(0)  # (1, H, W)

    # downsample to target_size for compact storage
    th, tw = target_size
    if (H, W) != (th, tw):
        logits_full = F.interpolate(
            logits_full.unsqueeze(0), size=(th, tw), mode="bilinear", align_corners=False
        ).squeeze(0)  # (1, th, tw)

    logits_256 = logits_full[0].cpu().numpy()  # (th, tw)

    return {
        "logits_256": logits_256.astype(np.float32),
        "sam_scores": scores.astype(np.float32),
        "selected_branch_index": selected,
    }


def verify_precompute_files(promptset_dir, required=None) -> bool:
    """Return True if all required files exist in promptset_dir."""
    from pathlib import Path

    if required is None:
        required = ["logits_256.npy", "pos_map.npy", "neg_map.npy", "meta.json"]
    d = Path(promptset_dir)
    return all((d / f).exists() for f in required)
