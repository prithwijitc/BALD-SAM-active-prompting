"""Frozen SAM2 wrapper — drop-in replacement for sam_wrapper.py.

Identical public API to sam_wrapper.py:
    load_sam_predictor(checkpoint, model_type, device) -> SAM2ImagePredictor
    run_sam_for_promptset(...) -> dict
    verify_precompute_files(...) -> bool

The key SAM2 difference: predictor.predict(return_logits=True) returns
    (masks, scores, low_res_masks)
where `masks` is already (N, H, W) float at ORIGINAL resolution — no
postprocess_masks() call needed, unlike SAM1.
"""

import logging
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

log = logging.getLogger(__name__)


def load_sam_predictor(checkpoint: str, model_type: str, device: str):
    """Load SAM2 and return a SAM2ImagePredictor.

    Args:
        checkpoint: Path to sam2 .pt checkpoint file.
        model_type: Hydra config name, e.g. "sam2.1_hiera_l".
                    Corresponds to a YAML in the sam2 package configs dir.
        device: e.g. "cuda:0"

    Returns:
        SAM2ImagePredictor instance (frozen, eval mode).
    """
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ImportError:
        raise ImportError(
            "sam2 not installed. "
            "Install with: pip install -e /path/to/sam2  (requires Python >= 3.10)"
        )

    log.info("Loading SAM2 model: config=%s  ckpt=%s  device=%s", model_type, checkpoint, device)
    model = build_sam2(model_type, checkpoint, device=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    predictor = SAM2ImagePredictor(model)
    log.info("SAM2 loaded. image_size=%d", model.image_size)
    return predictor


def run_sam_for_promptset(
    predictor,
    positive_points: List[List[int]],
    negative_points: List[List[int]],
    image_hw: Tuple[int, int],
    multimask_output: bool = True,
    target_size: Tuple[int, int] = (256, 256),
    select_by: str = "sam_score",
) -> dict:
    """Run SAM2 for one prompt set.

    Assumes predictor.set_image() has already been called.

    SAM2 difference vs SAM1: predict(return_logits=True) already returns
    full-resolution logits in `masks` — no postprocess_masks() needed.

    Returns dict with keys:
        logits_256         (target_h, target_w) float32
        sam_scores         (n_masks,) float32
        selected_branch_index int
    """
    all_points = np.array(positive_points + negative_points, dtype=np.float32)
    labels = np.array(
        [1] * len(positive_points) + [0] * len(negative_points), dtype=np.int32
    )

    if select_by != "sam_score":
        raise ValueError(
            f"select_by='{select_by}' is not supported. Only 'sam_score' is implemented."
        )

    with torch.no_grad():
        # SAM2: masks is (N, H, W) float at original image resolution when return_logits=True
        # scores: (N,) float
        # low_res_masks: (N, 256, 256) padded low-res — not used
        masks, scores, _low_res = predictor.predict(
            point_coords=all_points,
            point_labels=labels,
            multimask_output=multimask_output,
            return_logits=True,
            normalize_coords=True,  # points are in original image pixel space
        )

    selected = int(np.argmax(scores))
    H, W = image_hw
    th, tw = target_size

    # masks[selected] is already (H, W) float logits at original image resolution
    logits_full = torch.from_numpy(masks[selected]).float().unsqueeze(0)  # (1, H, W)

    # Downsample to target_size for compact storage
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
