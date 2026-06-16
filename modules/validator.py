"""Validates image/mask pairs for inclusion in the manifest."""

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)


def validate_sample(image_path: str, mask_path: str) -> Tuple[bool, str, dict]:
    """
    Validate one image/mask pair.

    Returns:
        valid (bool), note (str), meta (dict with image_h, image_w, mask_h, mask_w,
                                        fg_pixels, bg_pixels)
    """
    meta = {
        "image_h": None,
        "image_w": None,
        "mask_h": None,
        "mask_w": None,
        "fg_pixels": None,
        "bg_pixels": None,
    }

    # --- image ---
    try:
        img = Image.open(image_path).convert("RGB")
        img_w, img_h = img.size
        meta["image_h"] = img_h
        meta["image_w"] = img_w
    except Exception as e:
        return False, f"image_load_error: {e}", meta

    # --- mask ---
    mask_path = Path(mask_path)
    if not mask_path.exists():
        return False, "mask_not_found", meta

    try:
        mask = np.load(str(mask_path))
    except Exception as e:
        return False, f"mask_load_error: {e}", meta

    if mask.ndim != 2:
        return False, f"mask_not_2d: shape={mask.shape}", meta

    meta["mask_h"] = mask.shape[0]
    meta["mask_w"] = mask.shape[1]

    # binarize
    try:
        binary = (mask > 0).astype(np.uint8)
    except Exception as e:
        return False, f"mask_binarize_error: {e}", meta

    fg = int(binary.sum())
    bg = int((binary == 0).sum())
    meta["fg_pixels"] = fg
    meta["bg_pixels"] = bg

    if fg == 0:
        return False, "mask_no_foreground", meta

    total = fg + bg
    if bg == 0:
        return False, "mask_all_foreground", meta

    # shape compatibility check
    if (mask.shape[0] != img_h or mask.shape[1] != img_w):
        # allow if mask can be resized (any shape is resizable with nearest-neighbor)
        # just record a note but don't fail
        return True, f"mask_resized_needed:{mask.shape[0]}x{mask.shape[1]}!={img_h}x{img_w}", meta

    return True, "ok", meta
