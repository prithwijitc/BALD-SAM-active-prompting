"""Dataset scanner: pairs images with masks across nested subfolders."""

import logging
from pathlib import Path
from typing import List, Tuple

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

log = logging.getLogger(__name__)


def scan_dataset(images_root: str, masks_root: str) -> List[dict]:
    """
    Walk images_root, find all images, locate matching .npy mask in masks_root.

    Returns list of dicts with keys:
        sample_id, safe_id, group, image_path, mask_path
    """
    images_root = Path(images_root)
    masks_root = Path(masks_root)

    records = []
    for image_path in sorted(images_root.rglob("*")):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        # canonical ID: subfolder/stem
        rel = image_path.relative_to(images_root)
        parts = rel.parts
        if len(parts) < 2:
            log.warning("Skipping image not inside a subfolder: %s", image_path)
            continue

        group = parts[0]
        stem = image_path.stem
        # handle deeper nesting: join all parts except last as group
        group = str(Path(*parts[:-1]))
        canonical_id = f"{group}/{stem}"
        safe_id = canonical_id.replace("/", "__")

        mask_path = masks_root / group / f"{stem}.npy"
        records.append(
            {
                "sample_id": canonical_id,
                "safe_id": safe_id,
                "group": parts[0],  # top-level subfolder
                "image_path": str(image_path),
                "mask_path": str(mask_path),
            }
        )

    log.info("Scanned %d candidate images from %s", len(records), images_root)
    return records
