"""Prompt-set generation and Gaussian prompt map construction."""

import logging
from typing import List, Tuple

import numpy as np

log = logging.getLogger(__name__)


def generate_prompt_sets(
    mask: np.ndarray,
    num_sets: int,
    min_pairs: int,
    max_pairs: int,
    seed: int,
) -> List[dict]:
    """
    Generate prompt sets for one sample.

    Args:
        mask: 2D binary uint8 array (H, W)
        num_sets: number of prompt sets to generate
        min_pairs: minimum number of pos/neg point pairs per set
        max_pairs: maximum number of pos/neg point pairs per set
        seed: base RNG seed

    Returns:
        list of dicts, each with keys:
            promptset_id, k_pairs, positive_points, negative_points, sampling_note
    """
    rng = np.random.default_rng(seed)
    binary = (mask > 0).astype(np.uint8)

    fg_yx = np.argwhere(binary == 1)  # shape (N_fg, 2) in (row, col)
    bg_yx = np.argwhere(binary == 0)  # shape (N_bg, 2)

    prompt_sets = []
    for ps_id in range(num_sets):
        k = int(rng.integers(min_pairs, max_pairs + 1))

        pos_points, pos_note = _sample_points(rng, fg_yx, k)
        neg_points, neg_note = _sample_points(rng, bg_yx, k)

        # convert (row, col) -> (x, y) = (col, row)
        pos_xy = [[int(p[1]), int(p[0])] for p in pos_points]
        neg_xy = [[int(p[1]), int(p[0])] for p in neg_points]

        notes = []
        if pos_note:
            notes.append(f"pos:{pos_note}")
        if neg_note:
            notes.append(f"neg:{neg_note}")
        sampling_note = ",".join(notes) if notes else "without_replacement"

        prompt_sets.append(
            {
                "promptset_id": ps_id,
                "k_pairs": k,
                "positive_points": pos_xy,
                "negative_points": neg_xy,
                "sampling_note": sampling_note,
            }
        )

    return prompt_sets


def _sample_points(
    rng: np.random.Generator, candidates: np.ndarray, k: int
) -> Tuple[np.ndarray, str]:
    """Sample k rows from candidates. Returns array and a note."""
    n = len(candidates)
    if n == 0:
        return np.zeros((k, 2), dtype=int), "empty_region"
    if k <= n:
        idx = rng.choice(n, size=k, replace=False)
        return candidates[idx], ""
    else:
        idx = rng.choice(n, size=k, replace=True)
        return candidates[idx], "with_replacement"


def build_gaussian_map(
    points_xy: List[List[int]],
    height: int,
    width: int,
    sigma: float,
) -> np.ndarray:
    """
    Build a Gaussian prompt map by summing Gaussians at each point.

    Args:
        points_xy: list of [x, y] = [col, row] coordinates
        height, width: spatial dimensions
        sigma: Gaussian sigma in pixels

    Returns:
        float32 array of shape (H, W), values clamped to [0, 1]
    """
    acc = np.zeros((height, width), dtype=np.float32)
    if not points_xy:
        return acc

    ys = np.arange(height, dtype=np.float32)
    xs = np.arange(width, dtype=np.float32)
    XX, YY = np.meshgrid(xs, ys)  # both (H, W)

    two_sigma_sq = 2.0 * sigma * sigma
    for x_p, y_p in points_xy:
        acc += np.exp(-((XX - x_p) ** 2 + (YY - y_p) ** 2) / two_sigma_sq)

    return np.clip(acc, 0.0, 1.0)
