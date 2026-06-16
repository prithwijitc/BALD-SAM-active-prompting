"""
Active prompting evaluation: iteratively add prompt points using different
acquisition strategies (BALD, Entropy, Random, MaxDist, Oracle) and measure
segmentation improvement over steps.

Compares Laplace MC sampling vs post-hoc MC Dropout for uncertainty.

Usage:
    python active_prompting.py --config config.yaml --run_id v2
    python active_prompting.py --config config.yaml --run_id v2 --budget 20 --mc_samples 20
    python active_prompting.py --config config.yaml --run_id v2 --max_images 3 --budget 5
    python active_prompting.py --config config.yaml --run_id v2 --mc_ablation
"""

import argparse
import csv
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F_torch
import yaml
from PIL import Image
from scipy import ndimage

from modules.head import NoImageHead
from modules.laplace_wrapper import DiagLaplaceHead, load_laplace
from modules.promptset import build_gaussian_map
from modules.sam_wrapper import load_sam_predictor, run_sam_for_promptset

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ALL_STRATEGIES = ["bald_laplace", "bald_mcdropout", "entropy", "random", "maxdist", "oracle"]

STEP_CSV_FIELDS = [
    "sample_id", "group", "seed", "strategy", "step",
    "iou", "dice", "boundary_iou", "hausdorff95", "ece", "nll",
    "mean_bald", "mean_entropy",
    "point_x", "point_y", "point_label", "n_pos", "n_neg",
    "object_size_bin", "thinness_bin", "boundary_complexity_bin", "contrast_bin",
    "fg_pixels", "thinness_val", "boundary_complexity_val", "contrast_val",
]

MC_ABLATION_K_VALUES = [5, 10, 20, 50, 100]

_EPS = 1e-7


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------
def binary_entropy(p: torch.Tensor) -> torch.Tensor:
    """Binary entropy H(p), works for any shape. Returns same shape."""
    p = p.clamp(_EPS, 1.0 - _EPS)
    return -p * torch.log(p) - (1.0 - p) * torch.log(1.0 - p)


def build_exclusion_mask(
    points_xy: List[List[int]], H: int, W: int, radius: int = 5
) -> np.ndarray:
    """Returns (H, W) bool mask: True = excluded."""
    mask = np.zeros((H, W), dtype=bool)
    if not points_xy:
        return mask
    yy, xx = np.ogrid[:H, :W]
    for px, py in points_xy:
        dist_sq = (xx - px) ** 2 + (yy - py) ** 2
        mask |= dist_sq <= radius * radius
    return mask


def resize_gt_mask(
    mask_path: str, image_h: int, image_w: int, target_h: int = 256, target_w: int = 256
) -> np.ndarray:
    """Load GT mask and two-step resize: raw → image space → target space (nearest)."""
    mask_raw = np.load(mask_path)
    gt = (mask_raw > 0).astype(np.float32)
    # raw → image space
    if gt.shape != (image_h, image_w):
        t = torch.from_numpy(gt).unsqueeze(0).unsqueeze(0)
        t = F_torch.interpolate(t, size=(image_h, image_w), mode="nearest")
        gt = t.squeeze().numpy()
    # image space → target
    if gt.shape != (target_h, target_w):
        t = torch.from_numpy(gt).unsqueeze(0).unsqueeze(0)
        t = F_torch.interpolate(t, size=(target_h, target_w), mode="nearest")
        gt = t.squeeze().numpy()
    return gt


def compute_gt_centroid(gt_256: np.ndarray) -> Tuple[int, int]:
    """Return (x, y) centroid of FG pixels in 256x256 GT mask."""
    fg_yx = np.argwhere(gt_256 > 0)
    if len(fg_yx) == 0:
        return 128, 128  # fallback to center
    cy, cx = fg_yx.mean(axis=0)
    return int(round(cx)), int(round(cy))


def scale_points_to_original(
    points_256: List[List[int]], H_orig: int, W_orig: int, H_target: int = 256, W_target: int = 256
) -> List[List[float]]:
    """Scale (x,y) points from target space to original image space."""
    sx = W_orig / W_target
    sy = H_orig / H_target
    return [[p[0] * sx, p[1] * sy] for p in points_256]


# ---------------------------------------------------------------------------
# Sample attribute computation (for stratification)
# ---------------------------------------------------------------------------
def precompute_sample_attributes(
    gt_256: np.ndarray, image_path: str
) -> Dict:
    """Compute stratification attributes for a single sample."""
    binary = (gt_256 > 0).astype(np.uint8)
    fg_pixels = int(binary.sum())

    # Thinness = perimeter^2 / (4 * pi * area)
    # Perimeter approximated by contour length via binary erosion
    eroded = ndimage.binary_erosion(binary)
    perimeter_pixels = int(binary.sum() - eroded.sum())
    perimeter = max(perimeter_pixels, 1)
    area = max(fg_pixels, 1)
    thinness = (perimeter ** 2) / (4.0 * math.pi * area)

    # Boundary complexity = perimeter / sqrt(area)
    boundary_complexity = perimeter / math.sqrt(area)

    # Contrast from source image
    try:
        img = np.array(Image.open(image_path).convert("L"))
        # Resize grayscale to 256x256
        if img.shape != (256, 256):
            t = torch.from_numpy(img.astype(np.float32)).unsqueeze(0).unsqueeze(0)
            t = F_torch.interpolate(t, size=(256, 256), mode="bilinear", align_corners=False)
            img = t.squeeze().numpy()
        fg_mean = img[binary > 0].mean() if fg_pixels > 0 else 0.0
        bg_mean = img[binary == 0].mean() if fg_pixels < 256 * 256 else 0.0
        contrast = abs(float(fg_mean) - float(bg_mean))
    except Exception:
        contrast = -1.0  # unknown

    # Binning
    if fg_pixels < 1000:
        size_bin = "small"
    elif fg_pixels < 10000:
        size_bin = "medium"
    else:
        size_bin = "large"

    if thinness > 5:
        thinness_bin = "thin"
    elif thinness > 2:
        thinness_bin = "moderate"
    else:
        thinness_bin = "compact"

    if boundary_complexity > 10:
        bc_bin = "complex"
    elif boundary_complexity > 5:
        bc_bin = "moderate"
    else:
        bc_bin = "simple"

    if contrast < 0:
        contrast_bin = "unknown"
    elif contrast < 30:
        contrast_bin = "low"
    elif contrast < 80:
        contrast_bin = "medium"
    else:
        contrast_bin = "high"

    return {
        "fg_pixels": fg_pixels,
        "object_size_bin": size_bin,
        "thinness_val": round(thinness, 3),
        "thinness_bin": thinness_bin,
        "boundary_complexity_val": round(boundary_complexity, 3),
        "boundary_complexity_bin": bc_bin,
        "contrast_val": round(contrast, 2),
        "contrast_bin": contrast_bin,
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    pred = (pred_mask > 0).astype(np.float32)
    gt = (gt_mask > 0).astype(np.float32)
    inter = (pred * gt).sum()
    union = np.clip(pred + gt, 0, 1).sum()
    return float((inter + 1e-6) / (union + 1e-6))


def compute_dice(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    pred = (pred_mask > 0).astype(np.float32)
    gt = (gt_mask > 0).astype(np.float32)
    inter = (pred * gt).sum()
    denom = pred.sum() + gt.sum()
    return float((2.0 * inter + 1e-6) / (denom + 1e-6))


def compute_boundary_iou(
    pred_mask: np.ndarray, gt_mask: np.ndarray, dilation: int = 3
) -> float:
    """IoU computed only within `dilation` px of the GT boundary."""
    gt_bin = (gt_mask > 0).astype(np.uint8)
    gt_dilated = ndimage.binary_dilation(gt_bin, iterations=dilation).astype(np.uint8)
    gt_eroded = ndimage.binary_erosion(gt_bin, iterations=dilation).astype(np.uint8)
    boundary_band = gt_dilated - gt_eroded  # 1 near boundary

    if boundary_band.sum() == 0:
        return 1.0  # no boundary = trivial

    pred = (pred_mask > 0).astype(np.float32)
    gt_f = gt_bin.astype(np.float32)
    band = boundary_band.astype(np.float32)

    pred_b = pred * band
    gt_b = gt_f * band
    inter = (pred_b * gt_b).sum()
    union = np.clip(pred_b + gt_b, 0, 1).sum()
    return float((inter + 1e-6) / (union + 1e-6))


def compute_hausdorff95(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """95th percentile Hausdorff distance between contours."""
    pred_bin = (pred_mask > 0).astype(np.uint8)
    gt_bin = (gt_mask > 0).astype(np.uint8)

    # Edge pixels via XOR with erosion
    pred_edge = pred_bin ^ ndimage.binary_erosion(pred_bin).astype(np.uint8)
    gt_edge = gt_bin ^ ndimage.binary_erosion(gt_bin).astype(np.uint8)

    pred_pts = np.argwhere(pred_edge > 0)
    gt_pts = np.argwhere(gt_edge > 0)

    if len(pred_pts) == 0 and len(gt_pts) == 0:
        return 0.0
    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return float(max(pred_mask.shape))  # max possible

    # Distance from each pred edge pixel to nearest gt edge pixel and vice versa
    from scipy.spatial.distance import cdist

    d_pred_to_gt = cdist(pred_pts, gt_pts, metric="euclidean").min(axis=1)
    d_gt_to_pred = cdist(gt_pts, pred_pts, metric="euclidean").min(axis=1)
    all_distances = np.concatenate([d_pred_to_gt, d_gt_to_pred])
    return float(np.percentile(all_distances, 95))


def compute_ece(probs: np.ndarray, gt: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error (binary, 10 bins)."""
    probs_flat = probs.ravel()
    gt_flat = gt.ravel()

    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(probs_flat)
    for i in range(n_bins):
        in_bin = (probs_flat >= bin_edges[i]) & (probs_flat < bin_edges[i + 1])
        if i == n_bins - 1:
            in_bin |= probs_flat == bin_edges[i + 1]
        n_in_bin = in_bin.sum()
        if n_in_bin == 0:
            continue
        avg_conf = probs_flat[in_bin].mean()
        avg_acc = gt_flat[in_bin].mean()
        ece += (n_in_bin / total) * abs(avg_conf - avg_acc)
    return float(ece)


def compute_nll(probs: np.ndarray, gt: np.ndarray) -> float:
    """Mean per-pixel binary cross-entropy (NLL)."""
    p = np.clip(probs.ravel(), _EPS, 1.0 - _EPS)
    g = gt.ravel().astype(np.float32)
    nll = -(g * np.log(p) + (1.0 - g) * np.log(1.0 - p))
    return float(nll.mean())


def compute_all_metrics(
    pred_logits: np.ndarray, probs: np.ndarray, gt_256: np.ndarray
) -> Dict[str, float]:
    """Compute all per-step metrics. pred_logits and probs are (256,256)."""
    pred_mask = (pred_logits > 0).astype(np.uint8)
    return {
        "iou": compute_iou(pred_mask, gt_256),
        "dice": compute_dice(pred_mask, gt_256),
        "boundary_iou": compute_boundary_iou(pred_mask, gt_256),
        "hausdorff95": compute_hausdorff95(pred_mask, gt_256),
        "ece": compute_ece(probs, gt_256),
        "nll": compute_nll(probs, gt_256),
    }


# ---------------------------------------------------------------------------
# Uncertainty: BALD via analytical Laplace approximation
# ---------------------------------------------------------------------------
@torch.no_grad()
def compute_bald_laplace(
    la: DiagLaplaceHead, feat: torch.Tensor, K: int = 20
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Per-pixel BALD via MC mutual information under the Laplace posterior.

    Because the posterior is only over the 17-parameter `final_conv`, exact-ish
    MC BALD is cheap here and more faithful than the previous surrogate
    `H(p_probit) - H(p_map)`.

    Args:
        la: DiagLaplaceHead with diag_var_bald set
        feat: (1, C_in, H, W) backbone features
        K: number of posterior samples

    Returns:
        bald_map:    (H, W) — mutual information
        entropy_map: (H, W) — predictive entropy
    """
    device = feat.device
    _, _, H, W = feat.shape

    if la.diag_var_bald is None:
        raise RuntimeError("diag_var_bald is not set. Fit Laplace with a BALD prior first.")

    phi = la._phi(feat)  # (H*W, C_in+1)

    map_weight = la.head.final_conv.weight.detach().reshape(-1)
    map_bias = la.head.final_conv.bias.detach().reshape(-1)
    map_params = torch.cat([map_weight, map_bias], dim=0).to(device=device, dtype=phi.dtype)

    std = torch.sqrt(la.diag_var_bald.to(device=device, dtype=phi.dtype)).clamp_min(1e-12)
    eps = torch.randn(K, la._n_params, device=device, dtype=phi.dtype)
    sampled_params = map_params.unsqueeze(0) + eps * std.unsqueeze(0)  # (K, P)

    logits_samples = phi @ sampled_params.t()  # (H*W, K)
    probs_samples = torch.sigmoid(logits_samples)

    mean_probs = probs_samples.mean(dim=1)
    total_entropy = binary_entropy(mean_probs)
    expected_entropy = binary_entropy(probs_samples).mean(dim=1)
    bald = (total_entropy - expected_entropy).clamp(min=0)
    entropy_map = total_entropy

    return bald.reshape(H, W), entropy_map.reshape(H, W)


# ---------------------------------------------------------------------------
# Uncertainty: BALD via MC Dropout (post-hoc)
# ---------------------------------------------------------------------------
@torch.no_grad()
def compute_bald_mcdropout(
    head: NoImageHead, x: torch.Tensor, K: int = 20, dropout_rate: float = 0.1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Per-pixel BALD via MC Dropout before final_conv.

    Dropout is NOT trained — this is a post-hoc baseline.

    Args:
        head: NoImageHead in eval mode
        x: (1, 3, H, W) head input
        K: number of MC forward passes
        dropout_rate: dropout probability

    Returns:
        bald_map: (H, W)
        entropy_map: (H, W)
    """
    device = x.device
    dropout = nn.Dropout2d(p=dropout_rate).to(device)
    dropout.train()  # keep dropout active

    probs_list = []
    feat = head.backbone(x)  # (1, C_in, H, W)
    for _ in range(K):
        feat_dropped = dropout(feat)
        logits = head.final_conv(feat_dropped)  # (1, 1, H, W)
        probs_list.append(torch.sigmoid(logits).squeeze(0).squeeze(0))  # (H, W)

    probs_all = torch.stack(probs_list, dim=0)  # (K, H, W)
    H, W = probs_all.shape[1], probs_all.shape[2]

    mean_probs = probs_all.mean(dim=0)  # (H, W)
    H_mean = binary_entropy(mean_probs)
    per_sample_H = binary_entropy(probs_all)  # (K, H, W)
    E_H = per_sample_H.mean(dim=0)

    bald = (H_mean - E_H).clamp(min=0)
    return bald, H_mean


# ---------------------------------------------------------------------------
# Acquisition strategies
# ---------------------------------------------------------------------------
def select_bald(acq_map: np.ndarray, exclusion: np.ndarray) -> Tuple[int, int]:
    """Pick pixel with highest BALD outside exclusion zone. Returns (x, y)."""
    acq = acq_map.copy()
    acq[exclusion] = -np.inf
    flat_idx = np.argmax(acq)
    y, x = np.unravel_index(flat_idx, acq.shape)
    return int(x), int(y)


def select_entropy(entropy_map: np.ndarray, exclusion: np.ndarray) -> Tuple[int, int]:
    """Pick pixel with highest predictive entropy. Returns (x, y)."""
    acq = entropy_map.copy()
    acq[exclusion] = -np.inf
    flat_idx = np.argmax(acq)
    y, x = np.unravel_index(flat_idx, acq.shape)
    return int(x), int(y)


def select_random(H: int, W: int, exclusion: np.ndarray, rng: np.random.Generator) -> Tuple[int, int]:
    """Pick uniformly random non-excluded pixel. Returns (x, y)."""
    valid = ~exclusion
    valid_yx = np.argwhere(valid)
    if len(valid_yx) == 0:
        y, x = rng.integers(0, H), rng.integers(0, W)
    else:
        idx = rng.integers(0, len(valid_yx))
        y, x = valid_yx[idx]
    return int(x), int(y)


def select_maxdist(
    existing_points_xy: List[List[int]], H: int, W: int, exclusion: np.ndarray
) -> Tuple[int, int]:
    """Pick pixel farthest from all existing points. Returns (x, y)."""
    if not existing_points_xy:
        return W // 2, H // 2

    yy, xx = np.mgrid[:H, :W]
    min_dist = np.full((H, W), np.inf, dtype=np.float32)
    for px, py in existing_points_xy:
        d = np.sqrt((xx - px) ** 2 + (yy - py) ** 2).astype(np.float32)
        min_dist = np.minimum(min_dist, d)

    min_dist[exclusion] = -np.inf
    flat_idx = np.argmax(min_dist)
    y, x = np.unravel_index(flat_idx, min_dist.shape)
    return int(x), int(y)


def select_oracle(
    pred_logits: np.ndarray, gt_256: np.ndarray, exclusion: np.ndarray, rng: np.random.Generator
) -> Tuple[int, int]:
    """Centroid of largest misclassified connected component. Returns (x, y)."""
    pred_mask = (pred_logits > 0).astype(np.uint8)
    gt_bin = (gt_256 > 0).astype(np.uint8)

    misclassified = (pred_mask != gt_bin).astype(np.uint8)
    misclassified[exclusion] = 0

    if misclassified.sum() == 0:
        # Perfect — pick random valid point
        return select_random(256, 256, exclusion, rng)

    labeled, n_components = ndimage.label(misclassified)
    component_sizes = ndimage.sum(misclassified, labeled, range(1, n_components + 1))
    largest = int(np.argmax(component_sizes)) + 1

    yx = np.argwhere(labeled == largest)
    cy, cx = yx.mean(axis=0)
    return int(round(cx)), int(round(cy))


def dispatch_strategy(
    strategy: str,
    bald_map_np: np.ndarray,
    entropy_map_np: np.ndarray,
    pred_logits_np: np.ndarray,
    gt_256: np.ndarray,
    all_points_xy: List[List[int]],
    exclusion: np.ndarray,
    rng: np.random.Generator,
    H: int = 256,
    W: int = 256,
) -> Tuple[int, int]:
    """Dispatch to the right acquisition function. Returns (x, y)."""
    if strategy in ("bald_laplace", "bald_mcdropout"):
        return select_bald(bald_map_np, exclusion)
    elif strategy == "entropy":
        return select_entropy(entropy_map_np, exclusion)
    elif strategy == "random":
        return select_random(H, W, exclusion, rng)
    elif strategy == "maxdist":
        return select_maxdist(all_points_xy, H, W, exclusion)
    elif strategy == "oracle":
        return select_oracle(pred_logits_np, gt_256, exclusion, rng)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


# ---------------------------------------------------------------------------
# Core active prompting loop for one image x one strategy
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_active_loop(
    sample_row: pd.Series,
    la: DiagLaplaceHead,
    head: NoImageHead,
    predictor,
    strategy: str,
    budget: int,
    mc_samples: int,
    exclusion_radius: int,
    init_point_xy: Tuple[int, int],
    rng: np.random.Generator,
    device: str,
    sigma: float,
    dropout_rate: float,
    gt_256: np.ndarray,
    attrs: Dict,
    seed: int,
) -> List[Dict]:
    """Run active prompting for one image x one strategy. Returns list of step dicts."""
    H_orig = int(sample_row["image_h"])
    W_orig = int(sample_row["image_w"])
    sample_id = str(sample_row["sample_id"])
    group = sample_id.split("/")[0] if "/" in sample_id else ""

    # Initialize with common first positive point
    pos_points_256 = [list(init_point_xy)]
    neg_points_256: List[List[int]] = []
    results = []

    for step in range(budget):
        # Scale points to original image space for SAM
        all_pos = pos_points_256
        all_neg = neg_points_256
        pos_orig = scale_points_to_original(all_pos, H_orig, W_orig)
        neg_orig = scale_points_to_original(all_neg, H_orig, W_orig) if all_neg else []

        # Run SAM
        sam_out = run_sam_for_promptset(
            predictor,
            pos_orig,
            neg_orig if neg_orig else [],
            (H_orig, W_orig),
            multimask_output=True,
            target_size=(256, 256),
        )
        logits_256 = sam_out["logits_256"]  # (256, 256) float32

        # Build prompt maps at 256x256
        pos_map = build_gaussian_map(all_pos, 256, 256, sigma)
        neg_map = build_gaussian_map(all_neg, 256, 256, sigma) if all_neg else np.zeros((256, 256), dtype=np.float32)

        # Build head input
        logits_clipped = np.clip(logits_256, -20.0, 20.0)
        x_input = torch.from_numpy(
            np.stack([logits_clipped, pos_map, neg_map], axis=0)
        ).float().unsqueeze(0).to(device)  # (1, 3, 256, 256)

        # Forward through head
        feat = la._backbone(x_input)  # (1, C_in, 256, 256)
        map_logits = head.final_conv(feat)  # (1, 1, 256, 256)

        # Get probit predictive probability
        probit_probs = la._probit_predictive(feat, map_logits)  # (1, 1, 256, 256)
        probs_np = probit_probs.squeeze().cpu().numpy()
        logits_np = map_logits.squeeze().cpu().numpy()

        # Compute uncertainty maps based on strategy
        if strategy == "bald_laplace":
            bald_map, ent_map = compute_bald_laplace(la, feat, K=mc_samples)
            bald_np = bald_map.cpu().numpy()
            ent_np = ent_map.cpu().numpy()
        elif strategy == "bald_mcdropout":
            bald_map, ent_map = compute_bald_mcdropout(head, x_input, K=mc_samples, dropout_rate=dropout_rate)
            bald_np = bald_map.cpu().numpy()
            ent_np = ent_map.cpu().numpy()
        elif strategy == "entropy":
            ent_map_t = binary_entropy(probit_probs.squeeze())
            ent_np = ent_map_t.cpu().numpy()
            bald_np = np.zeros_like(ent_np)
        else:
            # random, maxdist, oracle don't need uncertainty maps
            ent_np = binary_entropy(probit_probs.squeeze()).cpu().numpy()
            bald_np = np.zeros((256, 256), dtype=np.float32)

        # Compute all metrics
        metrics = compute_all_metrics(logits_np, probs_np, gt_256)
        metrics["mean_bald"] = float(bald_np.mean()) if bald_np.any() else 0.0
        metrics["mean_entropy"] = float(ent_np.mean())

        # Record the point that was added at this step
        if step == 0:
            pt_x, pt_y = init_point_xy
            pt_label = 1  # always positive (FG centroid)
        else:
            pt_x, pt_y = results[-1]["_next_point"]
            pt_label = results[-1]["_next_label"]

        row = {
            "sample_id": sample_id,
            "group": group,
            "seed": seed,
            "strategy": strategy,
            "step": step,
            "point_x": pt_x,
            "point_y": pt_y,
            "point_label": pt_label,
            "n_pos": len(pos_points_256),
            "n_neg": len(neg_points_256),
        }
        row.update(metrics)
        row.update(attrs)
        results.append(row)

        # Select next point (for the NEXT step)
        if step < budget - 1:
            all_pts = pos_points_256 + neg_points_256
            exclusion = build_exclusion_mask(all_pts, 256, 256, exclusion_radius)
            nx, ny = dispatch_strategy(
                strategy, bald_np, ent_np, logits_np, gt_256,
                all_pts, exclusion, rng,
            )
            # Label from GT
            label = int(gt_256[ny, nx] > 0)
            results[-1]["_next_point"] = (nx, ny)
            results[-1]["_next_label"] = label
            if label == 1:
                pos_points_256.append([nx, ny])
            else:
                neg_points_256.append([nx, ny])

    # Clean internal keys
    for r in results:
        r.pop("_next_point", None)
        r.pop("_next_label", None)

    return results


# ---------------------------------------------------------------------------
# Incremental CSV writer
# ---------------------------------------------------------------------------
class IncrementalCSVWriter:
    """Append rows to a CSV file incrementally."""

    def __init__(self, path: Path, fieldnames: List[str]):
        self.path = path
        self.fieldnames = fieldnames
        write_header = not path.exists() or path.stat().st_size == 0
        self._file = open(path, "a", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            self._writer.writeheader()
            self._file.flush()

    def writerow(self, row: Dict):
        self._writer.writerow(row)

    def writerows(self, rows: List[Dict]):
        for r in rows:
            self._writer.writerow(r)
        self._file.flush()

    def close(self):
        self._file.close()


# ---------------------------------------------------------------------------
# Resumability
# ---------------------------------------------------------------------------
def load_completed(path: Path) -> set:
    if path.exists():
        with open(path) as f:
            return set(json.load(f))
    return set()


def save_completed(path: Path, completed: set):
    with open(path, "w") as f:
        json.dump(sorted(completed), f)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
DELTA_METRICS = ["iou", "dice", "boundary_iou", "hausdorff95", "ece", "nll"]


def _add_delta_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add delta columns: metric[step] - metric[step=0] per (sample_id, seed, strategy)."""
    df = df.copy()
    key_cols = ["sample_id", "seed", "strategy"]
    step0 = df[df["step"] == 0][key_cols + DELTA_METRICS].copy()
    step0_renamed = step0.rename(columns={m: f"{m}_step0" for m in DELTA_METRICS})
    df = df.merge(step0_renamed, on=key_cols, how="left")
    for m in DELTA_METRICS:
        df[f"delta_{m}"] = df[m] - df[f"{m}_step0"]
    df.drop(columns=[f"{m}_step0" for m in DELTA_METRICS], inplace=True)
    return df


def aggregate_results(step_csv: Path, out_dir: Path):
    """Read step_results.csv and produce aggregate, per-group, and per-stratum CSVs."""
    df = pd.read_csv(step_csv)

    # Add delta columns
    df = _add_delta_columns(df)

    metric_cols = ["iou", "dice", "boundary_iou", "hausdorff95", "ece", "nll", "mean_bald", "mean_entropy"]
    delta_cols = [f"delta_{m}" for m in DELTA_METRICS]
    all_metric_cols = metric_cols + delta_cols

    # Overall aggregate: strategy x step
    agg = df.groupby(["strategy", "step"])[all_metric_cols].agg(["mean", "std", "median"]).reset_index()
    agg.columns = ["_".join(c).rstrip("_") for c in agg.columns]
    agg.to_csv(out_dir / "aggregate.csv", index=False)
    log.info("Saved aggregate.csv (%d rows)", len(agg))

    # Per-group: strategy x step x group
    if "group" in df.columns and df["group"].nunique() > 1:
        grp = df.groupby(["strategy", "step", "group"])[all_metric_cols].agg(["mean", "std"]).reset_index()
        grp.columns = ["_".join(c).rstrip("_") for c in grp.columns]
        grp.to_csv(out_dir / "per_group.csv", index=False)
        log.info("Saved per_group.csv")

    # Per-stratum for each stratification attribute
    strata_cols = ["object_size_bin", "thinness_bin", "boundary_complexity_bin", "contrast_bin"]
    all_strata = []
    for scol in strata_cols:
        if scol in df.columns:
            s = df.groupby(["strategy", "step", scol])[all_metric_cols].agg(["mean", "std"]).reset_index()
            s.columns = ["_".join(c).rstrip("_") for c in s.columns]
            s["stratum_type"] = scol
            s.rename(columns={scol: "stratum_value"}, inplace=True)
            all_strata.append(s)
    if all_strata:
        strata_df = pd.concat(all_strata, ignore_index=True)
        strata_df.to_csv(out_dir / "per_stratum.csv", index=False)
        log.info("Saved per_stratum.csv")

    return df


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
STRATEGY_COLORS = {
    "bald_laplace": "#e41a1c",
    "bald_mcdropout": "#ff7f00",
    "entropy": "#377eb8",
    "random": "#999999",
    "maxdist": "#4daf4a",
    "oracle": "#984ea3",
}

STRATEGY_LABELS = {
    "bald_laplace": "BALD (Laplace)",
    "bald_mcdropout": "BALD (MC Dropout)",
    "entropy": "Entropy",
    "random": "Random",
    "maxdist": "MaxDist",
    "oracle": "Oracle",
}


def _plot_metric_vs_steps(
    df: pd.DataFrame,
    metric: str,
    out_dir: Path,
    ylabel: str = None,
    higher_is_better: bool = True,
):
    """Generate mean-only and mean+std plots for a metric vs steps."""
    ylabel = ylabel or metric.replace("_", " ").title()
    strategies = sorted(df["strategy"].unique())

    for variant in ("mean", "std"):
        fig, ax = plt.subplots(figsize=(8, 5))
        for strat in strategies:
            sdf = df[df["strategy"] == strat].groupby("step")[metric].agg(["mean", "std"]).reset_index()
            color = STRATEGY_COLORS.get(strat, "#000000")
            label = STRATEGY_LABELS.get(strat, strat)
            ax.plot(sdf["step"], sdf["mean"], color=color, label=label, linewidth=2)
            if variant == "std":
                ax.fill_between(
                    sdf["step"],
                    sdf["mean"] - sdf["std"],
                    sdf["mean"] + sdf["std"],
                    color=color,
                    alpha=0.15,
                )
        ax.set_xlabel("Prompt Step")
        ax.set_ylabel(ylabel)
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fname = f"{metric}_vs_steps_{variant}.png"
        fig.savefig(out_dir / fname, dpi=150)
        plt.close(fig)
    log.info("Plotted %s vs steps (mean + std)", metric)


def _plot_per_group_grid(df: pd.DataFrame, metric: str, out_dir: Path):
    """Small-multiples: one subplot per group."""
    groups = sorted(df["group"].unique())
    if len(groups) < 2:
        return
    n = len(groups)
    ncols = min(4, n)
    nrows = math.ceil(n / ncols)
    strategies = sorted(df["strategy"].unique())

    for variant in ("mean", "std"):
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
        for i, grp in enumerate(groups):
            ax = axes[i // ncols][i % ncols]
            gdf = df[df["group"] == grp]
            for strat in strategies:
                sdf = gdf[gdf["strategy"] == strat].groupby("step")[metric].agg(["mean", "std"]).reset_index()
                color = STRATEGY_COLORS.get(strat, "#000")
                ax.plot(sdf["step"], sdf["mean"], color=color, linewidth=1.5)
                if variant == "std":
                    ax.fill_between(sdf["step"], sdf["mean"] - sdf["std"], sdf["mean"] + sdf["std"],
                                    color=color, alpha=0.12)
            ax.set_title(grp, fontsize=9)
            ax.grid(True, alpha=0.2)
        # Hide unused axes
        for j in range(n, nrows * ncols):
            axes[j // ncols][j % ncols].set_visible(False)
        fig.suptitle(f"{metric.replace('_',' ').title()} per Group", fontsize=12)
        fig.tight_layout()
        fig.savefig(out_dir / f"per_group_{metric}_grid_{variant}.png", dpi=120)
        plt.close(fig)


def _plot_stratified_grid(
    df: pd.DataFrame, out_dir: Path, metric: str = "iou", title_prefix: str = "Stratified IoU"
):
    """Small-multiples grid: rows = stratification attributes, columns = bins."""
    strata_info = [
        ("object_size_bin", ["small", "medium", "large"]),
        ("thinness_bin", ["thin", "moderate", "compact"]),
        ("boundary_complexity_bin", ["simple", "moderate", "complex"]),
        ("contrast_bin", ["low", "medium", "high"]),
    ]
    strategies = sorted(df["strategy"].unique())
    safe_metric = metric.replace("delta_", "d_")

    for variant in ("mean", "std"):
        fig, axes = plt.subplots(4, 3, figsize=(15, 16), squeeze=False)
        for row_i, (scol, bins) in enumerate(strata_info):
            if scol not in df.columns:
                continue
            for col_j, bval in enumerate(bins):
                ax = axes[row_i][col_j]
                sub = df[df[scol] == bval]
                if len(sub) == 0:
                    ax.set_visible(False)
                    continue
                for strat in strategies:
                    sdf = sub[sub["strategy"] == strat].groupby("step")[metric].agg(["mean", "std"]).reset_index()
                    if len(sdf) == 0:
                        continue
                    color = STRATEGY_COLORS.get(strat, "#000")
                    label = STRATEGY_LABELS.get(strat, strat) if row_i == 0 and col_j == 0 else None
                    ax.plot(sdf["step"], sdf["mean"], color=color, label=label, linewidth=1.5)
                    if variant == "std":
                        ax.fill_between(sdf["step"], sdf["mean"] - sdf["std"], sdf["mean"] + sdf["std"],
                                        color=color, alpha=0.12)
                ax.set_title(f"{scol.replace('_bin','')}: {bval}", fontsize=9)
                ax.grid(True, alpha=0.2)
        if axes[0][0].get_legend_handles_labels()[1]:
            axes[0][0].legend(fontsize=7, loc="lower right")
        fig.suptitle(f"{title_prefix} vs Steps", fontsize=13)
        fig.tight_layout()
        fig.savefig(out_dir / f"stratified_{safe_metric}_grid_{variant}.png", dpi=120)
        plt.close(fig)
    log.info("Plotted stratified %s grid", metric)


def plot_all(step_csv: Path, out_dir: Path):
    """Generate all plots from step_results.csv."""
    df = pd.read_csv(step_csv)
    df = _add_delta_columns(df)

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Raw metric plots
    for metric, ylabel, hib in [
        ("iou", "IoU", True),
        ("dice", "Dice", True),
        ("boundary_iou", "Boundary IoU", True),
        ("hausdorff95", "Hausdorff 95", False),
        ("nll", "Per-pixel NLL", False),
        ("ece", "ECE", False),
        ("mean_bald", "Mean BALD", True),
        ("mean_entropy", "Mean Entropy", True),
    ]:
        _plot_metric_vs_steps(df, metric, plots_dir, ylabel=ylabel, higher_is_better=hib)

    # Delta metric plots (gain from step 0)
    for metric, ylabel, hib in [
        ("delta_iou", "\u0394 IoU (from step 0)", True),
        ("delta_dice", "\u0394 Dice (from step 0)", True),
        ("delta_boundary_iou", "\u0394 Boundary IoU (from step 0)", True),
        ("delta_hausdorff95", "\u0394 Hausdorff 95 (from step 0)", False),
        ("delta_nll", "\u0394 NLL (from step 0)", False),
        ("delta_ece", "\u0394 ECE (from step 0)", False),
    ]:
        _plot_metric_vs_steps(df, metric, plots_dir, ylabel=ylabel, higher_is_better=hib)

    # Per-group grid (raw + delta)
    _plot_per_group_grid(df, "iou", plots_dir)
    _plot_per_group_grid(df, "delta_iou", plots_dir)

    # Stratified grid (raw + delta)
    _plot_stratified_grid(df, plots_dir, metric="iou", title_prefix="Stratified IoU")
    _plot_stratified_grid(df, plots_dir, metric="delta_iou", title_prefix="Stratified \u0394 IoU")

    log.info("All plots saved to %s", plots_dir)


# ---------------------------------------------------------------------------
# Qualitative figures
# ---------------------------------------------------------------------------
def generate_qualitative_panels(
    step_csv: Path,
    out_dir: Path,
    config: dict,
    la: DiagLaplaceHead,
    head: NoImageHead,
    predictor,
    device: str,
    sigma: float,
    mc_samples: int,
    dropout_rate: float,
    n_best: int = 5,
    n_worst: int = 5,
    manifest_path: Path = None,
):
    """Generate qualitative panels for best and worst BALD-vs-Random cases."""
    df = pd.read_csv(step_csv)
    qual_dir = out_dir / "qualitative"
    qual_dir.mkdir(exist_ok=True)

    # Compute final-step IoU delta (BALD Laplace - Random) per (sample_id, seed)
    max_step = df["step"].max()
    final = df[df["step"] == max_step]
    bald_final = final[final["strategy"] == "bald_laplace"][["sample_id", "seed", "iou"]].rename(columns={"iou": "iou_bald"})
    random_final = final[final["strategy"] == "random"][["sample_id", "seed", "iou"]].rename(columns={"iou": "iou_random"})

    if len(bald_final) == 0 or len(random_final) == 0:
        log.warning("Cannot generate qualitative panels: missing bald_laplace or random strategy")
        return

    merged = bald_final.merge(random_final, on=["sample_id", "seed"])
    merged["delta"] = merged["iou_bald"] - merged["iou_random"]

    best = merged.nlargest(n_best, "delta")
    worst = merged.nsmallest(n_worst, "delta")

    if manifest_path is None:
        manifest_path = Path(config["data"]["project_root"]) / "manifests" / "test.csv"
    manifest = pd.read_csv(manifest_path)

    step_cols = [0, 1, 3, 5, 10, 15, 20]

    for label, subset in [("best", best), ("worst", worst)]:
        for rank, (_, case) in enumerate(subset.iterrows()):
            sid = case["sample_id"]
            seed = int(case["seed"])

            mrow = manifest[manifest["sample_id"] == sid].iloc[0]
            image_path = mrow["image_path"]
            mask_path = mrow["mask_path"]
            image_h, image_w = int(mrow["image_h"]), int(mrow["image_w"])

            gt_256 = resize_gt_mask(mask_path, image_h, image_w)

            # Load image for display
            try:
                img_display = np.array(Image.open(image_path).convert("RGB"))
            except Exception:
                continue

            # Get step data for BALD laplace
            case_df = df[(df["sample_id"] == sid) & (df["seed"] == seed) & (df["strategy"] == "bald_laplace")]
            avail_steps = sorted(case_df["step"].unique())
            plot_steps = [s for s in step_cols if s in avail_steps]
            if not plot_steps:
                continue

            n_cols = len(plot_steps)
            fig, axes = plt.subplots(2, n_cols, figsize=(3 * n_cols, 6))
            if n_cols == 1:
                axes = axes.reshape(2, 1)

            for ci, s in enumerate(plot_steps):
                srow = case_df[case_df["step"] == s].iloc[0]

                # Row 0: GT overlay (green = TP, red = FN, blue = FP)
                ax = axes[0, ci]
                ax.imshow(np.zeros((256, 256, 3), dtype=np.uint8))
                gt_vis = np.zeros((256, 256, 3), dtype=np.uint8)
                gt_vis[gt_256 > 0] = [0, 200, 0]
                ax.imshow(gt_vis, alpha=0.5)
                # Mark all points up to this step
                sub_steps = case_df[case_df["step"] <= s]
                for _, pr in sub_steps.iterrows():
                    color = "lime" if pr["point_label"] == 1 else "red"
                    ax.plot(pr["point_x"], pr["point_y"], "o", color=color, markersize=4)
                ax.set_title(f"Step {s}\nIoU={srow['iou']:.3f}", fontsize=8)
                ax.axis("off")

                # Row 1: prediction overlay
                ax2 = axes[1, ci]
                ax2.imshow(np.zeros((256, 256), dtype=np.float32), cmap="gray", vmin=0, vmax=1)
                ax2.set_title(f"Dice={srow['dice']:.3f}", fontsize=8)
                ax2.axis("off")

            safe_id = sid.replace("/", "__")
            fig.suptitle(f"{label.upper()} #{rank+1}: {sid} (seed={seed}, delta={case['delta']:.3f})", fontsize=10)
            fig.tight_layout()
            fig.savefig(qual_dir / f"{safe_id}_{label}_{rank+1}.png", dpi=150)
            plt.close(fig)

    log.info("Qualitative panels saved to %s", qual_dir)


# ---------------------------------------------------------------------------
# MC ablation
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_mc_ablation(
    la: DiagLaplaceHead,
    head: NoImageHead,
    predictor,
    manifest_df: pd.DataFrame,
    config: dict,
    device: str,
    sigma: float,
    dropout_rate: float,
    out_dir: Path,
    max_images: int = 20,
):
    """Sweep K values and measure BALD stability."""
    from scipy.stats import spearmanr

    ablation_dir = out_dir / "mc_ablation"
    ablation_dir.mkdir(exist_ok=True)

    rows = []
    images_root = config["data"]["images_root"]

    sample_df = manifest_df.head(max_images) if max_images > 0 else manifest_df

    for img_i, (_, mrow) in enumerate(sample_df.iterrows()):
        sid = str(mrow["sample_id"])
        image_h, image_w = int(mrow["image_h"]), int(mrow["image_w"])

        # Load image, set SAM embedding
        img = np.array(Image.open(mrow["image_path"]).convert("RGB"))
        predictor.set_image(img)

        gt_256 = resize_gt_mask(mrow["mask_path"], image_h, image_w)
        cx, cy = compute_gt_centroid(gt_256)

        # Run SAM with centroid point
        pos_orig = scale_points_to_original([[cx, cy]], image_h, image_w)
        sam_out = run_sam_for_promptset(predictor, pos_orig, [], (image_h, image_w), target_size=(256, 256))
        logits_256 = sam_out["logits_256"]

        pos_map = build_gaussian_map([[cx, cy]], 256, 256, sigma)
        neg_map = np.zeros((256, 256), dtype=np.float32)
        logits_clipped = np.clip(logits_256, -20.0, 20.0)
        x_input = torch.from_numpy(
            np.stack([logits_clipped, pos_map, neg_map], axis=0)
        ).float().unsqueeze(0).to(device)

        feat = la._backbone(x_input)

        # Sweep K for Laplace
        bald_maps_laplace = {}
        for K in MC_ABLATION_K_VALUES:
            t0 = time.time()
            bald_map, _ = compute_bald_laplace(la, feat, K=K)
            elapsed = time.time() - t0
            bald_maps_laplace[K] = bald_map.cpu().numpy().ravel()
            rows.append({
                "sample_id": sid, "method": "laplace", "K": K,
                "wall_time_s": round(elapsed, 4),
                "mean_bald": float(bald_maps_laplace[K].mean()),
            })

        # Sweep K for MC Dropout
        bald_maps_dropout = {}
        for K in MC_ABLATION_K_VALUES:
            t0 = time.time()
            bald_map, _ = compute_bald_mcdropout(head, x_input, K=K, dropout_rate=dropout_rate)
            elapsed = time.time() - t0
            bald_maps_dropout[K] = bald_map.cpu().numpy().ravel()
            rows.append({
                "sample_id": sid, "method": "mcdropout", "K": K,
                "wall_time_s": round(elapsed, 4),
                "mean_bald": float(bald_maps_dropout[K].mean()),
            })

        # Rank correlation with K=100 as reference
        ref_K = max(MC_ABLATION_K_VALUES)
        for method, maps in [("laplace", bald_maps_laplace), ("mcdropout", bald_maps_dropout)]:
            ref = maps[ref_K]
            for K in MC_ABLATION_K_VALUES:
                if K == ref_K:
                    continue
                rho, _ = spearmanr(maps[K], ref)
                # Do they agree on argmax?
                agree = int(np.argmax(maps[K]) == np.argmax(ref))
                # Update the row
                for r in rows:
                    if r["sample_id"] == sid and r["method"] == method and r["K"] == K:
                        r["spearman_vs_ref"] = round(rho, 4)
                        r["argmax_agree"] = agree

        if (img_i + 1) % 5 == 0:
            log.info("MC ablation: %d/%d images", img_i + 1, len(sample_df))

    abl_df = pd.DataFrame(rows)
    abl_df.to_csv(ablation_dir / "mc_ablation.csv", index=False)

    # Plot: wall time vs K
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, method in zip(axes, ["laplace", "mcdropout"]):
        mdf = abl_df[abl_df["method"] == method]
        wt = mdf.groupby("K")["wall_time_s"].agg(["mean", "std"]).reset_index()
        ax.errorbar(wt["K"], wt["mean"], yerr=wt["std"], marker="o", capsize=4)
        ax.set_xlabel("MC Samples (K)")
        ax.set_ylabel("Wall Time (s)")
        ax.set_title(f"{method}: Time vs K")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(ablation_dir / "wall_time_vs_K.png", dpi=150)
    plt.close(fig)

    # Plot: spearman correlation vs K
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, method in zip(axes, ["laplace", "mcdropout"]):
        mdf = abl_df[(abl_df["method"] == method) & abl_df["spearman_vs_ref"].notna()]
        if len(mdf) == 0:
            continue
        sp = mdf.groupby("K")["spearman_vs_ref"].agg(["mean", "std"]).reset_index()
        ax.errorbar(sp["K"], sp["mean"], yerr=sp["std"], marker="o", capsize=4)
        ax.set_xlabel("MC Samples (K)")
        ax.set_ylabel(f"Spearman rho vs K={ref_K}")
        ax.set_title(f"{method}: BALD Rank Stability")
        ax.set_ylim(0.5, 1.05)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(ablation_dir / "spearman_vs_K.png", dpi=150)
    plt.close(fig)

    log.info("MC ablation saved to %s", ablation_dir)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Active prompting evaluation")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--run_id", required=True, help="Run ID matching trained head")
    parser.add_argument("--budget", type=int, default=20, help="Max prompt steps per strategy")
    parser.add_argument("--mc_samples", type=int, default=20, help="MC samples for BALD")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0], help="Random seeds")
    parser.add_argument("--max_images", type=int, default=0, help="0=all, N=first N test images")
    parser.add_argument(
        "--strategies", nargs="+", default=ALL_STRATEGIES,
        help=f"Strategies to run. Choices: {ALL_STRATEGIES}",
    )
    parser.add_argument("--exclusion_radius", type=int, default=5, help="Pixel exclusion radius around existing points")
    parser.add_argument("--dropout_rate", type=float, default=0.1, help="Dropout rate for MC Dropout")
    parser.add_argument("--mc_ablation", action="store_true", help="Run MC samples ablation sweep")
    parser.add_argument("--mc_ablation_images", type=int, default=20, help="Images for MC ablation")
    parser.add_argument("--force", action="store_true", help="Clear previous results and restart")
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb")
    parser.add_argument("--device", default=None, help="Override device")
    parser.add_argument("--manifest", default=None, help="Override test manifest CSV path")
    args = parser.parse_args()

    # Validate strategies
    for s in args.strategies:
        if s not in ALL_STRATEGIES:
            parser.error(f"Unknown strategy: {s}. Choices: {ALL_STRATEGIES}")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    project_root = Path(cfg["data"]["project_root"])
    device = args.device or cfg["sam"]["device"]
    sigma = cfg["prompt_generation"]["gaussian_sigma"]

    out_dir = project_root / "active_prompting" / f"run_{args.run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = project_root / "logs" / "active_prompting.log"
    setup_logging(log_path)
    log.info("Active prompting: run_id=%s budget=%d mc=%d seeds=%s strategies=%s",
             args.run_id, args.budget, args.mc_samples, args.seeds, args.strategies)

    # Force: clear previous results
    completed_path = out_dir / "completed.json"
    step_csv_path = out_dir / "step_results.csv"
    if args.force:
        for f in [completed_path, step_csv_path]:
            if f.exists():
                f.unlink()
        log.info("Cleared previous results (--force)")

    # ---- Load models ----
    log.info("Loading NoImageHead from run_%s...", args.run_id)
    # Allow direct paths in config, else fall back to project_root convention
    if "head_checkpoint" in cfg.get("data", {}):
        best_ckpt = Path(cfg["data"]["head_checkpoint"])
    else:
        ckpt_dir = project_root / "checkpoints" / "no_image" / f"run_{args.run_id}"
        best_ckpt = ckpt_dir / "best.pt"
    if not best_ckpt.exists():
        raise FileNotFoundError(f"{best_ckpt} not found")

    head = NoImageHead(
        channels_in=cfg["head"]["channels_in"],
        hidden_channels=cfg["head"]["hidden_channels"],
    )
    state = torch.load(best_ckpt, map_location="cpu")
    if "model_state_dict" in state:
        head.load_state_dict(state["model_state_dict"])
    else:
        head.load_state_dict(state)
    head.eval()
    head.to(device)

    log.info("Loading Laplace posterior from run_%s...", args.run_id)
    if "laplace_dir" in cfg.get("data", {}):
        laplace_dir = Path(cfg["data"]["laplace_dir"])
    else:
        laplace_dir = project_root / "laplace" / "no_image" / f"run_{args.run_id}"
    la, la_meta = load_laplace(head, laplace_dir, device)
    if la.diag_var_bald is None:
        raise RuntimeError(
            "diag_var_bald is None — re-run fit_laplace_no_image.py with bald_prior_precision set"
        )
    log.info("BALD posterior: n_params=%d  diag_var_bald range=[%.4e, %.4e]  bald_prior=%.4e",
             la._n_params, la.diag_var_bald.min().item(), la.diag_var_bald.max().item(),
             la._bald_prior_precision)

    log.info("Loading SAM predictor...")
    predictor = load_sam_predictor(
        cfg["sam"]["checkpoint"], cfg["sam"]["model_type"], device
    )

    # ---- Load test manifest ----
    if args.manifest:
        manifest_path = Path(args.manifest)
    else:
        manifest_path = project_root / "manifests" / "test.csv"
    manifest = pd.read_csv(manifest_path)
    if args.max_images > 0:
        manifest = manifest.head(args.max_images)
    log.info("Test images: %d", len(manifest))

    # ---- Precompute sample attributes ----
    log.info("Precomputing sample attributes...")
    attrs_cache: Dict[str, Dict] = {}
    for _, mrow in manifest.iterrows():
        sid = str(mrow["sample_id"])
        if sid not in attrs_cache:
            gt_256 = resize_gt_mask(
                mrow["mask_path"],
                int(mrow["image_h"]),
                int(mrow["image_w"]),
            )
            attrs_cache[sid] = precompute_sample_attributes(gt_256, mrow["image_path"])
    log.info("Attributes computed for %d unique samples", len(attrs_cache))

    # ---- Resumability ----
    completed = load_completed(completed_path)
    csv_writer = IncrementalCSVWriter(step_csv_path, STEP_CSV_FIELDS)

    # ---- Save run metadata ----
    run_meta = {
        "run_id": args.run_id,
        "budget": args.budget,
        "mc_samples": args.mc_samples,
        "seeds": args.seeds,
        "strategies": args.strategies,
        "exclusion_radius": args.exclusion_radius,
        "dropout_rate": args.dropout_rate,
        "n_images": len(manifest),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(out_dir / "run_meta.json", "w") as f:
        json.dump(run_meta, f, indent=2)

    # ---- Main loop ----
    t_start = time.time()
    n_total = len(manifest) * len(args.seeds)
    n_done = 0

    for seed in args.seeds:
        for img_i, (_, mrow) in enumerate(manifest.iterrows()):
            sid = str(mrow["sample_id"])
            key = f"{sid}|{seed}"

            if key in completed:
                n_done += 1
                continue

            image_h = int(mrow["image_h"])
            image_w = int(mrow["image_w"])

            # Load image, set SAM embedding once per image
            img = np.array(Image.open(mrow["image_path"]).convert("RGB"))
            predictor.set_image(img)

            # Load GT and compute centroid
            gt_256 = resize_gt_mask(mrow["mask_path"], image_h, image_w)
            init_xy = compute_gt_centroid(gt_256)
            attrs = attrs_cache[sid]

            # Run each strategy
            for strat in args.strategies:
                rng = np.random.default_rng(seed)
                step_rows = run_active_loop(
                    sample_row=mrow,
                    la=la,
                    head=head,
                    predictor=predictor,
                    strategy=strat,
                    budget=args.budget,
                    mc_samples=args.mc_samples,
                    exclusion_radius=args.exclusion_radius,
                    init_point_xy=init_xy,
                    rng=rng,
                    device=device,
                    sigma=sigma,
                    dropout_rate=args.dropout_rate,
                    gt_256=gt_256,
                    attrs=attrs,
                    seed=seed,
                )
                csv_writer.writerows(step_rows)

            completed.add(key)
            save_completed(completed_path, completed)
            n_done += 1

            if n_done % 10 == 0 or n_done == n_total:
                elapsed = time.time() - t_start
                rate = elapsed / max(n_done, 1)
                eta = rate * (n_total - n_done)
                log.info(
                    "Progress: %d/%d images (%.1f%%)  elapsed=%.0fs  ETA=%.0fs",
                    n_done, n_total, 100 * n_done / n_total, elapsed, eta,
                )

    csv_writer.close()
    total_time = time.time() - t_start
    log.info("Active prompting complete: %.1f min", total_time / 60)

    # Update run_meta with timing
    run_meta["total_wall_time_s"] = round(total_time, 1)
    with open(out_dir / "run_meta.json", "w") as f:
        json.dump(run_meta, f, indent=2)

    # ---- Aggregation ----
    log.info("Aggregating results...")
    aggregate_results(step_csv_path, out_dir)

    # ---- Plotting ----
    log.info("Generating plots...")
    plot_all(step_csv_path, out_dir)

    # ---- Qualitative panels ----
    log.info("Generating qualitative panels...")
    generate_qualitative_panels(
        step_csv_path, out_dir, cfg, la, head, predictor, device,
        sigma, args.mc_samples, args.dropout_rate,
        manifest_path=manifest_path,
    )

    # ---- MC Ablation (optional) ----
    if args.mc_ablation:
        log.info("Running MC samples ablation (K=%s)...", MC_ABLATION_K_VALUES)
        run_mc_ablation(
            la, head, predictor, manifest, cfg, device, sigma,
            args.dropout_rate, out_dir, max_images=args.mc_ablation_images,
        )

    log.info("All outputs saved to %s", out_dir)


if __name__ == "__main__":
    main()
