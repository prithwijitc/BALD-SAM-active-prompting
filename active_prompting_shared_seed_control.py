#!/usr/bin/env python3
"""Controlled shared-seed active prompting evaluation.

This script evaluates all acquisition strategies from the same fixed,
model-independent prompt initialization so that differences can be attributed
to the acquisition rule rather than to strategy-specific prompt histories.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image

from active_prompting import (
    ALL_STRATEGIES,
    binary_entropy,
    build_exclusion_mask,
    build_gaussian_map,
    compute_all_metrics,
    compute_bald_laplace,
    compute_bald_mcdropout,
    dispatch_strategy,
    load_completed,
    load_sam_predictor,
    precompute_sample_attributes,
    resize_gt_mask,
    save_completed,
    scale_points_to_original,
    setup_logging,
)
from modules.head import NoImageHead
from modules.laplace_wrapper import load_laplace
from modules.sam_wrapper import run_sam_for_promptset


_BASE_FIELDS = [
    "sample_id",
    "group",
    "seed",
    "strategy",
    "shared_seed_k",
    "shared_seed_mode",
    "extra_step",
    "total_prompts",
    "point_x",
    "point_y",
    "point_label",
    "n_pos",
    "n_neg",
]

_METRIC_FIELDS = [
    "iou",
    "dice",
    "boundary_iou",
    "hausdorff95",
    "ece",
    "nll",
    "mean_bald",
    "mean_entropy",
]

_ATTR_FIELDS = [
    "object_size_bin",
    "thinness_bin",
    "boundary_complexity_bin",
    "contrast_bin",
    "fg_pixels",
    "thinness_val",
    "boundary_complexity_val",
    "contrast_val",
]

STEP_CSV_FIELDS = _BASE_FIELDS + _METRIC_FIELDS + _ATTR_FIELDS


class IncrementalCSVWriter:
    """Append rows to a CSV file incrementally."""

    def __init__(self, path: Path, fieldnames: Sequence[str]):
        self.path = path
        self.fieldnames = list(fieldnames)
        write_header = not path.exists() or path.stat().st_size == 0
        self._file = open(path, "a", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=self.fieldnames, extrasaction="ignore")
        if write_header:
            self._writer.writeheader()
            self._file.flush()

    def writerows(self, rows: Iterable[Dict]) -> None:
        for row in rows:
            self._writer.writerow(row)

    def close(self) -> None:
        self._file.flush()
        self._file.close()


def fps_points_from_mask(mask: np.ndarray, n_points: int, first_xy: Tuple[int, int] | None = None) -> List[List[int]]:
    """Greedy farthest-point sampling on a boolean mask in 256x256 space."""
    coords_yx = np.argwhere(mask)
    if len(coords_yx) == 0 or n_points <= 0:
        return []

    coords_xy = coords_yx[:, [1, 0]].astype(np.int32)
    selected: List[np.ndarray] = []

    if first_xy is not None:
        first = np.array(first_xy, dtype=np.int32)
        # Snap to nearest valid coordinate if needed.
        d2 = np.sum((coords_xy - first[None, :]) ** 2, axis=1)
        selected.append(coords_xy[int(np.argmin(d2))])
    else:
        center = coords_xy.mean(axis=0)
        d2 = np.sum((coords_xy - center[None, :]) ** 2, axis=1)
        selected.append(coords_xy[int(np.argmin(d2))])

    while len(selected) < min(n_points, len(coords_xy)):
        stacked = np.stack(selected, axis=0)
        d2 = ((coords_xy[:, None, :] - stacked[None, :, :]) ** 2).sum(axis=2)
        min_d2 = d2.min(axis=1)
        next_idx = int(np.argmax(min_d2))
        next_pt = coords_xy[next_idx]
        if not any(np.array_equal(next_pt, s) for s in selected):
            selected.append(next_pt)
        else:
            break

    return [[int(p[0]), int(p[1])] for p in selected]


def build_deterministic_shared_seed(gt_256: np.ndarray, k: int) -> Tuple[List[List[int]], List[List[int]]]:
    """Build a model-independent shared seed using deterministic FG/BG coverage."""
    fg_mask = gt_256 > 0
    bg_mask = ~fg_mask

    if fg_mask.sum() == 0:
        n_pos = 0
        n_neg = k
    elif bg_mask.sum() == 0:
        n_pos = k
        n_neg = 0
    else:
        n_pos = (k + 1) // 2
        n_neg = k // 2

    fg_coords = np.argwhere(fg_mask)
    if len(fg_coords) > 0:
        cy, cx = fg_coords.mean(axis=0)
        centroid_xy = (int(round(cx)), int(round(cy)))
    else:
        centroid_xy = (128, 128)

    pos_points = fps_points_from_mask(fg_mask, n_pos, first_xy=centroid_xy)

    bg_first = None
    if n_neg > 0:
        # Start the BG seed far from the FG centroid to ensure the seed encodes exclusion.
        bg_coords = np.argwhere(bg_mask)[:, [1, 0]]
        if len(bg_coords) > 0:
            d2 = np.sum((bg_coords - np.array(centroid_xy)[None, :]) ** 2, axis=1)
            bg_first = tuple(bg_coords[int(np.argmax(d2))].tolist())
    neg_points = fps_points_from_mask(bg_mask, n_neg, first_xy=bg_first)
    return pos_points, neg_points


@torch.no_grad()
def evaluate_state(
    sample_row: pd.Series,
    la,
    head: NoImageHead,
    predictor,
    strategy: str,
    mc_samples: int,
    pos_points_256: List[List[int]],
    neg_points_256: List[List[int]],
    device: str,
    sigma: float,
    dropout_rate: float,
    gt_256: np.ndarray,
) -> Dict:
    """Run SAM + head on the current prompt state and return predictions/maps."""
    H_orig = int(sample_row["image_h"])
    W_orig = int(sample_row["image_w"])
    pos_orig = scale_points_to_original(pos_points_256, H_orig, W_orig)
    neg_orig = scale_points_to_original(neg_points_256, H_orig, W_orig) if neg_points_256 else []

    sam_out = run_sam_for_promptset(
        predictor,
        pos_orig,
        neg_orig if neg_orig else [],
        (H_orig, W_orig),
        multimask_output=True,
        target_size=(256, 256),
    )
    logits_256 = sam_out["logits_256"]

    pos_map = build_gaussian_map(pos_points_256, 256, 256, sigma)
    neg_map = (
        build_gaussian_map(neg_points_256, 256, 256, sigma)
        if neg_points_256
        else np.zeros((256, 256), dtype=np.float32)
    )

    logits_clipped = np.clip(logits_256, -20.0, 20.0)
    x_input = torch.from_numpy(np.stack([logits_clipped, pos_map, neg_map], axis=0)).float().unsqueeze(0).to(device)
    feat = la._backbone(x_input)
    map_logits = head.final_conv(feat)
    probit_probs = la._probit_predictive(feat, map_logits)
    probs_np = probit_probs.squeeze().cpu().numpy()
    logits_np = map_logits.squeeze().cpu().numpy()

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
        ent_np = binary_entropy(probit_probs.squeeze()).cpu().numpy()
        bald_np = np.zeros((256, 256), dtype=np.float32)

    metrics = compute_all_metrics(logits_np, probs_np, gt_256)
    metrics["mean_bald"] = float(bald_np.mean()) if bald_np.any() else 0.0
    metrics["mean_entropy"] = float(ent_np.mean())

    return {
        "metrics": metrics,
        "bald_np": bald_np,
        "ent_np": ent_np,
        "logits_np": logits_np,
    }


@torch.no_grad()
def run_shared_seed_control(
    sample_row: pd.Series,
    la,
    head: NoImageHead,
    predictor,
    strategy: str,
    mc_samples: int,
    exclusion_radius: int,
    init_pos_points_256: List[List[int]],
    init_neg_points_256: List[List[int]],
    max_extra_steps: int,
    rng: np.random.Generator,
    device: str,
    sigma: float,
    dropout_rate: float,
    gt_256: np.ndarray,
    attrs: Dict,
    seed: int,
    shared_seed_k: int,
    shared_seed_mode: str,
) -> List[Dict]:
    """Evaluate a strategy from a fixed shared seed and record extra-step states."""
    sample_id = str(sample_row["sample_id"])
    group = sample_id.split("/")[0] if "/" in sample_id else ""
    pos_points_256 = [list(p) for p in init_pos_points_256]
    neg_points_256 = [list(p) for p in init_neg_points_256]
    results: List[Dict] = []

    for extra_step in range(max_extra_steps + 1):
        state = evaluate_state(
            sample_row=sample_row,
            la=la,
            head=head,
            predictor=predictor,
            strategy=strategy,
            mc_samples=mc_samples,
            pos_points_256=pos_points_256,
            neg_points_256=neg_points_256,
            device=device,
            sigma=sigma,
            dropout_rate=dropout_rate,
            gt_256=gt_256,
        )

        if extra_step == 0:
            pt_x = pt_y = -1
            pt_label = -1
        else:
            pt_x, pt_y = results[-1]["_next_point"]
            pt_label = results[-1]["_next_label"]

        row = {
            "sample_id": sample_id,
            "group": group,
            "seed": seed,
            "strategy": strategy,
            "shared_seed_k": shared_seed_k,
            "shared_seed_mode": shared_seed_mode,
            "extra_step": extra_step,
            "total_prompts": len(pos_points_256) + len(neg_points_256),
            "point_x": pt_x,
            "point_y": pt_y,
            "point_label": pt_label,
            "n_pos": len(pos_points_256),
            "n_neg": len(neg_points_256),
        }
        row.update(state["metrics"])
        row.update(attrs)
        results.append(row)

        if extra_step == max_extra_steps:
            continue

        all_pts = pos_points_256 + neg_points_256
        exclusion = build_exclusion_mask(all_pts, 256, 256, exclusion_radius)
        nx, ny = dispatch_strategy(
            strategy,
            state["bald_np"],
            state["ent_np"],
            state["logits_np"],
            gt_256,
            all_pts,
            exclusion,
            rng,
        )
        label = int(gt_256[ny, nx] > 0)
        results[-1]["_next_point"] = (nx, ny)
        results[-1]["_next_label"] = label
        if label == 1:
            pos_points_256.append([nx, ny])
        else:
            neg_points_256.append([nx, ny])

    for row in results:
        row.pop("_next_point", None)
        row.pop("_next_label", None)
    return results


def summarize_control_results(step_csv_path: Path, out_dir: Path) -> None:
    df = pd.read_csv(step_csv_path)
    metric_cols = ["iou", "dice", "boundary_iou", "hausdorff95", "ece", "nll", "mean_bald", "mean_entropy"]

    agg = (
        df.groupby(["shared_seed_k", "extra_step", "strategy"])[metric_cols]
        .agg(["mean", "std", "median"])
        .reset_index()
    )
    agg.columns = [
        "_".join(str(x) for x in col if str(x) != "").strip("_") if isinstance(col, tuple) else col
        for col in agg.columns
    ]
    agg.to_csv(out_dir / "aggregate.csv", index=False)

    per_group = (
        df.groupby(["shared_seed_k", "extra_step", "strategy", "group"])[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    per_group.columns = [
        "_".join(str(x) for x in col if str(x) != "").strip("_") if isinstance(col, tuple) else col
        for col in per_group.columns
    ]
    per_group.to_csv(out_dir / "per_group.csv", index=False)

    # Reviewer-facing summary: next-prompt effect only.
    next_prompt = df[df["extra_step"] == 1].copy()
    next_summary = (
        next_prompt.groupby(["shared_seed_k", "strategy"])[["iou", "ece", "hausdorff95", "nll"]]
        .agg(["mean", "std", "median"])
        .reset_index()
    )
    next_summary.columns = [
        "_".join(str(x) for x in col if str(x) != "").strip("_") if isinstance(col, tuple) else col
        for col in next_summary.columns
    ]
    next_summary.to_csv(out_dir / "next_prompt_summary.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Controlled shared-seed active prompting evaluation")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--run_id", required=True, help="Head/Laplace run id to evaluate")
    parser.add_argument("--output_run_id", default=None, help="Output folder suffix under active_prompting/")
    parser.add_argument("--manifest", default=None, help="Override test manifest CSV path")
    parser.add_argument("--device", default=None, help="Override CUDA device")
    parser.add_argument("--max_images", type=int, default=0, help="Restrict to the first N test images")
    parser.add_argument("--strategies", nargs="+", default=ALL_STRATEGIES, help="Strategies to evaluate")
    parser.add_argument("--shared_seed_ks", type=int, nargs="+", default=[3, 5, 8], help="Shared-seed sizes")
    parser.add_argument("--extra_steps", type=int, default=1, help="Number of strategy-specific prompts after the shared seed")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0], help="Evaluation seeds")
    parser.add_argument("--mc_samples", type=int, default=20, help="Posterior samples for BALD Laplace / MC Dropout")
    parser.add_argument("--exclusion_radius", type=int, default=5, help="Prompt exclusion radius in 256x256 space")
    parser.add_argument("--dropout_rate", type=float, default=0.1, help="Dropout rate for MC Dropout BALD")
    parser.add_argument("--force", action="store_true", help="Clear previous results")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    project_root = Path(cfg["data"]["project_root"])
    device = args.device or cfg["sam"]["device"]
    sigma = cfg["prompt_generation"]["gaussian_sigma"]

    output_run_id = args.output_run_id or f"{args.run_id}_sharedseed_control"
    out_dir = project_root / "active_prompting" / f"run_{output_run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = project_root / "logs" / "active_prompting_shared_seed_control.log"
    setup_logging(log_path)

    completed_path = out_dir / "completed.json"
    step_csv_path = out_dir / "step_results.csv"
    if args.force:
        for path in [completed_path, step_csv_path]:
            if path.exists():
                path.unlink()

    if args.manifest:
        manifest_path = Path(args.manifest)
    else:
        manifest_path = project_root / "manifests" / "test.csv"
    manifest = pd.read_csv(manifest_path)
    if args.max_images > 0:
        manifest = manifest.head(args.max_images)

    if "head_checkpoint" in cfg.get("data", {}):
        best_ckpt = Path(cfg["data"]["head_checkpoint"])
    else:
        best_ckpt = project_root / "checkpoints" / "no_image" / f"run_{args.run_id}" / "best.pt"
    if "laplace_dir" in cfg.get("data", {}):
        laplace_dir = Path(cfg["data"]["laplace_dir"])
    else:
        laplace_dir = project_root / "laplace" / "no_image" / f"run_{args.run_id}"

    head = NoImageHead(
        channels_in=cfg["head"]["channels_in"],
        hidden_channels=cfg["head"]["hidden_channels"],
    )
    state = torch.load(best_ckpt, map_location="cpu")
    head.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)
    head.eval()
    head.to(device)

    la, _ = load_laplace(head, laplace_dir, device)
    predictor = load_sam_predictor(cfg["sam"]["checkpoint"], cfg["sam"]["model_type"], device)

    attrs_cache: Dict[str, Dict] = {}
    for _, mrow in manifest.iterrows():
        sid = str(mrow["sample_id"])
        if sid in attrs_cache:
            continue
        gt_256 = resize_gt_mask(mrow["mask_path"], int(mrow["image_h"]), int(mrow["image_w"]))
        attrs_cache[sid] = precompute_sample_attributes(gt_256, mrow["image_path"])

    completed = load_completed(completed_path)
    csv_writer = IncrementalCSVWriter(step_csv_path, STEP_CSV_FIELDS)

    run_meta = {
        "run_id": args.run_id,
        "output_run_id": output_run_id,
        "shared_seed_mode": "deterministic_fps_fg_bg",
        "shared_seed_ks": args.shared_seed_ks,
        "extra_steps": args.extra_steps,
        "mc_samples": args.mc_samples,
        "seeds": args.seeds,
        "strategies": args.strategies,
        "exclusion_radius": args.exclusion_radius,
        "dropout_rate": args.dropout_rate,
        "n_images": len(manifest),
        "manifest_path": str(manifest_path),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2))

    t_start = time.time()
    n_total = len(manifest) * len(args.seeds) * len(args.shared_seed_ks)
    n_done = 0

    for seed in args.seeds:
        for shared_seed_k in args.shared_seed_ks:
            for _, mrow in manifest.iterrows():
                sid = str(mrow["sample_id"])
                key = f"{sid}|{seed}|k{shared_seed_k}"
                if key in completed:
                    n_done += 1
                    continue

                image_h = int(mrow["image_h"])
                image_w = int(mrow["image_w"])
                img = np.array(Image.open(mrow["image_path"]).convert("RGB"))
                predictor.set_image(img)
                gt_256 = resize_gt_mask(mrow["mask_path"], image_h, image_w)
                attrs = attrs_cache[sid]
                init_pos, init_neg = build_deterministic_shared_seed(gt_256, shared_seed_k)

                for strat in args.strategies:
                    rng = np.random.default_rng(seed)
                    rows = run_shared_seed_control(
                        sample_row=mrow,
                        la=la,
                        head=head,
                        predictor=predictor,
                        strategy=strat,
                        mc_samples=args.mc_samples,
                        exclusion_radius=args.exclusion_radius,
                        init_pos_points_256=init_pos,
                        init_neg_points_256=init_neg,
                        max_extra_steps=args.extra_steps,
                        rng=rng,
                        device=device,
                        sigma=sigma,
                        dropout_rate=args.dropout_rate,
                        gt_256=gt_256,
                        attrs=attrs,
                        seed=seed,
                        shared_seed_k=shared_seed_k,
                        shared_seed_mode="deterministic_fps_fg_bg",
                    )
                    csv_writer.writerows(rows)

                completed.add(key)
                save_completed(completed_path, completed)
                n_done += 1

                if n_done % 25 == 0 or n_done == n_total:
                    elapsed = time.time() - t_start
                    rate = elapsed / max(n_done, 1)
                    eta = rate * (n_total - n_done)
                    print(
                        f"Progress: {n_done}/{n_total} ({100.0 * n_done / n_total:.1f}%) "
                        f"elapsed={elapsed:.0f}s ETA={eta:.0f}s"
                    )

    csv_writer.close()
    total_time = time.time() - t_start
    run_meta["total_wall_time_s"] = round(total_time, 1)
    (out_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2))
    summarize_control_results(step_csv_path, out_dir)


if __name__ == "__main__":
    main()
