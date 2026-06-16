"""
Stage 3 (SAM2): Run frozen SAM2 over all (sample, promptset) pairs and save artifacts.

Drop-in replacement for precompute_sam_outputs.py — uses sam2_wrapper instead of sam_wrapper.
Output .npy format is identical; only the backbone logit values differ (SAM2 vs SAM1).

Usage:
    python precompute_sam2_outputs.py --config config_sam2.yaml
    python precompute_sam2_outputs.py --config config_sam2.yaml --split train
    python precompute_sam2_outputs.py --config config_sam2.yaml --force
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image

from modules.promptset import build_gaussian_map
from modules.sam2_wrapper import load_sam_predictor, run_sam_for_promptset, verify_precompute_files
from modules.wandb_utils import init_wandb

REQUIRED_FILES = ["logits_256.npy", "pos_map.npy", "neg_map.npy", "meta.json"]


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


def precompute_split(cfg: dict, split: str, predictor, force: bool, wandb_run=None):
    project_root = Path(cfg["data"]["project_root"])
    manifest_csv = project_root / "manifests" / f"{split}.csv"
    prompt_dir = project_root / "prompt_sets" / split
    precompute_dir = project_root / "sam_precompute" / split

    if not manifest_csv.exists():
        raise FileNotFoundError(f"{manifest_csv} not found. Run build_manifests.py first.")

    df = pd.read_csv(manifest_csv)
    sigma = cfg["prompt_generation"]["gaussian_sigma"]
    multimask = cfg["sam"].get("multimask_output", True)
    select_by = cfg["sam"].get("select_by", "sam_score")
    target_size = tuple(cfg["training"].get("target_size", [256, 256]))

    n_done = 0
    n_skipped = 0
    n_failed = 0

    for _, row in df.iterrows():
        safe_id = row["safe_id"]
        sample_id = row["sample_id"]
        image_h = int(row["image_h"])
        image_w = int(row["image_w"])

        prompt_json = prompt_dir / f"{safe_id}.json"
        if not prompt_json.exists():
            logging.warning("No prompt set found for %s — skipping.", sample_id)
            n_failed += 1
            continue

        with open(prompt_json) as f:
            prompt_data = json.load(f)

        # load image once per sample
        try:
            img_np = np.array(Image.open(row["image_path"]).convert("RGB"))
        except Exception as e:
            logging.error("Failed to load image %s: %s", row["image_path"], e)
            n_failed += 1
            continue

        # set image once for all prompt sets of this sample
        try:
            predictor.set_image(img_np)
        except Exception as e:
            logging.error("predictor.set_image failed for %s: %s", sample_id, e)
            n_failed += 1
            continue

        sample_dir = precompute_dir / safe_id
        sample_dir.mkdir(parents=True, exist_ok=True)

        # save image meta once
        image_meta_path = sample_dir / "image_meta.json"
        if not image_meta_path.exists():
            with open(image_meta_path, "w") as f:
                json.dump({"sample_id": sample_id, "image_h": image_h, "image_w": image_w}, f)

        for ps in prompt_data["prompt_sets"]:
            ps_id = ps["promptset_id"]
            ps_dir = sample_dir / f"promptset_{ps_id:04d}"
            ps_dir.mkdir(parents=True, exist_ok=True)

            if not force and verify_precompute_files(ps_dir, REQUIRED_FILES):
                n_skipped += 1
                continue

            try:
                result = run_sam_for_promptset(
                    predictor=predictor,
                    positive_points=ps["positive_points"],
                    negative_points=ps["negative_points"],
                    image_hw=(image_h, image_w),
                    multimask_output=multimask,
                    target_size=target_size,
                    select_by=select_by,
                )
            except Exception as e:
                logging.error("SAM2 failed for %s promptset %d: %s", sample_id, ps_id, e)
                n_failed += 1
                continue

            # build prompt maps at target_size with rescaled point coordinates
            th, tw = target_size
            pos_pts_t = [[x * tw / image_w, y * th / image_h] for x, y in ps["positive_points"]]
            neg_pts_t = [[x * tw / image_w, y * th / image_h] for x, y in ps["negative_points"]]
            pos_map = build_gaussian_map(pos_pts_t, th, tw, sigma)
            neg_map = build_gaussian_map(neg_pts_t, th, tw, sigma)

            # save only what is needed — no full-res logits, no lowres, no selected_mask
            np.save(ps_dir / "logits_256.npy", result["logits_256"].astype(np.float16))
            np.save(ps_dir / "pos_map.npy",    pos_map.astype(np.float16))
            np.save(ps_dir / "neg_map.npy",    neg_map.astype(np.float16))

            meta = {
                "sample_id": sample_id,
                "promptset_id": ps_id,
                "image_size": [image_h, image_w],
                "logits_size": list(target_size),
                "prompt_map_size": list(target_size),
                "positive_points": ps["positive_points"],
                "negative_points": ps["negative_points"],
                "positive_points_scaled": pos_pts_t,
                "negative_points_scaled": neg_pts_t,
                "selected_branch_index": int(result["selected_branch_index"]),
                "sam_scores": result["sam_scores"].tolist(),
                "logits_256_dtype": "float16",
                "pos_map_dtype": "float16",
                "neg_map_dtype": "float16",
                "sigma": float(sigma),
                "sam_version": "sam2",
            }
            with open(ps_dir / "meta.json", "w") as f:
                json.dump(meta, f)

            n_done += 1

        processed_total = n_done + n_skipped
        if processed_total % 100 == 0 and processed_total > 0:
            logging.info(
                "[%s] processed=%d  skipped=%d  failed=%d", split, n_done, n_skipped, n_failed
            )
            if wandb_run:
                wandb_run.log({
                    f"precompute/{split}/promptsets_done": n_done,
                    f"precompute/{split}/promptsets_skipped": n_skipped,
                    f"precompute/{split}/promptsets_failed": n_failed,
                })

    logging.info(
        "[%s] DONE: processed=%d  skipped=%d  failed=%d", split, n_done, n_skipped, n_failed
    )
    if wandb_run:
        wandb_run.log({
            f"precompute/{split}/final_done": n_done,
            f"precompute/{split}/final_skipped": n_skipped,
            f"precompute/{split}/final_failed": n_failed,
            f"precompute/{split}/total_samples": len(df),
        })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="all")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    log_path = Path(cfg["data"]["project_root"]) / "logs" / "sam2_precompute.log"
    setup_logging(log_path)

    sam_cfg = cfg["sam"]
    logging.info("Loading SAM2 model: config=%s  checkpoint=%s  device=%s",
                 sam_cfg["model_type"], sam_cfg["checkpoint"], sam_cfg["device"])
    predictor = load_sam_predictor(
        checkpoint=sam_cfg["checkpoint"],
        model_type=sam_cfg["model_type"],
        device=sam_cfg["device"],
    )

    run = init_wandb(cfg, job_type="sam2_precompute", run_name="precompute_sam2_outputs")

    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    for split in splits:
        logging.info("=== Processing split: %s ===", split)
        precompute_split(cfg, split, predictor, force=args.force, wandb_run=run)

    run.finish()


if __name__ == "__main__":
    main()
