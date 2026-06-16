"""
Stage 2: Generate and save prompt sets for all samples.

Usage:
    python generate_promptsets.py --config config.yaml
    python generate_promptsets.py --config config.yaml --force
"""

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image

from modules.promptset import generate_prompt_sets
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


def generate_all_promptsets(cfg: dict, force: bool = False, wandb_run=None):
    project_root = Path(cfg["data"]["project_root"])
    manifest_dir = project_root / "manifests"
    prompt_dir = project_root / "prompt_sets"
    pg = cfg["prompt_generation"]

    split_counts = {
        "train": pg["num_promptsets_train"],
        "val": pg["num_promptsets_val"],
        "test": pg["num_promptsets_test"],
    }

    for split, num_sets in split_counts.items():
        csv_path = manifest_dir / f"{split}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"{csv_path} not found. Run build_manifests.py first."
            )
        df = pd.read_csv(csv_path)
        split_prompt_dir = prompt_dir / split
        split_prompt_dir.mkdir(parents=True, exist_ok=True)

        logging.info("Generating prompt sets for %s split (%d samples, %d sets each)...",
                     split, len(df), num_sets)
        skipped = 0
        generated = 0

        for _, row in df.iterrows():
            out_path = split_prompt_dir / f"{row['safe_id']}.json"

            if not force and out_path.exists():
                skipped += 1
                continue

            mask = np.load(row["mask_path"])
            if mask.ndim != 2:
                logging.warning("Skipping %s: mask not 2D", row["sample_id"])
                continue

            # determine image size from the row (use mask size; resize handled at precompute)
            image_h = int(row["image_h"])
            image_w = int(row["image_w"])

            # if mask and image differ in size, resize mask to image size for prompt sampling
            if mask.shape != (image_h, image_w):
                from PIL import Image as PILImage
                m_img = PILImage.fromarray(mask.astype(np.uint8))
                m_img = m_img.resize((image_w, image_h), PILImage.NEAREST)
                mask = np.array(m_img)

            # derive a per-sample seed deterministically (hashlib avoids PYTHONHASHSEED randomness)
            sample_hash = int(hashlib.md5(row["sample_id"].encode()).hexdigest(), 16) % (2 ** 31)
            sample_seed = (pg["prompt_seed"] + sample_hash) % (2 ** 31)

            prompt_sets = generate_prompt_sets(
                mask=mask,
                num_sets=num_sets,
                min_pairs=pg["min_pairs"],
                max_pairs=pg["max_pairs"],
                seed=sample_seed,
            )

            payload = {
                "sample_id": row["sample_id"],
                "image_size": [image_h, image_w],
                "generation_seed": int(sample_seed),
                "prompt_sets": prompt_sets,
            }
            with open(out_path, "w") as f:
                json.dump(payload, f)

            generated += 1

        logging.info(
            "%s: generated=%d  skipped(already exist)=%d", split, generated, skipped
        )
        if wandb_run:
            wandb_run.log({
                f"promptsets/{split}/generated": generated,
                f"promptsets/{split}/skipped": skipped,
                f"promptsets/{split}/failed": len(df) - generated - skipped,
                f"promptsets/{split}/total_samples": len(df),
                f"promptsets/{split}/sets_per_sample": num_sets,
            })

    if wandb_run:
        total_sets = sum(
            cfg["prompt_generation"][f"num_promptsets_{s}"]
            * (len(pd.read_csv(project_root / "manifests" / f"{s}.csv")) if (project_root / "manifests" / f"{s}.csv").exists() else 0)
            for s in ["train", "val", "test"]
        )
        wandb_run.log({"promptsets/total_sets_expected": total_sets})

    logging.info("Prompt set generation complete.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    log_path = Path(cfg["data"]["project_root"]) / "logs" / "prompt_generation.log"
    setup_logging(log_path)

    run = init_wandb(cfg, job_type="prompt_generation", run_name="generate_promptsets")
    generate_all_promptsets(cfg, force=args.force, wandb_run=run)
    run.finish()


if __name__ == "__main__":
    main()
