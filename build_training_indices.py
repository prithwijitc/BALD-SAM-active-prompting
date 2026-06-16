"""
Stage 4: Build flat CSV indices for training from precomputed SAM artifacts.

Usage:
    python build_training_indices.py --config config.yaml
    python build_training_indices.py --config config.yaml --force
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd
import yaml

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


REQUIRED_FILES = ["logits_256.npy", "pos_map.npy", "neg_map.npy", "meta.json"]


def build_index_for_split(cfg: dict, split: str) -> pd.DataFrame:
    project_root = Path(cfg["data"]["project_root"])
    manifest_csv = project_root / "manifests" / f"{split}.csv"
    precompute_dir = project_root / "sam_precompute" / split
    prompt_dir = project_root / "prompt_sets" / split

    df = pd.read_csv(manifest_csv)
    rows = []
    missing_count = 0

    for _, row in df.iterrows():
        safe_id = row["safe_id"]
        sample_id = row["sample_id"]
        sample_dir = precompute_dir / safe_id

        prompt_json = prompt_dir / f"{safe_id}.json"
        if not prompt_json.exists():
            logging.warning("No prompt JSON for %s", sample_id)
            missing_count += 1
            continue

        with open(prompt_json) as f:
            prompt_data = json.load(f)

        for ps in prompt_data["prompt_sets"]:
            ps_id = ps["promptset_id"]
            ps_dir = sample_dir / f"promptset_{ps_id:04d}"

            # verify all required files exist
            missing_files = [f for f in REQUIRED_FILES if not (ps_dir / f).exists()]
            if missing_files:
                logging.warning(
                    "Skipping %s promptset %d: missing %s", sample_id, ps_id, missing_files
                )
                missing_count += 1
                continue

            rows.append({
                "sample_id": sample_id,
                "safe_id": safe_id,
                "split": split,
                "promptset_id": ps_id,
                "image_path": row["image_path"],
                "image_h": int(row["image_h"]),
                "image_w": int(row["image_w"]),
                "mask_path": row["mask_path"],
                "logits_256_path": str(ps_dir / "logits_256.npy"),
                "pos_map_path": str(ps_dir / "pos_map.npy"),
                "neg_map_path": str(ps_dir / "neg_map.npy"),
                "k_pairs": ps["k_pairs"],
            })

    logging.info(
        "[%s] index rows=%d  missing/skipped=%d", split, len(rows), missing_count
    )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    project_root = Path(cfg["data"]["project_root"])
    log_path = project_root / "logs" / "build_indices.log"
    setup_logging(log_path)

    out_dir = project_root / "training_data" / "no_image"
    out_dir.mkdir(parents=True, exist_ok=True)

    run = init_wandb(cfg, job_type="build_indices", run_name="build_training_indices")

    index_sizes = {}
    for split in ["train", "val", "test"]:
        out_csv = out_dir / f"{split}_index.csv"
        if not args.force and out_csv.exists():
            logging.info("%s index already exists. Skipping. Use --force to recompute.", split)
            continue

        manifest_csv = project_root / "manifests" / f"{split}.csv"
        if not manifest_csv.exists():
            logging.warning("Manifest for %s not found, skipping.", split)
            continue

        index_df = build_index_for_split(cfg, split)
        index_df.to_csv(out_csv, index=False)
        logging.info("Wrote %d rows to %s", len(index_df), out_csv)
        index_sizes[split] = len(index_df)

        if run:
            run.log({
                f"index/{split}/total_rows": len(index_df),
                f"index/{split}/unique_samples": index_df["sample_id"].nunique(),
                f"index/{split}/mean_k_pairs": float(index_df["k_pairs"].mean()),
            })

    if run and index_sizes:
        run.log({"index/total_rows": sum(index_sizes.values())})

    run.finish()
    logging.info("Training index construction complete.")


if __name__ == "__main__":
    main()
