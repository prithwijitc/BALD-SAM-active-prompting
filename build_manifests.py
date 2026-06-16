"""
Stage 1: Build sample manifests and train/val/test splits.

Usage:
    python build_manifests.py --config config.yaml
    python build_manifests.py --config config.yaml --force  # recompute even if manifests exist
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from modules.scanner import scan_dataset
from modules.validator import validate_sample
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


def build_manifests(cfg: dict, force: bool = False, wandb_run=None):
    project_root = Path(cfg["data"]["project_root"])
    manifest_dir = project_root / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    all_csv = manifest_dir / "all_samples.csv"
    train_csv = manifest_dir / "train.csv"
    val_csv = manifest_dir / "val.csv"
    test_csv = manifest_dir / "test.csv"

    if not force and all([p.exists() for p in [all_csv, train_csv, val_csv, test_csv]]):
        logging.info("Manifests already exist. Skipping. Use --force to recompute.")
        if wandb_run:
            wandb_run.log({"status": "skipped_existing"})
        return

    # Stage 1a: scan
    records = scan_dataset(cfg["data"]["images_root"], cfg["data"]["masks_root"])
    logging.info("Found %d candidate samples.", len(records))

    # Stage 1b: validate
    rows = []
    for rec in records:
        valid, note, meta = validate_sample(rec["image_path"], rec["mask_path"])
        row = {**rec, **meta, "valid": int(valid), "note": note}
        rows.append(row)
        if not valid:
            logging.warning("INVALID  %s  reason=%s", rec["sample_id"], note)

    df_all = pd.DataFrame(rows, columns=[
        "sample_id", "safe_id", "group", "image_path", "mask_path",
        "image_h", "image_w", "mask_h", "mask_w",
        "fg_pixels", "bg_pixels", "valid", "note",
    ])
    df_all.to_csv(all_csv, index=False)

    n_valid = int(df_all["valid"].sum())
    n_invalid = len(df_all) - n_valid
    logging.info("Validated: %d valid, %d invalid", n_valid, n_invalid)

    # Stage 1c: splits (valid samples only)
    df_valid = df_all[df_all["valid"] == 1].reset_index(drop=True)
    train_df, val_df, test_df = _split(
        df_valid,
        train_frac=cfg["data"].get("train_frac", 0.80),
        val_frac=cfg["data"].get("val_frac", 0.10),
        test_frac=cfg["data"].get("test_frac", 0.10),
        seed=cfg["data"]["split_seed"],
    )

    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    logging.info(
        "Splits: train=%d  val=%d  test=%d", len(train_df), len(val_df), len(test_df)
    )
    logging.info("Manifests written to %s", manifest_dir)

    # Log to wandb
    if wandb_run:
        # per-group breakdown
        group_stats = df_valid.groupby("group").agg(
            total=("sample_id", "count"),
            fg_mean=("fg_pixels", "mean"),
        ).reset_index()

        wandb_run.log({
            "manifest/total_candidates": len(df_all),
            "manifest/valid": n_valid,
            "manifest/invalid": n_invalid,
            "manifest/train": len(train_df),
            "manifest/val": len(val_df),
            "manifest/test": len(test_df),
            "manifest/n_groups": df_valid["group"].nunique(),
            "manifest/mean_fg_pixels": float(df_valid["fg_pixels"].mean()),
            "manifest/mean_bg_pixels": float(df_valid["bg_pixels"].mean()),
        })

        try:
            import wandb
            table = wandb.Table(dataframe=group_stats)
            wandb_run.log({"manifest/group_breakdown": table})

            # log invalid sample table
            inv = df_all[df_all["valid"] == 0][["sample_id", "note"]]
            if len(inv):
                wandb_run.log({"manifest/invalid_samples": wandb.Table(dataframe=inv)})
        except Exception:
            pass  # wandb Table logging is best-effort

    # Write summary report
    _write_report(project_root, df_all, train_df, val_df, test_df)


def _split(df: pd.DataFrame, train_frac: float, val_frac: float, test_frac: float, seed: int):
    rng = np.random.default_rng(seed)
    groups = df["group"].unique()

    train_ids, val_ids, test_ids = [], [], []
    for group in sorted(groups):
        idx = df[df["group"] == group].index.tolist()
        idx = rng.permutation(idx).tolist()
        n = len(idx)
        n_train = max(0, int(np.round(n * train_frac)))
        n_val   = max(0, int(np.round(n * val_frac)))
        n_test  = max(0, int(np.round(n * test_frac)))
        # resolve rounding so all three sum to n: assign surplus/deficit to train
        total = n_train + n_val + n_test
        n_train += n - total
        # safety: ensure no split goes negative after adjustment
        if n_train < 0:
            n_val  = max(0, n_val  + n_train // 2)
            n_test = max(0, n_test + n_train - n_train // 2)
            n_train = 0
        train_ids.extend(idx[:n_train])
        val_ids.extend(idx[n_train: n_train + n_val])
        test_ids.extend(idx[n_train + n_val: n_train + n_val + n_test])

    return df.loc[train_ids].reset_index(drop=True), \
           df.loc[val_ids].reset_index(drop=True), \
           df.loc[test_ids].reset_index(drop=True)


def _write_report(project_root, df_all, train_df, val_df, test_df):
    report_dir = project_root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Preprocessing Summary\n",
        f"- Total candidate samples: {len(df_all)}",
        f"- Valid: {df_all['valid'].sum()}",
        f"- Invalid: {(df_all['valid'] == 0).sum()}",
        f"- Train: {len(train_df)}",
        f"- Val: {len(val_df)}",
        f"- Test: {len(test_df)}",
        "",
        "## Invalid samples",
    ]
    inv = df_all[df_all["valid"] == 0][["sample_id", "note"]]
    for _, r in inv.iterrows():
        lines.append(f"- {r['sample_id']}: {r['note']}")
    (report_dir / "preprocessing_summary.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    log_path = Path(cfg["data"]["project_root"]) / "logs" / "manifest.log"
    setup_logging(log_path)

    run = init_wandb(cfg, job_type="manifest", run_name="build_manifests")
    build_manifests(cfg, force=args.force, wandb_run=run)
    run.finish()


if __name__ == "__main__":
    main()
