"""
Stage 6: Fit Laplace approximation on the final layer of the trained no-image head.

Usage:
    python fit_laplace_no_image.py --config config.yaml --run_id v1
    python fit_laplace_no_image.py --config config.yaml --run_id v1 --force
"""

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from modules.dataset import HDF5HeadDataset, NoImageHeadDataset
from modules.head import NoImageHead
from modules.laplace_wrapper import fit_laplace, load_laplace
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


def checkpoint_hash(ckpt_path: Path) -> str:
    """Compute MD5 of a checkpoint file for provenance tracking."""
    h = hashlib.md5()
    with open(ckpt_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    project_root = Path(cfg["data"]["project_root"])
    log_path = project_root / "logs" / "laplace_no_image.log"
    setup_logging(log_path)

    run_id = args.run_id
    ckpt_dir = project_root / "checkpoints" / "no_image" / f"run_{run_id}"
    laplace_dir = project_root / "laplace" / "no_image" / f"run_{run_id}"

    best_ckpt = ckpt_dir / "best.pt"
    if not best_ckpt.exists():
        raise FileNotFoundError(
            f"{best_ckpt} not found. Run train_no_image_head.py first."
        )

    # check if already fitted
    if not args.force and (laplace_dir / "laplace_state.pt").exists():
        meta_path = laplace_dir / "posterior_meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            ckpt_md5 = checkpoint_hash(best_ckpt)
            if meta.get("source_checkpoint_md5") == ckpt_md5:
                logging.info(
                    "Laplace artifacts already exist and match checkpoint. "
                    "Skipping. Use --force to refit."
                )
                return

    device = cfg["sam"].get("device", "cuda") if torch.cuda.is_available() else "cpu"
    hcfg = cfg["head"]

    # load model with best weights
    model = NoImageHead(
        channels_in=hcfg.get("channels_in", 3),
        hidden_channels=hcfg.get("hidden_channels", 16),
    )
    model.load_state_dict(torch.load(best_ckpt, map_location="cpu"))
    model.eval()
    model.to(device)
    logging.info("Loaded best checkpoint from %s", best_ckpt)

    tcfg     = cfg["training"]
    data_dir = project_root / "training_data" / "no_image"
    nw       = tcfg.get("num_workers", 4)
    clip     = tcfg.get("logit_clip", 20.0)
    use_hdf5 = cfg.get("hdf5", {}).get("use_hdf5", True)

    h5_train, h5_val = data_dir / "train.h5", data_dir / "val.h5"
    if use_hdf5 and h5_train.exists() and h5_val.exists():
        logging.info("Using HDF5 datasets for Laplace fitting.")
        train_ds = HDF5HeadDataset(str(h5_train), logit_clip=clip)
        val_ds   = HDF5HeadDataset(str(h5_val),   logit_clip=clip)
        persistent = nw > 0
    else:
        if use_hdf5 and (h5_train.exists() or h5_val.exists()):
            logging.warning(
                "HDF5 mode requested but not all required files exist "
                "(train.h5=%s, val.h5=%s). Falling back to per-file loading. "
                "Run pack_to_hdf5.py for all splits first.",
                h5_train.exists(), h5_val.exists(),
            )
        target_size = tuple(tcfg.get("target_size", [256, 256]))
        train_ds = NoImageHeadDataset(str(data_dir / "train_index.csv"), logit_clip=clip, target_size=target_size)
        val_ds   = NoImageHeadDataset(str(data_dir / "val_index.csv"),   logit_clip=clip, target_size=target_size)
        persistent = False

    fit_batch = max(1, tcfg["batch_size"] // 2)
    train_loader = DataLoader(
        train_ds, batch_size=fit_batch, shuffle=False,
        num_workers=nw, pin_memory=True, persistent_workers=persistent,
    )
    val_loader = DataLoader(
        val_ds, batch_size=fit_batch, shuffle=False,
        num_workers=nw, pin_memory=True, persistent_workers=persistent,
    )

    lcfg = cfg["laplace"]
    laplace_dir.mkdir(parents=True, exist_ok=True)

    wandb_run = init_wandb(
        cfg,
        job_type="laplace",
        run_name=f"laplace_no_image_{run_id}",
        extra_config={"run_id": run_id, "source_checkpoint": str(best_ckpt)},
    )

    la = fit_laplace(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        prior_precision_grid=lcfg["prior_precision_grid"],
        device=device,
        run_dir=laplace_dir,
        source_checkpoint=str(best_ckpt),
        bald_prior_precision=lcfg.get("bald_prior_precision", None),
        wandb_run=wandb_run,
    )

    # patch posterior_meta.json with checkpoint hash
    meta_path = laplace_dir / "posterior_meta.json"
    with open(meta_path) as f:
        meta = json.load(f)
    meta["source_checkpoint_md5"] = checkpoint_hash(best_ckpt)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    wandb_run.log({
        "laplace/chosen_prior_precision": meta["chosen_prior_precision"],
        "laplace/best_val_nll": meta["best_val_nll"],
        "laplace/target_layer": meta["target_layer"],
        "laplace/approx_type": meta["approx_type"],
    })

    # log Laplace artifacts
    try:
        import wandb as _wandb
        if _wandb.run is not None:
            artifact = _wandb.Artifact(
                name=f"laplace_posterior_{run_id}",
                type="laplace",
                description="Diagonal last-layer Laplace posterior",
                metadata=meta,
            )
            artifact.add_file(str(laplace_dir / "laplace_state.pt"))
            artifact.add_file(str(laplace_dir / "posterior_meta.json"))
            _wandb.log_artifact(artifact)
    except Exception as e:
        logging.warning("Could not log Laplace artifact to wandb: %s", e)

    wandb_run.finish()
    logging.info("Laplace fitting complete. Artifacts at %s", laplace_dir)

    # write report
    _write_laplace_report(project_root, run_id, meta)


def _write_laplace_report(project_root, run_id, meta):
    report_dir = project_root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Laplace Summary: No-Image Head\n",
        f"- Run ID: {run_id}",
        f"- Source checkpoint: {meta.get('source_checkpoint', 'N/A')}",
        f"- Target layer: {meta.get('target_layer', 'N/A')}",
        f"- Approximation: {meta.get('approx_type', 'N/A')}",
        f"- Chosen prior precision: {meta.get('chosen_prior_precision', 'N/A')}",
        f"- Best val NLL: {meta.get('best_val_nll', 'N/A')}",
        f"- Selection metric: {meta.get('selection_metric', 'N/A')}",
    ]
    (report_dir / "laplace_summary_no_image.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
