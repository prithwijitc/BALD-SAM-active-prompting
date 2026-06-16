"""
Stage 5: Train the deterministic no-image segmentation head.

Usage:
    python train_no_image_head.py --config config.yaml --run_id v1
    python train_no_image_head.py --config config.yaml --run_id v1 --resume
    python train_no_image_head.py --config config.yaml --run_id v1 --sanity_check
"""

import argparse
import json
import logging
import sys
import uuid
from pathlib import Path

import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
import yaml

import pandas as pd

from modules.checkpoint import load_checkpoint, save_checkpoint
from modules.dataset import HDF5HeadDataset, NoImageHeadDataset
from modules.head import NoImageHead
from modules.losses import CombinedLoss
from modules.metrics import dice_from_logits, iou_from_logits
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


def compute_pos_weight(manifest_train_csv: str, device: str) -> torch.Tensor:
    """
    Compute pos_weight = total_bg_pixels / total_fg_pixels from the training manifest.
    This corrects for background dominance in BCE loss.
    """
    df = pd.read_csv(manifest_train_csv)
    total_fg = df["fg_pixels"].sum()
    total_bg = df["bg_pixels"].sum()
    ratio = total_bg / max(total_fg, 1)
    logging.info("pos_weight = bg/fg = %.1f / %.1f = %.4f", total_bg, total_fg, ratio)
    return torch.tensor([ratio], dtype=torch.float32, device=device)


def build_scheduler(optimizer, tcfg: dict, num_epochs: int):
    sched_type = tcfg.get("scheduler", "cosine").lower()
    if sched_type == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_epochs,
            eta_min=tcfg.get("scheduler_min_lr", 1e-6),
        )
    elif sched_type == "plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",   # maximise val IoU
            patience=tcfg.get("scheduler_plateau_patience", 3),
            min_lr=tcfg.get("scheduler_min_lr", 1e-6),
        )
    elif sched_type == "none":
        return None
    else:
        raise ValueError(f"Unknown scheduler: {sched_type}. Choose cosine | plateau | none.")


def run_sanity_check(model, train_loader, device, cfg):
    """Overfit a tiny batch to verify the head can learn."""
    logging.info("=== Sanity check: overfitting 2 batches ===")
    model.train()
    criterion = CombinedLoss(dice_lambda=cfg["training"]["dice_lambda"])
    opt = optim.AdamW(model.parameters(), lr=1e-2)

    batch = next(iter(train_loader))
    x = batch["input"][:4].to(device)
    y = batch["target"][:4].to(device)

    for step in range(200):
        opt.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        opt.step()
        if step % 50 == 0:
            iou = iou_from_logits(pred.detach(), y)
            logging.info("  step=%d  loss=%.4f  iou=%.4f", step, loss.item(), iou)

    final_iou = iou_from_logits(model(x).detach(), y)
    if final_iou < 0.5:
        logging.error("Sanity check FAILED: final IoU=%.4f (expected >0.5)", final_iou)
        sys.exit(1)
    logging.info("Sanity check PASSED: final IoU=%.4f", final_iou)


def train_one_epoch(model, loader, optimizer, criterion, scaler, device, use_amp):
    model.train()
    total_loss = 0.0
    n = 0
    for batch in loader:
        x = batch["input"].to(device, non_blocking=True)
        y = batch["target"].to(device, non_blocking=True)

        optimizer.zero_grad()
        with autocast(enabled=use_amp):
            pred = model(x)
            loss = criterion(pred, y)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * x.size(0)
        n += x.size(0)

    return total_loss / n if n > 0 else 0.0


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp):
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0
    n_batches = 0

    for batch in loader:
        x = batch["input"].to(device, non_blocking=True)
        y = batch["target"].to(device, non_blocking=True)

        with autocast(enabled=use_amp):
            pred = model(x)
            loss = criterion(pred, y)

        total_loss += loss.item()
        total_iou += iou_from_logits(pred, y)
        total_dice += dice_from_logits(pred, y)
        n_batches += 1

    n = max(n_batches, 1)
    return total_loss / n, total_iou / n, total_dice / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run_id", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--sanity_check", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    project_root = Path(cfg["data"]["project_root"])
    run_id = args.run_id or str(uuid.uuid4())[:8]
    run_dir = project_root / "checkpoints" / "no_image" / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    log_path = project_root / "logs" / "train_no_image.log"
    setup_logging(log_path)
    logging.info("Run ID: %s", run_id)

    # save config snapshot
    (run_dir / "config.yaml").write_text(yaml.dump(cfg))

    wandb_run = init_wandb(
        cfg,
        job_type="train",
        run_name=f"train_no_image_{run_id}",
        extra_config={"run_id": run_id, "resumed": args.resume},
    )

    tcfg = cfg["training"]
    hcfg = cfg["head"]
    device = cfg["sam"].get("device", "cuda") if torch.cuda.is_available() else "cpu"

    data_dir  = project_root / "training_data" / "no_image"
    use_hdf5  = cfg.get("hdf5", {}).get("use_hdf5", True)
    nw        = tcfg.get("num_workers", 4)
    clip      = tcfg.get("logit_clip", 20.0)
    pf        = tcfg.get("prefetch_factor", 2) if nw > 0 else None

    h5_train, h5_val = data_dir / "train.h5", data_dir / "val.h5"
    if use_hdf5 and h5_train.exists() and h5_val.exists():
        logging.info("Using HDF5 datasets.")
        train_ds = HDF5HeadDataset(str(h5_train), logit_clip=clip)
        val_ds   = HDF5HeadDataset(str(h5_val),   logit_clip=clip)
        # HDF5 workers open their own file handle after fork — persistent_workers is safe
        persistent = nw > 0
    else:
        if use_hdf5 and (h5_train.exists() or h5_val.exists()):
            logging.warning(
                "HDF5 mode requested but not all required files exist "
                "(train.h5=%s, val.h5=%s). Falling back to per-file loading. "
                "Run pack_to_hdf5.py for all splits first.",
                h5_train.exists(), h5_val.exists(),
            )
        logging.info("HDF5 not found — falling back to per-file .npy loading.")
        target_size = tuple(tcfg.get("target_size", [256, 256]))
        train_ds = NoImageHeadDataset(str(data_dir / "train_index.csv"), logit_clip=clip, target_size=target_size)
        val_ds   = NoImageHeadDataset(str(data_dir / "val_index.csv"),   logit_clip=clip, target_size=target_size)
        persistent = False  # npy workers accumulate memory

    loader_kwargs = dict(num_workers=nw, pin_memory=True, persistent_workers=persistent)
    if nw > 0 and pf:
        loader_kwargs["prefetch_factor"] = pf

    train_loader = DataLoader(train_ds, batch_size=tcfg["batch_size"], shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   batch_size=tcfg["batch_size"], shuffle=False, **loader_kwargs)

    # model
    model = NoImageHead(
        channels_in=hcfg.get("channels_in", 3),
        hidden_channels=hcfg.get("hidden_channels", 16),
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logging.info("Model: NoImageHead (%d params)", n_params)
    wandb_run.log({"model/n_params": n_params,
                   "model/n_params_final_conv": sum(p.numel() for p in model.final_conv.parameters())})

    # sanity check before full training
    if args.sanity_check:
        run_sanity_check(model, train_loader, device, cfg)
        logging.info("Sanity check done. Exiting.")
        wandb_run.finish()
        return

    optimizer = optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    scaler = GradScaler(enabled=tcfg.get("amp", True))

    # pos_weight for class-imbalance correction
    pos_weight = compute_pos_weight(
        str(project_root / "manifests" / "train.csv"), device
    )
    criterion = CombinedLoss(dice_lambda=tcfg["dice_lambda"], pos_weight=pos_weight)
    wandb_run.log({"train/pos_weight": pos_weight.item()})

    num_epochs = tcfg["num_epochs"]
    scheduler = build_scheduler(optimizer, tcfg, num_epochs)

    start_epoch = 0
    best_val_iou = 0.0
    patience_counter = 0

    # watch model gradients/params
    try:
        import wandb as _wandb
        if _wandb.run is not None:
            _wandb.watch(model, log="all", log_freq=50)
    except Exception:
        pass

    if args.resume:
        state = load_checkpoint(run_dir, model, optimizer, scheduler=scheduler, scaler=scaler)
        start_epoch = state["epoch"] + 1
        best_val_iou = state["best_val_iou"]
        logging.info("Resuming from epoch %d", start_epoch)

    use_amp = tcfg.get("amp", True) and device != "cpu"
    patience = tcfg.get("early_stop_patience", 5)
    sched_type = tcfg.get("scheduler", "cosine").lower()

    for epoch in range(start_epoch, num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device, use_amp)
        val_loss, val_iou, val_dice = evaluate(model, val_loader, criterion, device, use_amp)

        # step scheduler
        if scheduler is not None:
            if sched_type == "plateau":
                scheduler.step(val_iou)
            else:
                scheduler.step()

        is_best = val_iou > best_val_iou
        if is_best:
            best_val_iou = val_iou
            patience_counter = 0
        else:
            patience_counter += 1

        current_lr = optimizer.param_groups[0]["lr"]
        is_best_str = "[BEST]" if is_best else f"[patience {patience_counter}/{patience}]"
        logging.info(
            "Epoch %3d/%d  train_loss=%.4f  val_loss=%.4f  val_iou=%.4f  val_dice=%.4f  lr=%.2e  %s",
            epoch + 1, num_epochs, train_loss, val_loss, val_iou, val_dice, current_lr, is_best_str,
        )

        wandb_run.log({
            "epoch": epoch + 1,
            "train/loss": train_loss,
            "val/loss": val_loss,
            "val/iou": val_iou,
            "val/dice": val_dice,
            "val/best_iou": best_val_iou,
            "train/patience_counter": patience_counter,
            "train/lr": current_lr,
        })

        save_checkpoint(
            run_dir=run_dir,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            best_val_iou=best_val_iou,
            is_best=is_best,
        )

        if patience_counter >= patience:
            logging.info("Early stopping triggered after %d epochs without improvement.", patience)
            break

    wandb_run.log({
        "summary/best_val_iou": best_val_iou,
        "summary/epochs_run": epoch + 1,
    })

    # log best checkpoint as a wandb artifact
    try:
        import wandb as _wandb
        if _wandb.run is not None:
            artifact = _wandb.Artifact(
                name=f"no_image_head_{run_id}",
                type="model",
                description="Best NoImageHead checkpoint",
                metadata={"best_val_iou": best_val_iou, "run_id": run_id},
            )
            artifact.add_file(str(run_dir / "best.pt"))
            _wandb.log_artifact(artifact)
    except Exception as e:
        logging.warning("Could not log model artifact to wandb: %s", e)

    wandb_run.finish()
    logging.info("Training complete. Best val IoU: %.4f", best_val_iou)
    _write_training_report(project_root, run_id, best_val_iou, epoch + 1)


def _write_training_report(project_root, run_id, best_val_iou, epochs_run):
    report_dir = project_root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Training Summary: No-Image Head\n",
        f"- Run ID: {run_id}",
        f"- Epochs run: {epochs_run}",
        f"- Best validation IoU: {best_val_iou:.4f}",
    ]
    (report_dir / "training_summary_no_image.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
