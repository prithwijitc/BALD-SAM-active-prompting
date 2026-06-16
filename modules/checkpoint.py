"""Checkpoint save/load utilities for fully resumable training."""

import json
import logging
from pathlib import Path
from typing import Optional

import torch

log = logging.getLogger(__name__)


def save_checkpoint(
    run_dir: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    epoch: int,
    best_val_iou: float,
    is_best: bool,
):
    """Save last.pt and optionally best.pt, plus all optimizer/scaler states."""
    run_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), run_dir / "last.pt")
    torch.save(optimizer.state_dict(), run_dir / "optimizer_last.pt")

    if scheduler is not None:
        torch.save(scheduler.state_dict(), run_dir / "scheduler_last.pt")

    if scaler is not None:
        torch.save(scaler.state_dict(), run_dir / "scaler_last.pt")

    epoch_state = {"epoch": epoch, "best_val_iou": best_val_iou}
    with open(run_dir / "epoch_state.json", "w") as f:
        json.dump(epoch_state, f, indent=2)

    if is_best:
        torch.save(model.state_dict(), run_dir / "best.pt")
        log.info("Saved best checkpoint at epoch %d (val_iou=%.4f)", epoch, best_val_iou)


def load_checkpoint(
    run_dir: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
) -> dict:
    """
    Load training state from run_dir. Fails loudly if any expected file is missing.

    Returns:
        dict with keys: epoch, best_val_iou
    """
    required = ["last.pt", "optimizer_last.pt", "epoch_state.json"]
    for fname in required:
        if not (run_dir / fname).exists():
            raise FileNotFoundError(
                f"Resume requested but {run_dir / fname} is missing. "
                "Cannot resume — check checkpoint directory."
            )

    model.load_state_dict(torch.load(run_dir / "last.pt", map_location="cpu"))
    optimizer.load_state_dict(torch.load(run_dir / "optimizer_last.pt", map_location="cpu"))

    sched_path = run_dir / "scheduler_last.pt"
    if scheduler is not None and sched_path.exists():
        scheduler.load_state_dict(torch.load(sched_path, map_location="cpu"))

    scaler_path = run_dir / "scaler_last.pt"
    if scaler is not None and scaler_path.exists():
        scaler.load_state_dict(torch.load(scaler_path, map_location="cpu"))

    with open(run_dir / "epoch_state.json") as f:
        state = json.load(f)

    log.info(
        "Resumed from epoch %d (best_val_iou=%.4f)", state["epoch"], state["best_val_iou"]
    )
    return state
