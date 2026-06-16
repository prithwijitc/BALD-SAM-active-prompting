"""
Diagonal Laplace approximation for the final 1×1 conv layer of NoImageHead.

Implemented manually (no laplace-torch dependency for fitting) because:
- laplace-torch requires nn.Linear as last layer (not Conv2d)
- pixel-flattening workaround causes CUBLAS failures at B×H×W scale
- With only C_in+1 = 17 parameters, the GGN diagonal is trivially computable

Math
----
Model:  f_i = W @ phi_i + b   (1×1 conv = linear per pixel)
        phi_i ∈ R^{C_in}       (backbone feature at pixel i)

GGN diagonal (Bernoulli/sigmoid likelihood):
        diag_H[j] = Σ_i  σ'(f_i) · phi_i[j]²
        where σ'(f) = σ(f)(1-σ(f)),  phi augmented with 1 for bias term

Posterior (diagonal):
        diag_precision = prior_precision + diag_H
        diag_var       = 1 / diag_precision

Predictive (probit approximation):
        var_i  = Σ_j  phi_i[j]² · diag_var[j]
        p_i    = σ(f_i / √(1 + π/8 · var_i))
"""

import json
import logging
import math
from pathlib import Path
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)

_PI_OVER_8 = math.pi / 8.0


class DiagLaplaceHead:
    """
    Diagonal Laplace posterior over the final conv layer of NoImageHead.

    Parameters are the weights and bias of `head.final_conv` (a 1×1 Conv2d),
    treated as a linear map phi -> f = W @ phi + b.

    Usage::
        la = DiagLaplaceHead(head, device)
        la.fit(train_loader)
        best_pp, best_nll = la.tune_prior(val_loader, prior_grid)
        probs = la.predict_proba(x)          # (B, 1, H, W)
        torch.save(la.state_dict(), path)
    """

    def __init__(self, head: nn.Module, device: str, prior_precision: float = 1.0):
        self.head = head
        self.device = device
        self.prior_precision = prior_precision

        # Infer number of input features from the conv weight shape
        self._C_in = head.final_conv.weight.shape[1]   # e.g. 16
        self._n_params = self._C_in + 1                 # weights + bias

        self._diag_H:        Optional[torch.Tensor] = None  # normalized GGN diagonal (per-pixel)
        self.diag_var:       Optional[torch.Tensor] = None  # posterior variance (NLL-tuned prior)
        self.diag_var_bald:  Optional[torch.Tensor] = None  # posterior variance (BALD prior)
        self._n_pixels:      Optional[int]          = None  # total pixels seen during fit()
        self._bald_prior_precision: Optional[float] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _backbone(self, x: torch.Tensor) -> torch.Tensor:
        """Run head.backbone → (B, C_in, H, W)."""
        return self.head.backbone(x)

    def _phi(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Build augmented feature matrix [features | 1] for bias.

        feat: (B, C_in, H, W)
        returns: (B*H*W, C_in+1)
        """
        B, C, H, W = feat.shape
        phi_feat = feat.permute(0, 2, 3, 1).reshape(B * H * W, C)
        phi_bias = torch.ones(B * H * W, 1, device=feat.device, dtype=feat.dtype)
        return torch.cat([phi_feat, phi_bias], dim=1)  # (B*H*W, C_in+1)

    def _update_posterior(self, prior_precision: float):
        """Recompute diag_var from stored diag_H and the given prior precision."""
        if self._diag_H is None:
            raise RuntimeError("Call fit() before updating posterior.")
        self.prior_precision = prior_precision
        diag_prec = prior_precision + self._diag_H
        self.diag_var = 1.0 / diag_prec

    def set_bald_prior(self, bald_prior_precision: float) -> None:
        """
        Set a separate posterior variance for BALD computation.

        The NLL-tuned prior_precision is typically large (model confident → good NLL),
        which collapses diag_var_bald → 0 and kills epistemic uncertainty.
        This method sets a data-scale prior (≈ diag_H magnitude) for BALD only,
        while leaving diag_var (used for prediction) unchanged.
        """
        if self._diag_H is None:
            raise RuntimeError("Call fit() before set_bald_prior().")
        self._bald_prior_precision = bald_prior_precision
        self.diag_var_bald = 1.0 / (bald_prior_precision + self._diag_H)
        log.info("BALD prior set: precision=%.4e  diag_var_bald range: [%.4e, %.4e]",
                 bald_prior_precision,
                 self.diag_var_bald.min().item(),
                 self.diag_var_bald.max().item())

    @torch.no_grad()
    def calibrate_bald_prior(
        self,
        train_loader,
        target_std: float = 2.0,
        max_batches: int = 50,
    ) -> float:
        """
        Auto-calibrate bald_prior_precision so that the average per-pixel
        logit standard deviation under the BALD posterior equals target_std.

        Uses bisection on bald_prior_precision in [1e-6, 1e6].
        Returns the calibrated bald_prior_precision.
        """
        if self._diag_H is None:
            raise RuntimeError("Call fit() before calibrate_bald_prior().")

        self.head.eval()

        # Accumulate mean phi^2 over a sample of training data
        phi_sq_sum = torch.zeros(self._n_params, device=self.device)
        n_pixels = 0
        for i, batch in enumerate(train_loader):
            if i >= max_batches:
                break
            x = batch["input"].to(self.device)
            feat = self._backbone(x)
            phi = self._phi(feat)
            phi_sq_sum += phi.pow(2).sum(dim=0)
            n_pixels += phi.shape[0]

        mean_phi_sq = phi_sq_sum / max(n_pixels, 1)  # (n_params,)
        target_var = target_std ** 2

        def logit_var_at(pp):
            dv = 1.0 / (pp + self._diag_H)
            return (mean_phi_sq * dv).sum().item()

        # Bisection: find pp where logit_var == target_var
        lo, hi = 1e-6, 1e6
        for _ in range(100):
            mid = (lo + hi) / 2.0
            if logit_var_at(mid) > target_var:
                lo = mid
            else:
                hi = mid

        bald_pp = (lo + hi) / 2.0
        self.set_bald_prior(bald_pp)
        actual_std = math.sqrt(logit_var_at(bald_pp))
        log.info(
            "Auto-calibrated BALD prior: precision=%.4e  logit_std=%.3f (target=%.3f)",
            bald_pp, actual_std, target_std,
        )
        return bald_pp

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, train_loader) -> None:
        """
        Accumulate the GGN diagonal over the full training set, normalized by
        total pixel count so that prior_precision is on a per-datapoint scale.

        diag_H[j] = (1/N) Σ_i  σ'(f_i) · phi_i[j]²

        Without this normalization, diag_H grows linearly with dataset size
        (billions of pixels), collapsing diag_var → 0 and making BALD ≈ 0.
        """
        self.head.eval()
        self.head.to(self.device)

        diag_H = torch.zeros(self._n_params, device=self.device)
        n_pixels = 0
        n_batches = len(train_loader)

        log.info("Accumulating GGN diagonal over %d batches...", n_batches)
        for i, batch in enumerate(train_loader):
            x = batch["input"].to(self.device)   # (B, 3, H, W)

            with torch.no_grad():
                feat   = self._backbone(x)                       # (B, C_in, H, W)
                logits = self.head.final_conv(feat)              # (B, 1, H, W)
                p      = torch.sigmoid(logits)                   # (B, 1, H, W)
                sp     = (p * (1.0 - p)).squeeze(1)             # (B, H, W) σ'(f)

            phi     = self._phi(feat)                            # (B*H*W, C_in+1)
            sp_flat = sp.reshape(-1, 1)                          # (B*H*W, 1)

            # diag_H += Σ_i σ'(f_i) · phi_i²
            diag_H  += (sp_flat * phi.pow(2)).sum(dim=0)
            n_pixels += sp_flat.shape[0]

            if (i + 1) % max(1, n_batches // 10) == 0:
                log.info("  GGN accumulation: %d/%d batches done", i + 1, n_batches)

        # Normalize by total pixels → per-datapoint Fisher
        diag_H = diag_H / max(n_pixels, 1)
        self._diag_H    = diag_H
        self._n_pixels  = n_pixels
        self._update_posterior(self.prior_precision)
        log.info("GGN diagonal accumulated (normalized by %d pixels). diag_H range: [%.4e, %.4e]",
                 n_pixels, diag_H.min().item(), diag_H.max().item())

    def tune_prior(
        self,
        val_loader,
        prior_grid: List[float],
        wandb_run=None,
    ):
        """
        Grid search for best prior precision on validation NLL.

        Returns:
            best_prior_precision (float), best_val_nll (float)
        """
        if self._diag_H is None:
            raise RuntimeError("Call fit() before tune_prior().")

        best_pp, best_nll = None, float("inf")

        for step, pp in enumerate(prior_grid):
            self._update_posterior(pp)
            nll = self._eval_val_nll(val_loader)
            log.info("  prior_precision=%.6f  val_NLL=%.6f", pp, nll)

            if wandb_run:
                wandb_run.log({
                    "laplace/prior_precision_search/prior_precision": pp,
                    "laplace/prior_precision_search/val_nll": nll,
                    "laplace/prior_precision_search/step": step,
                })

            if nll < best_nll:
                best_nll, best_pp = nll, pp

        self._update_posterior(best_pp)
        log.info("Selected prior_precision=%.6f  (val_NLL=%.6f)", best_pp, best_nll)
        return best_pp, best_nll

    @torch.no_grad()
    def _eval_val_nll(self, val_loader) -> float:
        """Mean binary cross-entropy using the probit-corrected Laplace predictive."""
        self.head.eval()
        total_nll, n = 0.0, 0

        for batch in val_loader:
            x = batch["input"].to(self.device)
            y = batch["target"].to(self.device)
            if y.dim() == 4:
                y = y.squeeze(1)          # (B, H, W)

            feat   = self._backbone(x)                    # (B, C_in, H, W)
            logits = self.head.final_conv(feat)           # (B, 1, H, W)
            probs  = self._probit_predictive(feat, logits)  # (B, 1, H, W)

            probs  = probs.clamp(1e-6, 1.0 - 1e-6).reshape(-1)
            y_flat = y.reshape(-1).float()
            total_nll += F.binary_cross_entropy(probs, y_flat, reduction="sum").item()
            n += y_flat.numel()

        return total_nll / n if n > 0 else float("inf")

    def _probit_predictive(
        self, feat: torch.Tensor, logits: torch.Tensor
    ) -> torch.Tensor:
        """
        Probit approximation to the Laplace predictive probability.

        feat:   (B, C_in, H, W)
        logits: (B, 1, H, W)  MAP logits
        returns (B, 1, H, W)  predictive probabilities
        """
        B, C, H, W = feat.shape
        phi = self._phi(feat)                                      # (B*H*W, C_in+1)
        var = (phi.pow(2) * self.diag_var.to(feat.device)).sum(dim=1)  # (B*H*W,)
        var = var.reshape(B, 1, H, W)
        kappa = 1.0 / torch.sqrt(1.0 + _PI_OVER_8 * var)
        return torch.sigmoid(kappa * logits)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full predictive pipeline: image → Laplace-corrected probability map.

        x: (B, 3, H, W)
        returns: (B, 1, H, W) predictive probabilities
        """
        self.head.eval()
        feat   = self._backbone(x)
        logits = self.head.final_conv(feat)
        return self._probit_predictive(feat, logits)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        return {
            "diag_H":                self._diag_H.cpu(),
            "diag_var":              self.diag_var.cpu(),
            "diag_var_bald":         self.diag_var_bald.cpu() if self.diag_var_bald is not None else None,
            "prior_precision":       self.prior_precision,
            "bald_prior_precision":  self._bald_prior_precision,
            "n_params":              self._n_params,
            "C_in":                  self._C_in,
            "n_pixels":              getattr(self, "_n_pixels", None),
        }

    def load_state_dict(self, state: dict) -> None:
        self._diag_H              = state["diag_H"].to(self.device)
        self.diag_var             = state["diag_var"].to(self.device)
        self.prior_precision      = state["prior_precision"]
        self._n_params            = state["n_params"]
        self._C_in                = state["C_in"]
        self._n_pixels            = state.get("n_pixels", None)
        self._bald_prior_precision = state.get("bald_prior_precision", None)
        dv_bald = state.get("diag_var_bald", None)
        self.diag_var_bald = dv_bald.to(self.device) if dv_bald is not None else None
        if self._n_pixels is None:
            log.warning(
                "Loaded a legacy Laplace state (no 'n_pixels'): diag_H is NOT normalized. "
                "Re-fit with fit_laplace_no_image.py --force to fix BALD degenerate issue."
            )


# ---------------------------------------------------------------------------
# Public interface (called from fit_laplace_no_image.py)
# ---------------------------------------------------------------------------

def fit_laplace(
    model: nn.Module,
    train_loader,
    val_loader,
    prior_precision_grid: List[float],
    device: str,
    run_dir: Path,
    source_checkpoint: str,
    bald_prior_precision: Optional[Union[float, str]] = None,
    wandb_run=None,
) -> DiagLaplaceHead:
    """
    Fit diagonal Laplace on the final conv layer of model (a NoImageHead).

    Steps:
      1. Accumulate GGN diagonal over training set
      2. Grid-search prior precision on val NLL
      3. Set BALD prior (auto-calibrated or fixed)
      4. Save artifacts to run_dir
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    model.to(device)

    la = DiagLaplaceHead(model, device=device)

    log.info("Fitting diagonal Laplace on final_conv (%d params)...", la._n_params)
    la.fit(train_loader)

    log.info("Tuning prior precision over grid: %s", prior_precision_grid)
    best_pp, best_nll = la.tune_prior(val_loader, prior_precision_grid, wandb_run)

    # Set separate BALD prior (data-scale, not NLL-tuned)
    if bald_prior_precision is not None:
        if isinstance(bald_prior_precision, str) and bald_prior_precision.lower() == "auto":
            calibrated_pp = la.calibrate_bald_prior(train_loader, target_std=2.0)
            log.info("BALD prior auto-calibrated to %.4e (NLL-tuned was %.4e)",
                     calibrated_pp, best_pp)
        else:
            la.set_bald_prior(float(bald_prior_precision))
            log.info("BALD prior precision set to %.4e (separate from NLL-tuned %.4e)",
                     bald_prior_precision, best_pp)

    # Save
    torch.save(la.state_dict(), run_dir / "laplace_state.pt")

    meta = {
        "source_checkpoint":    source_checkpoint,
        "target_layer":         "final_conv",
        "approx_type":          "diagonal",
        "likelihood":           "bernoulli",
        "subset_of_weights":    "last_layer",
        "chosen_prior_precision": best_pp,
        "best_val_nll":         best_nll,
        "selection_metric":     "val_nll",
        "prior_precision_grid": prior_precision_grid,
        "n_params":             la._n_params,
        "bald_prior_precision": la._bald_prior_precision,
    }
    with open(run_dir / "posterior_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    _save_laplace_config(run_dir, best_pp)
    log.info("Laplace artifacts saved to %s", run_dir)
    return la


def load_laplace(
    model: nn.Module,
    run_dir: Path,
    device: str,
) -> tuple:
    """
    Reload a saved DiagLaplaceHead posterior.

    Returns:
        la   (DiagLaplaceHead): fitted posterior
        meta (dict):            contents of posterior_meta.json
    """
    state_path = run_dir / "laplace_state.pt"
    meta_path  = run_dir / "posterior_meta.json"

    if not state_path.exists():
        raise FileNotFoundError(f"laplace_state.pt not found in {run_dir}")

    la = DiagLaplaceHead(model, device=device)
    saved = torch.load(state_path, map_location=device)
    la.load_state_dict(saved)

    meta = {}
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)

    log.info("Loaded Laplace posterior from %s  (prior_precision=%.4e)",
             run_dir, la.prior_precision)
    return la, meta


def _save_laplace_config(run_dir: Path, best_prior: float) -> None:
    import yaml
    cfg = {
        "target_layer":      "final_conv",
        "approx":            "diagonal",
        "likelihood":        "bernoulli",
        "subset_of_weights": "last_layer",
        "chosen_prior_precision": best_prior,
    }
    with open(run_dir / "laplace_config.yaml", "w") as f:
        yaml.dump(cfg, f)
