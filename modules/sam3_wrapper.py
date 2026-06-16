"""Frozen SAM3 image wrapper — drop-in replacement for sam_wrapper.py.

Identical public API to sam_wrapper.py:
    load_sam_predictor(checkpoint, model_type, device) -> _SAM3Predictor
    run_sam_for_promptset(...) -> dict
    verify_precompute_files(...) -> bool

This wrapper intentionally uses the official facebookresearch/sam3 package for
image point prompting. The previous Transformers-based Sam3TrackerModel path was
not a checkpoint/architecture match and produced partially initialized models.

Supported checkpoints:
  - "facebook/sam3"
  - local SAM3 image checkpoint files such as "sam3.pt"

Not supported:
  - "facebook/sam3.1" / multiplex checkpoints. SAM3.1 is a video-tracking
    release, and this repo does not implement a validated single-image
    SAM1-style predictor for it.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

log = logging.getLogger(__name__)

_SAM3_INSTALL_HINT = (
    "SAM3 requires the official facebookresearch/sam3 package in sam3-env. "
    "Install with: "
    "pip install 'git+https://github.com/facebookresearch/sam3.git@"
    "bfbed072a07a6a52c8d5fdc75a7a186251a835b1' "
    "and install the runtime dependency with: pip install pycocotools"
)


class _SAM3Predictor:
    """Thin adapter around the official SAM3 image model + processor."""

    def __init__(self, model, processor, device: str):
        self.model = model
        self.processor = processor
        self.device = device
        self._inference_state = None

    def set_image(self, img_np: np.ndarray):
        """Cache image embeddings for subsequent predict() calls."""
        self._inference_state = self.processor.set_image(Image.fromarray(img_np))

    def predict(
        self,
        positive_points: List[List[int]],
        negative_points: List[List[int]],
        multimask_output: bool = True,
    ) -> Dict[str, np.ndarray]:
        """Run SAM3 inference for one prompt set."""
        all_points = positive_points + negative_points
        if not all_points:
            raise ValueError("SAM3 requires at least one positive or negative point.")

        point_coords = np.asarray(all_points, dtype=np.float32)
        point_labels = np.asarray(
            [1] * len(positive_points) + [0] * len(negative_points), dtype=np.int32
        )

        if self._inference_state is None:
            raise RuntimeError("set_image() must be called before predict().")

        with torch.no_grad():
            masks, scores, low_res_masks = self.model.predict_inst(
                self._inference_state,
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=multimask_output,
            )

        return {
            "masks": np.asarray(masks, dtype=np.float32),
            "scores": np.asarray(scores, dtype=np.float32),
            "low_res_masks": np.asarray(low_res_masks, dtype=np.float32),
        }


def _is_unsupported_sam31_ref(ref: str) -> bool:
    if not ref:
        return False
    ref_l = str(ref).lower()
    ref_name = Path(ref_l).name
    return "sam3.1" in ref_l or "multiplex" in ref_name or ref_name == "sam31.pt"


def _resolve_checkpoint(checkpoint: str, model_type: str) -> Tuple[str | None, bool]:
    if _is_unsupported_sam31_ref(checkpoint) or _is_unsupported_sam31_ref(model_type):
        raise NotImplementedError(
            "SAM3.1 is not supported by this single-image point-prompt pipeline. "
            f"Received checkpoint={checkpoint!r}, model_type={model_type!r}. "
            "The official facebook/sam3.1 release is an Object Multiplex "
            "video-tracking checkpoint, not a validated SAM1-style image "
            "predictor in this repo. Use facebook/sam3 here."
        )

    if checkpoint == "facebook/sam3":
        return None, True

    checkpoint_path = Path(checkpoint).expanduser()
    if checkpoint_path.is_file():
        return str(checkpoint_path), False

    raise ValueError(
        "Unsupported SAM3 checkpoint. Use 'facebook/sam3' or a local SAM3 image "
        f"checkpoint file such as 'sam3.pt'. Got checkpoint={checkpoint!r}."
    )


def load_sam_predictor(checkpoint: str, model_type: str, device: str) -> _SAM3Predictor:
    """Load official SAM3 image predictor and return a _SAM3Predictor."""
    checkpoint_path, load_from_hf = _resolve_checkpoint(checkpoint, model_type)

    try:
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model
    except ImportError as exc:
        raise ImportError(_SAM3_INSTALL_HINT) from exc

    log.info(
        "Loading official SAM3 image predictor: checkpoint=%s model_type=%s device=%s",
        checkpoint,
        model_type,
        device,
    )

    build_device = "cuda" if str(device).startswith("cuda") else "cpu"
    model = build_sam3_image_model(
        device=build_device,
        eval_mode=True,
        checkpoint_path=checkpoint_path,
        load_from_HF=load_from_hf,
        enable_inst_interactivity=True,
    )

    if getattr(model, "inst_interactive_predictor", None) is None:
        raise RuntimeError(
            "Official SAM3 image model was built without inst_interactive_predictor."
        )

    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    processor = Sam3Processor(model, device=device)

    log.info("SAM3 image predictor loaded: %s", checkpoint)
    return _SAM3Predictor(model, processor, device)


def run_sam_for_promptset(
    predictor: _SAM3Predictor,
    positive_points: List[List[int]],
    negative_points: List[List[int]],
    image_hw: Tuple[int, int],
    multimask_output: bool = True,
    target_size: Tuple[int, int] = (256, 256),
    select_by: str = "sam_score",
) -> dict:
    """Run SAM3 for one prompt set.

    Assumes predictor.set_image() has already been called.

    Returns dict with keys:
        logits_256         (target_h, target_w) float32
        sam_scores         (n_masks,) float32
        selected_branch_index int
    """
    del image_hw  # Unused for the official SAM3 predictor path.

    if select_by != "sam_score":
        raise ValueError(
            f"select_by='{select_by}' is not supported. Only 'sam_score' is implemented."
        )

    outputs = predictor.predict(positive_points, negative_points, multimask_output)

    scores = np.asarray(outputs["scores"], dtype=np.float32).reshape(-1)
    if scores.size == 0:
        raise RuntimeError("SAM3 returned no mask scores for the provided prompt set.")
    selected = int(np.argmax(scores))

    low_res_masks = torch.from_numpy(np.asarray(outputs["low_res_masks"], dtype=np.float32))
    if low_res_masks.ndim == 2:
        low_res_masks = low_res_masks.unsqueeze(0)
    if low_res_masks.ndim != 3:
        raise RuntimeError(
            "SAM3 returned low_res_masks with unexpected shape "
            f"{tuple(low_res_masks.shape)}; expected (N, H, W)."
        )

    if selected >= low_res_masks.shape[0]:
        raise RuntimeError(
            f"SAM3 selected branch {selected} but only {low_res_masks.shape[0]} masks were returned."
        )

    logits = low_res_masks[selected].unsqueeze(0).unsqueeze(0)
    th, tw = target_size
    if tuple(logits.shape[-2:]) != (th, tw):
        logits = F.interpolate(
            logits,
            size=(th, tw),
            mode="bilinear",
            align_corners=False,
        )
    logits_256 = logits[0, 0].cpu().numpy()

    return {
        "logits_256": logits_256.astype(np.float32),
        "sam_scores": scores.astype(np.float32),
        "selected_branch_index": selected,
    }


def verify_precompute_files(promptset_dir, required=None) -> bool:
    """Return True if all required files exist in promptset_dir."""
    from pathlib import Path

    if required is None:
        required = ["logits_256.npy", "pos_map.npy", "neg_map.npy", "meta.json"]
    d = Path(promptset_dir)
    return all((d / f).exists() for f in required)
