from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F_torch
import yaml
from PIL import Image, ImageDraw

from modules.head import NoImageHead
from modules.laplace_wrapper import load_laplace
from modules.promptset import build_gaussian_map
from modules.sam_wrapper import load_sam_predictor, run_sam_for_promptset


FIGURES = {
    "medical": {
        "config": "/mnt/HD_18T_pt2/medical_data/active_configs/config_baseline_v3.yaml",
        "manifest": "/mnt/HD_18T_pt2/medical_data/manifests/test.csv",
        "step_results": "/mnt/HD_18T_pt2/medical_data/active_prompting/run_baseline_v3/step_results.csv",
        "caption": (
            "Representative medical active-prompting examples on baseline SAM1. "
            "Two successful cases and two failure cases are shown. "
            "Each row contains the original image crop, the ground-truth mask, the final BALD-Laplace prediction, "
            "and the final prompt set rendered in the same image space."
        ),
        "label": "fig:medical_qualitative_reconstructed",
        "cases": [
            {"key": "good_busi_390", "sample_id": "BUSI/benign_benign_390", "subcaption": "Good case: BUSI/benign\\_benign\\_390."},
            {"key": "good_kvasir_cju2hewssldzx0835ep795xu0", "sample_id": "KvasirSEG/cju2hewssldzx0835ep795xu0", "subcaption": "Good case: KvasirSEG/cju2hewssldzx0835ep795xu0."},
            {"key": "fail_isic_0014784", "sample_id": "ISIC2018/ISIC_0014784", "subcaption": "Failure case: ISIC2018/ISIC\\_0014784."},
            {"key": "fail_chexmask_jpcln113", "sample_id": "CheXmask/JPCLN113", "subcaption": "Failure case: CheXmask/JPCLN113."},
        ],
    },
    "pointprompt": {
        "config": "/mnt/HD_18T_pt2/prithwijit/bald/checkpoints/no_image/run_v3/config.yaml",
        "manifest": "/mnt/HD_18T_pt2/prithwijit/bald/manifests/test.csv",
        "step_results": "/mnt/HD_18T_pt2/prithwijit/bald/active_prompting/run_v3/point_prompt/step_results.csv",
        "caption": (
            "Representative PointPrompt active-prompting examples. "
            "Two successful cases and two failure cases are shown. "
            "Each row contains the original image crop, the ground-truth mask, the final BALD-Laplace prediction, "
            "and the final prompt set rendered in the same image space."
        ),
        "label": "fig:pointprompt_qualitative_reconstructed",
        "cases": [
            {"key": "good_baseball_376", "sample_id": "Baseball_bat/sample_000376", "subcaption": "Good case: Baseball\\_bat/sample\\_000376."},
            {"key": "good_stopsign_046", "sample_id": "Stop_sign/sample_000046", "subcaption": "Good case: Stop\\_sign/sample\\_000046."},
            {"key": "fail_clock_204", "sample_id": "Clock/sample_000204", "subcaption": "Failure case: Clock/sample\\_000204."},
            {"key": "fail_bus_092", "sample_id": "Bus/sample_000092", "subcaption": "Failure case: Bus/sample\\_000092."},
        ],
    },
}


def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def scale_points_to_original(points_256, H_orig, W_orig, H_target=256, W_target=256):
    sx = W_orig / W_target
    sy = H_orig / H_target
    return [[p[0] * sx, p[1] * sy] for p in points_256]


def load_gt(mask_path: str, image_h: int, image_w: int) -> np.ndarray:
    gt = (np.load(mask_path) > 0).astype(np.float32)
    if gt.shape != (image_h, image_w):
        t = torch.from_numpy(gt).unsqueeze(0).unsqueeze(0)
        t = F_torch.interpolate(t, size=(image_h, image_w), mode="nearest")
        gt = t.squeeze().numpy()
    return gt > 0


def fit_square(img: Image.Image, size: int, bg=(255, 255, 255)) -> Image.Image:
    canvas = Image.new("RGB", (size, size), bg)
    scale = min(size / img.width, size / img.height)
    w = max(1, int(img.width * scale))
    h = max(1, int(img.height * scale))
    resized = img.resize((w, h), Image.Resampling.LANCZOS)
    canvas.paste(resized, ((size - w) // 2, (size - h) // 2))
    return canvas


def mask_bbox(mask: np.ndarray, pad_frac: float = 0.15) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        h, w = mask.shape
        return 0, 0, w, h
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    w = x1 - x0 + 1
    h = y1 - y0 + 1
    pad = int(max(w, h) * pad_frac)
    return (
        max(0, x0 - pad),
        max(0, y0 - pad),
        min(mask.shape[1], x1 + pad + 1),
        min(mask.shape[0], y1 + pad + 1),
    )


def union_bbox(a: np.ndarray, b: np.ndarray, pad_frac: float = 0.15) -> Tuple[int, int, int, int]:
    return mask_bbox(a | b, pad_frac=pad_frac)


def render_gt(mask: np.ndarray) -> Image.Image:
    arr = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    arr[mask] = np.array([0, 180, 0], dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def render_prediction(pred: np.ndarray) -> Image.Image:
    arr = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
    arr[pred] = np.array([50, 120, 255], dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def render_prompts(image: np.ndarray, pos_orig, neg_orig, crop_box, radius: int = 5) -> Image.Image:
    x0, y0, x1, y1 = crop_box
    canvas = Image.fromarray(image[y0:y1, x0:x1], mode="RGB")
    draw = ImageDraw.Draw(canvas)

    def draw_point(pt, color):
        x, y = pt
        cx = x - x0
        cy = y - y0
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=color, outline=(255, 255, 255), width=1)

    for pt in pos_orig:
        draw_point(pt, (0, 220, 0))
    for pt in neg_orig:
        draw_point(pt, (255, 50, 50))
    return canvas


class DomainRuntime:
    def __init__(self, config_path: str, device: str):
        self.cfg = load_yaml(config_path)
        self.device = device
        data_cfg = self.cfg.get("data", {})
        project_root = Path(data_cfg["project_root"])
        head_ckpt = Path(
            data_cfg.get(
                "head_checkpoint",
                project_root / "checkpoints" / "no_image" / "run_v3" / "best.pt",
            )
        )
        laplace_dir = Path(
            data_cfg.get(
                "laplace_dir",
                project_root / "laplace" / "no_image" / "run_v3",
            )
        )

        self.head = NoImageHead(
            channels_in=self.cfg["head"].get("channels_in", 3),
            hidden_channels=self.cfg["head"].get("hidden_channels", 16),
        ).to(device)
        self.head.load_state_dict(torch.load(head_ckpt, map_location="cpu"))
        self.head.eval()
        self.la, _ = load_laplace(self.head, laplace_dir, device)
        self.predictor = load_sam_predictor(
            self.cfg["sam"]["checkpoint"],
            self.cfg["sam"]["model_type"],
            device,
        )
        self.sigma = float(self.cfg["prompt_generation"].get("gaussian_sigma", 8))


def reconstruct_prediction(runtime: DomainRuntime, image: np.ndarray, image_h: int, image_w: int, pos256, neg256) -> np.ndarray:
    pos_orig = scale_points_to_original(pos256, image_h, image_w)
    neg_orig = scale_points_to_original(neg256, image_h, image_w) if neg256 else []
    runtime.predictor.set_image(image)
    sam_out = run_sam_for_promptset(
        runtime.predictor,
        pos_orig,
        neg_orig,
        (image_h, image_w),
        multimask_output=True,
        target_size=(256, 256),
    )
    logits_256 = sam_out["logits_256"]
    pos_map = build_gaussian_map(pos256, 256, 256, runtime.sigma)
    neg_map = build_gaussian_map(neg256, 256, 256, runtime.sigma) if neg256 else np.zeros((256, 256), dtype=np.float32)
    x_input = torch.from_numpy(np.stack([np.clip(logits_256, -20, 20), pos_map, neg_map], axis=0)).float().unsqueeze(0).to(runtime.device)
    with torch.no_grad():
        feat = runtime.la._backbone(x_input)
        map_logits = runtime.head.final_conv(feat)
        probs = runtime.la._probit_predictive(feat, map_logits).squeeze().cpu().numpy()
    probs_t = torch.from_numpy(probs).unsqueeze(0).unsqueeze(0)
    probs_orig = F_torch.interpolate(probs_t, size=(image_h, image_w), mode="bilinear", align_corners=False).squeeze().numpy()
    return probs_orig >= 0.5


def save_case_row(runtime: DomainRuntime, manifest: pd.DataFrame, step_df: pd.DataFrame, case: dict, out_path: Path, tile_size: int = 340, gap: int = 16):
    sid = case["sample_id"]
    mrow = manifest[manifest["sample_id"] == sid].iloc[0]
    image_path = str(mrow["image_path"])
    mask_path = str(mrow["mask_path"])
    image = np.array(Image.open(image_path).convert("RGB"))
    H, W = image.shape[:2]
    gt = load_gt(mask_path, H, W)

    cdf = step_df[(step_df["sample_id"] == sid) & (step_df["strategy"] == "bald_laplace")].sort_values(["seed", "step"])
    seed = int(cdf.iloc[0]["seed"])
    cdf = cdf[cdf["seed"] == seed].sort_values("step")
    pos256 = cdf[cdf["point_label"] == 1][["point_x", "point_y"]].to_numpy(dtype=float).tolist()
    neg256 = cdf[cdf["point_label"] == 0][["point_x", "point_y"]].to_numpy(dtype=float).tolist()

    pos_orig = scale_points_to_original(pos256, H, W)
    neg_orig = scale_points_to_original(neg256, H, W) if neg256 else []
    pred = reconstruct_prediction(runtime, image, H, W, pos256, neg256)
    box = union_bbox(gt, pred, pad_frac=0.18)
    x0, y0, x1, y1 = box
    orig_crop = Image.fromarray(image[y0:y1, x0:x1], mode="RGB")
    gt_crop = render_gt(gt[y0:y1, x0:x1])
    ov_crop = render_prediction(pred[y0:y1, x0:x1])
    prompts_crop = render_prompts(image, pos_orig, neg_orig, box)

    tiles = [fit_square(img, tile_size) for img in [orig_crop, gt_crop, ov_crop, prompts_crop]]
    row = Image.new("RGB", (tile_size * 4 + gap * 3, tile_size), (255, 255, 255))
    x = 0
    for tile in tiles:
        row.paste(tile, (x, 0))
        x += tile_size + gap
    out_path.parent.mkdir(parents=True, exist_ok=True)
    row.save(out_path)


def save_panel(case_paths, out_path: Path, gap_y: int = 18):
    imgs = [Image.open(p).convert("RGB") for p in case_paths]
    width = max(img.width for img in imgs)
    total_h = sum(img.height for img in imgs) + gap_y * (len(imgs) - 1)
    panel = Image.new("RGB", (width, total_h), (255, 255, 255))
    y = 0
    for img in imgs:
        panel.paste(img, ((width - img.width) // 2, y))
        y += img.height + gap_y
    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(out_path)


def save_tex_input(kind: str, figure_dir: Path):
    spec = FIGURES[kind]
    lines = [r"\begin{figure*}[!t]", r"\centering"]
    for case in spec["cases"]:
        fname = f"{kind}_{case['key']}.png"
        lines.extend([
            r"\begin{subfigure}{\textwidth}",
            r"\centering",
            rf"\includegraphics[width=\textwidth]{{{fname}}}",
            rf"\caption{{{case['subcaption']}}}",
            r"\end{subfigure}",
            r"\vspace{0.5em}",
        ])
    lines = lines[:-1]
    lines.extend([
        rf"\caption{{{spec['caption']}}}",
        rf"\label{{{spec['label']}}}",
        r"\end{figure*}",
    ])
    (figure_dir / f"{kind}_figure.tex").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("/mnt/HD_18T_pt2/medical_data/active_prompting/sam_story/qualitative_reconstructed"))
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    for kind, spec in FIGURES.items():
        runtime = DomainRuntime(spec["config"], args.device)
        manifest = pd.read_csv(spec["manifest"])
        step_df = pd.read_csv(spec["step_results"], usecols=["sample_id", "seed", "strategy", "step", "point_x", "point_y", "point_label"])
        figure_dir = args.out_dir / kind
        case_paths = []
        for case in spec["cases"]:
            out_path = figure_dir / f"{kind}_{case['key']}.png"
            save_case_row(runtime, manifest, step_df, case, out_path)
            case_paths.append(out_path)
        save_panel(case_paths, figure_dir / f"{kind}_panel.png")
        save_tex_input(kind, figure_dir)


if __name__ == "__main__":
    main()
