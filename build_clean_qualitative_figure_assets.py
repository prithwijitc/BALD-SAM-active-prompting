from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


FIGURES = {
    "medical": {
        "caption": (
            "Representative medical active-prompting examples on baseline SAM1. "
            "Two successful cases are followed by two failure cases. "
            "Each row shows the original image crop, the ground-truth mask, and the final-step overlay."
        ),
        "label": "fig:medical_qualitative_cases",
        "cases": [
            {
                "key": "good_busi_390",
                "subcaption": "Good case: BUSI/benign\\_benign\\_390.",
                "original": "/mnt/HD_18T_pt2/medical_data/images/BUSI/benign_benign_390.jpg",
                "mask": "/mnt/HD_18T_pt2/medical_data/masks/BUSI/benign_benign_390.npy",
                "overlay": "/mnt/HD_18T_pt2/medical_data/active_prompting/run_baseline_v3/qualitative/BUSI__benign_benign_390_best_1.png",
            },
            {
                "key": "good_busi_291",
                "subcaption": "Good case: BUSI/benign\\_benign\\_291.",
                "original": "/mnt/HD_18T_pt2/medical_data/images/BUSI/benign_benign_291.jpg",
                "mask": "/mnt/HD_18T_pt2/medical_data/masks/BUSI/benign_benign_291.npy",
                "overlay": "/mnt/HD_18T_pt2/medical_data/active_prompting/run_baseline_v3/qualitative/BUSI__benign_benign_291_best_2.png",
            },
            {
                "key": "fail_isic_0014784",
                "subcaption": "Failure case: ISIC2018/ISIC\\_0014784.",
                "original": "/mnt/HD_18T_pt2/medical_data/images/ISIC2018/ISIC_0014784.jpg",
                "mask": "/mnt/HD_18T_pt2/medical_data/masks/ISIC2018/ISIC_0014784.npy",
                "overlay": "/mnt/HD_18T_pt2/medical_data/active_prompting/run_baseline_v3/qualitative/ISIC2018__ISIC_0014784_worst_1.png",
            },
            {
                "key": "fail_busi_415",
                "subcaption": "Failure case: BUSI/benign\\_benign\\_415.",
                "original": "/mnt/HD_18T_pt2/medical_data/images/BUSI/benign_benign_415.jpg",
                "mask": "/mnt/HD_18T_pt2/medical_data/masks/BUSI/benign_benign_415.npy",
                "overlay": "/mnt/HD_18T_pt2/medical_data/active_prompting/run_baseline_v3/qualitative/BUSI__benign_benign_415_worst_5.png",
            },
        ],
    },
    "pointprompt": {
        "caption": (
            "Representative PointPrompt active-prompting examples. "
            "Two successful cases are followed by two failure cases. "
            "Each row shows the original image crop, the ground-truth mask, and the final-step overlay."
        ),
        "label": "fig:pointprompt_qualitative_cases",
        "cases": [
            {
                "key": "good_baseball_376",
                "subcaption": "Good case: Baseball\\_bat/sample\\_000376.",
                "original": "/mnt/HD_18T_pt2/prithwijit/pace_backup/pointprompt/Images/Baseball_bat/sample_000376.jpg",
                "mask": "/mnt/HD_18T_pt2/prithwijit/pace_backup/pointprompt/Masks/Baseball_bat/sample_000376.npy",
                "overlay": "/mnt/HD_18T_pt2/prithwijit/bald/active_prompting/run_v3/point_prompt/qualitative/Baseball_bat__sample_000376_best_1.png",
            },
            {
                "key": "good_stopsign_046",
                "subcaption": "Good case: Stop\\_sign/sample\\_000046.",
                "original": "/mnt/HD_18T_pt2/prithwijit/pace_backup/pointprompt/Images/Stop_sign/sample_000046.jpg",
                "mask": "/mnt/HD_18T_pt2/prithwijit/pace_backup/pointprompt/Masks/Stop_sign/sample_000046.npy",
                "overlay": "/mnt/HD_18T_pt2/prithwijit/bald/active_prompting/run_v3/point_prompt/qualitative/Stop_sign__sample_000046_best_2.png",
            },
            {
                "key": "fail_clock_204",
                "subcaption": "Failure case: Clock/sample\\_000204.",
                "original": "/mnt/HD_18T_pt2/prithwijit/pace_backup/pointprompt/Images/Clock/sample_000204.jpg",
                "mask": "/mnt/HD_18T_pt2/prithwijit/pace_backup/pointprompt/Masks/Clock/sample_000204.npy",
                "overlay": "/mnt/HD_18T_pt2/prithwijit/bald/active_prompting/run_v3/point_prompt/qualitative/Clock__sample_000204_worst_4.png",
            },
            {
                "key": "fail_bus_092",
                "subcaption": "Failure case: Bus/sample\\_000092.",
                "original": "/mnt/HD_18T_pt2/prithwijit/pace_backup/pointprompt/Images/Bus/sample_000092.jpg",
                "mask": "/mnt/HD_18T_pt2/prithwijit/pace_backup/pointprompt/Masks/Bus/sample_000092.npy",
                "overlay": "/mnt/HD_18T_pt2/prithwijit/bald/active_prompting/run_v3/point_prompt/qualitative/Bus__sample_000092_worst_1.png",
            },
        ],
    },
}


def crop_box_from_mask(mask: np.ndarray, pad_frac: float = 0.18) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        h, w = mask.shape[:2]
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


def fit_square(img: Image.Image, size: int, bg=(255, 255, 255)) -> Image.Image:
    canvas = Image.new("RGB", (size, size), bg)
    scale = min(size / img.width, size / img.height)
    w = max(1, int(img.width * scale))
    h = max(1, int(img.height * scale))
    resized = img.resize((w, h), Image.Resampling.LANCZOS)
    canvas.paste(resized, ((size - w) // 2, (size - h) // 2))
    return canvas


def crop_original(original_path: str, mask_path: str) -> tuple[Image.Image, tuple[int, int, int, int]]:
    img = Image.open(original_path).convert("RGB")
    mask = np.load(mask_path)
    box = crop_box_from_mask(mask)
    return img.crop(box), box


def render_mask(mask_path: str, box: tuple[int, int, int, int]) -> Image.Image:
    mask = np.load(mask_path)
    x0, y0, x1, y1 = box
    crop = mask[y0:y1, x0:x1] > 0
    img = np.zeros((crop.shape[0], crop.shape[1], 3), dtype=np.uint8)
    img[crop] = np.array([0, 170, 0], dtype=np.uint8)
    return Image.fromarray(img, mode="RGB")


def find_top_row_tiles(overlay_img: Image.Image) -> list[tuple[int, int, int, int]]:
    arr = np.asarray(overlay_img.convert("RGB"))
    dark = (arr[..., 0] < 20) & (arr[..., 1] < 20) & (arr[..., 2] < 20)
    row_counts = dark.sum(axis=1)
    row_mask = row_counts > arr.shape[1] * 0.15
    row_ranges = []
    start = None
    for i, flag in enumerate(row_mask):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            row_ranges.append((start, i))
            start = None
    if start is not None:
        row_ranges.append((start, len(row_mask)))
    if not row_ranges:
        raise RuntimeError("Could not detect tile rows")
    y0, y1 = row_ranges[0]
    sub = dark[y0:y1]
    col_counts = sub.sum(axis=0)
    col_mask = col_counts > (y1 - y0) * 0.4
    col_ranges = []
    start = None
    for i, flag in enumerate(col_mask):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            col_ranges.append((start, i))
            start = None
    if start is not None:
        col_ranges.append((start, len(col_mask)))
    return [(x0, y0, x1, y1) for x0, x1 in col_ranges]


def crop_nonblack(img: Image.Image, pad_frac: float = 0.2) -> Image.Image:
    arr = np.asarray(img)
    fg = np.any(arr > 20, axis=2)
    ys, xs = np.where(fg)
    if len(xs) == 0:
        return img
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    w = x1 - x0 + 1
    h = y1 - y0 + 1
    pad = int(max(w, h) * pad_frac)
    return img.crop((
        max(0, x0 - pad),
        max(0, y0 - pad),
        min(arr.shape[1], x1 + pad + 1),
        min(arr.shape[0], y1 + pad + 1),
    ))


def crop_final_overlay(overlay_path: str) -> Image.Image:
    img = Image.open(overlay_path).convert("RGB")
    tiles = find_top_row_tiles(img)
    final = img.crop(tiles[-1])
    return crop_nonblack(final)


def save_case_row(case: dict, out_path: Path, tile_size: int = 340, gap: int = 16) -> None:
    original, box = crop_original(case["original"], case["mask"])
    gt = render_mask(case["mask"], box)
    final = crop_final_overlay(case["overlay"])
    tiles = [fit_square(img, tile_size) for img in [original, gt, final]]
    row = Image.new("RGB", (tile_size * 3 + gap * 2, tile_size), (255, 255, 255))
    x = 0
    for tile in tiles:
        row.paste(tile, (x, 0))
        x += tile_size + gap
    out_path.parent.mkdir(parents=True, exist_ok=True)
    row.save(out_path)


def save_panel(case_paths: list[Path], out_path: Path, gap_y: int = 18) -> None:
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


def save_tex_input(kind: str, raw_dir: Path, out_tex: Path) -> None:
    spec = FIGURES[kind]
    lines = [
        r"\begin{figure*}[!t]",
        r"\centering",
    ]
    for case in spec["cases"]:
        fname = f"{kind}_{case['key']}.png"
        lines.extend([
            rf"\begin{{subfigure}}{{\textwidth}}",
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
    out_tex.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/mnt/HD_18T_pt2/medical_data/active_prompting/sam_story/qualitative_clean"),
    )
    args = parser.parse_args()

    for kind, spec in FIGURES.items():
        figure_dir = args.out_dir / kind
        case_paths = []
        for case in spec["cases"]:
            out_path = figure_dir / f"{kind}_{case['key']}.png"
            save_case_row(case, out_path)
            case_paths.append(out_path)
        save_panel(case_paths, figure_dir / f"{kind}_panel.png")
        save_tex_input(kind, figure_dir, figure_dir / f"{kind}_figure.tex")


if __name__ == "__main__":
    main()
