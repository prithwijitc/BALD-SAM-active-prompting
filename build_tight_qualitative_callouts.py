from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


CASES = {
    "medical": {
        "title": "Medical qualitative callouts",
        "cases": [
            {
                "label": "Good case 1",
                "name": "BUSI/benign_benign_390",
                "original": "/mnt/HD_18T_pt2/medical_data/images/BUSI/benign_benign_390.jpg",
                "mask": "/mnt/HD_18T_pt2/medical_data/masks/BUSI/benign_benign_390.npy",
                "overlay": "/mnt/HD_18T_pt2/medical_data/active_prompting/run_baseline_v3/qualitative/BUSI__benign_benign_390_best_1.png",
                "note": "Stable high-IoU refinement",
            },
            {
                "label": "Good case 2",
                "name": "BUSI/benign_benign_291",
                "original": "/mnt/HD_18T_pt2/medical_data/images/BUSI/benign_benign_291.jpg",
                "mask": "/mnt/HD_18T_pt2/medical_data/masks/BUSI/benign_benign_291.npy",
                "overlay": "/mnt/HD_18T_pt2/medical_data/active_prompting/run_baseline_v3/qualitative/BUSI__benign_benign_291_best_2.png",
                "note": "Early gain with sustained refinement",
            },
            {
                "label": "Failure case",
                "name": "ISIC2018/ISIC_0014784",
                "original": "/mnt/HD_18T_pt2/medical_data/images/ISIC2018/ISIC_0014784.jpg",
                "mask": "/mnt/HD_18T_pt2/medical_data/masks/ISIC2018/ISIC_0014784.npy",
                "overlay": "/mnt/HD_18T_pt2/medical_data/active_prompting/run_baseline_v3/qualitative/ISIC2018__ISIC_0014784_worst_1.png",
                "note": "Prompts fail to recover from a poor initial mask",
            },
        ],
    },
    "pointprompt": {
        "title": "PointPrompt qualitative callouts",
        "cases": [
            {
                "label": "Good case 1",
                "name": "Baseball_bat/sample_000376",
                "original": "/mnt/HD_18T_pt2/prithwijit/pace_backup/pointprompt/Images/Baseball_bat/sample_000376.jpg",
                "mask": "/mnt/HD_18T_pt2/prithwijit/pace_backup/pointprompt/Masks/Baseball_bat/sample_000376.npy",
                "overlay": "/mnt/HD_18T_pt2/prithwijit/bald/active_prompting/run_v3/point_prompt/qualitative/Baseball_bat__sample_000376_best_1.png",
                "note": "Strong boundary correction after a few prompts",
            },
            {
                "label": "Good case 2",
                "name": "Stop_sign/sample_000046",
                "original": "/mnt/HD_18T_pt2/prithwijit/pace_backup/pointprompt/Images/Stop_sign/sample_000046.jpg",
                "mask": "/mnt/HD_18T_pt2/prithwijit/pace_backup/pointprompt/Masks/Stop_sign/sample_000046.npy",
                "overlay": "/mnt/HD_18T_pt2/prithwijit/bald/active_prompting/run_v3/point_prompt/qualitative/Stop_sign__sample_000046_best_2.png",
                "note": "Consistent refinement with sparse prompts",
            },
            {
                "label": "Failure case",
                "name": "Clock/sample_000204",
                "original": "/mnt/HD_18T_pt2/prithwijit/pace_backup/pointprompt/Images/Clock/sample_000204.jpg",
                "mask": "/mnt/HD_18T_pt2/prithwijit/pace_backup/pointprompt/Masks/Clock/sample_000204.npy",
                "overlay": "/mnt/HD_18T_pt2/prithwijit/bald/active_prompting/run_v3/point_prompt/qualitative/Clock__sample_000204_worst_4.png",
                "note": "Late-budget degradation after a strong start",
            },
        ],
    },
}


def load_font(size: int):
    for name in ["DejaVuSans.ttf", "Arial.ttf", "LiberationSans-Regular.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def bbox_from_mask(mask: np.ndarray, pad_frac: float = 0.18) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        h, w = mask.shape[:2]
        return 0, 0, w, h
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    w = x1 - x0 + 1
    h = y1 - y0 + 1
    pad = int(max(w, h) * pad_frac)
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(mask.shape[1], x1 + pad + 1)
    y1 = min(mask.shape[0], y1 + pad + 1)
    return x0, y0, x1, y1


def crop_original(original_path: str, mask_path: str) -> Image.Image:
    img = Image.open(original_path).convert("RGB")
    mask = np.load(mask_path)
    x0, y0, x1, y1 = bbox_from_mask(mask)
    return img.crop((x0, y0, x1, y1))


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
        raise RuntimeError("Could not detect top row tiles")
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
    boxes = [(x0, y0, x1, y1) for x0, x1 in col_ranges]
    if len(boxes) < 2:
        raise RuntimeError("Could not detect enough tiles")
    return boxes


def crop_overlay_tiles(overlay_path: str) -> tuple[Image.Image, Image.Image]:
    img = Image.open(overlay_path).convert("RGB")
    tiles = find_top_row_tiles(img)
    step0 = img.crop(tiles[0])
    final = img.crop(tiles[-1])
    return crop_nonblack(step0), crop_nonblack(final)


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
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(arr.shape[1], x1 + pad + 1)
    y1 = min(arr.shape[0], y1 + pad + 1)
    return img.crop((x0, y0, x1, y1))


def fit_square(img: Image.Image, size: int, bg=(255, 255, 255)) -> Image.Image:
    canvas = Image.new("RGB", (size, size), bg)
    scale = min(size / img.width, size / img.height)
    w = max(1, int(img.width * scale))
    h = max(1, int(img.height * scale))
    resized = img.resize((w, h), Image.Resampling.LANCZOS)
    x = (size - w) // 2
    y = (size - h) // 2
    canvas.paste(resized, (x, y))
    return canvas


def build_panel(kind: str, out_path: Path) -> None:
    spec = CASES[kind]
    title_font = load_font(40)
    header_font = load_font(26)
    body_font = load_font(22)
    small_font = load_font(20)
    margin = 36
    gap_x = 18
    gap_y = 26
    tile_size = 360
    bg = (255, 255, 255)
    fg = (20, 20, 20)
    muted = (90, 90, 90)

    rows = []
    for case in spec["cases"]:
        original = fit_square(crop_original(case["original"], case["mask"]), tile_size)
        step0, final = crop_overlay_tiles(case["overlay"])
        step0 = fit_square(step0, tile_size)
        final = fit_square(final, tile_size)
        rows.append((case, original, step0, final))

    width = margin * 2 + tile_size * 3 + gap_x * 2
    row_header_h = 58
    subheader_h = 28
    title_h = 56
    height = margin + title_h + gap_y
    height += len(rows) * (row_header_h + subheader_h + tile_size + gap_y) - gap_y
    height += margin

    canvas = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(canvas)
    y = margin
    draw.text((margin, y), spec["title"], font=title_font, fill=fg)
    y += title_h + gap_y

    col_titles = ["Original image crop", "Step 0 overlay", "Final-step overlay"]
    for case, original, step0, final in rows:
        draw.text((margin, y), f"{case['label']}: {case['name']}", font=header_font, fill=fg)
        draw.text((margin, y + 30), case["note"], font=small_font, fill=muted)
        y += row_header_h
        x = margin
        for title, panel in zip(col_titles, [original, step0, final]):
            draw.text((x, y), title, font=body_font, fill=fg)
            canvas.paste(panel, (x, y + subheader_h))
            x += tile_size + gap_x
        y += subheader_h + tile_size + gap_y

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/mnt/HD_18T_pt2/medical_data/active_prompting/sam_story/qualitative_cases"),
    )
    args = parser.parse_args()
    for kind in CASES:
        build_panel(kind, args.out_dir / f"{kind}_qualitative_callouts.png")


if __name__ == "__main__":
    main()
