from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont


PANEL_SPECS = {
    "medical": {
        "title": "Medical active prompting qualitative cases",
        "cases": [
            (
                "/mnt/HD_18T_pt2/medical_data/images/BUSI/benign_benign_390.jpg",
                "/mnt/HD_18T_pt2/medical_data/active_prompting/run_baseline_v3/qualitative/BUSI__benign_benign_390_best_1.png",
                "Good case 1: BUSI/benign_benign_390 (stable high IoU throughout prompting)",
            ),
            (
                "/mnt/HD_18T_pt2/medical_data/images/BUSI/benign_benign_291.jpg",
                "/mnt/HD_18T_pt2/medical_data/active_prompting/run_baseline_v3/qualitative/BUSI__benign_benign_291_best_2.png",
                "Good case 2: BUSI/benign_benign_291 (early gain and sustained refinement)",
            ),
            (
                "/mnt/HD_18T_pt2/medical_data/images/ISIC2018/ISIC_0014784.jpg",
                "/mnt/HD_18T_pt2/medical_data/active_prompting/run_baseline_v3/qualitative/ISIC2018__ISIC_0014784_worst_1.png",
                "Failure case: ISIC2018/ISIC_0014784 (prompts fail to recover from poor initial mask)",
            ),
        ],
    },
    "pointprompt": {
        "title": "PointPrompt active prompting qualitative cases",
        "cases": [
            (
                "/mnt/HD_18T_pt2/prithwijit/pace_backup/pointprompt/Images/Baseball_bat/sample_000376.jpg",
                "/mnt/HD_18T_pt2/prithwijit/bald/active_prompting/run_v3/point_prompt/qualitative/Baseball_bat__sample_000376_best_1.png",
                "Good case 1: Baseball_bat/sample_000376 (strong boundary correction after a few prompts)",
            ),
            (
                "/mnt/HD_18T_pt2/prithwijit/pace_backup/pointprompt/Images/Stop_sign/sample_000046.jpg",
                "/mnt/HD_18T_pt2/prithwijit/bald/active_prompting/run_v3/point_prompt/qualitative/Stop_sign__sample_000046_best_2.png",
                "Good case 2: Stop_sign/sample_000046 (consistent refinement with sparse prompts)",
            ),
            (
                "/mnt/HD_18T_pt2/prithwijit/pace_backup/pointprompt/Images/Clock/sample_000204.jpg",
                "/mnt/HD_18T_pt2/prithwijit/bald/active_prompting/run_v3/point_prompt/qualitative/Clock__sample_000204_worst_4.png",
                "Failure case: Clock/sample_000204 (late-budget degradation after initially strong overlap)",
            ),
        ],
    },
}


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for font_name in ["DejaVuSans.ttf", "Arial.ttf", "LiberationSans-Regular.ttf"]:
        try:
            return ImageFont.truetype(font_name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def ensure_rgb(img: Image.Image) -> Image.Image:
    if img.mode != "RGB":
        return img.convert("RGB")
    return img


def compose_panel(title: str, cases: Iterable[tuple[str, str, str]], out_path: Path, target_width: int = 1800) -> None:
    title_font = load_font(36)
    caption_font = load_font(26)
    body_font = load_font(22)
    margin = 36
    gap = 28
    text_gap = 14
    pair_gap = 20
    bg = (255, 255, 255)
    fg = (20, 20, 20)

    loaded = []
    panel_inner_width = target_width
    single_width = (panel_inner_width - pair_gap) // 2
    for original_path, overlay_path, caption in cases:
        original = ensure_rgb(Image.open(original_path))
        overlay = ensure_rgb(Image.open(overlay_path))
        original_scale = single_width / original.width
        overlay_scale = single_width / overlay.width
        original_resized = original.resize(
            (single_width, int(original.height * original_scale)), Image.Resampling.LANCZOS
        )
        overlay_resized = overlay.resize(
            (single_width, int(overlay.height * overlay_scale)), Image.Resampling.LANCZOS
        )
        row_height = max(original_resized.height, overlay_resized.height)
        loaded.append(
            (
                original_resized,
                overlay_resized,
                row_height,
                caption,
                Path(original_path).name,
                Path(overlay_path).name,
            )
        )

    tmp = Image.new("RGB", (target_width + margin * 2, 1000), bg)
    draw = ImageDraw.Draw(tmp)
    title_h = int(draw.textbbox((0, 0), title, font=title_font)[3]) + 8

    case_heights = []
    for _, _, row_height, caption, original_name, overlay_name in loaded:
        cap_h = int(draw.textbbox((0, 0), caption, font=caption_font)[3]) + 6
        file_h = max(
            int(draw.textbbox((0, 0), original_name, font=body_font)[3]),
            int(draw.textbbox((0, 0), overlay_name, font=body_font)[3]),
        ) + 4
        case_heights.append(cap_h + text_gap + file_h + text_gap + row_height)

    total_h = margin + title_h + gap
    for text_h in case_heights:
        total_h += text_h + gap
    total_h -= gap
    total_h += margin

    canvas = Image.new("RGB", (target_width + margin * 2, total_h), bg)
    draw = ImageDraw.Draw(canvas)

    y = margin
    draw.text((margin, y), title, font=title_font, fill=fg)
    y += title_h + gap

    for (original, overlay, row_height, caption, original_name, overlay_name), text_h in zip(loaded, case_heights):
        draw.text((margin, y), caption, font=caption_font, fill=fg)
        y += int(draw.textbbox((0, 0), caption, font=caption_font)[3]) + text_gap
        draw.text((margin, y), f"Original: {original_name}", font=body_font, fill=(90, 90, 90))
        draw.text((margin + single_width + pair_gap, y), f"Overlay: {overlay_name}", font=body_font, fill=(90, 90, 90))
        y += int(draw.textbbox((0, 0), original_name, font=body_font)[3]) + text_gap
        canvas.paste(original, (margin, y))
        canvas.paste(overlay, (margin + single_width + pair_gap, y))
        y += row_height + gap

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

    for key, spec in PANEL_SPECS.items():
        compose_panel(
            title=spec["title"],
            cases=spec["cases"],
            out_path=args.out_dir / f"{key}_qualitative_cases.png",
        )


if __name__ == "__main__":
    main()
