from __future__ import annotations

import re
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


PROMPTING_ROOT = Path(os.environ.get("POINTPROMPT_RESULTS_ROOT", "/mnt/HD_18T_pt2/prithwijit/pace_backup/pointprompt/Prompting results")).expanduser()
MANIFEST_PATH = Path(os.environ.get("POINTPROMPT_MANIFEST", "/mnt/HD_18T_pt2/prithwijit/bald/manifests/test.csv")).expanduser()
BALD_RUN_DIR = Path(os.environ.get("POINTPROMPT_BALD_RUN", "/mnt/HD_18T_pt2/prithwijit/bald/active_prompting/run_v3/point_prompt")).expanduser()
OUT_DIR = Path(os.environ.get("POINTPROMPT_OUT_DIR", "/mnt/HD_18T_pt2/medical_data/IEEE/point_prompt_human_compare")).expanduser()
MAX_MATCHED_STEP = 19

MASK_RE = re.compile(r"^(\d+)_(\d+)_mask\.png$")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def compute_iou_and_dice(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    union = tp + fp + fn
    iou = float(tp / union) if union > 0 else 1.0
    denom = 2 * tp + fp + fn
    dice = float((2 * tp) / denom) if denom > 0 else 1.0
    return iou, dice


def load_gt(mask_path: str) -> np.ndarray:
    return np.load(mask_path) > 0


def load_binary_png(path: Path, target_shape: tuple[int, int]) -> tuple[np.ndarray, bool]:
    arr = np.array(Image.open(path).convert("L")) > 127
    resized = False
    if arr.shape != target_shape:
        arr = np.array(Image.fromarray(arr.astype(np.uint8) * 255).resize((target_shape[1], target_shape[0]), Image.Resampling.NEAREST)) > 127
        resized = True
    return arr, resized


def build_human_trajectory_csv(manifest: pd.DataFrame) -> Path:
    common_groups = sorted(set(manifest["group"].unique()) & {p.name for p in PROMPTING_ROOT.iterdir() if p.is_dir()})
    manifest_by_id = manifest.set_index("sample_id")
    rows: list[dict] = []

    for group in common_groups:
        group_dir = PROMPTING_ROOT / group
        prompters = sorted([p for p in group_dir.iterdir() if p.is_dir() and p.name.startswith("st")])
        for prompter_dir in prompters:
            mask_dir = prompter_dir / "masks"
            for mask_file in sorted(mask_dir.iterdir()):
                match = MASK_RE.match(mask_file.name)
                if not match:
                    continue
                image_index = int(match.group(1))
                step = int(match.group(2))
                sample_id = f"{group}/sample_{image_index:06d}"
                if sample_id not in manifest_by_id.index:
                    continue
                mrow = manifest_by_id.loc[sample_id]
                gt = load_gt(mrow["mask_path"])
                pred, resized = load_binary_png(mask_file, gt.shape)
                iou, dice = compute_iou_and_dice(pred, gt)
                rows.append(
                    {
                        "group": group,
                        "prompter": prompter_dir.name,
                        "sample_id": sample_id,
                        "image_index": image_index,
                        "step": step,
                        "prompt_count": step + 1,
                        "iou": iou,
                        "dice": dice,
                        "gt_h": gt.shape[0],
                        "gt_w": gt.shape[1],
                        "pred_h": pred.shape[0],
                        "pred_w": pred.shape[1],
                        "was_resized": int(resized),
                        "mask_path": str(mask_file),
                        "gt_path": str(mrow["mask_path"]),
                    }
                )

    df = pd.DataFrame(rows).sort_values(["group", "prompter", "sample_id", "step"]).reset_index(drop=True)
    ensure_dir(OUT_DIR)
    out_path = OUT_DIR / "human_pointprompt_trajectory.csv"
    df.to_csv(out_path, index=False)
    return out_path


def summarize_from_curve(
    df: pd.DataFrame,
    value_col: str,
    group: str,
    strategy: str,
    strategy_label: str,
    full_step_override: int | None = None,
) -> dict:
    curve = df.sort_values("step")
    full_step_used = int(min(MAX_MATCHED_STEP, curve["step"].max()) if full_step_override is None else full_step_override)
    step0 = float(curve.loc[curve["step"] == 0, value_col].iloc[0])
    step2 = float(curve.loc[curve["step"] == 2, value_col].iloc[0]) if (curve["step"] == 2).any() else np.nan
    step5 = float(curve.loc[curve["step"] == 5, value_col].iloc[0]) if (curve["step"] == 5).any() else np.nan
    step10 = float(curve.loc[curve["step"] == 10, value_col].iloc[0]) if (curve["step"] == 10).any() else np.nan
    step15 = float(curve.loc[curve["step"] == 15, value_col].iloc[0]) if (curve["step"] == 15).any() else np.nan
    full_row = curve.loc[curve["step"] == full_step_used]
    if full_row.empty:
        raise ValueError(f"Missing step {full_step_used} for {group} {strategy} {value_col}")
    step_full = float(full_row[value_col].iloc[0])
    improvement = step_full - step0
    auc = float(np.trapz(curve[value_col].to_numpy(), curve["step"].to_numpy()))
    mean = float(curve[value_col].mean())
    return {
        "metric": "IoU" if value_col == "iou_mean" else "Dice",
        "direction": "higher_is_better",
        "group": group,
        "strategy": strategy,
        "strategy_label": strategy_label,
        "step_0": step0,
        "step_2": step2,
        "step_5": step5,
        "step_10": step10,
        "step_15": step15,
        "step_full_used": full_step_used,
        "step_full": step_full,
        "raw_change_0_to_full": step_full - step0,
        "improvement_0_to_full": improvement,
        "budget_auc": auc,
        "budget_mean": mean,
    }


def build_human_summary(human_csv: Path) -> tuple[Path, Path, dict[str, int]]:
    human = pd.read_csv(human_csv)
    human = human[human["step"] <= MAX_MATCHED_STEP].copy()

    by_group_curve = (
        human.groupby(["group", "step"], as_index=False)[["iou", "dice"]]
        .mean()
        .rename(columns={"iou": "iou_mean", "dice": "dice_mean"})
    )
    overall_curve = (
        human.groupby(["step"], as_index=False)[["iou", "dice"]]
        .mean()
        .rename(columns={"iou": "iou_mean", "dice": "dice_mean"})
    )

    rows = []
    full_steps_by_group: dict[str, int] = {}
    for group, gdf in by_group_curve.groupby("group"):
        full_steps_by_group[group] = int(min(MAX_MATCHED_STEP, gdf["step"].max()))
        rows.append(summarize_from_curve(gdf, "iou_mean", group, "human", "Human", full_step_override=full_steps_by_group[group]))
        rows.append(summarize_from_curve(gdf, "dice_mean", group, "human", "Human", full_step_override=full_steps_by_group[group]))
    full_steps_by_group["Overall"] = int(min(MAX_MATCHED_STEP, overall_curve["step"].max()))
    rows.append(summarize_from_curve(overall_curve, "iou_mean", "Overall", "human", "Human", full_step_override=full_steps_by_group["Overall"]))
    rows.append(summarize_from_curve(overall_curve, "dice_mean", "Overall", "human", "Human", full_step_override=full_steps_by_group["Overall"]))

    out_dir = OUT_DIR
    by_dataset_path = out_dir / "human_curve_summary_by_dataset.csv"
    overall_path = out_dir / "human_curve_summary.csv"
    df = pd.DataFrame(rows)
    df[df["group"] != "Overall"].to_csv(by_dataset_path, index=False)
    df[df["group"] == "Overall"].to_csv(overall_path, index=False)
    return overall_path, by_dataset_path, full_steps_by_group


def build_bald_summary(full_steps_by_group: dict[str, int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    agg = pd.read_csv(BALD_RUN_DIR / "aggregate.csv")
    per_group = pd.read_csv(BALD_RUN_DIR / "per_group.csv")

    agg = agg[(agg["strategy"] == "bald_laplace") & (agg["step"] <= MAX_MATCHED_STEP)]
    per_group = per_group[(per_group["strategy"] == "bald_laplace") & (per_group["step"] <= MAX_MATCHED_STEP)]

    rows = []
    for group, gdf in per_group.groupby("group"):
        full_step = full_steps_by_group.get(group, MAX_MATCHED_STEP)
        rows.append(summarize_from_curve(gdf[["step", "iou_mean"]], "iou_mean", group, "bald_laplace", "BALD (Laplace)", full_step_override=full_step))
        rows.append(summarize_from_curve(gdf[["step", "dice_mean"]], "dice_mean", group, "bald_laplace", "BALD (Laplace)", full_step_override=full_step))
    overall_full_step = full_steps_by_group.get("Overall", MAX_MATCHED_STEP)
    rows.append(summarize_from_curve(agg[["step", "iou_mean"]], "iou_mean", "Overall", "bald_laplace", "BALD (Laplace)", full_step_override=overall_full_step))
    rows.append(summarize_from_curve(agg[["step", "dice_mean"]], "dice_mean", "Overall", "bald_laplace", "BALD (Laplace)", full_step_override=overall_full_step))

    df = pd.DataFrame(rows)
    return df[df["group"] == "Overall"].copy(), df[df["group"] != "Overall"].copy()


def build_human_vs_bald_table(human_by_dataset_path: Path, human_overall_path: Path, full_steps_by_group: dict[str, int]) -> tuple[Path, Path]:
    human_by = pd.read_csv(human_by_dataset_path)
    human_over = pd.read_csv(human_overall_path)
    bald_over, bald_by = build_bald_summary(full_steps_by_group)

    human = pd.concat([human_by, human_over], ignore_index=True)
    bald = pd.concat([bald_by, bald_over], ignore_index=True)

    merged = human.merge(
        bald,
        on=["group", "metric"],
        suffixes=("_human", "_bald"),
        how="inner",
    )

    rows = []
    for _, r in merged.iterrows():
        rows.append(
            {
                "group": r["group"],
                "metric": r["metric"],
                "full_step_used": r["step_full_used_human"],
                "human_improvement": r["improvement_0_to_full_human"],
                "bald_laplace_improvement": r["improvement_0_to_full_bald"],
                "human_auc": r["budget_auc_human"],
                "bald_laplace_auc": r["budget_auc_bald"],
                "human_mean": r["budget_mean_human"],
                "bald_laplace_mean": r["budget_mean_bald"],
            }
        )
    compare = pd.DataFrame(rows).sort_values(["group", "metric"]).reset_index(drop=True)

    out_csv = OUT_DIR / "human_vs_bald_laplace_iou_dice_summary.csv"
    compare.to_csv(out_csv, index=False)

    wide_rows = []
    for group in list(compare["group"].drop_duplicates()):
        g = compare[compare["group"] == group].set_index("metric")
        wide_rows.append(
            {
                "Dataset": group,
                "Full Step Used": g.loc["IoU", "full_step_used"],
                "IoU Human Improvement": g.loc["IoU", "human_improvement"],
                "IoU BALD Laplace Improvement": g.loc["IoU", "bald_laplace_improvement"],
                "IoU Human AUC": g.loc["IoU", "human_auc"],
                "IoU BALD Laplace AUC": g.loc["IoU", "bald_laplace_auc"],
                "IoU Human Mean": g.loc["IoU", "human_mean"],
                "IoU BALD Laplace Mean": g.loc["IoU", "bald_laplace_mean"],
                "Dice Human Improvement": g.loc["Dice", "human_improvement"],
                "Dice BALD Laplace Improvement": g.loc["Dice", "bald_laplace_improvement"],
                "Dice Human AUC": g.loc["Dice", "human_auc"],
                "Dice BALD Laplace AUC": g.loc["Dice", "bald_laplace_auc"],
                "Dice Human Mean": g.loc["Dice", "human_mean"],
                "Dice BALD Laplace Mean": g.loc["Dice", "bald_laplace_mean"],
            }
        )
    wide = pd.DataFrame(wide_rows)
    wide_csv = OUT_DIR / "human_vs_bald_laplace_iou_dice_table.csv"
    wide.to_csv(wide_csv, index=False)
    return out_csv, wide_csv


def main() -> None:
    manifest = pd.read_csv(MANIFEST_PATH)
    human_csv = build_human_trajectory_csv(manifest)
    human_overall, human_by, full_steps_by_group = build_human_summary(human_csv)
    build_human_vs_bald_table(human_by, human_overall, full_steps_by_group)
    print(f"Saved {human_csv}")
    print(f"Saved {human_overall}")
    print(f"Saved {human_by}")
    print(f"Saved {OUT_DIR / 'human_vs_bald_laplace_iou_dice_summary.csv'}")
    print(f"Saved {OUT_DIR / 'human_vs_bald_laplace_iou_dice_table.csv'}")


if __name__ == "__main__":
    main()
