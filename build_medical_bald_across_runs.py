#!/usr/bin/env python3
"""
Compare BALD (Laplace) across the medical active-prompting runs.

Inputs:
    /mnt/HD_18T_pt2/medical_data/active_prompting/

Outputs:
    /mnt/HD_18T_pt2/medical_data/IEEE/bald_laplace_across_runs/
        overall/
            00_legend.png
            02_curves.png
            02_iou_curve.png
            02_ece_curve.png
            02_hausdorff95_curve.png
            02_nll_curve.png
            03_budget_mean_gains_vs_baseline.png
        per_dataset/<dataset_slug>/
            00_legend.png
            02_curves.png
            02_iou_curve.png
            02_ece_curve.png
            02_hausdorff95_curve.png
            02_nll_curve.png
        tables/
            bald_laplace_overall_curve_summary.csv
            bald_laplace_per_dataset_curve_summary.csv
            bald_laplace_overall_budget_mean_gains_vs_baseline.csv
            bald_laplace_dataset_budget_mean_gains_vs_baseline.csv
        README.md
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd


BASE = Path(os.environ.get("BALD_ACTIVE_ROOT", "/mnt/HD_18T_pt2/medical_data/active_prompting")).expanduser()
OUT = Path(os.environ.get("BALD_MEDICAL_BALD_OUT", "/mnt/HD_18T_pt2/medical_data/IEEE/bald_laplace_across_runs")).expanduser()
BALD = "bald_laplace"
KEY_STEPS = [0, 2, 5, 10, 15]
METRICS = [
    ("iou", "IoU", False, 100.0),
    ("ece", "ECE", True, 100.0),
    ("hausdorff95", "HD95", True, 1.0),
    ("nll", "NLL", True, 1.0),
]
STANDALONE_RUNS = [
    "run_baseline_v3",
    "run_ft_combined",
    "run_ft_busi",
    "run_ft_chexmask",
    "run_ft_drive",
    "run_ft_isic2018",
    "run_ft_kvasir_seg",
    "run_sam2_v1",
]
FULL_RUNS = ["run_baseline_v3", "run_ft_combined", "run_sam2_v1"]
GROUP_TO_FT_RUN = {
    "BUSI": "run_ft_busi",
    "CheXmask": "run_ft_chexmask",
    "DRIVE": "run_ft_drive",
    "ISIC2018": "run_ft_isic2018",
    "KvasirSEG": "run_ft_kvasir_seg",
}
VARIANT_ORDER = ["baseline", "combined_ft", "dataset_ft", "sam2"]
VARIANT_LABELS = {
    "baseline": "Baseline SAM",
    "combined_ft": "Combined FT SAM",
    "dataset_ft": "Dataset FT SAM",
    "sam2": "SAM2 Checkpoint",
}
RUN_LABELS = {
    "run_baseline_v3": "Baseline SAM",
    "run_ft_combined": "Combined FT SAM",
    "run_ft_busi": "BUSI FT SAM",
    "run_ft_chexmask": "CheXmask FT SAM",
    "run_ft_drive": "DRIVE FT SAM",
    "run_ft_isic2018": "ISIC2018 FT SAM",
    "run_ft_kvasir_seg": "KvasirSEG FT SAM",
    "run_sam2_v1": "SAM2 Checkpoint",
}
RUN_COLORS = {
    "run_baseline_v3": "#2f2f2f",
    "run_ft_combined": "#1f77b4",
    "run_ft_busi": "#c73333",
    "run_ft_chexmask": "#ff8c42",
    "run_ft_drive": "#9467bd",
    "run_ft_isic2018": "#8c564b",
    "run_ft_kvasir_seg": "#e377c2",
    "run_sam2_v1": "#2f855a",
}
RUN_MARKERS = {
    "run_baseline_v3": "o",
    "run_ft_combined": "s",
    "run_ft_busi": "D",
    "run_ft_chexmask": "P",
    "run_ft_drive": "X",
    "run_ft_isic2018": "v",
    "run_ft_kvasir_seg": "<",
    "run_sam2_v1": "^",
}
RUN_LINESTYLES = {
    "run_baseline_v3": "-",
    "run_ft_combined": "-",
    "run_ft_busi": "-",
    "run_ft_chexmask": "-",
    "run_ft_drive": "-",
    "run_ft_isic2018": "-",
    "run_ft_kvasir_seg": "-",
    "run_sam2_v1": "--",
}
RUN_LINEWIDTHS = {
    "run_baseline_v3": 2.5,
    "run_ft_combined": 2.7,
    "run_ft_busi": 2.7,
    "run_ft_chexmask": 2.7,
    "run_ft_drive": 2.7,
    "run_ft_isic2018": 2.7,
    "run_ft_kvasir_seg": 2.7,
    "run_sam2_v1": 2.8,
}
CURVE_LABEL_FONTSIZE = 18
CURVE_TICK_FONTSIZE = 14

plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "DejaVu Serif",
})


@dataclass(frozen=True)
class RunData:
    aggregate: pd.DataFrame
    per_group: pd.DataFrame
    max_step: int


def metric_col(metric_key: str) -> str:
    return f"{metric_key}_mean"


def slugify(text: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in text).strip("_")


def trapz_area(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.trapezoid(y, x) if hasattr(np, "trapezoid") else np.trapz(y, x))


def summarize_per_group(step_results: pd.DataFrame) -> pd.DataFrame:
    return (
        step_results.groupby(["strategy", "step", "group"], as_index=False)[["iou", "ece", "hausdorff95", "nll"]]
        .mean()
        .rename(columns={
            "iou": "iou_mean",
            "ece": "ece_mean",
            "hausdorff95": "hausdorff95_mean",
            "nll": "nll_mean",
        })
    )


def summarize_aggregate(per_group: pd.DataFrame) -> pd.DataFrame:
    return (
        per_group.groupby(["strategy", "step"], as_index=False)[[metric_col(m[0]) for m in METRICS]]
        .mean()
    )


def load_run(run: str) -> RunData:
    run_dir = BASE / run
    if (run_dir / "per_group.csv").exists():
        per_group = pd.read_csv(run_dir / "per_group.csv")
    else:
        per_group = summarize_per_group(pd.read_csv(run_dir / "step_results.csv"))
    if (run_dir / "aggregate.csv").exists():
        aggregate = pd.read_csv(run_dir / "aggregate.csv")
    else:
        aggregate = summarize_aggregate(per_group)
    max_step = int(aggregate["step"].max())
    return RunData(aggregate=aggregate, per_group=per_group, max_step=max_step)


def run_variant(run: str) -> str:
    if run == "run_baseline_v3":
        return "baseline"
    if run == "run_ft_combined":
        return "combined_ft"
    if run == "run_sam2_v1":
        return "sam2"
    return "dataset_ft"


def run_scope(run: str) -> str:
    if run in FULL_RUNS:
        return "All datasets"
    for group, ft_run in GROUP_TO_FT_RUN.items():
        if ft_run == run:
            return group
    return "Unknown"


def savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def plot_legend(out_path: Path, runs: list[str]) -> None:
    handles = [
        Line2D(
            [0],
            [0],
            color=RUN_COLORS[run],
            linewidth=RUN_LINEWIDTHS[run],
            linestyle=RUN_LINESTYLES[run],
            marker=RUN_MARKERS[run],
            markersize=7,
            label=RUN_LABELS[run],
        )
        for run in runs
    ]
    ncols = min(4, len(handles))
    fig_h = 1.6 if len(handles) <= 4 else 2.4
    fig, ax = plt.subplots(figsize=(12.0, fig_h))
    ax.axis("off")
    ax.legend(
        handles=handles,
        loc="center",
        frameon=False,
        ncol=ncols,
        fontsize=12,
        handlelength=2.3,
        columnspacing=1.7,
    )
    savefig(fig, out_path)


def plot_metric_overlay(series_map: dict[str, pd.DataFrame], metric_key: str, ylabel: str, out_path: Path, run_order: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    for run in run_order:
        if run not in series_map:
            continue
        sub = series_map[run].sort_values("step")
        ax.plot(
            sub["step"],
            sub[metric_col(metric_key)],
            color=RUN_COLORS[run],
            linewidth=RUN_LINEWIDTHS[run],
            linestyle=RUN_LINESTYLES[run],
            marker=RUN_MARKERS[run],
            markersize=4.5,
            markevery=2,
        )
    ax.set_xlabel("Prompting Step", fontsize=CURVE_LABEL_FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=CURVE_LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=CURVE_TICK_FONTSIZE)
    ax.grid(alpha=0.25)
    savefig(fig, out_path)


def plot_metric_overlay_grid(series_map: dict[str, pd.DataFrame], out_path: Path, run_order: list[str]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.5))
    for ax, (metric_key, ylabel, _, _) in zip(axes.flatten(), METRICS):
        for run in run_order:
            if run not in series_map:
                continue
            sub = series_map[run].sort_values("step")
            ax.plot(
                sub["step"],
                sub[metric_col(metric_key)],
                color=RUN_COLORS[run],
                linewidth=RUN_LINEWIDTHS[run],
                linestyle=RUN_LINESTYLES[run],
                marker=RUN_MARKERS[run],
                markersize=4.5,
                markevery=2,
            )
        ax.set_xlabel("Prompting Step", fontsize=CURVE_LABEL_FONTSIZE)
        ax.set_ylabel(ylabel, fontsize=CURVE_LABEL_FONTSIZE)
        ax.tick_params(axis="both", labelsize=CURVE_TICK_FONTSIZE)
        ax.grid(alpha=0.25)
    savefig(fig, out_path)


def build_curve_summary_row(frame: pd.DataFrame, metric_key: str, label: str, lower_is_better: bool) -> dict[str, float | str]:
    sub = frame.sort_values("step")
    value_by_step = sub.set_index("step")[metric_col(metric_key)]
    row: dict[str, float | str] = {
        "metric": label,
        "direction": "lower_is_better" if lower_is_better else "higher_is_better",
    }
    for step in KEY_STEPS:
        row[f"step_{step}"] = float(value_by_step.loc[step])
    full_step = int(sub["step"].max())
    row["step_full"] = float(value_by_step.loc[full_step])
    row["raw_change_0_to_full"] = row["step_full"] - row["step_0"]
    row["improvement_0_to_full"] = (row["step_0"] - row["step_full"]) if lower_is_better else (row["step_full"] - row["step_0"])
    x = sub["step"].to_numpy(dtype=float)
    y = sub[metric_col(metric_key)].to_numpy(dtype=float)
    row["budget_auc"] = trapz_area(x, y)
    row["budget_mean"] = float(y.mean())
    return row


def build_summary_table(series_by_run: dict[str, pd.DataFrame], include_group: str | None = None) -> pd.DataFrame:
    rows = []
    for run, frame in series_by_run.items():
        for metric_key, label, lower_is_better, _ in METRICS:
            row = build_curve_summary_row(frame, metric_key, label, lower_is_better)
            if include_group is not None:
                row["group"] = include_group
            row["run"] = run
            row["scope"] = run_scope(run)
            row["variant"] = run_variant(run)
            row["run_label"] = RUN_LABELS[run]
            rows.append(row)
    cols = ["metric", "direction"]
    if include_group is not None:
        cols.append("group")
    cols.extend([
        "run",
        "scope",
        "variant",
        "run_label",
        "step_0",
        "step_2",
        "step_5",
        "step_10",
        "step_15",
        "step_full",
        "raw_change_0_to_full",
        "improvement_0_to_full",
        "budget_auc",
        "budget_mean",
    ])
    return pd.DataFrame(rows)[cols]


def direction_aligned_gain(new_value: float, base_value: float, lower_is_better: bool, scale: float) -> float:
    if lower_is_better:
        return scale * (base_value - new_value)
    return scale * (new_value - base_value)


def plot_gain_bars(gain_table: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.5))
    run_order = ["combined_ft", "sam2"]
    for ax, (metric_key, label, lower_is_better, scale) in zip(axes.flatten(), METRICS):
        col = f"{metric_key}_budget_mean_gain"
        sub = gain_table.set_index("variant").loc[run_order]
        vals = sub[col].to_numpy(dtype=float)
        bars = ax.bar(
            np.arange(len(run_order)),
            vals,
            color=[RUN_COLORS["run_ft_combined"], RUN_COLORS["run_sam2_v1"]],
            edgecolor="white",
        )
        pad = max(np.max(np.abs(vals)) * 0.04 if len(vals) else 0.0, 0.2)
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + (pad if val >= 0 else -pad),
                f"{val:+.1f}",
                ha="center",
                va="bottom" if val >= 0 else "top",
                fontsize=10,
            )
        ax.axhline(0, color="black", lw=1.0, alpha=0.5)
        ax.set_xticks(np.arange(len(run_order)))
        ax.set_xticklabels([VARIANT_LABELS[v] for v in run_order], rotation=18, ha="right")
        ax.set_ylabel(f"Budget-Mean {label} gain", fontsize=CURVE_LABEL_FONTSIZE)
        ax.tick_params(axis="both", labelsize=CURVE_TICK_FONTSIZE)
        ax.grid(axis="y", alpha=0.25)
    savefig(fig, out_path)


def plot_gain_heatmaps(gain_table: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 8.5))
    col_order = ["combined_ft", "dataset_ft", "sam2"]
    col_labels = [VARIANT_LABELS[v] for v in col_order]
    row_order = list(GROUP_TO_FT_RUN.keys())
    for ax, (metric_key, label, _, _) in zip(axes.flatten(), METRICS):
        col = f"{metric_key}_budget_mean_gain"
        mat = (
            gain_table.pivot(index="group", columns="variant", values=col)
            .reindex(index=row_order, columns=col_order)
        )
        data = mat.to_numpy(dtype=float)
        vmax = float(np.nanpercentile(np.abs(data), 90)) if data.size else 1.0
        vmax = max(vmax, 1.0)
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
        im = ax.imshow(data, cmap="RdBu_r", norm=norm, aspect="auto")
        ax.set_xticks(np.arange(len(col_labels)))
        ax.set_xticklabels(col_labels, rotation=18, ha="right")
        ax.set_yticks(np.arange(len(row_order)))
        ax.set_yticklabels(row_order)
        ax.set_title(f"Budget-Mean {label} gain", fontweight="bold")
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax.text(j, i, f"{data[i, j]:+.1f}", ha="center", va="center", fontsize=9)
        cbar = fig.colorbar(im, ax=ax, shrink=0.9)
        cbar.set_label("Positive means better than baseline")
    savefig(fig, out_path)


def write_readme(overall_curve: pd.DataFrame, overall_gains: pd.DataFrame, dataset_gains: pd.DataFrame) -> None:
    metric_best = {}
    for metric_key, label, _, _ in METRICS:
        col = f"{metric_key}_budget_mean_gain"
        best = overall_gains.sort_values(col, ascending=False).iloc[0]
        metric_best[label] = f"{best['run_label']} ({best[col]:+.1f})"

    lines = [
        "# BALD (Laplace) Across Medical Runs",
        "",
        "This package compares only `BALD (Laplace)` across the medical active-prompting runs.",
        "",
        "Scope:",
        "- Every run folder is treated as its own standalone checkpoint in the overall curves and overall summary table.",
        "- The `scope` column in the tables tells you whether a row comes from `All datasets` or from a single dataset checkpoint such as `BUSI` or `DRIVE`.",
        "- Baseline-vs-gain plots are only used where the comparison is scope-matched.",
        "",
        "Best budget-mean gains vs baseline:",
        f"- IoU: {metric_best['IoU']}",
        f"- ECE: {metric_best['ECE']}",
        f"- HD95: {metric_best['HD95']}",
        f"- NLL: {metric_best['NLL']}",
        "",
        "Key files:",
        "- `overall/02_curves.png`",
        "- `overall/03_budget_mean_gains_vs_baseline.png`",
        "- `dataset_budget_mean_gains_vs_baseline.png`",
        "- `tables/bald_laplace_overall_curve_summary.csv`",
        "- `tables/bald_laplace_per_dataset_curve_summary.csv`",
        "- `tables/bald_laplace_overall_budget_mean_gains_vs_baseline.csv`",
        "- `tables/bald_laplace_dataset_budget_mean_gains_vs_baseline.csv`",
    ]
    (OUT / "README.md").write_text("\n".join(lines))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "overall").mkdir(exist_ok=True)
    (OUT / "per_dataset").mkdir(exist_ok=True)
    (OUT / "tables").mkdir(exist_ok=True)

    runs = {run: load_run(run) for run in STANDALONE_RUNS}

    overall_series = {
        run: runs[run].aggregate[runs[run].aggregate["strategy"] == BALD].copy()
        for run in STANDALONE_RUNS
    }
    plot_legend(OUT / "overall" / "00_legend.png", STANDALONE_RUNS)
    plot_metric_overlay_grid(overall_series, OUT / "overall" / "02_curves.png", STANDALONE_RUNS)
    for metric_key, label, _, _ in METRICS:
        plot_metric_overlay(overall_series, metric_key, label, OUT / "overall" / f"02_{metric_key}_curve.png", STANDALONE_RUNS)

    overall_summary = build_summary_table(overall_series).sort_values(["metric", "scope", "run_label"])
    overall_summary.to_csv(OUT / "tables" / "bald_laplace_overall_curve_summary.csv", index=False)

    overall_gain_rows = []
    base_overall = overall_summary[overall_summary["run"] == "run_baseline_v3"].set_index("metric")
    for run in ["run_ft_combined", "run_sam2_v1"]:
        sub = overall_summary[overall_summary["run"] == run].set_index("metric")
        row = {
            "run": run,
            "variant": run_variant(run),
            "run_label": RUN_LABELS[run],
        }
        for metric_key, label, lower_is_better, scale in METRICS:
            row[f"{metric_key}_step_full_gain"] = direction_aligned_gain(
                float(sub.loc[label, "step_full"]),
                float(base_overall.loc[label, "step_full"]),
                lower_is_better,
                scale,
            )
            row[f"{metric_key}_budget_mean_gain"] = direction_aligned_gain(
                float(sub.loc[label, "budget_mean"]),
                float(base_overall.loc[label, "budget_mean"]),
                lower_is_better,
                scale,
            )
            row[f"{metric_key}_budget_auc_gain"] = direction_aligned_gain(
                float(sub.loc[label, "budget_auc"]),
                float(base_overall.loc[label, "budget_auc"]),
                lower_is_better,
                scale,
            )
        overall_gain_rows.append(row)
    overall_gains = pd.DataFrame(overall_gain_rows).sort_values("variant")
    overall_gains.to_csv(OUT / "tables" / "bald_laplace_overall_budget_mean_gains_vs_baseline.csv", index=False)
    plot_gain_bars(overall_gains, OUT / "overall" / "03_budget_mean_gains_vs_baseline.png")

    dataset_rows = []
    dataset_gain_rows = []
    for group, ft_run in GROUP_TO_FT_RUN.items():
        group_slug = slugify(group)
        series_map = {
            "run_baseline_v3": runs["run_baseline_v3"].per_group[
                (runs["run_baseline_v3"].per_group["strategy"] == BALD) &
                (runs["run_baseline_v3"].per_group["group"] == group)
            ].copy(),
            "run_ft_combined": runs["run_ft_combined"].per_group[
                (runs["run_ft_combined"].per_group["strategy"] == BALD) &
                (runs["run_ft_combined"].per_group["group"] == group)
            ].copy(),
            ft_run: runs[ft_run].per_group[
                (runs[ft_run].per_group["strategy"] == BALD) &
                (runs[ft_run].per_group["group"] == group)
            ].copy(),
            "run_sam2_v1": runs["run_sam2_v1"].per_group[
                (runs["run_sam2_v1"].per_group["strategy"] == BALD) &
                (runs["run_sam2_v1"].per_group["group"] == group)
            ].copy(),
        }
        dataset_rows.append(build_summary_table(series_map, include_group=group))

        dataset_run_order = ["run_baseline_v3", "run_ft_combined", ft_run, "run_sam2_v1"]
        out_dir = OUT / "per_dataset" / group_slug
        plot_legend(out_dir / "00_legend.png", dataset_run_order)
        plot_metric_overlay_grid(series_map, out_dir / "02_curves.png", dataset_run_order)
        for metric_key, label, _, _ in METRICS:
            plot_metric_overlay(series_map, metric_key, label, out_dir / f"02_{metric_key}_curve.png", dataset_run_order)

        dataset_summary = build_summary_table(series_map, include_group=group)
        base = dataset_summary[dataset_summary["run"] == "run_baseline_v3"].set_index("metric")
        for run in ["run_ft_combined", ft_run, "run_sam2_v1"]:
            sub = dataset_summary[dataset_summary["run"] == run].set_index("metric")
            row = {
                "group": group,
                "run": run,
                "variant": run_variant(run),
                "run_label": RUN_LABELS[run],
            }
            for metric_key, label, lower_is_better, scale in METRICS:
                row[f"{metric_key}_step_full_gain"] = direction_aligned_gain(
                    float(sub.loc[label, "step_full"]),
                    float(base.loc[label, "step_full"]),
                    lower_is_better,
                    scale,
                )
                row[f"{metric_key}_budget_mean_gain"] = direction_aligned_gain(
                    float(sub.loc[label, "budget_mean"]),
                    float(base.loc[label, "budget_mean"]),
                    lower_is_better,
                    scale,
                )
                row[f"{metric_key}_budget_auc_gain"] = direction_aligned_gain(
                    float(sub.loc[label, "budget_auc"]),
                    float(base.loc[label, "budget_auc"]),
                    lower_is_better,
                    scale,
                )
            dataset_gain_rows.append(row)

    per_dataset_summary = pd.concat(dataset_rows, ignore_index=True).sort_values(["metric", "group", "run_label"])
    per_dataset_summary.to_csv(OUT / "tables" / "bald_laplace_per_dataset_curve_summary.csv", index=False)

    dataset_gains = pd.DataFrame(dataset_gain_rows).sort_values(["group", "variant"])
    dataset_gains.to_csv(OUT / "tables" / "bald_laplace_dataset_budget_mean_gains_vs_baseline.csv", index=False)
    plot_gain_heatmaps(dataset_gains, OUT / "dataset_budget_mean_gains_vs_baseline.png")

    write_readme(overall_summary, overall_gains, dataset_gains)
    print(f"Wrote BALD-only cross-run report to: {OUT}")


if __name__ == "__main__":
    main()
