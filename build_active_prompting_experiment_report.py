#!/usr/bin/env python3
"""
Build a reusable active-prompting report for a single experiment directory.

Inputs:
    <experiment_dir>/
        aggregate.csv
        per_group.csv        (optional if step_results.csv exists)
        per_stratum.csv
        step_results.csv     (optional if per_group.csv exists)

Outputs:
    <plots_dir>/
        02_curves_legend.png
        02_<curve_tag>_curves.png
        02_iou_curve.png
        02_ece_curve.png
        02_hausdorff95_curve.png
        02_nll_curve.png
        05_strata_dominance_matrix.png
        05_strata_dominance_legend.png
        per_dataset/<dataset_slug>/
            02_curves.png
            02_iou_curve.png
            02_ece_curve.png
            02_hausdorff95_curve.png
            02_nll_curve.png

    <tables_dir>/
        <table_prefix>_curve_summary.csv
        <table_prefix>_curve_summary_by_dataset.csv
        <table_prefix>_strata_dominance_table.csv
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MetricSpec:
    key: str
    label: str
    lower_is_better: bool


@dataclass(frozen=True)
class ExperimentData:
    aggregate: pd.DataFrame
    per_group: pd.DataFrame
    per_stratum: pd.DataFrame
    max_step: int


METRICS: list[MetricSpec] = [
    MetricSpec("iou", "IoU", False),
    MetricSpec("ece", "ECE", True),
    MetricSpec("hausdorff95", "HD95", True),
    MetricSpec("nll", "NLL", True),
]
DEFAULT_STRATEGIES = ["bald_laplace", "bald_mcdropout", "entropy"]
CANONICAL_STRATEGY_ORDER = ["bald_laplace", "entropy", "oracle", "bald_mcdropout", "random", "maxdist"]
DEFAULT_KEY_STEPS = [0, 2, 5, 10, 15]
CURVE_LABEL_FONTSIZE = 18
CURVE_TICK_FONTSIZE = 14
MATRIX_HEADER_FONTSIZE = 15
MATRIX_TICK_FONTSIZE = 14

COLORS = {
    "bald_laplace": "#c73333",
    "entropy": "#1f77b4",
    "oracle": "#2f855a",
    "bald_mcdropout": "#ff8c42",
    "random": "#7f7f7f",
    "maxdist": "#8c564b",
    "tie": "#767676",
}
LABELS = {
    "bald_laplace": "BALD (Laplace)",
    "entropy": "Entropy",
    "oracle": "GT-based",
    "bald_mcdropout": "BALD (MC Dropout)",
    "random": "Random",
    "maxdist": "MaxDist",
    "tie": "Tie",
}
SHORT_LABELS = {
    "bald_laplace": "BALD",
    "entropy": "ENT",
    "oracle": "GT",
    "bald_mcdropout": "MCD",
    "random": "RAND",
    "maxdist": "MD",
    "tie": "TIE",
}
MARKERS = {
    "bald_laplace": "o",
    "entropy": "s",
    "oracle": "^",
    "bald_mcdropout": "D",
    "random": "x",
    "maxdist": "v",
}
LINESTYLES = {
    "bald_laplace": "-",
    "entropy": "-",
    "oracle": "--",
    "bald_mcdropout": "-.",
    "random": ":",
    "maxdist": ":",
}
LINEWIDTHS = {
    "bald_laplace": 3.2,
    "entropy": 2.2,
    "oracle": 2.2,
    "bald_mcdropout": 2.0,
    "random": 1.3,
    "maxdist": 1.3,
}
STRATUM_FAMILY_ORDER = {
    "boundary_complexity_bin": 0,
    "contrast_bin": 1,
    "thinness_bin": 2,
    "object_size_bin": 3,
}

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True, help="Directory containing aggregate/per_group/per_stratum CSVs.")
    parser.add_argument("--plots-dir", type=Path, required=True, help="Directory to write PNG outputs.")
    parser.add_argument("--tables-dir", type=Path, required=True, help="Directory to write CSV outputs.")
    parser.add_argument("--strategies", nargs="+", default=DEFAULT_STRATEGIES, help="Strategies to use for the strata matrix and strata CSV, in display order.")
    parser.add_argument("--curve-strategies", nargs="+", default=None, help="Strategies to use for curves and curve summary tables. Default: all strategies found in the experiment.")
    parser.add_argument("--focus-strategy", default="bald_laplace", help="Strategy to emphasize in curves and matrix sorting.")
    parser.add_argument("--curve-tag", default="", help="Optional tag for the combined overall curves filename. Example: baseline -> 02_baseline_curves.png.")
    parser.add_argument("--table-prefix", default=None, help="Prefix for output CSV filenames. Default: experiment directory name.")
    parser.add_argument("--key-steps", nargs="+", type=int, default=DEFAULT_KEY_STEPS, help="Checkpoint steps to report in curve tables.")
    parser.add_argument("--groups", nargs="+", default=None, help="Optional subset of dataset groups to include. Requires step_results.csv so summaries can be recomputed faithfully.")
    parser.add_argument("--curve-overlay-experiment", type=Path, default=None, help="Optional extra experiment directory to overlay on the curves and curve summary tables only.")
    parser.add_argument("--curve-overlay-strategy", default="bald_laplace", help="Strategy to take from --curve-overlay-experiment.")
    parser.add_argument("--curve-overlay-id", default=None, help="Strategy id to assign to the overlay series inside the plots/tables. Default: <strategy>_overlay.")
    parser.add_argument("--curve-overlay-label", default=None, help="Display label for the overlay series. Default: '<label> (Overlay)'.")
    return parser.parse_args()


def metric_col(metric_key: str) -> str:
    return f"{metric_key}_mean"


def display_label(strategy: str) -> str:
    return LABELS.get(strategy, strategy)


def short_label(strategy: str) -> str:
    return SHORT_LABELS.get(strategy, strategy)


def safe_linewidth(strategy: str, focus_strategy: str) -> float:
    if strategy in LINEWIDTHS:
        return LINEWIDTHS[strategy]
    return 3.0 if strategy == focus_strategy else 2.0


def safe_linestyle(strategy: str) -> str:
    return LINESTYLES.get(strategy, "-")


def safe_marker(strategy: str) -> str:
    return MARKERS.get(strategy, "o")


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_") or "group"


def savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def add_curve_overlay(
    data: ExperimentData,
    overlay_data: ExperimentData,
    overlay_source_strategy: str,
    overlay_strategy_id: str,
) -> ExperimentData:
    if overlay_source_strategy not in set(overlay_data.aggregate["strategy"].unique()):
        raise ValueError(f"--curve-overlay-strategy {overlay_source_strategy!r} is missing from overlay aggregate.csv")
    if overlay_source_strategy not in set(overlay_data.per_group["strategy"].unique()):
        raise ValueError(f"--curve-overlay-strategy {overlay_source_strategy!r} is missing from overlay per_group.csv")
    if overlay_strategy_id in set(data.aggregate["strategy"].unique()):
        raise ValueError(f"--curve-overlay-id {overlay_strategy_id!r} collides with an existing strategy in the base experiment")

    overlay_aggregate = overlay_data.aggregate[overlay_data.aggregate["strategy"] == overlay_source_strategy].copy()
    overlay_per_group = overlay_data.per_group[overlay_data.per_group["strategy"] == overlay_source_strategy].copy()

    overlay_aggregate["strategy"] = overlay_strategy_id
    overlay_per_group["strategy"] = overlay_strategy_id

    combined_aggregate = pd.concat([data.aggregate.copy(), overlay_aggregate], ignore_index=True)
    combined_per_group = pd.concat([data.per_group.copy(), overlay_per_group], ignore_index=True)
    return ExperimentData(
        aggregate=combined_aggregate,
        per_group=combined_per_group,
        per_stratum=data.per_stratum.copy(),
        max_step=data.max_step,
    )


def summarize_per_group(step_results: pd.DataFrame) -> pd.DataFrame:
    needed = {"strategy", "step", "group", "iou", "ece", "hausdorff95", "nll"}
    missing = needed.difference(step_results.columns)
    if missing:
        raise ValueError(f"step_results.csv is missing columns required to summarize per_group: {sorted(missing)}")
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


def summarize_per_stratum(step_results: pd.DataFrame) -> pd.DataFrame:
    metric_map = {
        "iou": "iou_mean",
        "ece": "ece_mean",
        "hausdorff95": "hausdorff95_mean",
        "nll": "nll_mean",
    }
    stratum_cols = [col for col in step_results.columns if col.endswith("_bin")]
    if not stratum_cols:
        raise ValueError("step_results.csv does not contain any *_bin stratum columns needed to rebuild per_stratum.")

    parts = []
    for col in stratum_cols:
        part = step_results[["strategy", "step", col, *metric_map.keys()]].copy()
        part = part.rename(columns={col: "stratum_value"})
        part["stratum_type"] = col
        parts.append(part)

    long_df = pd.concat(parts, ignore_index=True)
    return (
        long_df.groupby(["strategy", "step", "stratum_type", "stratum_value"], as_index=False)[list(metric_map.keys())]
        .mean()
        .rename(columns=metric_map)
    )


def summarize_aggregate(per_group: pd.DataFrame) -> pd.DataFrame:
    return (
        per_group.groupby(["strategy", "step"], as_index=False)[[metric_col(m.key) for m in METRICS]]
        .mean()
    )


def validate_columns(df: pd.DataFrame, required: Sequence[str], name: str) -> None:
    missing = sorted(set(required).difference(df.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def load_experiment(experiment_dir: Path, groups: Sequence[str] | None = None) -> ExperimentData:
    aggregate_path = experiment_dir / "aggregate.csv"
    per_group_path = experiment_dir / "per_group.csv"
    per_stratum_path = experiment_dir / "per_stratum.csv"
    step_results_path = experiment_dir / "step_results.csv"

    if groups and not step_results_path.exists():
        raise FileNotFoundError(f"--groups requires step_results.csv under {experiment_dir} so per_group/per_stratum can be recomputed faithfully.")

    if not groups and not per_stratum_path.exists():
        raise FileNotFoundError(f"Missing required file: {per_stratum_path}")

    step_results = pd.read_csv(step_results_path) if step_results_path.exists() else None
    if groups:
        validate_columns(step_results, ["group"], "step_results.csv")
        requested = list(groups)
        present_groups = step_results["group"].drop_duplicates().tolist()
        missing = [group for group in requested if group not in set(present_groups)]
        if missing:
            raise ValueError(f"Requested groups are missing from step_results.csv: {missing}")
        step_results = step_results[step_results["group"].isin(requested)].copy()
        per_group = summarize_per_group(step_results)
        aggregate = summarize_aggregate(per_group)
        per_stratum = summarize_per_stratum(step_results)
    else:
        if per_group_path.exists():
            per_group = pd.read_csv(per_group_path)
        elif step_results is not None:
            per_group = summarize_per_group(step_results)
        else:
            raise FileNotFoundError(f"Need either per_group.csv or step_results.csv under {experiment_dir}")

        if aggregate_path.exists():
            aggregate = pd.read_csv(aggregate_path)
        else:
            aggregate = summarize_aggregate(per_group)

        per_stratum = pd.read_csv(per_stratum_path)

    validate_columns(aggregate, ["strategy", "step"] + [metric_col(m.key) for m in METRICS], "aggregate.csv")
    validate_columns(per_group, ["strategy", "step", "group"] + [metric_col(m.key) for m in METRICS], "per_group.csv")
    validate_columns(per_stratum, ["strategy", "step", "stratum_type", "stratum_value"] + [metric_col(m.key) for m in METRICS], "per_stratum.csv")

    max_step = int(aggregate["step"].max())
    return ExperimentData(
        aggregate=aggregate,
        per_group=per_group,
        per_stratum=per_stratum,
        max_step=max_step,
    )


def resolve_strategies(data: ExperimentData, requested: Sequence[str]) -> list[str]:
    present = set(data.aggregate["strategy"].unique())
    missing = [strategy for strategy in requested if strategy not in present]
    if missing:
        raise ValueError(f"Requested strategies are missing from aggregate.csv: {missing}")
    return list(requested)


def resolve_curve_strategies(data: ExperimentData, requested: Sequence[str] | None) -> list[str]:
    present_order = data.aggregate["strategy"].drop_duplicates().tolist()
    present = set(present_order)
    if requested:
        missing = [strategy for strategy in requested if strategy not in present]
        if missing:
            raise ValueError(f"Requested curve strategies are missing from aggregate.csv: {missing}")
        return list(requested)

    ordered = [strategy for strategy in CANONICAL_STRATEGY_ORDER if strategy in present]
    ordered.extend([strategy for strategy in present_order if strategy not in ordered])
    return ordered


def step_value(step_map: dict[int, float], step: int) -> float:
    value = step_map.get(step, np.nan)
    return float(value) if pd.notna(value) else np.nan


def strategy_value_suffix(strategy: str) -> str:
    label = display_label(strategy)
    label = re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_")
    label = label.replace("BALD_MC_Dropout", "BALD_MC")
    label = label.replace("BALD_Laplace", "BALD_Laplace")
    return label


def unique_winner_from_series(values: pd.Series, lower_is_better: bool) -> tuple[str, list[str]]:
    ordered = values.sort_values(ascending=lower_is_better)
    best_value = float(ordered.iloc[0])
    top = [idx for idx, value in ordered.items() if np.isclose(float(value), best_value)]
    if len(top) > 1:
        return "tie", top
    return top[0], top


def build_curve_summary_table(data: ExperimentData, strategies: Sequence[str], key_steps: Sequence[int]) -> pd.DataFrame:
    rows = []
    checkpoints = sorted(set(int(step) for step in key_steps if int(step) <= data.max_step))
    for metric in METRICS:
        for strategy in strategies:
            sub = data.aggregate[data.aggregate["strategy"] == strategy].sort_values("step")
            step_map = sub.set_index("step")[metric_col(metric.key)].to_dict()
            series = sub[metric_col(metric.key)].to_numpy(dtype=float)
            x = sub["step"].to_numpy(dtype=float)
            start_val = step_value(step_map, 0)
            end_val = step_value(step_map, data.max_step)
            row = {
                "metric": metric.label,
                "direction": "lower_is_better" if metric.lower_is_better else "higher_is_better",
                "strategy": strategy,
                "strategy_label": display_label(strategy),
                "step_0": start_val,
                "step_full": end_val,
                "raw_change_0_to_full": end_val - start_val,
                "improvement_0_to_full": (start_val - end_val) if metric.lower_is_better else (end_val - start_val),
                "budget_auc": float(np.trapezoid(series, x) if hasattr(np, "trapezoid") else np.trapz(series, x)),
                "budget_mean": float(series.mean()),
            }
            for step in checkpoints:
                if step == 0:
                    continue
                row[f"step_{step}"] = step_value(step_map, step)
            rows.append(row)

    ordered_cols = ["metric", "direction", "strategy", "strategy_label", "step_0"]
    ordered_cols.extend([f"step_{step}" for step in checkpoints if step != 0])
    ordered_cols.extend(["step_full", "raw_change_0_to_full", "improvement_0_to_full", "budget_auc", "budget_mean"])
    return pd.DataFrame(rows)[ordered_cols]


def build_curve_summary_by_dataset_table(data: ExperimentData, strategies: Sequence[str], key_steps: Sequence[int]) -> pd.DataFrame:
    overall = build_curve_summary_table(data, strategies, key_steps).copy()
    overall.insert(2, "group", "Overall")

    checkpoints = sorted(set(int(step) for step in key_steps if int(step) <= data.max_step))
    rows = []
    groups = data.per_group["group"].drop_duplicates().tolist()
    for metric in METRICS:
        col = metric_col(metric.key)
        for group in groups:
            group_df = data.per_group[data.per_group["group"] == group]
            for strategy in strategies:
                sub = group_df[group_df["strategy"] == strategy].sort_values("step")
                step_map = sub.set_index("step")[col].to_dict()
                series = sub[col].to_numpy(dtype=float)
                x = sub["step"].to_numpy(dtype=float)
                start_val = step_value(step_map, 0)
                end_val = step_value(step_map, data.max_step)
                row = {
                    "metric": metric.label,
                    "direction": "lower_is_better" if metric.lower_is_better else "higher_is_better",
                    "group": group,
                    "strategy": strategy,
                    "strategy_label": display_label(strategy),
                    "step_0": start_val,
                    "step_full": end_val,
                    "raw_change_0_to_full": end_val - start_val,
                    "improvement_0_to_full": (start_val - end_val) if metric.lower_is_better else (end_val - start_val),
                    "budget_auc": float(np.trapezoid(series, x) if hasattr(np, "trapezoid") else np.trapz(series, x)),
                    "budget_mean": float(series.mean()),
                }
                for step in checkpoints:
                    if step == 0:
                        continue
                    row[f"step_{step}"] = step_value(step_map, step)
                rows.append(row)

    dataset_df = pd.DataFrame(rows)
    combined = pd.concat([overall, dataset_df], ignore_index=True)
    group_order = ["Overall"] + groups
    combined["group"] = pd.Categorical(combined["group"], categories=group_order, ordered=True)
    return combined.sort_values(["metric", "group", "strategy"]).reset_index(drop=True)


def compute_strata_dominance_rows(data: ExperimentData, strategies: Sequence[str], focus_strategy: str) -> pd.DataFrame:
    final = data.per_stratum[
        (data.per_stratum["step"] == data.max_step) &
        (data.per_stratum["strategy"].isin(strategies))
    ].copy()

    rows = []
    for (stratum_type, stratum_value), group_df in final.groupby(["stratum_type", "stratum_value"]):
        values = group_df.set_index("strategy")
        rank_sum = {strategy: 0 for strategy in strategies}
        focus_wins = 0
        row = {
            "stratum_type": stratum_type,
            "stratum_value": stratum_value,
            "stratum_family": stratum_type.replace("_bin", "").replace("_", " ").title(),
            "pretty_name": f"{stratum_type.replace('_bin', '').replace('_', ' ').title()}: {stratum_value.title()}",
        }
        for metric in METRICS:
            col = metric_col(metric.key)
            ordered = values[col].sort_values(ascending=metric.lower_is_better).index.tolist()
            for rank, strategy in enumerate(ordered, start=1):
                rank_sum[strategy] += rank
            winner_id, _ = unique_winner_from_series(values[col], metric.lower_is_better)
            row[f"{metric.label}_winner_id"] = winner_id
            row[f"{metric.label}_winner"] = display_label(winner_id)
            focus_wins += int(winner_id == focus_strategy)
            for strategy in strategies:
                row[f"{metric.label}_{strategy_value_suffix(strategy)}"] = float(values.loc[strategy, col])

        rank_series = pd.Series(rank_sum, dtype=float)
        balanced_winner_id, _ = unique_winner_from_series(rank_series, lower_is_better=True)
        row["Balanced_winner_id"] = balanced_winner_id
        row["Balanced_winner"] = display_label(balanced_winner_id)
        row["_focus_wins"] = focus_wins + int(balanced_winner_id == focus_strategy)
        row["_balanced_is_focus"] = int(balanced_winner_id == focus_strategy)
        row["_family_order"] = STRATUM_FAMILY_ORDER.get(stratum_type, 99)
        rows.append(row)

    dominance = pd.DataFrame(rows).sort_values(
        ["_balanced_is_focus", "_focus_wins", "_family_order", "pretty_name"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)
    dominance.insert(0, "order_in_matrix", np.arange(1, len(dominance) + 1))
    return dominance


def build_strata_dominance_table(data: ExperimentData, strategies: Sequence[str], focus_strategy: str) -> pd.DataFrame:
    rows = compute_strata_dominance_rows(data, strategies, focus_strategy).copy()
    value_cols = []
    for metric in METRICS:
        for strategy in strategies:
            col = f"{metric.label}_{strategy_value_suffix(strategy)}"
            value_cols.append(col)
    rows[value_cols] = rows[value_cols].round(4)

    ordered_cols = [
        "order_in_matrix",
        "stratum_type",
        "stratum_value",
        "stratum_family",
        "pretty_name",
    ]
    for metric in METRICS:
        for strategy in strategies:
            ordered_cols.append(f"{metric.label}_{strategy_value_suffix(strategy)}")
        ordered_cols.append(f"{metric.label}_winner")
    ordered_cols.append("Balanced_winner")
    return rows[ordered_cols]


def plot_curve_legend(out_path: Path, strategies: Sequence[str], focus_strategy: str) -> None:
    handles = [
        Line2D(
            [0],
            [0],
            color=COLORS[strategy],
            linewidth=safe_linewidth(strategy, focus_strategy),
            linestyle=safe_linestyle(strategy),
            marker=safe_marker(strategy),
            markersize=8 if strategy == focus_strategy else 6.5,
            markerfacecolor=COLORS[strategy],
            markeredgecolor="black" if strategy == focus_strategy else "white",
            markeredgewidth=1.2 if strategy == focus_strategy else 0.8,
            label=display_label(strategy),
        )
        for strategy in strategies
    ]
    fig, ax = plt.subplots(figsize=(8.5, 1.8))
    ax.axis("off")
    ax.legend(
        handles=handles,
        loc="center",
        frameon=False,
        ncol=min(3, len(strategies)),
        fontsize=13,
        handlelength=2.3,
        columnspacing=1.7,
        labelspacing=1.0,
    )
    savefig(fig, out_path)


def plot_strata_legend(out_path: Path, strategies: Sequence[str], focus_strategy: str) -> None:
    handles = [
        Patch(
            facecolor=COLORS[strategy],
            edgecolor="black" if strategy == focus_strategy else "white",
            linewidth=1.8 if strategy == focus_strategy else 1.0,
            label=f"{short_label(strategy)} = {display_label(strategy)}",
        )
        for strategy in strategies
    ]
    fig, ax = plt.subplots(figsize=(8.5, 1.6))
    ax.axis("off")
    ax.legend(
        handles=handles,
        loc="center",
        frameon=False,
        ncol=min(3, len(strategies)),
        fontsize=13,
        handlelength=1.6,
        columnspacing=1.6,
        labelspacing=1.0,
    )
    savefig(fig, out_path)


def plot_metric_curve(
    df: pd.DataFrame,
    metric: MetricSpec,
    strategies: Sequence[str],
    focus_strategy: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    col = metric_col(metric.key)
    for strategy in strategies:
        sub = df[df["strategy"] == strategy].sort_values("step")
        ax.plot(
            sub["step"],
            sub[col],
            color=COLORS[strategy],
            linewidth=safe_linewidth(strategy, focus_strategy),
            linestyle=safe_linestyle(strategy),
            marker=safe_marker(strategy),
            markersize=4,
            markevery=2,
        )
    ax.set_xlabel("Prompting Step", fontsize=CURVE_LABEL_FONTSIZE)
    ax.set_ylabel(metric.label, fontsize=CURVE_LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=CURVE_TICK_FONTSIZE)
    ax.grid(alpha=0.25)
    savefig(fig, out_path)


def plot_combined_curves(
    df: pd.DataFrame,
    strategies: Sequence[str],
    focus_strategy: str,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.5))
    for ax, metric in zip(axes.flatten(), METRICS):
        col = metric_col(metric.key)
        for strategy in strategies:
            sub = df[df["strategy"] == strategy].sort_values("step")
            ax.plot(
                sub["step"],
                sub[col],
                color=COLORS[strategy],
                linewidth=safe_linewidth(strategy, focus_strategy),
                linestyle=safe_linestyle(strategy),
                marker=safe_marker(strategy),
                markersize=4,
                markevery=2,
            )
        ax.set_xlabel("Prompting Step", fontsize=CURVE_LABEL_FONTSIZE)
        ax.set_ylabel(metric.label, fontsize=CURVE_LABEL_FONTSIZE)
        ax.tick_params(axis="both", labelsize=CURVE_TICK_FONTSIZE)
        ax.grid(alpha=0.25)
    savefig(fig, out_path)


def plot_per_dataset_curves(
    data: ExperimentData,
    plots_dir: Path,
    strategies: Sequence[str],
    focus_strategy: str,
) -> None:
    for group in data.per_group["group"].drop_duplicates().tolist():
        group_slug = slugify(group)
        group_dir = plots_dir / "per_dataset" / group_slug
        group_df = data.per_group[data.per_group["group"] == group].copy()
        plot_combined_curves(group_df, strategies, focus_strategy, group_dir / "02_curves.png")
        for metric in METRICS:
            plot_metric_curve(group_df, metric, strategies, focus_strategy, group_dir / f"02_{metric.key}_curve.png")


def plot_strata_dominance_matrix(
    dominance_rows: pd.DataFrame,
    strategies: Sequence[str],
    focus_strategy: str,
    out_path: Path,
) -> None:
    n_rows = len(dominance_rows)
    n_cols = 5
    fig_h = max(7.2, 0.54 * n_rows + 2.1)
    fig, ax = plt.subplots(figsize=(10.6, fig_h))
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.invert_yaxis()

    metric_labels = [metric.label for metric in METRICS]
    display_cols = metric_labels + ["Balanced"]

    for i, row in dominance_rows.iterrows():
        if i % 2 == 1:
            ax.add_patch(Rectangle((0, i), n_cols, 1, facecolor="#f7f3ee", edgecolor="none", zorder=0))
        for j, label in enumerate(metric_labels):
            winner_id = row[f"{label}_winner_id"]
            ax.add_patch(
                Rectangle(
                    (j, i),
                    1,
                    1,
                    facecolor=COLORS[winner_id],
                    alpha=0.9,
                    edgecolor="black" if winner_id == focus_strategy else "white",
                    linewidth=1.9 if winner_id == focus_strategy else 1.2,
                )
            )
            ax.text(
                j + 0.5,
                i + 0.5,
                short_label(winner_id),
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                color="white",
            )

        balanced_winner_id = row["Balanced_winner_id"]
        ax.add_patch(
            Rectangle(
                (4, i),
                1,
                1,
                facecolor=COLORS[balanced_winner_id],
                alpha=0.92,
                edgecolor="black" if balanced_winner_id == focus_strategy else "white",
                linewidth=2.1 if balanced_winner_id == focus_strategy else 1.2,
            )
        )
        ax.text(
            4.5,
            i + 0.5,
            short_label(balanced_winner_id),
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="white",
        )

    for x in range(n_cols + 1):
        ax.axvline(x, color="white", lw=1.2, zorder=5)
    for y in range(n_rows + 1):
        ax.axhline(y, color="white", lw=1.0, zorder=5)

    ax.set_xticks(np.arange(n_cols) + 0.5)
    ax.set_xticklabels(display_cols, fontweight="bold", fontsize=MATRIX_HEADER_FONTSIZE)
    ax.tick_params(axis="x", top=True, bottom=False, labeltop=True, labelbottom=False, length=0, pad=6)
    ax.set_yticks(np.arange(n_rows) + 0.5)
    ax.set_yticklabels(dominance_rows["pretty_name"], fontsize=MATRIX_TICK_FONTSIZE)
    ax.tick_params(axis="y", length=0)
    savefig(fig, out_path)


def build_outputs(
    data: ExperimentData,
    plots_dir: Path,
    tables_dir: Path,
    curve_strategies: Sequence[str],
    matrix_strategies: Sequence[str],
    focus_strategy: str,
    curve_tag: str,
    table_prefix: str,
    key_steps: Sequence[int],
) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    plot_curve_legend(plots_dir / "02_curves_legend.png", curve_strategies, focus_strategy)
    plot_strata_legend(plots_dir / "05_strata_dominance_legend.png", matrix_strategies, focus_strategy)

    for metric in METRICS:
        plot_metric_curve(data.aggregate, metric, curve_strategies, focus_strategy, plots_dir / f"02_{metric.key}_curve.png")

    combined_name = f"02_{curve_tag}_curves.png" if curve_tag else "02_curves.png"
    plot_combined_curves(data.aggregate, curve_strategies, focus_strategy, plots_dir / combined_name)
    plot_per_dataset_curves(data, plots_dir, curve_strategies, focus_strategy)

    curve_summary = build_curve_summary_table(data, curve_strategies, key_steps)
    curve_summary_by_dataset = build_curve_summary_by_dataset_table(data, curve_strategies, key_steps)
    strata_dominance_table = build_strata_dominance_table(data, matrix_strategies, focus_strategy)
    dominance_rows = compute_strata_dominance_rows(data, matrix_strategies, focus_strategy)

    curve_summary.to_csv(tables_dir / f"{table_prefix}_curve_summary.csv", index=False)
    curve_summary_by_dataset.to_csv(tables_dir / f"{table_prefix}_curve_summary_by_dataset.csv", index=False)
    strata_dominance_table.to_csv(tables_dir / f"{table_prefix}_strata_dominance_table.csv", index=False)
    plot_strata_dominance_matrix(dominance_rows, matrix_strategies, focus_strategy, plots_dir / "05_strata_dominance_matrix.png")


def main() -> None:
    args = parse_args()
    data = load_experiment(args.experiment_dir, groups=args.groups)
    overlay_strategy_id = None
    if args.curve_overlay_experiment is not None:
        overlay_data = load_experiment(args.curve_overlay_experiment, groups=args.groups)
        overlay_strategy_id = args.curve_overlay_id or f"{args.curve_overlay_strategy}_overlay"
        overlay_label = args.curve_overlay_label or f"{display_label(args.curve_overlay_strategy)} (Overlay)"
        LABELS[overlay_strategy_id] = overlay_label
        SHORT_LABELS[overlay_strategy_id] = overlay_label
        COLORS[overlay_strategy_id] = "#7d1f1f" if args.curve_overlay_strategy == "bald_laplace" else "#444444"
        LINESTYLES[overlay_strategy_id] = "--"
        MARKERS[overlay_strategy_id] = "o" if args.curve_overlay_strategy == "bald_laplace" else "s"
        LINEWIDTHS[overlay_strategy_id] = 3.0
        data = add_curve_overlay(data, overlay_data, args.curve_overlay_strategy, overlay_strategy_id)

    matrix_strategies = resolve_strategies(data, args.strategies)
    curve_strategies = resolve_curve_strategies(data, args.curve_strategies)
    if overlay_strategy_id is not None and overlay_strategy_id not in curve_strategies:
        curve_strategies.append(overlay_strategy_id)
    focus_strategy = args.focus_strategy
    if focus_strategy not in matrix_strategies:
        raise ValueError(f"--focus-strategy must be included in --strategies. Got {focus_strategy!r} not in {matrix_strategies!r}.")
    if focus_strategy not in curve_strategies:
        raise ValueError(f"--focus-strategy must be included in --curve-strategies when that flag is used. Got {focus_strategy!r} not in {curve_strategies!r}.")
    table_prefix = args.table_prefix or slugify(args.experiment_dir.name)
    build_outputs(
        data=data,
        plots_dir=args.plots_dir,
        tables_dir=args.tables_dir,
        curve_strategies=curve_strategies,
        matrix_strategies=matrix_strategies,
        focus_strategy=focus_strategy,
        curve_tag=args.curve_tag.strip(),
        table_prefix=table_prefix,
        key_steps=args.key_steps,
    )
    print(f"Wrote plots to: {args.plots_dir}")
    print(f"Wrote tables to: {args.tables_dir}")


if __name__ == "__main__":
    main()
