#!/usr/bin/env python3
"""
Build two medical active-prompting comparison packages:

1. A BALD-victory package centered on the strongest faithful wins.
2. A baseline-SAM -> finetuned-SAM transition package showing how the
   strategy landscape changes.

Inputs:
    /mnt/HD_18T_pt2/medical_data/active_prompting/

Outputs:
    /mnt/HD_18T_pt2/medical_data/active_prompting/sam_story/
        victory/*.png
        transition/*.png
        tables/*.csv
        README.md
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from math import ceil
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Patch
import numpy as np
import pandas as pd


BASE = Path(os.environ.get("BALD_ACTIVE_ROOT", "/mnt/HD_18T_pt2/medical_data/active_prompting")).expanduser()
OUT = Path(os.environ.get("BALD_MEDICAL_STORY_OUT", BASE / "sam_story")).expanduser()
RUNS = [
    "run_baseline_v3",
    "run_ft_combined",
    "run_ft_busi",
    "run_ft_chexmask",
    "run_ft_drive",
    "run_ft_isic2018",
    "run_ft_kvasir_seg",
]
GROUP_TO_RUN = {
    "BUSI": "run_ft_busi",
    "CheXmask": "run_ft_chexmask",
    "DRIVE": "run_ft_drive",
    "ISIC2018": "run_ft_isic2018",
    "KvasirSEG": "run_ft_kvasir_seg",
}
RUN_LABELS = {
    "run_baseline_v3": "Baseline SAM",
    "run_ft_combined": "Combined FT SAM",
    "run_ft_busi": "BUSI FT SAM",
    "run_ft_chexmask": "CheXmask FT SAM",
    "run_ft_drive": "DRIVE FT SAM",
    "run_ft_isic2018": "ISIC2018 FT SAM",
    "run_ft_kvasir_seg": "KvasirSEG FT SAM",
}
STRATEGIES = ["bald_laplace", "entropy", "oracle", "bald_mcdropout", "random", "maxdist"]
NO_GT_STRATEGIES = ["bald_laplace", "entropy", "bald_mcdropout", "random", "maxdist"]
TRIO = ["bald_laplace", "entropy", "bald_mcdropout"]
KEY_BUDGETS = [2, 5, 10, 15, 19]
METRICS = [
    ("iou", "IoU", False),
    ("ece", "ECE", True),
    ("hausdorff95", "HD95", True),
    ("nll", "NLL", True),
]
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
CURVE_LABEL_FONTSIZE = 18
CURVE_TICK_FONTSIZE = 14
MATRIX_HEADER_FONTSIZE = 15
MATRIX_TICK_FONTSIZE = 14

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 15,
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
    step_results: pd.DataFrame
    aggregate: pd.DataFrame
    per_group: pd.DataFrame
    per_stratum: pd.DataFrame
    max_step: int


def ensure_dirs() -> None:
    for sub in ["victory", "transition", "tables", "all_strategies", "no_gt"]:
        (OUT / sub).mkdir(parents=True, exist_ok=True)


def savefig(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path)
    plt.close(fig)


def trapz_area(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.trapezoid(y, x) if hasattr(np, "trapezoid") else np.trapz(y, x))


def summarize_per_group(step_results: pd.DataFrame) -> pd.DataFrame:
    return (
        step_results.groupby(["strategy", "step", "group"], as_index=False)[["iou", "ece", "hausdorff95", "nll"]]
        .mean()
        .rename(columns={"iou": "iou_mean", "ece": "ece_mean", "hausdorff95": "hausdorff95_mean", "nll": "nll_mean"})
    )


def load_run(run: str) -> RunData:
    run_dir = BASE / run
    step_results = pd.read_csv(run_dir / "step_results.csv")
    aggregate = pd.read_csv(run_dir / "aggregate.csv")
    if (run_dir / "per_group.csv").exists():
        per_group = pd.read_csv(run_dir / "per_group.csv")
    else:
        per_group = summarize_per_group(step_results)
    per_stratum = pd.read_csv(run_dir / "per_stratum.csv")
    return RunData(
        step_results=step_results,
        aggregate=aggregate,
        per_group=per_group,
        per_stratum=per_stratum,
        max_step=int(aggregate["step"].max()),
    )


def rank_table(df: pd.DataFrame, strategies: Sequence[str], group_cols: Sequence[str], metric_cols: Sequence[tuple[str, str, bool]]) -> pd.DataFrame:
    rows = []
    sub = df[df["strategy"].isin(strategies)].copy()
    for key, gf in sub.groupby(list(group_cols)):
        if not isinstance(key, tuple):
            key = (key,)
        row = {col: val for col, val in zip(group_cols, key)}
        rank_sum = {s: 0.0 for s in strategies}
        for metric, label, lower_is_better in metric_cols:
            ordered = gf.sort_values(metric, ascending=lower_is_better)["strategy"].tolist()
            ranks = {s: i + 1 for i, s in enumerate(ordered)}
            row[f"{label}_winner"] = ordered[0]
            for s in strategies:
                row[f"{label}_rank__{s}"] = ranks[s]
                rank_sum[s] += ranks[s]
        for s in strategies:
            row[f"balanced_rank__{s}"] = rank_sum[s] / len(metric_cols)
        row["balanced_winner"] = min(strategies, key=lambda s: row[f"balanced_rank__{s}"])
        rows.append(row)
    return pd.DataFrame(rows)


def wins_table(rank_df: pd.DataFrame, strategies: Sequence[str], metric_labels: Sequence[str]) -> pd.DataFrame:
    rows = []
    for label in metric_labels:
        counts = rank_df[f"{label}_winner"].value_counts()
        row = {"metric": label}
        for s in strategies:
            row[s] = int(counts.get(s, 0))
        rows.append(row)
    counts = rank_df["balanced_winner"].value_counts()
    row = {"metric": "Balanced"}
    for s in strategies:
        row[s] = int(counts.get(s, 0))
    rows.append(row)
    return pd.DataFrame(rows)


def budget_wins_table(df: pd.DataFrame, strategies: Sequence[str], metric_col: str, lower_is_better: bool, budgets: Sequence[int]) -> pd.DataFrame:
    rows = []
    for step in budgets:
        sdf = df[(df["step"] == step) & (df["strategy"].isin(strategies))].copy()
        counts = {s: 0 for s in strategies}
        for _, gf in sdf.groupby("group"):
            ordered = gf.sort_values(metric_col, ascending=lower_is_better)["strategy"].tolist()
            counts[ordered[0]] += 1
        rows.append({"step": step, **counts})
    return pd.DataFrame(rows)


def plot_grouped_counts(ax: plt.Axes, counts_df: pd.DataFrame, strategies: Sequence[str], total: int, title: str) -> None:
    x = np.arange(len(counts_df))
    width = 0.22 if len(strategies) == 3 else 0.15
    offsets = np.linspace(-(len(strategies) - 1) / 2, (len(strategies) - 1) / 2, len(strategies)) * width
    for off, strategy in zip(offsets, strategies):
        vals = counts_df[strategy].to_numpy()
        bars = ax.bar(
            x + off,
            vals,
            width=width,
            color=COLORS[strategy],
            edgecolor="black" if strategy == "bald_laplace" else "white",
            linewidth=2.0 if strategy == "bald_laplace" else 1.0,
            label=LABELS[strategy],
        )
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    val + 0.1,
                    str(int(val)),
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color=COLORS[strategy],
                    fontweight="bold",
                )
    ax.set_xticks(x)
    ax.set_xticklabels(counts_df["metric"])
    ax.set_ylim(0, total + 1.5)
    ax.set_title(title, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)


def plot_budget_wins(
    budget_df: pd.DataFrame,
    strategies: Sequence[str],
    title: str,
    ylabel: str,
    out_path: Path,
    total: int,
) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    x = np.arange(len(budget_df))
    width = 0.24 if len(strategies) == 3 else 0.16
    offsets = np.linspace(-(len(strategies) - 1) / 2, (len(strategies) - 1) / 2, len(strategies)) * width
    for off, strategy in zip(offsets, strategies):
        vals = budget_df[strategy].to_numpy()
        bars = ax.bar(
            x + off,
            vals,
            width=width,
            color=COLORS[strategy],
            edgecolor="black" if strategy == "bald_laplace" else "white",
            linewidth=2.0 if strategy == "bald_laplace" else 1.0,
            label=LABELS[strategy],
        )
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + 0.08,
                str(int(val)),
                ha="center",
                va="bottom",
                fontsize=8,
                color=COLORS[strategy],
                fontweight="bold",
            )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}" if s != max(budget_df['step']) else "full" for s in budget_df["step"]])
    ax.set_ylim(0, total + 0.8)
    ax.set_xlabel("Prompt budget step")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=len(strategies), loc="upper center", bbox_to_anchor=(0.5, 1.18))
    savefig(fig, out_path)


def plot_heatmap(fig: plt.Figure, ax: plt.Axes, matrix: pd.DataFrame, title: str, cbar_label: str, fmt: str = "{:+.1f}") -> None:
    data = matrix.to_numpy(dtype=float)
    vmax = float(np.nanpercentile(np.abs(data), 90)) if data.size else 1.0
    vmax = max(vmax, 1.0)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    im = ax.imshow(data, cmap="RdBu_r", norm=norm, aspect="auto")
    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=22, ha="right")
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, fmt.format(data[i, j]), ha="center", va="center", fontsize=8, color="black")
    ax.set_title(title, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_label(cbar_label)


def plot_categorical_winner_map(
    winner_df: pd.DataFrame,
    title: str,
    out_path: Path,
    strategies: Sequence[str] = TRIO,
) -> None:
    rows = winner_df.copy()
    strategy_to_code = {s: i for i, s in enumerate(strategies)}
    metric_cols = ["IoU_winner", "ECE_winner", "HD95_winner", "NLL_winner", "Balanced_winner"]
    display_cols = ["IoU", "ECE", "HD95", "NLL", "Balanced"]
    matrix = rows[metric_cols].apply(lambda col: col.map(strategy_to_code)).to_numpy(dtype=float)
    cmap = ListedColormap([COLORS[s] for s in strategies])
    norm = BoundaryNorm(np.arange(-0.5, len(strategies) + 0.5, 1), cmap.N)

    fig_h = max(5.0, 0.42 * len(rows) + 2.5)
    fig, ax = plt.subplots(figsize=(9.5, fig_h))
    im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(np.arange(len(display_cols)))
    ax.set_xticklabels(display_cols)
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(rows["pretty_name"])
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            winner = rows.iloc[i][metric_cols[j]]
            text = "BALD" if winner == "bald_laplace" else ("Entropy" if winner == "entropy" else "MCD")
            ax.text(j, i, text, ha="center", va="center", fontsize=8, color="white", fontweight="bold")
    ax.set_title(title, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, ticks=np.arange(len(strategies)), shrink=0.9)
    cbar.ax.set_yticklabels([LABELS[s] for s in strategies])
    savefig(fig, out_path)


def plot_dominance_matrix_legend(out_path: Path, strategies: Sequence[str] = TRIO, include_tie: bool = False) -> None:
    items = list(strategies) + (["tie"] if include_tie else [])
    handles = [
        Patch(
            facecolor=COLORS[s],
            edgecolor="black" if s in {"bald_laplace", "tie"} else "white",
            linewidth=1.8 if s == "bald_laplace" else 1.0,
            label=f"{SHORT_LABELS[s]} = {'Tie for top spot' if s == 'tie' else LABELS[s]}",
        )
        for s in items
    ]

    ncols = 2 if len(handles) > 4 else len(handles)
    nrows = ceil(len(handles) / max(ncols, 1))
    fig_h = 1.8 if nrows == 1 else 2.6
    fig_w = 10.8 if nrows == 1 else 12.6
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax_leg = fig.add_axes([0.03, 0.10, 0.66 if nrows == 1 else 0.72, 0.82])
    ax_leg.set_axis_off()
    ax_leg.legend(
        handles=handles,
        loc="center left",
        frameon=False,
        ncol=ncols,
        fontsize=13 if nrows == 1 else 12,
        handlelength=1.6,
        handleheight=1.2,
        columnspacing=1.6,
        labelspacing=1.0,
    )

    grad = np.linspace(0.2, 1.0, 256).reshape(1, -1)
    grad_ax = fig.add_axes([0.75 if nrows == 1 else 0.76, 0.34, 0.21, 0.28])
    grad_ax.imshow(grad, cmap="Reds", aspect="auto")
    grad_ax.set_xticks([])
    grad_ax.set_yticks([])
    for spine in grad_ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_edgecolor("black")
    fig.text(0.855, 0.70, "Darker = larger lead", ha="center", va="bottom", fontsize=12, fontweight="bold")
    fig.text(0.75 if nrows == 1 else 0.76, 0.23, "smaller", ha="left", va="top", fontsize=10)
    fig.text(0.96, 0.23, "larger", ha="right", va="top", fontsize=10)
    savefig(fig, out_path)


def plot_curve_legend(out_path: Path, strategies: Sequence[str] = STRATEGIES) -> None:
    handles = [
        Line2D(
            [0],
            [0],
            color=COLORS[s],
            linewidth=LINEWIDTHS[s],
            linestyle=LINESTYLES[s],
            marker=MARKERS[s],
            markersize=8 if s == "bald_laplace" else 6.5,
            markerfacecolor=COLORS[s],
            markeredgecolor="black" if s == "bald_laplace" else "white",
            markeredgewidth=1.2 if s == "bald_laplace" else 0.8,
            label=LABELS[s],
        )
        for s in strategies
    ]
    fig, ax = plt.subplots(figsize=(11.5, 1.8))
    ax.axis("off")
    ax.legend(
        handles=handles,
        loc="center",
        frameon=False,
        ncol=3,
        fontsize=13,
        handlelength=2.3,
        columnspacing=1.7,
        labelspacing=1.0,
    )
    savefig(fig, out_path)


def plot_strategy_gain_bars(
    values: pd.Series,
    yerrs: pd.Series | None,
    ylabel: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10.2, 5.8))
    x = np.arange(len(values))
    err = None if yerrs is None else np.asarray(yerrs, dtype=float)
    bars = ax.bar(
        x,
        values.to_numpy(dtype=float),
        yerr=err,
        color=[COLORS[s] for s in values.index],
        edgecolor="white",
        error_kw={"elinewidth": 1.5, "ecolor": "black", "capsize": 4, "capthick": 1.5},
    )
    vals_np = values.to_numpy(dtype=float)
    errs_np = np.zeros_like(vals_np) if err is None else err
    span = float(np.max(np.abs(vals_np) + errs_np)) if len(vals_np) else 1.0
    text_pad = max(0.02 * span, 0.25)
    for bar, val, strategy, err_val in zip(bars, vals_np, values.index, errs_np):
        anchor = val + err_val
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            anchor + (text_pad if val >= 0 else -text_pad),
            f"{val:+.1f}",
            ha="center",
            va="bottom" if val >= 0 else "top",
            fontsize=10,
            color=COLORS[strategy],
        )
    ax.axhline(0, color="black", lw=1.0, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[s] for s in values.index], rotation=20, ha="right")
    ax.set_ylabel(ylabel, fontsize=CURVE_LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=CURVE_TICK_FONTSIZE)
    ax.grid(axis="y", alpha=0.25)
    savefig(fig, out_path)


def unique_winner_from_metric(values: pd.Series, lower_is_better: bool) -> tuple[str, float, list[str]]:
    ordered = values.sort_values(ascending=lower_is_better)
    best_value = float(ordered.iloc[0])
    top = [idx for idx, value in ordered.items() if np.isclose(value, best_value)]
    if len(top) > 1:
        return "tie", 0.0, top
    remaining = ordered.iloc[1:]
    if remaining.empty:
        margin = 0.0
    else:
        runner = float(remaining.iloc[0])
        margin = (runner - best_value) if lower_is_better else (best_value - runner)
    return top[0], float(margin), top


def unique_winner_from_counts(counts: dict[str, int], strategies: Sequence[str]) -> tuple[str, int, int, list[str]]:
    ordered = sorted(((strategy, int(counts.get(strategy, 0))) for strategy in strategies), key=lambda kv: (-kv[1], kv[0]))
    top_count = ordered[0][1]
    top = [strategy for strategy, count in ordered if count == top_count]
    if len(top) > 1:
        return "tie", top_count, 0, top
    runner_up = ordered[1][1] if len(ordered) > 1 else top_count
    return top[0], top_count, top_count - runner_up, top


def tie_text(strategies: Sequence[str]) -> str:
    if len(strategies) <= 2:
        return "/".join(SHORT_LABELS[s] for s in strategies)
    return f"{len(strategies)}-way"


def build_budget_timeline_df(baseline: RunData, strategies: Sequence[str]) -> pd.DataFrame:
    rows = []
    for step in KEY_BUDGETS:
        row = {
            "step": step,
            "pretty_step": "Full" if step == baseline.max_step else f"Step {step}",
            "bald_total_wins": 0,
        }
        sdf = baseline.per_group[
            (baseline.per_group["step"] == step) &
            (baseline.per_group["strategy"].isin(strategies))
        ].copy()
        for metric_col, label, lower_is_better in [
            ("iou_mean", "IoU", False),
            ("ece_mean", "ECE", True),
            ("hausdorff95_mean", "HD95", True),
            ("nll_mean", "NLL", True),
        ]:
            counts = {s: 0 for s in strategies}
            for _, gf in sdf.groupby("group"):
                winner, _, _ = unique_winner_from_metric(gf.set_index("strategy")[metric_col], lower_is_better)
                if winner != "tie":
                    counts[winner] += 1
            winner, top_count, margin, top = unique_winner_from_counts(counts, strategies)
            row[f"{label}_winner"] = winner
            row[f"{label}_count"] = top_count
            row[f"{label}_margin"] = margin
            row[f"{label}_tie"] = tie_text(top) if winner == "tie" else ""
            row["bald_total_wins"] += counts.get("bald_laplace", 0)
        rows.append(row)
    return pd.DataFrame(rows)


def build_budget_wins_long(baseline: RunData, strategies: Sequence[str]) -> pd.DataFrame:
    rows = []
    metric_specs = [
        ("iou_mean", "IoU", False),
        ("ece_mean", "ECE", True),
        ("hausdorff95_mean", "HD95", True),
        ("nll_mean", "NLL", True),
    ]
    for step in KEY_BUDGETS:
        pretty_step = "Full" if step == baseline.max_step else f"Step {step}"
        sdf = baseline.per_group[
            (baseline.per_group["step"] == step) &
            (baseline.per_group["strategy"].isin(strategies))
        ].copy()
        metric_totals = {s: 0.0 for s in strategies}
        for metric_col, label, lower_is_better in metric_specs:
            counts = {s: 0.0 for s in strategies}
            for _, gf in sdf.groupby("group"):
                winner, _, top = unique_winner_from_metric(gf.set_index("strategy")[metric_col], lower_is_better)
                if winner == "tie":
                    share = 1.0 / len(top)
                    for strategy in top:
                        counts[strategy] += share
                else:
                    counts[winner] += 1.0
            for strategy in strategies:
                metric_totals[strategy] += counts[strategy]
                rows.append({
                    "step": step,
                    "pretty_step": pretty_step,
                    "panel": label,
                    "strategy": strategy,
                    "wins": counts[strategy],
                })
        for strategy in strategies:
            rows.append({
                "step": step,
                "pretty_step": pretty_step,
                "panel": "Total",
                "strategy": strategy,
                "wins": metric_totals[strategy],
            })
    return pd.DataFrame(rows)


def plot_budget_takeover_timeline(
    budget_df: pd.DataFrame,
    out_path: Path,
    title_line: str,
    subtitle_line: str,
) -> None:
    step_labels = budget_df["pretty_step"].tolist()
    metric_labels = ["IoU", "ECE", "HD95", "NLL"]
    row_labels = metric_labels + ["BALD total"]
    n_rows = len(row_labels)
    n_cols = len(step_labels)

    fig, ax = plt.subplots(figsize=(12.2, 5.8))
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.invert_yaxis()

    for j in range(n_cols):
        if j % 2 == 1:
            ax.add_patch(Rectangle((j, 0), 1, n_rows, facecolor="#f7f3ee", edgecolor="none", zorder=0))

    for j, row in budget_df.iterrows():
        for i, label in enumerate(metric_labels):
            winner = row[f"{label}_winner"]
            face = COLORS[winner]
            ax.add_patch(
                Rectangle(
                    (j, i),
                    1,
                    1,
                    facecolor=face,
                    alpha=0.90,
                    edgecolor="black" if winner in {"bald_laplace", "tie"} else "white",
                    linewidth=2.0 if winner == "bald_laplace" else 1.2,
                )
            )
            if winner == "tie":
                ax.text(
                    j + 0.5,
                    i + 0.34,
                    "TIE",
                    ha="center",
                    va="center",
                    fontsize=10,
                    fontweight="bold",
                    color="white",
                )
                ax.text(
                    j + 0.5,
                    i + 0.58,
                    row[f"{label}_tie"],
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    fontweight="bold",
                    color="white",
                )
                ax.text(
                    j + 0.5,
                    i + 0.80,
                    f"{int(row[f'{label}_count'])}/5",
                    ha="center",
                    va="center",
                    fontsize=9,
                    fontweight="bold",
                    color="white",
                )
            else:
                ax.text(
                    j + 0.5,
                    i + 0.40,
                    SHORT_LABELS[winner],
                    ha="center",
                    va="center",
                    fontsize=11,
                    fontweight="bold",
                    color="white",
                )
                ax.text(
                    j + 0.5,
                    i + 0.73,
                    f"{int(row[f'{label}_count'])}/5",
                    ha="center",
                    va="center",
                    fontsize=10,
                    fontweight="bold",
                    color="white",
                )

        frac = row["bald_total_wins"] / 20.0
        count_color = plt.cm.Reds(0.18 + 0.72 * frac)
        ax.add_patch(Rectangle((j, 4), 1, 1, facecolor=count_color, edgecolor="white", linewidth=1.2))
        ax.text(
            j + 0.5,
            4.5,
            f"{int(row['bald_total_wins'])}/20",
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            color="white" if frac >= 0.55 else "#611919",
        )

    for x in range(n_cols + 1):
        ax.axvline(x, color="white", lw=1.8, zorder=5)
    for y in range(n_rows + 1):
        ax.axhline(y, color="white", lw=1.2, zorder=5)

    ax.set_xticks(np.arange(n_cols) + 0.5)
    ax.set_xticklabels(step_labels, fontweight="bold")
    ax.tick_params(axis="x", top=True, bottom=False, labeltop=True, labelbottom=False, length=0)
    ax.set_yticks(np.arange(n_rows) + 0.5)
    ax.set_yticklabels(row_labels, fontweight="bold")
    ax.tick_params(axis="y", length=0)
    ax.set_title(f"{title_line}\n{subtitle_line}", fontweight="bold", pad=28)
    ax.annotate(
        "More prompts",
        xy=(n_cols - 0.1, -0.22),
        xytext=(0.1, -0.22),
        xycoords=("data", "data"),
        textcoords=("data", "data"),
        arrowprops=dict(arrowstyle="->", lw=1.7, color="#444"),
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color="#444",
        annotation_clip=False,
    )
    savefig(fig, out_path)


def plot_budget_bump_chart(
    wins_long: pd.DataFrame,
    strategies: Sequence[str],
    out_path: Path,
    title_line: str,
    subtitle_line: str,
) -> None:
    panel_order = ["IoU", "ECE", "HD95", "NLL", "Total"]
    step_order = list(KEY_BUDGETS)
    step_labels = [
        wins_long[wins_long["step"] == step]["pretty_step"].iloc[0]
        for step in step_order
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.2))
    axes = axes.flatten()
    legend_ax = axes[-1]

    for ax, panel in zip(axes[:-1], panel_order):
        sub = wins_long[wins_long["panel"] == panel].copy()
        ymax = 20 if panel == "Total" else 5
        for strategy in strategies:
            ssub = sub[sub["strategy"] == strategy].sort_values("step")
            ax.plot(
                ssub["step"],
                ssub["wins"],
                color=COLORS[strategy],
                linewidth=3.6 if strategy == "bald_laplace" else 2.1,
                marker=MARKERS.get(strategy, "o"),
                markersize=7 if strategy == "bald_laplace" else 5.5,
                markerfacecolor=COLORS[strategy],
                markeredgecolor="black" if strategy == "bald_laplace" else "white",
                markeredgewidth=1.2 if strategy == "bald_laplace" else 0.8,
                alpha=0.98 if strategy == "bald_laplace" else 0.92,
                label=LABELS[strategy],
                zorder=4 if strategy == "bald_laplace" else 3,
            )
        ax.set_title(panel, fontweight="bold")
        ax.set_xticks(step_order)
        ax.set_xticklabels(step_labels)
        ax.set_ylim(0, ymax)
        ax.set_xlim(min(step_order), max(step_order))
        ax.grid(axis="y", alpha=0.25)
        ax.grid(axis="x", alpha=0.10)
        if panel == "Total":
            ax.set_ylabel("Wins across 4 metrics")
            ax.set_yticks(np.arange(0, ymax + 0.1, 4))
        else:
            ax.set_ylabel("Dataset wins")
            ax.set_yticks(np.arange(0, ymax + 0.1, 1))

    handles, labels = axes[0].get_legend_handles_labels()
    legend_ax.axis("off")
    legend_ax.legend(
        handles,
        labels,
        loc="center",
        frameon=False,
        ncol=2 if len(strategies) > 4 else 1,
        fontsize=11,
        columnspacing=1.5,
        handlelength=2.2,
        labelspacing=1.0,
    )
    legend_ax.text(
        0.5,
        0.12,
        "Higher lines mean more datasets won at that budget.",
        ha="center",
        va="center",
        fontsize=10,
        color="#444",
    )
    fig.suptitle(f"{title_line}\n{subtitle_line}", fontweight="bold", y=0.98)
    savefig(fig, out_path)


def plot_strata_dominance_matrix(baseline: RunData, out_path: Path) -> None:
    final = baseline.per_stratum[
        (baseline.per_stratum["step"] == baseline.max_step) &
        (baseline.per_stratum["strategy"].isin(TRIO))
    ].copy()
    metric_specs = [
        ("iou_mean", "IoU", False),
        ("ece_mean", "ECE", True),
        ("hausdorff95_mean", "HD95", True),
        ("nll_mean", "NLL", True),
    ]
    short_labels = {"bald_laplace": "BALD", "entropy": "ENT", "bald_mcdropout": "MCD"}
    family_order = {
        "boundary_complexity_bin": 0,
        "contrast_bin": 1,
        "thinness_bin": 2,
        "object_size_bin": 3,
    }

    rows = []
    for (stratum_type, stratum_value), gf in final.groupby(["stratum_type", "stratum_value"]):
        vals = gf.set_index("strategy")
        row = {
            "stratum_type": stratum_type,
            "stratum_value": stratum_value,
            "pretty_name": f"{stratum_type.replace('_bin', '').replace('_', ' ').title()}: {stratum_value.title()}",
        }
        rank_sum = {s: 0 for s in TRIO}
        bald_win_count = 0
        for metric_col, label, lower_is_better in metric_specs:
            series = vals[metric_col]
            ordered = series.sort_values(ascending=lower_is_better).index.tolist()
            winner = ordered[0]
            row[f"{label}_winner"] = winner
            rival_series = series.drop("bald_laplace")
            rival_best = rival_series.min() if lower_is_better else rival_series.max()
            row[f"{label}_edge"] = (rival_best - series["bald_laplace"]) if lower_is_better else (series["bald_laplace"] - rival_best)
            for rank, strategy in enumerate(ordered, start=1):
                rank_sum[strategy] += rank
            bald_win_count += int(winner == "bald_laplace")
        balanced_winner = min(TRIO, key=lambda s: rank_sum[s])
        row["Balanced_winner"] = balanced_winner
        row["bald_win_count"] = bald_win_count + int(balanced_winner == "bald_laplace")
        rows.append(row)

    dominance = pd.DataFrame(rows)
    dominance["balanced_is_bald"] = (dominance["Balanced_winner"] == "bald_laplace").astype(int)
    dominance["family_order"] = dominance["stratum_type"].map(family_order).fillna(99)
    dominance = dominance.sort_values(
        ["balanced_is_bald", "bald_win_count", "family_order", "pretty_name"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)

    n_rows = len(dominance)
    n_cols = 6
    fig_h = max(7.6, 0.54 * n_rows + 2.6)
    fig, ax = plt.subplots(figsize=(11.8, fig_h))
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.invert_yaxis()

    for i, row in dominance.iterrows():
        if i % 2 == 1:
            ax.add_patch(Rectangle((0, i), n_cols, 1, facecolor="#f7f3ee", edgecolor="none", zorder=0))
        for j, (_, label, _) in enumerate(metric_specs):
            winner = row[f"{label}_winner"]
            ax.add_patch(
                Rectangle(
                    (j, i),
                    1,
                    1,
                    facecolor=COLORS[winner],
                    alpha=0.88,
                    edgecolor="black" if winner == "bald_laplace" else "white",
                    linewidth=1.9 if winner == "bald_laplace" else 1.2,
                )
            )
            ax.text(
                j + 0.5,
                i + 0.5,
                short_labels[winner],
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                color="white",
            )

        balanced_winner = row["Balanced_winner"]
        ax.add_patch(
            Rectangle(
                (4, i),
                1,
                1,
                facecolor=COLORS[balanced_winner],
                alpha=0.92,
                edgecolor="black" if balanced_winner == "bald_laplace" else "white",
                linewidth=2.1 if balanced_winner == "bald_laplace" else 1.2,
            )
        )
        ax.text(
            4.5,
            i + 0.5,
            short_labels[balanced_winner],
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="white",
        )

        frac = row["bald_win_count"] / 5.0
        count_color = plt.cm.Reds(0.18 + 0.72 * frac)
        ax.add_patch(Rectangle((5, i), 1, 1, facecolor=count_color, edgecolor="white", linewidth=1.2))
        ax.text(
            5.5,
            i + 0.5,
            f"{int(row['bald_win_count'])}/5",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="white" if frac >= 0.55 else "#611919",
        )

    for x in range(n_cols + 1):
        ax.axvline(x, color="white", lw=1.2, zorder=5)
    for y in range(n_rows + 1):
        ax.axhline(y, color="white", lw=1.0, zorder=5)

    ax.set_xticks(np.arange(n_cols) + 0.5)
    ax.set_xticklabels(["IoU", "ECE", "HD95", "NLL", "Balanced", "BALD wins"], fontweight="bold", fontsize=MATRIX_HEADER_FONTSIZE)
    ax.tick_params(axis="x", top=True, bottom=False, labeltop=True, labelbottom=False, length=0, pad=6)
    ax.set_yticks(np.arange(n_rows) + 0.5)
    ax.set_yticklabels(dominance["pretty_name"], fontsize=MATRIX_TICK_FONTSIZE)
    ax.tick_params(axis="y", length=0)
    savefig(fig, out_path)


def plot_budget_dominance_matrix(tables: dict[str, pd.DataFrame], out_path: Path) -> None:
    metric_files = [
        ("victory_baseline_budget_wins_iou.csv", "IoU"),
        ("victory_baseline_budget_wins_ece.csv", "ECE"),
        ("victory_baseline_budget_wins_hd95.csv", "HD95"),
        ("victory_baseline_budget_wins_nll.csv", "NLL"),
    ]
    short_labels = {"bald_laplace": "BALD", "entropy": "ENT", "bald_mcdropout": "MCD", "tie": "TIE"}
    rows = []
    for step in KEY_BUDGETS:
        row = {
            "step": step,
            "pretty_step": "Full" if step == max(KEY_BUDGETS) else f"Step {step}",
            "bald_total_wins": 0,
        }
        for file_name, label in metric_files:
            sub = tables[file_name]
            counts = sub[sub["step"] == step][TRIO].iloc[0].to_dict()
            ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
            top_count = ordered[0][1]
            top_strategies = [strategy for strategy, count in ordered if count == top_count]
            winner = top_strategies[0] if len(top_strategies) == 1 else "tie"
            runner_up = ordered[len(top_strategies)][1] if len(top_strategies) < len(ordered) else top_count
            margin = top_count - runner_up
            row[f"{label}_winner"] = winner
            row[f"{label}_count"] = top_count
            row[f"{label}_margin"] = margin
            row["bald_total_wins"] += int(counts["bald_laplace"])
        rows.append(row)

    budget_df = pd.DataFrame(rows)
    margin_scale = max(float(budget_df[[f"{label}_margin" for _, label in metric_files]].to_numpy().max()), 1.0)

    n_rows = len(budget_df)
    n_cols = 5
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.invert_yaxis()

    for i, row in budget_df.iterrows():
        if i % 2 == 1:
            ax.add_patch(Rectangle((0, i), n_cols, 1, facecolor="#f7f3ee", edgecolor="none", zorder=0))
        for j, (_, label) in enumerate(metric_files):
            winner = row[f"{label}_winner"]
            margin = row[f"{label}_margin"]
            alpha = 0.28 + 0.62 * min(1.0, float(margin) / margin_scale)
            ax.add_patch(
                Rectangle(
                    (j, i),
                    1,
                    1,
                    facecolor=COLORS[winner],
                    alpha=alpha,
                    edgecolor="black" if winner == "bald_laplace" else "white",
                    linewidth=1.9 if winner == "bald_laplace" else 1.2,
                )
            )
            ax.text(
                j + 0.5,
                i + 0.44,
                short_labels[winner],
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                color="white",
            )
            ax.text(
                j + 0.5,
                i + 0.72,
                f"{int(row[f'{label}_count'])}/5",
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                color="white",
            )

        frac = row["bald_total_wins"] / 20.0
        count_color = plt.cm.Reds(0.18 + 0.72 * frac)
        ax.add_patch(Rectangle((4, i), 1, 1, facecolor=count_color, edgecolor="white", linewidth=1.2))
        ax.text(
            4.5,
            i + 0.5,
            f"{int(row['bald_total_wins'])}/20",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="white" if frac >= 0.55 else "#611919",
        )

    for x in range(n_cols + 1):
        ax.axvline(x, color="white", lw=1.2, zorder=5)
    for y in range(n_rows + 1):
        ax.axhline(y, color="white", lw=1.0, zorder=5)

    ax.set_xticks(np.arange(n_cols) + 0.5)
    ax.set_xticklabels(["IoU", "ECE", "HD95", "NLL", "BALD total"], fontweight="bold", fontsize=MATRIX_HEADER_FONTSIZE)
    ax.tick_params(axis="x", top=True, bottom=False, labeltop=True, labelbottom=False, length=0, pad=6)
    ax.set_yticks(np.arange(n_rows) + 0.5)
    ax.set_yticklabels(budget_df["pretty_step"], fontsize=MATRIX_TICK_FONTSIZE)
    ax.tick_params(axis="y", length=0)
    savefig(fig, out_path)


def plot_all_strategies_budget_dominance_matrix(
    baseline: RunData,
    out_path: Path,
    strategies: Sequence[str] = STRATEGIES,
    title_line: str = "Baseline SAM: budget dominance across all strategies",
    subtitle_line: str = "Winner among BALD / Entropy / GT-based / MC Dropout / Random / MaxDist",
) -> None:
    rows = []
    for step in KEY_BUDGETS:
        row = {
            "step": step,
            "pretty_step": "Full" if step == baseline.max_step else f"Step {step}",
            "bald_total_wins": 0,
        }
        sdf = baseline.per_group[(baseline.per_group["step"] == step) & (baseline.per_group["strategy"].isin(strategies))].copy()
        for metric_col, label, lower_is_better in [("iou_mean", "IoU", False), ("ece_mean", "ECE", True), ("hausdorff95_mean", "HD95", True), ("nll_mean", "NLL", True)]:
            counts = {s: 0 for s in strategies}
            for _, gf in sdf.groupby("group"):
                winner, _, _ = unique_winner_from_metric(gf.set_index("strategy")[metric_col], lower_is_better)
                if winner != "tie":
                    counts[winner] += 1
            winner, top_count, margin, top = unique_winner_from_counts(counts, strategies)
            row[f"{label}_winner"] = winner
            row[f"{label}_count"] = top_count
            row[f"{label}_margin"] = margin
            row[f"{label}_tie"] = tie_text(top) if winner == "tie" else ""
            row["bald_total_wins"] += counts["bald_laplace"]
        rows.append(row)

    budget_df = pd.DataFrame(rows)
    margin_scale = max(float(budget_df[[f"{label}_margin" for label in ["IoU", "ECE", "HD95", "NLL"]]].to_numpy().max()), 1.0)

    n_rows = len(budget_df)
    n_cols = 5
    fig, ax = plt.subplots(figsize=(9.4, 5.2))
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.invert_yaxis()

    for i, row in budget_df.iterrows():
        if i % 2 == 1:
            ax.add_patch(Rectangle((0, i), n_cols, 1, facecolor="#f7f3ee", edgecolor="none", zorder=0))
        for j, label in enumerate(["IoU", "ECE", "HD95", "NLL"]):
            winner = row[f"{label}_winner"]
            margin = row[f"{label}_margin"]
            alpha = 0.80 if winner == "tie" else 0.28 + 0.62 * min(1.0, float(margin) / margin_scale)
            ax.add_patch(
                Rectangle(
                    (j, i),
                    1,
                    1,
                    facecolor=COLORS[winner],
                    alpha=alpha,
                    edgecolor="black" if winner in {"bald_laplace", "tie"} else "white",
                    linewidth=1.9 if winner == "bald_laplace" else 1.3,
                )
            )
            if winner == "tie":
                ax.text(j + 0.5, i + 0.30, "TIE", ha="center", va="center", fontsize=9, fontweight="bold", color="white")
                ax.text(j + 0.5, i + 0.54, row[f"{label}_tie"], ha="center", va="center", fontsize=7.5, fontweight="bold", color="white")
                ax.text(j + 0.5, i + 0.78, f"{int(row[f'{label}_count'])}/5", ha="center", va="center", fontsize=8.5, fontweight="bold", color="white")
            else:
                ax.text(j + 0.5, i + 0.44, SHORT_LABELS[winner], ha="center", va="center", fontsize=9, fontweight="bold", color="white")
                ax.text(j + 0.5, i + 0.72, f"{int(row[f'{label}_count'])}/5", ha="center", va="center", fontsize=9, fontweight="bold", color="white")

        frac = row["bald_total_wins"] / 20.0
        count_color = plt.cm.Reds(0.18 + 0.72 * frac)
        ax.add_patch(Rectangle((4, i), 1, 1, facecolor=count_color, edgecolor="white", linewidth=1.2))
        ax.text(
            4.5,
            i + 0.5,
            f"{int(row['bald_total_wins'])}/20",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="white" if frac >= 0.55 else "#611919",
        )

    for x in range(n_cols + 1):
        ax.axvline(x, color="white", lw=1.2, zorder=5)
    for y in range(n_rows + 1):
        ax.axhline(y, color="white", lw=1.0, zorder=5)

    ax.set_xticks(np.arange(n_cols) + 0.5)
    ax.set_xticklabels(["IoU", "ECE", "HD95", "NLL", "BALD wins"], fontweight="bold", fontsize=MATRIX_HEADER_FONTSIZE)
    ax.tick_params(axis="x", top=True, bottom=False, labeltop=True, labelbottom=False, length=0, pad=6)
    ax.set_yticks(np.arange(n_rows) + 0.5)
    ax.set_yticklabels(budget_df["pretty_step"], fontsize=MATRIX_TICK_FONTSIZE)
    ax.tick_params(axis="y", length=0)
    savefig(fig, out_path)


def plot_all_strategies_strata_dominance_matrix(
    baseline: RunData,
    out_path: Path,
    strategies: Sequence[str] = STRATEGIES,
    title_line: str = "Baseline SAM: strata dominance across all strategies",
) -> None:
    final = baseline.per_stratum[
        (baseline.per_stratum["step"] == baseline.max_step) &
        (baseline.per_stratum["strategy"].isin(strategies))
    ].copy()
    metric_specs = [
        ("iou_mean", "IoU", False),
        ("ece_mean", "ECE", True),
        ("hausdorff95_mean", "HD95", True),
        ("nll_mean", "NLL", True),
    ]
    family_order = {
        "boundary_complexity_bin": 0,
        "contrast_bin": 1,
        "thinness_bin": 2,
        "object_size_bin": 3,
    }

    rows = []
    for (stratum_type, stratum_value), gf in final.groupby(["stratum_type", "stratum_value"]):
        vals = gf.set_index("strategy")
        row = {
            "stratum_type": stratum_type,
            "stratum_value": stratum_value,
            "pretty_name": f"{stratum_type.replace('_bin', '').replace('_', ' ').title()}: {stratum_value.title()}",
        }
        rank_sum = {s: 0 for s in strategies}
        bald_win_count = 0
        for metric_col, label, lower_is_better in metric_specs:
            winner, margin, top = unique_winner_from_metric(vals[metric_col], lower_is_better)
            row[f"{label}_winner"] = winner
            row[f"{label}_margin"] = margin
            row[f"{label}_tie"] = tie_text(top) if winner == "tie" else ""
            ordered = vals[metric_col].sort_values(ascending=lower_is_better).index.tolist()
            for rank, strategy in enumerate(ordered, start=1):
                rank_sum[strategy] += rank
            bald_win_count += int(winner == "bald_laplace")

        min_rank = min(rank_sum.values())
        top_balanced = [s for s, value in rank_sum.items() if value == min_rank]
        if len(top_balanced) > 1:
            balanced_winner = "tie"
            balanced_margin = 0.0
        else:
            balanced_winner = top_balanced[0]
            second = min(value for s, value in rank_sum.items() if s != balanced_winner)
            balanced_margin = second - min_rank
        row["Balanced_winner"] = balanced_winner
        row["Balanced_margin"] = balanced_margin
        row["Balanced_tie"] = tie_text(top_balanced) if balanced_winner == "tie" else ""
        row["bald_win_count"] = bald_win_count + int(balanced_winner == "bald_laplace")
        rows.append(row)

    dominance = pd.DataFrame(rows)
    dominance["balanced_is_bald"] = (dominance["Balanced_winner"] == "bald_laplace").astype(int)
    dominance["family_order"] = dominance["stratum_type"].map(family_order).fillna(99)
    dominance = dominance.sort_values(
        ["balanced_is_bald", "bald_win_count", "family_order", "pretty_name"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)

    n_rows = len(dominance)
    n_cols = 6
    fig_h = max(7.6, 0.54 * n_rows + 2.6)
    fig, ax = plt.subplots(figsize=(11.8, fig_h))
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.invert_yaxis()

    for i, row in dominance.iterrows():
        if i % 2 == 1:
            ax.add_patch(Rectangle((0, i), n_cols, 1, facecolor="#f7f3ee", edgecolor="none", zorder=0))
        for j, label in enumerate(["IoU", "ECE", "HD95", "NLL", "Balanced"]):
            winner = row[f"{label}_winner"]
            ax.add_patch(
                Rectangle(
                    (j, i),
                    1,
                    1,
                    facecolor=COLORS[winner],
                    alpha=0.88,
                    edgecolor="black" if winner in {"bald_laplace", "tie"} else "white",
                    linewidth=1.9 if winner == "bald_laplace" else 1.2,
                )
            )
            if winner == "tie":
                ax.text(j + 0.5, i + 0.35, "TIE", ha="center", va="center", fontsize=8.5, fontweight="bold", color="white")
                ax.text(j + 0.5, i + 0.64, row[f"{label}_tie"], ha="center", va="center", fontsize=7.2, fontweight="bold", color="white")
            else:
                ax.text(j + 0.5, i + 0.5, SHORT_LABELS[winner], ha="center", va="center", fontsize=9, fontweight="bold", color="white")

        frac = row["bald_win_count"] / 5.0
        count_color = plt.cm.Reds(0.18 + 0.72 * frac)
        ax.add_patch(Rectangle((5, i), 1, 1, facecolor=count_color, edgecolor="white", linewidth=1.2))
        ax.text(
            5.5,
            i + 0.5,
            f"{int(row['bald_win_count'])}/5",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="white" if frac >= 0.55 else "#611919",
        )

    for x in range(n_cols + 1):
        ax.axvline(x, color="white", lw=1.2, zorder=5)
    for y in range(n_rows + 1):
        ax.axhline(y, color="white", lw=1.0, zorder=5)

    ax.set_xticks(np.arange(n_cols) + 0.5)
    ax.set_xticklabels(["IoU", "ECE", "HD95", "NLL", "Balanced", "BALD wins"], fontweight="bold", fontsize=MATRIX_HEADER_FONTSIZE)
    ax.tick_params(axis="x", top=True, bottom=False, labeltop=True, labelbottom=False, length=0, pad=6)
    ax.set_yticks(np.arange(n_rows) + 0.5)
    ax.set_yticklabels(dominance["pretty_name"], fontsize=MATRIX_TICK_FONTSIZE)
    ax.tick_params(axis="y", length=0)
    savefig(fig, out_path)


def plot_metric_curve(
    df: pd.DataFrame,
    metric_col: str,
    ylabel: str,
    title: str,
    out_path: Path,
    strategies: Sequence[str] = STRATEGIES,
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for strategy in strategies:
        sub = df[df["strategy"] == strategy].sort_values("step")
        ax.plot(
            sub["step"],
            sub[metric_col],
            color=COLORS[strategy],
            linewidth=LINEWIDTHS[strategy],
            linestyle=LINESTYLES[strategy],
            marker=MARKERS[strategy],
            markersize=4,
            markevery=2,
        )
    ax.set_xlabel("Prompting Step", fontsize=CURVE_LABEL_FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=CURVE_LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=CURVE_TICK_FONTSIZE)
    ax.grid(alpha=0.25)
    savefig(fig, out_path)


def plot_transition_delta_curve(
    merged: pd.DataFrame,
    metric_col: str,
    ylabel: str,
    lower_is_better: bool,
    scale: float,
    out_path: Path,
    strategies: Sequence[str] = STRATEGIES,
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for strategy in strategies:
        sub = merged[merged["strategy"] == strategy].sort_values("step").copy()
        if lower_is_better:
            delta = scale * (sub[f"{metric_col}_baseline"] - sub[f"{metric_col}_ftcombined"])
        else:
            delta = scale * (sub[f"{metric_col}_ftcombined"] - sub[f"{metric_col}_baseline"])
        ax.plot(
            sub["step"],
            delta,
            color=COLORS[strategy],
            linewidth=LINEWIDTHS[strategy],
            linestyle=LINESTYLES[strategy],
            marker=MARKERS[strategy],
            markersize=4,
            markevery=2,
        )
    ax.axhline(0, color="black", lw=1.0, alpha=0.55)
    ax.set_xlabel("Prompting Step", fontsize=CURVE_LABEL_FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=CURVE_LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=CURVE_TICK_FONTSIZE)
    ax.grid(alpha=0.25)
    savefig(fig, out_path)


def plot_metric_curve_per_dataset(
    per_group: pd.DataFrame,
    metric_col: str,
    ylabel: str,
    title: str,
    out_path: Path,
    strategies: Sequence[str] = STRATEGIES,
) -> None:
    groups = per_group["group"].drop_duplicates().tolist()
    fig, axes = plt.subplots(1, len(groups), figsize=(4.8 * len(groups), 4.8), sharex=True, sharey=True)
    if len(groups) == 1:
        axes = [axes]
    for ax, group in zip(axes, groups):
        gdf = per_group[per_group["group"] == group]
        for strategy in strategies:
            sub = gdf[gdf["strategy"] == strategy].sort_values("step")
            ax.plot(
                sub["step"],
                sub[metric_col],
                color=COLORS[strategy],
                linewidth=LINEWIDTHS[strategy],
                linestyle=LINESTYLES[strategy],
                marker=MARKERS[strategy],
                markersize=3.5,
                markevery=2,
            )
        ax.set_title(group, fontweight="bold", fontsize=CURVE_LABEL_FONTSIZE - 1)
        ax.set_xlabel("Prompting Step", fontsize=CURVE_LABEL_FONTSIZE)
        ax.tick_params(axis="both", labelsize=CURVE_TICK_FONTSIZE)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel(ylabel, fontsize=CURVE_LABEL_FONTSIZE)
    savefig(fig, out_path)


def plot_overall_curves(
    aggregate: pd.DataFrame,
    out_path: Path,
    strategies: Sequence[str] = STRATEGIES,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.5))
    for ax, (metric, label, _) in zip(axes.flatten(), METRICS):
        for strategy in strategies:
            sub = aggregate[aggregate["strategy"] == strategy].sort_values("step")
            ax.plot(
                sub["step"],
                sub[f"{metric}_mean"],
                color=COLORS[strategy],
                linewidth=LINEWIDTHS[strategy],
                linestyle=LINESTYLES[strategy],
                marker=MARKERS[strategy],
                markersize=4,
                markevery=2,
            )
        ax.set_xlabel("Prompting Step", fontsize=CURVE_LABEL_FONTSIZE)
        ax.set_ylabel(label, fontsize=CURVE_LABEL_FONTSIZE)
        ax.tick_params(axis="both", labelsize=CURVE_TICK_FONTSIZE)
        ax.grid(alpha=0.25)
    savefig(fig, out_path)


def build_run_tables(run_data: RunData) -> dict[str, pd.DataFrame]:
    final_groups = run_data.per_group[run_data.per_group["step"] == run_data.max_step].copy()
    final_strata = run_data.per_stratum[run_data.per_stratum["step"] == run_data.max_step].copy()
    trio_metric_cols = [("iou_mean", "IoU", False), ("ece_mean", "ECE", True), ("hausdorff95_mean", "HD95", True), ("nll_mean", "NLL", True)]

    group_rank = rank_table(final_groups, TRIO, ["group"], trio_metric_cols)
    strata_rank = rank_table(final_strata, TRIO, ["stratum_type", "stratum_value"], trio_metric_cols)

    budget_auc_rows = []
    for strategy, sdf in run_data.aggregate.groupby("strategy"):
        sdf = sdf.sort_values("step")
        for metric, label, lower_is_better in METRICS:
            col = f"{metric}_mean"
            x = sdf["step"].to_numpy(dtype=float)
            y = sdf[col].to_numpy(dtype=float)
            auc = float(np.trapezoid(y, x) if hasattr(np, "trapezoid") else np.trapz(y, x))
            budget_auc_rows.append({
                "strategy": strategy,
                "metric": label,
                "direction": "lower_is_better" if lower_is_better else "higher_is_better",
                "budget_auc": auc,
                "budget_mean": float(y.mean()),
            })
    budget_auc = pd.DataFrame(budget_auc_rows)

    budget_rank_rows = []
    trio_pg = run_data.per_group[run_data.per_group["strategy"].isin(TRIO)].copy()
    for (group, step), gf in trio_pg.groupby(["group", "step"]):
        ranks = {s: 0.0 for s in TRIO}
        for metric, _, lower_is_better in trio_metric_cols:
            ordered = gf.sort_values(metric, ascending=lower_is_better)["strategy"].tolist()
            metric_ranks = {s: i + 1 for i, s in enumerate(ordered)}
            for s in TRIO:
                ranks[s] += metric_ranks[s]
        for s in TRIO:
            budget_rank_rows.append({"group": group, "step": step, "strategy": s, "balanced_rank": ranks[s] / len(trio_metric_cols)})
    budget_rank = (
        pd.DataFrame(budget_rank_rows)
        .groupby("strategy", as_index=False)["balanced_rank"]
        .mean()
        .sort_values("balanced_rank")
    )

    return {
        "dataset_wins": wins_table(group_rank, TRIO, ["IoU", "ECE", "HD95", "NLL"]),
        "strata_wins": wins_table(strata_rank, TRIO, ["IoU", "ECE", "HD95", "NLL"]),
        "budget_auc": budget_auc,
        "budget_rank": budget_rank,
        "budget_wins_iou": budget_wins_table(run_data.per_group, TRIO, "iou_mean", False, KEY_BUDGETS),
        "budget_wins_ece": budget_wins_table(run_data.per_group, TRIO, "ece_mean", True, KEY_BUDGETS),
        "budget_wins_hd95": budget_wins_table(run_data.per_group, TRIO, "hausdorff95_mean", True, KEY_BUDGETS),
        "budget_wins_nll": budget_wins_table(run_data.per_group, TRIO, "nll_mean", True, KEY_BUDGETS),
        "curve_summary": build_curve_summary_table(run_data, STRATEGIES),
        "curve_summary_by_dataset": build_curve_summary_by_dataset_table(run_data, STRATEGIES),
        "budget_summary": build_budget_summary_table(run_data, TRIO),
        "strata_summary": build_strata_summary_table(run_data, TRIO),
        "strata_dominance_table": build_strata_dominance_table(run_data, TRIO),
    }


def stitch_dataset_specific_run(data: dict[str, RunData]) -> RunData:
    per_group_parts = []
    per_stratum_parts = []
    step_parts = []
    for run in GROUP_TO_RUN.values():
        rd = data[run]
        per_group_parts.append(rd.per_group.copy())
        per_stratum_parts.append(rd.per_stratum.copy())
        step_parts.append(rd.step_results.copy())

    per_group = pd.concat(per_group_parts, ignore_index=True)
    per_stratum = (
        pd.concat(per_stratum_parts, ignore_index=True)
        .groupby(["strategy", "step", "stratum_type", "stratum_value"], as_index=False)
        .mean(numeric_only=True)
    )
    aggregate = (
        per_group.groupby(["strategy", "step"], as_index=False)[["iou_mean", "ece_mean", "hausdorff95_mean", "nll_mean"]]
        .mean()
    )
    step_results = pd.concat(step_parts, ignore_index=True)
    return RunData(
        step_results=step_results,
        aggregate=aggregate,
        per_group=per_group,
        per_stratum=per_stratum,
        max_step=int(aggregate["step"].max()),
    )


def write_prefixed_tables(prefix: str, tables: dict[str, pd.DataFrame]) -> None:
    for name, df in tables.items():
        df.to_csv(OUT / "tables" / f"{prefix}_{name}.csv", index=False)


def plot_finetuned_bundle(run_data: RunData, base_dir: Path) -> None:
    base_dir.mkdir(parents=True, exist_ok=True)
    plot_curve_legend(base_dir / "02_curves_legend.png")
    for metric, label, lower_is_better in METRICS:
        plot_metric_curve(
            run_data.aggregate,
            f"{metric}_mean",
            ylabel=label,
            title=f"{label} ({'lower' if lower_is_better else 'higher'} is better)",
            out_path=base_dir / f"02_{metric}_curve.png",
        )
        plot_metric_curve_per_dataset(
            run_data.per_group,
            f"{metric}_mean",
            ylabel=label,
            title=f"{label} by dataset ({'lower' if lower_is_better else 'higher'} is better)",
            out_path=base_dir / f"02_{metric}_curve_by_dataset.png",
        )
    plot_overall_curves(run_data.aggregate, base_dir / "02_curves.png")

    trio_tables = build_run_tables(run_data)
    trio_budget_table_map = {
        "victory_baseline_budget_wins_iou.csv": trio_tables["budget_wins_iou"],
        "victory_baseline_budget_wins_ece.csv": trio_tables["budget_wins_ece"],
        "victory_baseline_budget_wins_hd95.csv": trio_tables["budget_wins_hd95"],
        "victory_baseline_budget_wins_nll.csv": trio_tables["budget_wins_nll"],
    }
    plot_dominance_matrix_legend(base_dir / "00_dominance_matrix_legend.png")
    plot_budget_dominance_matrix(trio_budget_table_map, base_dir / "01_budget_dominance_matrix.png")
    plot_strata_dominance_matrix(run_data, base_dir / "02_strata_dominance_matrix.png")

    all_dir = base_dir / "all_strategies"
    all_dir.mkdir(parents=True, exist_ok=True)
    plot_dominance_matrix_legend(all_dir / "00_dominance_matrix_legend.png", strategies=STRATEGIES, include_tie=True)
    plot_all_strategies_budget_dominance_matrix(run_data, all_dir / "01_budget_dominance_matrix.png", strategies=STRATEGIES)
    plot_all_strategies_strata_dominance_matrix(run_data, all_dir / "02_strata_dominance_matrix.png", strategies=STRATEGIES)

    no_gt_dir = base_dir / "no_gt"
    no_gt_dir.mkdir(parents=True, exist_ok=True)
    plot_dominance_matrix_legend(no_gt_dir / "00_dominance_matrix_legend.png", strategies=NO_GT_STRATEGIES, include_tie=True)
    plot_all_strategies_budget_dominance_matrix(run_data, no_gt_dir / "01_budget_dominance_matrix.png", strategies=NO_GT_STRATEGIES)
    plot_all_strategies_strata_dominance_matrix(run_data, no_gt_dir / "02_strata_dominance_matrix.png", strategies=NO_GT_STRATEGIES)


def build_curve_summary_table(run_data: RunData, strategies: Sequence[str]) -> pd.DataFrame:
    key_steps = [0, 2, 5, 10, 15, run_data.max_step]
    rows = []
    for metric, label, lower_is_better in METRICS:
        for strategy in strategies:
            sub = run_data.aggregate[run_data.aggregate["strategy"] == strategy].sort_values("step")
            step_vals = sub[sub["step"].isin(key_steps)].set_index("step")[f"{metric}_mean"].to_dict()
            series = sub[f"{metric}_mean"].to_numpy(dtype=float)
            x = sub["step"].to_numpy(dtype=float)
            start_val = float(step_vals[0])
            end_val = float(step_vals[run_data.max_step])
            rows.append({
                "metric": label,
                "direction": "lower_is_better" if lower_is_better else "higher_is_better",
                "strategy": strategy,
                "step_0": start_val,
                "step_2": float(step_vals[2]),
                "step_5": float(step_vals[5]),
                "step_10": float(step_vals[10]),
                "step_15": float(step_vals[15]),
                "step_full": end_val,
                "raw_change_0_to_full": end_val - start_val,
                "improvement_0_to_full": (start_val - end_val) if lower_is_better else (end_val - start_val),
                "budget_auc": float(np.trapezoid(series, x) if hasattr(np, "trapezoid") else np.trapz(series, x)),
                "budget_mean": float(series.mean()),
            })
    return pd.DataFrame(rows)


def build_curve_summary_by_dataset_table(run_data: RunData, strategies: Sequence[str]) -> pd.DataFrame:
    overall = build_curve_summary_table(run_data, strategies).copy()
    overall.insert(2, "group", "Overall")

    key_steps = [0, 2, 5, 10, 15, run_data.max_step]
    rows = []
    for metric, label, lower_is_better in METRICS:
        for group in run_data.per_group["group"].drop_duplicates().tolist():
            gdf = run_data.per_group[run_data.per_group["group"] == group].copy()
            for strategy in strategies:
                sub = gdf[gdf["strategy"] == strategy].sort_values("step")
                step_vals = sub[sub["step"].isin(key_steps)].set_index("step")[f"{metric}_mean"].to_dict()
                series = sub[f"{metric}_mean"].to_numpy(dtype=float)
                x = sub["step"].to_numpy(dtype=float)
                start_val = float(step_vals[0])
                end_val = float(step_vals[run_data.max_step])
                rows.append({
                    "metric": label,
                    "direction": "lower_is_better" if lower_is_better else "higher_is_better",
                    "group": group,
                    "strategy": strategy,
                    "step_0": start_val,
                    "step_2": float(step_vals[2]),
                    "step_5": float(step_vals[5]),
                    "step_10": float(step_vals[10]),
                    "step_15": float(step_vals[15]),
                    "step_full": end_val,
                    "raw_change_0_to_full": end_val - start_val,
                    "improvement_0_to_full": (start_val - end_val) if lower_is_better else (end_val - start_val),
                    "budget_auc": float(np.trapezoid(series, x) if hasattr(np, "trapezoid") else np.trapz(series, x)),
                    "budget_mean": float(series.mean()),
                })

    dataset_df = pd.DataFrame(rows)
    combined = pd.concat([overall, dataset_df], ignore_index=True)
    group_order = ["Overall"] + run_data.per_group["group"].drop_duplicates().tolist()
    combined["group"] = pd.Categorical(combined["group"], categories=group_order, ordered=True)
    return combined.sort_values(["metric", "group", "strategy"]).reset_index(drop=True)


def build_budget_summary_table(run_data: RunData, strategies: Sequence[str]) -> pd.DataFrame:
    rows = []
    for step in KEY_BUDGETS:
        sdf = run_data.per_group[(run_data.per_group["step"] == step) & (run_data.per_group["strategy"].isin(strategies))].copy()
        pretty_step = "Full" if step == run_data.max_step else f"Step {step}"
        for metric_col, label, lower_is_better in [("iou_mean", "IoU", False), ("ece_mean", "ECE", True), ("hausdorff95_mean", "HD95", True), ("nll_mean", "NLL", True)]:
            counts = {s: 0 for s in strategies}
            for _, gf in sdf.groupby("group"):
                ordered = gf.sort_values(metric_col, ascending=lower_is_better)["strategy"].tolist()
                counts[ordered[0]] += 1
            winner, winner_wins, _, tied = unique_winner_from_counts(counts, strategies)
            rows.append({
                "step": step,
                "pretty_step": pretty_step,
                "metric": label,
                **counts,
                "winner": winner,
                "winner_label": tie_text(tied) if winner == "tie" else LABELS[winner],
                "winner_wins": winner_wins,
            })
    return pd.DataFrame(rows)


def build_strata_summary_table(run_data: RunData, strategies: Sequence[str]) -> pd.DataFrame:
    final_strata = run_data.per_stratum[(run_data.per_stratum["step"] == run_data.max_step) & (run_data.per_stratum["strategy"].isin(strategies))].copy()
    metric_cols = [("iou_mean", "IoU", False), ("ece_mean", "ECE", True), ("hausdorff95_mean", "HD95", True), ("nll_mean", "NLL", True)]
    strata_rank = rank_table(final_strata, strategies, ["stratum_type", "stratum_value"], metric_cols)
    strata_rank["pretty_name"] = strata_rank["stratum_type"].str.replace("_bin", "", regex=False).str.replace("_", " ").str.title() + ": " + strata_rank["stratum_value"].str.title()
    summary = strata_rank[[
        "stratum_type",
        "stratum_value",
        "pretty_name",
        "IoU_winner",
        "ECE_winner",
        "HD95_winner",
        "NLL_winner",
        "balanced_winner",
    ]].rename(columns={"balanced_winner": "Balanced_winner"})
    summary["bald_wins"] = (
        (summary["IoU_winner"] == "bald_laplace").astype(int) +
        (summary["ECE_winner"] == "bald_laplace").astype(int) +
        (summary["HD95_winner"] == "bald_laplace").astype(int) +
        (summary["NLL_winner"] == "bald_laplace").astype(int) +
        (summary["Balanced_winner"] == "bald_laplace").astype(int)
    )
    summary = summary.sort_values(["bald_wins", "pretty_name"], ascending=[False, True]).reset_index(drop=True)
    return summary


def build_strata_dominance_table(run_data: RunData, strategies: Sequence[str]) -> pd.DataFrame:
    final_strata = run_data.per_stratum[
        (run_data.per_stratum["step"] == run_data.max_step) &
        (run_data.per_stratum["strategy"].isin(strategies))
    ].copy()
    metric_cols = [("iou_mean", "IoU", False), ("ece_mean", "ECE", True), ("hausdorff95_mean", "HD95", True), ("nll_mean", "NLL", True)]
    family_order = {
        "boundary_complexity_bin": 0,
        "contrast_bin": 1,
        "thinness_bin": 2,
        "object_size_bin": 3,
    }
    value_strategy_order = [s for s in ["bald_laplace", "bald_mcdropout", "entropy"] if s in strategies]
    value_labels = {
        "bald_laplace": "BALD_Laplace",
        "bald_mcdropout": "BALD_MC",
        "entropy": "Entropy",
        "oracle": "GT_based",
        "random": "Random",
        "maxdist": "MaxDist",
    }

    rows = []
    for (stratum_type, stratum_value), gf in final_strata.groupby(["stratum_type", "stratum_value"]):
        vals = gf.set_index("strategy")
        pretty_family = stratum_type.replace("_bin", "").replace("_", " ").title()
        pretty_name = f"{pretty_family}: {stratum_value.title()}"
        rank_sum = {s: 0 for s in strategies}
        bald_wins = 0
        row = {
            "stratum_type": stratum_type,
            "stratum_value": stratum_value,
            "stratum_family": pretty_family,
            "pretty_name": pretty_name,
        }
        for metric_col, label, lower_is_better in metric_cols:
            ordered = vals[metric_col].sort_values(ascending=lower_is_better).index.tolist()
            winner = ordered[0]
            for strategy in value_strategy_order:
                row[f"{label}_{value_labels[strategy]}"] = float(vals.loc[strategy, metric_col])
            row[f"{label}_winner"] = LABELS[winner]
            for rank, strategy in enumerate(ordered, start=1):
                rank_sum[strategy] += rank
            bald_wins += int(winner == "bald_laplace")
        balanced_winner = min(strategies, key=lambda s: rank_sum[s])
        row["Balanced_winner"] = LABELS[balanced_winner]
        row["bald_wins"] = bald_wins + int(balanced_winner == "bald_laplace")
        row["balanced_is_bald"] = int(balanced_winner == "bald_laplace")
        row["family_order"] = family_order.get(stratum_type, 99)
        rows.append(row)

    dominance = pd.DataFrame(rows).sort_values(
        ["balanced_is_bald", "bald_wins", "family_order", "pretty_name"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)
    dominance.insert(0, "order_in_matrix", np.arange(1, len(dominance) + 1))
    value_cols = []
    for _, label, _ in metric_cols:
        for strategy in value_strategy_order:
            value_cols.append(f"{label}_{value_labels[strategy]}")
    dominance[value_cols] = dominance[value_cols].round(4)
    ordered_cols = [
        "order_in_matrix",
        "stratum_type",
        "stratum_value",
        "stratum_family",
        "pretty_name",
        "bald_wins",
    ]
    for _, label, _ in metric_cols:
        ordered_cols.extend([f"{label}_{value_labels[strategy]}" for strategy in value_strategy_order])
        ordered_cols.append(f"{label}_winner")
    ordered_cols.append("Balanced_winner")
    return dominance[ordered_cols]


def build_victory_tables(baseline: RunData) -> dict[str, pd.DataFrame]:
    baseline_final_groups = baseline.per_group[baseline.per_group["step"] == baseline.max_step].copy()
    baseline_final_strata = baseline.per_stratum[baseline.per_stratum["step"] == baseline.max_step].copy()
    trio_metric_cols = [("iou_mean", "IoU", False), ("ece_mean", "ECE", True), ("hausdorff95_mean", "HD95", True), ("nll_mean", "NLL", True)]
    group_rank = rank_table(baseline_final_groups, TRIO, ["group"], trio_metric_cols)
    strata_rank = rank_table(baseline_final_strata, TRIO, ["stratum_type", "stratum_value"], trio_metric_cols)
    strata_rank["pretty_name"] = strata_rank["stratum_type"].str.replace("_bin", "", regex=False) + ": " + strata_rank["stratum_value"]

    rows = []
    for group, gf in baseline_final_groups.groupby("group"):
        vals = gf.set_index("strategy")
        rows.append({
            "group": group,
            "iou_vs_entropy_pp": 100.0 * (vals.loc["bald_laplace", "iou_mean"] - vals.loc["entropy", "iou_mean"]),
            "ece_gain_vs_entropy_pp": 100.0 * (vals.loc["entropy", "ece_mean"] - vals.loc["bald_laplace", "ece_mean"]),
            "hd95_gain_vs_entropy": vals.loc["entropy", "hausdorff95_mean"] - vals.loc["bald_laplace", "hausdorff95_mean"],
            "nll_gain_vs_entropy": vals.loc["entropy", "nll_mean"] - vals.loc["bald_laplace", "nll_mean"],
            "iou_vs_mcd_pp": 100.0 * (vals.loc["bald_laplace", "iou_mean"] - vals.loc["bald_mcdropout", "iou_mean"]),
            "ece_gain_vs_mcd_pp": 100.0 * (vals.loc["bald_mcdropout", "ece_mean"] - vals.loc["bald_laplace", "ece_mean"]),
            "hd95_gain_vs_mcd": vals.loc["bald_mcdropout", "hausdorff95_mean"] - vals.loc["bald_laplace", "hausdorff95_mean"],
            "nll_gain_vs_mcd": vals.loc["bald_mcdropout", "nll_mean"] - vals.loc["bald_laplace", "nll_mean"],
            "iou_vs_gt_pp": 100.0 * (vals.loc["bald_laplace", "iou_mean"] - vals.loc["oracle", "iou_mean"]),
            "ece_gain_vs_gt_pp": 100.0 * (vals.loc["oracle", "ece_mean"] - vals.loc["bald_laplace", "ece_mean"]),
            "hd95_gain_vs_gt": vals.loc["oracle", "hausdorff95_mean"] - vals.loc["bald_laplace", "hausdorff95_mean"],
            "nll_gain_vs_gt": vals.loc["oracle", "nll_mean"] - vals.loc["bald_laplace", "nll_mean"],
        })
    advantage = pd.DataFrame(rows).sort_values("ece_gain_vs_entropy_pp", ascending=False)

    hh_rows = []
    for opponent in ["entropy", "bald_mcdropout", "oracle"]:
        record = {"opponent": opponent}
        for metric, label, lower_is_better in METRICS:
            bald_wins = 0
            opp_wins = 0
            for _, gf in baseline.per_group.groupby(["group", "step"]):
                vals = gf.set_index("strategy")[f"{metric}_mean"]
                a = vals["bald_laplace"]
                b = vals[opponent]
                if np.isclose(a, b):
                    continue
                if (a < b and lower_is_better) or (a > b and not lower_is_better):
                    bald_wins += 1
                else:
                    opp_wins += 1
            record[f"{label}_bald_wins"] = bald_wins
            record[f"{label}_{opponent}_wins"] = opp_wins
        hh_rows.append(record)
    head_to_head = pd.DataFrame(hh_rows)

    budget_auc_rows = []
    for strategy, sdf in baseline.aggregate.groupby("strategy"):
        sdf = sdf.sort_values("step")
        for metric, label, lower_is_better in METRICS:
            col = f"{metric}_mean"
            x = sdf["step"].to_numpy(dtype=float)
            y = sdf[col].to_numpy(dtype=float)
            auc = float(np.trapezoid(y, x) if hasattr(np, "trapezoid") else np.trapz(y, x))
            mean_over_steps = float(y.mean())
            budget_auc_rows.append({
                "strategy": strategy,
                "metric": label,
                "direction": "lower_is_better" if lower_is_better else "higher_is_better",
                "budget_auc": auc,
                "budget_mean": mean_over_steps,
            })
    budget_auc = pd.DataFrame(budget_auc_rows)

    budget_rank_rows = []
    baseline_trio = baseline.per_group[baseline.per_group["strategy"].isin(TRIO)].copy()
    for (group, step), gf in baseline_trio.groupby(["group", "step"]):
        ranks = {s: 0.0 for s in TRIO}
        for metric, label, lower_is_better in trio_metric_cols:
            ordered = gf.sort_values(metric, ascending=lower_is_better)["strategy"].tolist()
            metric_ranks = {s: i + 1 for i, s in enumerate(ordered)}
            for s in TRIO:
                ranks[s] += metric_ranks[s]
        for s in TRIO:
            budget_rank_rows.append({"group": group, "step": step, "strategy": s, "balanced_rank": ranks[s] / len(trio_metric_cols)})
    budget_rank = (
        pd.DataFrame(budget_rank_rows)
        .groupby("strategy", as_index=False)["balanced_rank"]
        .mean()
        .sort_values("balanced_rank")
    )

    budget_metric_tables = {}
    for metric, label, lower_is_better in trio_metric_cols:
        budget_metric_tables[f"victory_baseline_budget_wins_{label.lower()}.csv"] = budget_wins_table(
            baseline.per_group,
            TRIO,
            metric,
            lower_is_better=lower_is_better,
            budgets=KEY_BUDGETS,
        )

    family_rows = []
    for metric_label in ["IoU", "ECE", "HD95", "NLL", "Balanced"]:
        for stratum_type, gf in strata_rank.groupby("stratum_type"):
            counts = gf[f"{metric_label}_winner" if metric_label != "Balanced" else "balanced_winner"].value_counts()
            family_rows.append({
                "stratum_type": stratum_type,
                "metric": metric_label,
                **{s: int(counts.get(s, 0)) for s in TRIO},
            })
    strata_family_wins = pd.DataFrame(family_rows)

    strata_bin_winners = strata_rank[[
        "stratum_type",
        "stratum_value",
        "pretty_name",
        "IoU_winner",
        "ECE_winner",
        "HD95_winner",
        "NLL_winner",
        "balanced_winner",
    ]].rename(columns={"balanced_winner": "Balanced_winner"})

    tables = {
        "victory_baseline_dataset_wins.csv": wins_table(group_rank, TRIO, ["IoU", "ECE", "HD95", "NLL"]),
        "victory_baseline_strata_wins.csv": wins_table(strata_rank, TRIO, ["IoU", "ECE", "HD95", "NLL"]),
        "victory_baseline_strata_family_wins.csv": strata_family_wins,
        "victory_baseline_strata_bin_winners.csv": strata_bin_winners,
        "victory_baseline_curve_summary.csv": build_curve_summary_table(baseline, STRATEGIES),
        "victory_baseline_curve_summary_by_dataset.csv": build_curve_summary_by_dataset_table(baseline, STRATEGIES),
        "victory_baseline_budget_summary.csv": build_budget_summary_table(baseline, TRIO),
        "victory_baseline_strata_summary.csv": build_strata_summary_table(baseline, TRIO),
        "victory_baseline_strata_dominance_table.csv": build_strata_dominance_table(baseline, TRIO),
        "victory_baseline_dataset_advantages.csv": advantage,
        "victory_baseline_head_to_head_over_steps.csv": head_to_head,
        "victory_baseline_budget_auc.csv": budget_auc,
        "victory_baseline_budget_rank.csv": budget_rank,
    }
    tables.update(budget_metric_tables)
    return tables


def build_transition_tables(data: dict[str, RunData]) -> dict[str, pd.DataFrame]:
    baseline = data["run_baseline_v3"]
    ft_combined = data["run_ft_combined"]
    base_final = baseline.per_group[baseline.per_group["step"] == baseline.max_step].groupby("strategy", as_index=False)[["iou_mean", "ece_mean", "hausdorff95_mean", "nll_mean"]].mean()
    ft_final = ft_combined.per_group[ft_combined.per_group["step"] == ft_combined.max_step].groupby("strategy", as_index=False)[["iou_mean", "ece_mean", "hausdorff95_mean", "nll_mean"]].mean()
    overall = base_final.merge(ft_final, on="strategy", suffixes=("_baseline", "_ftcombined"))
    overall["iou_gain_pp"] = 100.0 * (overall["iou_mean_ftcombined"] - overall["iou_mean_baseline"])
    overall["ece_gain_pp"] = 100.0 * (overall["ece_mean_baseline"] - overall["ece_mean_ftcombined"])
    overall["hd95_gain"] = overall["hausdorff95_mean_baseline"] - overall["hausdorff95_mean_ftcombined"]
    overall["nll_gain"] = overall["nll_mean_baseline"] - overall["nll_mean_ftcombined"]

    base_budget_rows = []
    for strategy, sdf in baseline.aggregate.groupby("strategy"):
        sdf = sdf.sort_values("step")
        row = {"strategy": strategy}
        for metric, _, lower_is_better in METRICS:
            col = f"{metric}_mean"
            x = sdf["step"].to_numpy(dtype=float)
            y = sdf[col].to_numpy(dtype=float)
            row[f"{metric}_budget_mean_baseline"] = float(y.mean())
            row[f"{metric}_budget_auc_baseline"] = trapz_area(x, y)
        base_budget_rows.append(row)
    ft_budget_rows = []
    for strategy, sdf in ft_combined.aggregate.groupby("strategy"):
        sdf = sdf.sort_values("step")
        row = {"strategy": strategy}
        for metric, _, lower_is_better in METRICS:
            col = f"{metric}_mean"
            x = sdf["step"].to_numpy(dtype=float)
            y = sdf[col].to_numpy(dtype=float)
            row[f"{metric}_budget_mean_ftcombined"] = float(y.mean())
            row[f"{metric}_budget_auc_ftcombined"] = trapz_area(x, y)
        ft_budget_rows.append(row)
    overall_budget = pd.DataFrame(base_budget_rows).merge(pd.DataFrame(ft_budget_rows), on="strategy")
    overall_budget_mean = overall_budget[["strategy"]].copy()
    overall_budget_auc = overall_budget[["strategy"]].copy()
    for metric, _, lower_is_better in METRICS:
        mean_base_col = f"{metric}_budget_mean_baseline"
        mean_ft_col = f"{metric}_budget_mean_ftcombined"
        auc_base_col = f"{metric}_budget_auc_baseline"
        auc_ft_col = f"{metric}_budget_auc_ftcombined"
        mean_gain_col = f"{metric}_budget_mean_gain"
        auc_gain_col = f"{metric}_budget_auc_gain"
        overall_budget_mean[mean_base_col] = overall_budget[mean_base_col]
        overall_budget_mean[mean_ft_col] = overall_budget[mean_ft_col]
        overall_budget_auc[auc_base_col] = overall_budget[auc_base_col]
        overall_budget_auc[auc_ft_col] = overall_budget[auc_ft_col]
        if lower_is_better:
            overall_budget_mean[mean_gain_col] = overall_budget[mean_base_col] - overall_budget[mean_ft_col]
            overall_budget_auc[auc_gain_col] = overall_budget[auc_base_col] - overall_budget[auc_ft_col]
        else:
            overall_budget_mean[mean_gain_col] = overall_budget[mean_ft_col] - overall_budget[mean_base_col]
            overall_budget_auc[auc_gain_col] = overall_budget[auc_ft_col] - overall_budget[auc_base_col]

    mean_rank_cols = []
    auc_rank_cols = []
    for metric, _, _ in METRICS:
        mean_gain_col = f"{metric}_budget_mean_gain"
        auc_gain_col = f"{metric}_budget_auc_gain"
        mean_rank_col = f"{metric}_budget_mean_rank"
        auc_rank_col = f"{metric}_budget_auc_rank"
        overall_budget_mean[mean_rank_col] = overall_budget_mean[mean_gain_col].rank(ascending=False, method="min")
        overall_budget_auc[auc_rank_col] = overall_budget_auc[auc_gain_col].rank(ascending=False, method="min")
        mean_rank_cols.append(mean_rank_col)
        auc_rank_cols.append(auc_rank_col)
    overall_budget_mean["avg_gain_rank"] = overall_budget_mean[mean_rank_cols].mean(axis=1)
    overall_budget_auc["avg_gain_rank"] = overall_budget_auc[auc_rank_cols].mean(axis=1)
    overall_budget_mean = overall_budget_mean.sort_values("avg_gain_rank")
    overall_budget_auc = overall_budget_auc.sort_values("avg_gain_rank")

    baseline_subset = (
        baseline.step_results[baseline.step_results["step"] == baseline.max_step]
        .groupby(["group", "strategy"], as_index=False)[["iou", "ece", "hausdorff95", "nll"]]
        .mean()
    )
    rows = []
    budget_mean_rows = []
    budget_auc_rows = []
    winner_rows = []
    rank_rows = []
    for group, ft_run in GROUP_TO_RUN.items():
        ft = data[ft_run]
        ft_final_group = (
            ft.step_results[ft.step_results["step"] == ft.max_step]
            .groupby(["group", "strategy"], as_index=False)[["iou", "ece", "hausdorff95", "nll"]]
            .mean()
        )
        b = baseline_subset[baseline_subset["group"] == group].set_index("strategy")
        f = ft_final_group.set_index("strategy")
        for strategy in STRATEGIES:
            rows.append({
                "group": group,
                "ft_run": ft_run,
                "strategy": strategy,
                "iou_gain_pp": 100.0 * (f.loc[strategy, "iou"] - b.loc[strategy, "iou"]),
                "ece_gain_pp": 100.0 * (b.loc[strategy, "ece"] - f.loc[strategy, "ece"]),
                "hd95_gain": b.loc[strategy, "hausdorff95"] - f.loc[strategy, "hausdorff95"],
                "nll_gain": b.loc[strategy, "nll"] - f.loc[strategy, "nll"],
            })
        base_budget = baseline.per_group[baseline.per_group["group"] == group].copy()
        ft_budget = ft.per_group[ft.per_group["group"] == group].copy()
        for strategy in STRATEGIES:
            b_budget = base_budget[base_budget["strategy"] == strategy].sort_values("step")
            f_budget = ft_budget[ft_budget["strategy"] == strategy].sort_values("step")
            b_x = b_budget["step"].to_numpy(dtype=float)
            f_x = f_budget["step"].to_numpy(dtype=float)
            budget_mean_rows.append({
                "group": group,
                "ft_run": ft_run,
                "strategy": strategy,
                "iou_budget_mean_gain_pp": 100.0 * (f_budget["iou_mean"].mean() - b_budget["iou_mean"].mean()),
                "ece_budget_mean_gain_pp": 100.0 * (b_budget["ece_mean"].mean() - f_budget["ece_mean"].mean()),
                "hd95_budget_mean_gain": b_budget["hausdorff95_mean"].mean() - f_budget["hausdorff95_mean"].mean(),
                "nll_budget_mean_gain": b_budget["nll_mean"].mean() - f_budget["nll_mean"].mean(),
            })
            budget_auc_rows.append({
                "group": group,
                "ft_run": ft_run,
                "strategy": strategy,
                "iou_budget_auc_gain": trapz_area(f_x, f_budget["iou_mean"].to_numpy(dtype=float)) - trapz_area(b_x, b_budget["iou_mean"].to_numpy(dtype=float)),
                "ece_budget_auc_gain": trapz_area(b_x, b_budget["ece_mean"].to_numpy(dtype=float)) - trapz_area(f_x, f_budget["ece_mean"].to_numpy(dtype=float)),
                "hausdorff95_budget_auc_gain": trapz_area(b_x, b_budget["hausdorff95_mean"].to_numpy(dtype=float)) - trapz_area(f_x, f_budget["hausdorff95_mean"].to_numpy(dtype=float)),
                "nll_budget_auc_gain": trapz_area(b_x, b_budget["nll_mean"].to_numpy(dtype=float)) - trapz_area(f_x, f_budget["nll_mean"].to_numpy(dtype=float)),
            })

        for label, gf in [("baseline", baseline_subset[baseline_subset["group"] == group]), ("finetuned", ft_final_group)]:
            metric_cols = [("iou", "IoU", False), ("ece", "ECE", True), ("hausdorff95", "HD95", True), ("nll", "NLL", True)]
            all_rank = rank_table(gf, STRATEGIES, ["group"], metric_cols)
            trio_rank = rank_table(gf, TRIO, ["group"], metric_cols)
            winner_rows.append({
                "group": group,
                "stage": label,
                "all_strategies_winner": all_rank.iloc[0]["balanced_winner"],
                "trio_winner": trio_rank.iloc[0]["balanced_winner"],
            })
            for strategy in STRATEGIES:
                rank_rows.append({
                    "group": group,
                    "stage": label,
                    "strategy": strategy,
                    "balanced_rank": float(all_rank.iloc[0][f"balanced_rank__{strategy}"]),
                })

    rank_df = pd.DataFrame(rank_rows)
    shift = rank_df.pivot_table(index="group", columns=["stage", "strategy"], values="balanced_rank")
    shift_rows = []
    for group in shift.index:
        row = {"group": group}
        for strategy in STRATEGIES:
            row[strategy] = shift.loc[group, ("baseline", strategy)] - shift.loc[group, ("finetuned", strategy)]
        shift_rows.append(row)
    rank_shift = pd.DataFrame(shift_rows)

    budget_mean_by_group = pd.DataFrame(budget_mean_rows).sort_values(["group", "strategy"])
    budget_auc_by_group = pd.DataFrame(budget_auc_rows).sort_values(["group", "strategy"])

    mean_std = (
        budget_mean_by_group
        .groupby("strategy", as_index=False)[["iou_budget_mean_gain_pp", "ece_budget_mean_gain_pp", "hd95_budget_mean_gain", "nll_budget_mean_gain"]]
        .std(ddof=1)
        .rename(columns={
            "iou_budget_mean_gain_pp": "iou_budget_mean_gain_std",
            "ece_budget_mean_gain_pp": "ece_budget_mean_gain_std",
            "hd95_budget_mean_gain": "hausdorff95_budget_mean_gain_std",
            "nll_budget_mean_gain": "nll_budget_mean_gain_std",
        })
    )
    mean_std["iou_budget_mean_gain_std"] = mean_std["iou_budget_mean_gain_std"] / 100.0
    mean_std["ece_budget_mean_gain_std"] = mean_std["ece_budget_mean_gain_std"] / 100.0
    auc_std = (
        budget_auc_by_group
        .groupby("strategy", as_index=False)[["iou_budget_auc_gain", "ece_budget_auc_gain", "hausdorff95_budget_auc_gain", "nll_budget_auc_gain"]]
        .std(ddof=1)
        .rename(columns={
            "iou_budget_auc_gain": "iou_budget_auc_gain_std",
            "ece_budget_auc_gain": "ece_budget_auc_gain_std",
            "hausdorff95_budget_auc_gain": "hausdorff95_budget_auc_gain_std",
            "nll_budget_auc_gain": "nll_budget_auc_gain_std",
        })
    )
    overall_budget_mean = overall_budget_mean.merge(mean_std, on="strategy", how="left")
    overall_budget_auc = overall_budget_auc.merge(auc_std, on="strategy", how="left")

    return {
        "transition_overall_ftcombined_gains.csv": overall.sort_values("iou_gain_pp", ascending=False),
        "transition_overall_ftcombined_budget_mean_gains.csv": overall_budget_mean,
        "transition_overall_ftcombined_budget_auc_gains.csv": overall_budget_auc,
        "transition_dataset_specific_gains.csv": pd.DataFrame(rows).sort_values(["group", "strategy"]),
        "transition_dataset_specific_budget_mean_gains.csv": budget_mean_by_group,
        "transition_dataset_specific_budget_auc_gains.csv": budget_auc_by_group,
        "transition_winner_shift.csv": pd.DataFrame(winner_rows),
        "transition_rank_shift.csv": rank_shift,
    }


def plot_victory_package(baseline: RunData, tables: dict[str, pd.DataFrame]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    plot_grouped_counts(axes[0], tables["victory_baseline_dataset_wins.csv"], TRIO, total=5, title="Baseline SAM: dataset wins")
    plot_grouped_counts(axes[1], tables["victory_baseline_strata_wins.csv"], TRIO, total=12, title="Baseline SAM: strata wins")
    axes[0].set_ylabel("Wins")
    axes[0].legend(ncol=3, loc="upper center", bbox_to_anchor=(1.05, 1.22))
    fig.suptitle(
        "BALD (Laplace) dominates the baseline medical run among uncertainty strategies\n"
        "Final-budget wins over BUSI, DRIVE, ISIC2018, KvasirSEG and most strata",
        fontweight="bold",
        y=1.02,
    )
    savefig(fig, OUT / "victory" / "01_baseline_scoreboard.png")

    plot_budget_wins(
        tables["victory_baseline_budget_wins_iou.csv"],
        TRIO,
        title="Baseline SAM: IoU wins by prompt budget",
        ylabel="Datasets won out of 5",
        out_path=OUT / "victory" / "01b_budget_wins_iou.png",
        total=5,
    )
    plot_budget_wins(
        tables["victory_baseline_budget_wins_ece.csv"],
        TRIO,
        title="Baseline SAM: ECE wins by prompt budget",
        ylabel="Datasets won out of 5",
        out_path=OUT / "victory" / "01b_budget_wins_ece.png",
        total=5,
    )
    plot_budget_wins(
        tables["victory_baseline_budget_wins_hd95.csv"],
        TRIO,
        title="Baseline SAM: HD95 wins by prompt budget",
        ylabel="Datasets won out of 5",
        out_path=OUT / "victory" / "01b_budget_wins_hd95.png",
        total=5,
    )
    plot_budget_wins(
        tables["victory_baseline_budget_wins_nll.csv"],
        TRIO,
        title="Baseline SAM: NLL wins by prompt budget",
        ylabel="Datasets won out of 5",
        out_path=OUT / "victory" / "01b_budget_wins_nll.png",
        total=5,
    )
    plot_budget_dominance_matrix(
        tables,
        out_path=OUT / "victory" / "01c_budget_dominance_matrix.png",
    )
    plot_dominance_matrix_legend(OUT / "victory" / "00_dominance_matrix_legend.png")
    plot_budget_takeover_timeline(
        build_budget_timeline_df(baseline, TRIO),
        out_path=OUT / "victory" / "01d_budget_takeover_timeline.png",
        title_line="Baseline SAM: BALD takeover across the prompt budget",
        subtitle_line="Winner among BALD / Entropy / MC Dropout, with dataset wins shown out of 5",
    )
    plot_budget_bump_chart(
        build_budget_wins_long(baseline, TRIO),
        TRIO,
        out_path=OUT / "victory" / "01e_budget_bump_chart.png",
        title_line="Baseline SAM: BALD leads the budget race",
        subtitle_line="Dataset wins by budget among BALD / Entropy / MC Dropout",
    )
    plot_curve_legend(OUT / "victory" / "02_curves_legend.png")

    for metric, label, lower_is_better in METRICS:
        plot_metric_curve(
            baseline.aggregate,
            f"{metric}_mean",
            ylabel=label,
            title=f"Baseline SAM: {label} over active prompting steps ({'lower' if lower_is_better else 'higher'} is better)",
            out_path=OUT / "victory" / f"02_{metric}_curve.png",
        )
        plot_metric_curve_per_dataset(
            baseline.per_group,
            f"{metric}_mean",
            ylabel=label,
            title=f"Baseline SAM: {label} by dataset ({'lower' if lower_is_better else 'higher'} is better)",
            out_path=OUT / "victory" / f"02_{metric}_curve_by_dataset.png",
        )

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.5))
    for ax, (metric, label, lower_is_better) in zip(axes.flatten(), METRICS):
        for strategy in STRATEGIES:
            sub = baseline.aggregate[baseline.aggregate["strategy"] == strategy].sort_values("step")
            ax.plot(
                sub["step"],
                sub[f"{metric}_mean"],
                color=COLORS[strategy],
                linewidth=LINEWIDTHS[strategy],
                linestyle=LINESTYLES[strategy],
                marker=MARKERS[strategy],
                markersize=4,
                markevery=2,
            )
        ax.set_xlabel("Prompting Step", fontsize=CURVE_LABEL_FONTSIZE)
        ax.set_ylabel(label, fontsize=CURVE_LABEL_FONTSIZE)
        ax.tick_params(axis="both", labelsize=CURVE_TICK_FONTSIZE)
        ax.grid(alpha=0.25)
    savefig(fig, OUT / "victory" / "02_baseline_curves.png")

    adv = tables["victory_baseline_dataset_advantages.csv"].copy().set_index("group")
    heat = adv[[
        "iou_vs_entropy_pp",
        "ece_gain_vs_entropy_pp",
        "hd95_gain_vs_entropy",
        "nll_gain_vs_entropy",
        "iou_vs_mcd_pp",
        "ece_gain_vs_mcd_pp",
        "hd95_gain_vs_mcd",
        "nll_gain_vs_mcd",
        "iou_vs_gt_pp",
        "ece_gain_vs_gt_pp",
        "hd95_gain_vs_gt",
        "nll_gain_vs_gt",
    ]]
    heat.columns = [
        "IoU vs Ent", "ECE vs Ent", "HD95 vs Ent", "NLL vs Ent",
        "IoU vs MCD", "ECE vs MCD", "HD95 vs MCD", "NLL vs MCD",
        "IoU vs GT", "ECE vs GT", "HD95 vs GT", "NLL vs GT",
    ]
    fig, ax = plt.subplots(figsize=(15.5, 4.8))
    plot_heatmap(
        fig,
        ax,
        heat,
        "Baseline SAM: BALD advantage map across medical datasets\nPositive means BALD is better",
        cbar_label="Positive means BALD is better",
    )
    savefig(fig, OUT / "victory" / "03_baseline_advantage_heatmap.png")

    hh = tables["victory_baseline_head_to_head_over_steps.csv"].copy()
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5))
    for ax, opponent in zip(axes, ["entropy", "bald_mcdropout", "oracle"]):
        sub = hh[hh["opponent"] == opponent]
        labels = [m[1] for m in METRICS]
        bald_vals = [int(sub.iloc[0][f"{label}_bald_wins"]) for label in labels]
        opp_vals = [int(sub.iloc[0][f"{label}_{opponent}_wins"]) for label in labels]
        x = np.arange(len(labels))
        width = 0.36
        ax.bar(x - width / 2, bald_vals, width=width, color=COLORS["bald_laplace"], label="BALD (Laplace)")
        ax.bar(x + width / 2, opp_vals, width=width, color=COLORS[opponent], label=LABELS[opponent])
        for xi, bv, ov in zip(x, bald_vals, opp_vals):
            ax.text(xi - width / 2, bv + 1, str(bv), ha="center", va="bottom", fontsize=9, color=COLORS["bald_laplace"])
            ax.text(xi + width / 2, ov + 1, str(ov), ha="center", va="bottom", fontsize=9, color=COLORS[opponent])
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(f"BALD vs {LABELS[opponent]}", fontweight="bold")
        ax.set_ylim(0, 105)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Dataset-step wins out of 100")
    axes[0].legend(loc="upper right")
    fig.suptitle("Across the full baseline budget trajectory, BALD keeps winning on trust metrics", fontweight="bold", y=1.02)
    savefig(fig, OUT / "victory" / "04_baseline_head_to_head_budget_wins.png")

    family = tables["victory_baseline_strata_family_wins.csv"].copy()
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 8.5), sharey=True)
    for ax, metric in zip(axes.flatten(), ["IoU", "ECE", "HD95", "NLL"]):
        sub = family[family["metric"] == metric].copy()
        sub["pretty_type"] = sub["stratum_type"].str.replace("_bin", "", regex=False).str.replace("_", " ").str.title()
        x = np.arange(len(sub))
        width = 0.24
        offsets = np.linspace(-1, 1, len(TRIO)) * width
        for off, strategy in zip(offsets, TRIO):
            vals = sub[strategy].to_numpy()
            bars = ax.bar(
                x + off,
                vals,
                width=width,
                color=COLORS[strategy],
                edgecolor="black" if strategy == "bald_laplace" else "white",
                linewidth=2.0 if strategy == "bald_laplace" else 1.0,
                label=LABELS[strategy] if metric == "IoU" else None,
            )
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, val + 0.03, str(int(val)), ha="center", va="bottom", fontsize=8, color=COLORS[strategy])
        ax.set_xticks(x)
        ax.set_xticklabels(sub["pretty_type"], rotation=18, ha="right")
        ax.set_ylim(0, 3.4)
        ax.set_title(metric, fontweight="bold")
        ax.grid(axis="y", alpha=0.25)
    axes[0, 0].set_ylabel("Bins won within stratum family")
    axes[0, 0].legend(loc="upper right", fontsize=8)
    fig.suptitle(
        "Baseline SAM: BALD wins across nearly every stratum family\n"
        "Each family contains 3 bins (e.g. size: small / medium / large)",
        fontweight="bold",
        y=1.02,
    )
    savefig(fig, OUT / "victory" / "05_strata_family_wins.png")

    plot_categorical_winner_map(
        tables["victory_baseline_strata_bin_winners.csv"],
        title="Baseline SAM: winner map for every stratum bin",
        out_path=OUT / "victory" / "05_strata_bin_winner_map.png",
    )
    plot_strata_dominance_matrix(
        baseline,
        out_path=OUT / "victory" / "05_strata_dominance_matrix.png",
    )


def plot_transition_package(data: dict[str, RunData], tables: dict[str, pd.DataFrame]) -> None:
    baseline = data["run_baseline_v3"]
    ft_combined = data["run_ft_combined"]

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.5))
    for ax, (metric, label, lower_is_better) in zip(axes.flatten(), METRICS):
        for strategy in TRIO:
            b = baseline.aggregate[baseline.aggregate["strategy"] == strategy].sort_values("step")
            f = ft_combined.aggregate[ft_combined.aggregate["strategy"] == strategy].sort_values("step")
            ax.plot(
                b["step"],
                b[f"{metric}_mean"],
                color=COLORS[strategy],
                linestyle="--",
                linewidth=2.0,
                alpha=0.8,
            )
            ax.plot(
                f["step"],
                f[f"{metric}_mean"],
                color=COLORS[strategy],
                linestyle="-",
                linewidth=3.0 if strategy == "bald_laplace" else 2.3,
                marker=MARKERS[strategy],
                markersize=4,
                markevery=2,
                label=f"{LABELS[strategy]} (FT)" if ax is axes[0, 0] else None,
            )
        ax.set_xlabel("Prompting step")
        ax.set_ylabel(label)
        ax.set_title(f"{label} ({'lower' if lower_is_better else 'higher'} is better)", fontweight="bold")
        ax.grid(alpha=0.25)
    handles = []
    for strategy in TRIO:
        handles.append(plt.Line2D([0], [0], color=COLORS[strategy], lw=3, linestyle="-", label=f"{LABELS[strategy]} (FT)"))
        handles.append(plt.Line2D([0], [0], color=COLORS[strategy], lw=2, linestyle="--", label=f"{LABELS[strategy]} (Baseline)"))
    axes[0, 0].legend(handles=handles, fontsize=8, ncol=2)
    fig.suptitle(
        "Baseline SAM -> combined finetuned SAM: the curves move up, but the winner changes\n"
        "Finetuning raises absolute quality while eroding BALD's relative edge",
        fontweight="bold",
        y=1.01,
    )
    savefig(fig, OUT / "transition" / "01_trio_baseline_vs_ftcombined_curves.png")

    overall = tables["transition_overall_ftcombined_gains.csv"].copy()
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 8.5))
    panels = [
        ("iou_gain_pp", "IoU gain (pp)"),
        ("ece_gain_pp", "ECE reduction (pp)"),
        ("hd95_gain", "HD95 reduction"),
        ("nll_gain", "NLL reduction"),
    ]
    for ax, (col, title) in zip(axes.flatten(), panels):
        vals = overall.set_index("strategy").loc[STRATEGIES, col]
        bars = ax.bar(np.arange(len(STRATEGIES)), vals, color=[COLORS[s] for s in STRATEGIES], edgecolor="white")
        for bar, val, strategy in zip(bars, vals, STRATEGIES):
            ax.text(bar.get_x() + bar.get_width() / 2, val + (0.4 if val >= 0 else -0.4), f"{val:+.1f}", ha="center", va="bottom" if val >= 0 else "top", fontsize=8, color=COLORS[strategy])
        ax.axhline(0, color="black", lw=1.0, alpha=0.5)
        ax.set_xticks(np.arange(len(STRATEGIES)))
        ax.set_xticklabels([LABELS[s] for s in STRATEGIES], rotation=20, ha="right")
        ax.set_title(title, fontweight="bold")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Overall effect of switching from baseline SAM to combined finetuned SAM", fontweight="bold", y=1.01)
    savefig(fig, OUT / "transition" / "02_overall_strategy_gains_ftcombined.png")

    overall_budget_mean = tables["transition_overall_ftcombined_budget_mean_gains.csv"].copy().set_index("strategy").loc[STRATEGIES]
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 8.5))
    mean_panels = [
        ("iou_budget_mean_gain", "Budget-Mean IoU gain (pp)", 100.0, "iou"),
        ("ece_budget_mean_gain", "Budget-Mean ECE reduction (pp)", 100.0, "ece"),
        ("hausdorff95_budget_mean_gain", "Budget-Mean HD95 reduction", 1.0, "hausdorff95"),
        ("nll_budget_mean_gain", "Budget-Mean NLL reduction", 1.0, "nll"),
    ]
    for ax, (col, ylabel, scale, slug) in zip(axes.flatten(), mean_panels):
        vals = scale * overall_budget_mean[col]
        errs = scale * overall_budget_mean[f"{col}_std"].fillna(0.0)
        bars = ax.bar(
            np.arange(len(STRATEGIES)),
            vals,
            yerr=errs,
            color=[COLORS[s] for s in STRATEGIES],
            edgecolor="white",
            error_kw={"elinewidth": 1.5, "ecolor": "black", "capsize": 4, "capthick": 1.5},
        )
        for bar, val, err, strategy in zip(bars, vals, errs, STRATEGIES):
            anchor = val + err
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                anchor + (0.4 if val >= 0 else -0.4),
                f"{val:+.1f}",
                ha="center",
                va="bottom" if val >= 0 else "top",
                fontsize=8,
                color=COLORS[strategy],
            )
        ax.axhline(0, color="black", lw=1.0, alpha=0.5)
        ax.set_xticks(np.arange(len(STRATEGIES)))
        ax.set_xticklabels([LABELS[s] for s in STRATEGIES], rotation=20, ha="right")
        ax.set_ylabel(ylabel, fontsize=CURVE_LABEL_FONTSIZE)
        ax.tick_params(axis="both", labelsize=CURVE_TICK_FONTSIZE)
        ax.grid(axis="y", alpha=0.25)
    savefig(fig, OUT / "transition" / "09_overall_budget_mean_gains_ftcombined.png")
    for col, ylabel, scale, slug in mean_panels:
        plot_strategy_gain_bars(
            scale * overall_budget_mean[col],
            scale * overall_budget_mean[f"{col}_std"].fillna(0.0),
            ylabel=ylabel,
            out_path=OUT / "transition" / f"09_{slug}_budget_mean_gain.png",
        )

    overall_budget_auc = tables["transition_overall_ftcombined_budget_auc_gains.csv"].copy().set_index("strategy").loc[STRATEGIES]
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 8.5))
    auc_panels = [
        ("iou_budget_auc_gain", "Budget-AUC IoU gain", 1.0, "iou"),
        ("ece_budget_auc_gain", "Budget-AUC ECE reduction", 1.0, "ece"),
        ("hausdorff95_budget_auc_gain", "Budget-AUC HD95 reduction", 1.0, "hausdorff95"),
        ("nll_budget_auc_gain", "Budget-AUC NLL reduction", 1.0, "nll"),
    ]
    for ax, (col, ylabel, scale, slug) in zip(axes.flatten(), auc_panels):
        vals = scale * overall_budget_auc[col]
        errs = scale * overall_budget_auc[f"{col}_std"].fillna(0.0)
        bars = ax.bar(
            np.arange(len(STRATEGIES)),
            vals,
            yerr=errs,
            color=[COLORS[s] for s in STRATEGIES],
            edgecolor="white",
            error_kw={"elinewidth": 1.5, "ecolor": "black", "capsize": 4, "capthick": 1.5},
        )
        for bar, val, err, strategy in zip(bars, vals, errs, STRATEGIES):
            anchor = val + err
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                anchor + (0.4 if val >= 0 else -0.4),
                f"{val:+.1f}",
                ha="center",
                va="bottom" if val >= 0 else "top",
                fontsize=8,
                color=COLORS[strategy],
            )
        ax.axhline(0, color="black", lw=1.0, alpha=0.5)
        ax.set_xticks(np.arange(len(STRATEGIES)))
        ax.set_xticklabels([LABELS[s] for s in STRATEGIES], rotation=20, ha="right")
        ax.set_ylabel(ylabel, fontsize=CURVE_LABEL_FONTSIZE)
        ax.tick_params(axis="both", labelsize=CURVE_TICK_FONTSIZE)
        ax.grid(axis="y", alpha=0.25)
    savefig(fig, OUT / "transition" / "10_overall_budget_auc_gains_ftcombined.png")
    for col, ylabel, scale, slug in auc_panels:
        plot_strategy_gain_bars(
            scale * overall_budget_auc[col],
            scale * overall_budget_auc[f"{col}_std"].fillna(0.0),
            ylabel=ylabel,
            out_path=OUT / "transition" / f"10_{slug}_budget_auc_gain.png",
        )

    dataset_gains = tables["transition_dataset_specific_gains.csv"].copy()
    gain_specs = [
        ("iou_gain_pp", "IoU gain (pp)"),
        ("ece_gain_pp", "ECE reduction (pp)"),
        ("hd95_gain", "HD95 reduction"),
        ("nll_gain", "NLL reduction"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(16, 8.5))
    for ax, (col, title) in zip(axes.flatten(), gain_specs):
        mat = dataset_gains.pivot(index="group", columns="strategy", values=col).loc[list(GROUP_TO_RUN.keys()), STRATEGIES]
        plot_heatmap(fig, ax, mat, title, cbar_label="Positive means finetuned SAM is better")
    fig.suptitle(
        "Dataset-specific finetuned SAM vs baseline SAM subset\n"
        "All strategies improve, but BALD usually improves less than Entropy / MC Dropout / Random",
        fontweight="bold",
        y=1.02,
    )
    savefig(fig, OUT / "transition" / "03_dataset_specific_gain_heatmaps.png")

    rank_shift = tables["transition_rank_shift.csv"].copy().set_index("group").loc[list(GROUP_TO_RUN.keys()), STRATEGIES]
    rank_shift.columns = [LABELS[s] for s in rank_shift.columns]
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    plot_heatmap(
        fig,
        ax,
        rank_shift,
        "Balanced rank improvement after dataset-specific finetuning\nPositive means the strategy ranks better than it did under baseline SAM",
        cbar_label="Positive means rank improved",
    )
    savefig(fig, OUT / "transition" / "04_rank_shift_heatmap.png")

    budget_mean_gains = tables["transition_dataset_specific_budget_mean_gains.csv"].copy()
    mean_gain_specs = [
        ("iou_budget_mean_gain_pp", "Budget-Mean IoU gain (pp)"),
        ("ece_budget_mean_gain_pp", "Budget-Mean ECE reduction (pp)"),
        ("hd95_budget_mean_gain", "Budget-Mean HD95 reduction"),
        ("nll_budget_mean_gain", "Budget-Mean NLL reduction"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(16, 8.5))
    for ax, (col, title) in zip(axes.flatten(), mean_gain_specs):
        mat = budget_mean_gains.pivot(index="group", columns="strategy", values=col).loc[list(GROUP_TO_RUN.keys()), STRATEGIES]
        plot_heatmap(
            fig,
            ax,
            mat,
            title,
            cbar_label="Positive means the finetuned run is better on average over the prompting budget",
        )
    fig.suptitle(
        "Dataset-specific finetuning effect over the full active prompting budget\n"
        "This is the average-over-budget view, not just the final-step view",
        fontweight="bold",
        y=1.02,
    )
    savefig(fig, OUT / "transition" / "08_dataset_specific_budget_mean_gain_heatmaps.png")

    merged = baseline.aggregate.merge(
        ft_combined.aggregate,
        on=["strategy", "step"],
        suffixes=("_baseline", "_ftcombined"),
    )
    plot_curve_legend(OUT / "transition" / "05_delta_curves_legend.png")
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 8.8))
    delta_specs = [
        ("iou_mean", "IoU gain vs baseline (pp)", False, 100.0),
        ("ece_mean", "ECE reduction vs baseline (pp)", True, 100.0),
        ("hausdorff95_mean", "HD95 reduction vs baseline", True, 1.0),
        ("nll_mean", "NLL reduction vs baseline", True, 1.0),
    ]
    for ax, (metric_col, title, lower_is_better, scale) in zip(axes.flatten(), delta_specs):
        for strategy in STRATEGIES:
            sub = merged[merged["strategy"] == strategy].sort_values("step").copy()
            if lower_is_better:
                delta = scale * (sub[f"{metric_col}_baseline"] - sub[f"{metric_col}_ftcombined"])
            else:
                delta = scale * (sub[f"{metric_col}_ftcombined"] - sub[f"{metric_col}_baseline"])
            ax.plot(
                sub["step"],
                delta,
                color=COLORS[strategy],
                linewidth=LINEWIDTHS[strategy],
                linestyle=LINESTYLES[strategy],
                marker=MARKERS[strategy],
                markersize=4,
                markevery=2,
                label=LABELS[strategy] if ax is axes[0, 0] else None,
            )
        ax.axhline(0, color="black", lw=1.0, alpha=0.55)
        ax.set_xlabel("Prompting step")
        ax.set_ylabel(title)
        ax.set_title(title, fontweight="bold")
        ax.grid(alpha=0.25)
        plot_transition_delta_curve(
            merged,
            metric_col=metric_col,
            ylabel=title,
            lower_is_better=lower_is_better,
            scale=scale,
            out_path=OUT / "transition" / f"05_{metric_col.replace('_mean', '')}_delta_curve.png",
        )
    axes[0, 0].legend(loc="best", fontsize=8, ncol=2)
    fig.suptitle(
        "Combined finetuned SAM minus baseline SAM over the active prompting budget\n"
        "Positive values mean the finetuned run is better",
        fontweight="bold",
        y=1.01,
    )
    savefig(fig, OUT / "transition" / "05_ftcombined_delta_curves.png")

    dataset_delta_root = OUT / "transition" / "07_dataset_specific_delta_curves"
    dataset_delta_root.mkdir(parents=True, exist_ok=True)
    plot_curve_legend(dataset_delta_root / "00_delta_curves_legend.png")
    for group, ft_run in GROUP_TO_RUN.items():
        ft_group = data[ft_run]
        baseline_group = baseline.per_group[baseline.per_group["group"] == group].copy()
        finetuned_group = ft_group.per_group[ft_group.per_group["group"] == group].copy()
        merged_group = baseline_group.merge(
            finetuned_group,
            on=["strategy", "step", "group"],
            suffixes=("_baseline", "_ftcombined"),
        )
        group_dir = dataset_delta_root / group.lower().replace("-", "_").replace(" ", "_")
        group_dir.mkdir(parents=True, exist_ok=True)
        for metric_col, title, lower_is_better, scale in delta_specs:
            plot_transition_delta_curve(
                merged_group,
                metric_col=metric_col,
                ylabel=title,
                lower_is_better=lower_is_better,
                scale=scale,
                out_path=group_dir / f"05_{metric_col.replace('_mean', '')}_delta_curve.png",
            )

    fig, axes = plt.subplots(len(GROUP_TO_RUN), len(delta_specs), figsize=(18.0, 15.5), sharex=True)
    for row_idx, (group, ft_run) in enumerate(GROUP_TO_RUN.items()):
        ft_group = data[ft_run]
        baseline_group = baseline.per_group[baseline.per_group["group"] == group].copy()
        finetuned_group = ft_group.per_group[ft_group.per_group["group"] == group].copy()
        merged_group = baseline_group.merge(
            finetuned_group,
            on=["strategy", "step", "group"],
            suffixes=("_baseline", "_ftcombined"),
        )
        for col_idx, (metric_col, title, lower_is_better, scale) in enumerate(delta_specs):
            ax = axes[row_idx, col_idx]
            for strategy in STRATEGIES:
                sub = merged_group[merged_group["strategy"] == strategy].sort_values("step").copy()
                if lower_is_better:
                    delta = scale * (sub[f"{metric_col}_baseline"] - sub[f"{metric_col}_ftcombined"])
                else:
                    delta = scale * (sub[f"{metric_col}_ftcombined"] - sub[f"{metric_col}_baseline"])
                ax.plot(
                    sub["step"],
                    delta,
                    color=COLORS[strategy],
                    linewidth=LINEWIDTHS[strategy],
                    linestyle=LINESTYLES[strategy],
                    marker=MARKERS[strategy],
                    markersize=3.2,
                    markevery=2,
                )
            ax.axhline(0, color="black", lw=0.9, alpha=0.55)
            ax.grid(alpha=0.22)
            ax.tick_params(axis="both", labelsize=11)
            if row_idx == 0:
                ax.set_title(title, fontweight="bold", fontsize=15)
            if col_idx == 0:
                ax.set_ylabel(f"{group}\n{title}", fontsize=13)
            if row_idx == len(GROUP_TO_RUN) - 1:
                ax.set_xlabel("Prompting Step", fontsize=14)
    savefig(fig, OUT / "transition" / "08_dataset_specific_delta_grid.png")

    overall = tables["transition_overall_ftcombined_gains.csv"].copy().set_index("strategy").loc[STRATEGIES]
    dumbbell_specs = [
        ("iou_mean_baseline", "iou_mean_ftcombined", "IoU", False),
        ("ece_mean_baseline", "ece_mean_ftcombined", "ECE", True),
        ("hausdorff95_mean_baseline", "hausdorff95_mean_ftcombined", "HD95", True),
        ("nll_mean_baseline", "nll_mean_ftcombined", "NLL", True),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 8.8))
    for ax, (base_col, ft_col, title, lower_is_better) in zip(axes.flatten(), dumbbell_specs):
        y = np.arange(len(STRATEGIES))
        for yi, strategy in zip(y, STRATEGIES):
            base_val = float(overall.loc[strategy, base_col])
            ft_val = float(overall.loc[strategy, ft_col])
            ax.plot([base_val, ft_val], [yi, yi], color=COLORS[strategy], lw=2.4, alpha=0.85)
            ax.scatter(base_val, yi, s=55, color="white", edgecolor=COLORS[strategy], linewidth=2.0, zorder=3)
            ax.scatter(ft_val, yi, s=70, color=COLORS[strategy], edgecolor="black" if strategy == "bald_laplace" else "white", linewidth=1.0, zorder=4)
        ax.set_yticks(y)
        ax.set_yticklabels([LABELS[s] for s in STRATEGIES])
        ax.set_title(f"{title} ({'lower' if lower_is_better else 'higher'} is better)", fontweight="bold")
        ax.grid(axis="x", alpha=0.25)
        ax.set_xlabel("Baseline  o   FT  ●")
    fig.suptitle(
        "Final-step shift from baseline SAM to combined finetuned SAM\n"
        "Open marker = baseline, filled marker = finetuned",
        fontweight="bold",
        y=1.01,
    )
    savefig(fig, OUT / "transition" / "06_ftcombined_final_dumbbells.png")


def plot_all_strategies_package(baseline: RunData) -> None:
    folder = OUT / "all_strategies"
    plot_dominance_matrix_legend(
        folder / "00_dominance_matrix_legend.png",
        strategies=STRATEGIES,
        include_tie=True,
    )
    plot_all_strategies_budget_dominance_matrix(
        baseline,
        folder / "01_budget_dominance_matrix.png",
    )
    plot_all_strategies_strata_dominance_matrix(
        baseline,
        folder / "02_strata_dominance_matrix.png",
    )
    plot_budget_takeover_timeline(
        build_budget_timeline_df(baseline, STRATEGIES),
        out_path=folder / "03_budget_takeover_timeline.png",
        title_line="Baseline SAM: budget takeover across all strategies",
        subtitle_line="Winner among BALD / Entropy / GT-based / MC Dropout / Random / MaxDist",
    )
    plot_budget_bump_chart(
        build_budget_wins_long(baseline, STRATEGIES),
        STRATEGIES,
        out_path=folder / "04_budget_bump_chart.png",
        title_line="Baseline SAM: budget race across all strategies",
        subtitle_line="Dataset wins by budget among all six strategies",
    )


def plot_no_gt_package(baseline: RunData) -> None:
    folder = OUT / "no_gt"
    plot_dominance_matrix_legend(
        folder / "00_dominance_matrix_legend.png",
        strategies=NO_GT_STRATEGIES,
        include_tie=True,
    )
    plot_all_strategies_budget_dominance_matrix(
        baseline,
        folder / "01_budget_dominance_matrix.png",
        strategies=NO_GT_STRATEGIES,
        title_line="Baseline SAM: budget dominance without GT-based",
        subtitle_line="Winner among BALD / Entropy / MC Dropout / Random / MaxDist",
    )
    plot_all_strategies_strata_dominance_matrix(
        baseline,
        folder / "02_strata_dominance_matrix.png",
        strategies=NO_GT_STRATEGIES,
        title_line="Baseline SAM: strata dominance without GT-based",
    )
    plot_budget_takeover_timeline(
        build_budget_timeline_df(baseline, NO_GT_STRATEGIES),
        out_path=folder / "03_budget_takeover_timeline.png",
        title_line="Baseline SAM: budget takeover without GT-based",
        subtitle_line="Winner among BALD / Entropy / MC Dropout / Random / MaxDist",
    )
    plot_budget_bump_chart(
        build_budget_wins_long(baseline, NO_GT_STRATEGIES),
        NO_GT_STRATEGIES,
        out_path=folder / "04_budget_bump_chart.png",
        title_line="Baseline SAM: budget race without GT-based",
        subtitle_line="Dataset wins by budget among BALD / Entropy / MC Dropout / Random / MaxDist",
    )


def write_tables(tables: dict[str, pd.DataFrame]) -> None:
    for name, df in tables.items():
        df.to_csv(OUT / "tables" / name, index=False)


def write_readme(victory_tables: dict[str, pd.DataFrame], transition_tables: dict[str, pd.DataFrame]) -> None:
    dataset_wins = victory_tables["victory_baseline_dataset_wins.csv"].set_index("metric")
    strata_wins = victory_tables["victory_baseline_strata_wins.csv"].set_index("metric")
    overall_gains = transition_tables["transition_overall_ftcombined_gains.csv"].set_index("strategy")
    winner_shift = transition_tables["transition_winner_shift.csv"]

    lines = [
        "# Medical SAM story",
        "",
        "This folder splits the new medical results into two packages:",
        "- `victory/`: where BALD (Laplace) wins clearly.",
        "- `transition/`: how the strategy ranking changes when moving from baseline SAM to finetuned SAM.",
        "",
        "## Package 1: BALD victory",
        "",
        f"- In `run_baseline_v3`, BALD wins **{int(dataset_wins.loc['Balanced', 'bald_laplace'])}/5** datasets on the uncertainty-only balanced score.",
        f"- In the same run, BALD wins **ECE {int(dataset_wins.loc['ECE', 'bald_laplace'])}/5**, **HD95 {int(dataset_wins.loc['HD95', 'bald_laplace'])}/5**, and **NLL {int(dataset_wins.loc['NLL', 'bald_laplace'])}/5** datasets among `BALD / Entropy / MC Dropout`.",
        f"- Across morphology strata, BALD wins **ECE {int(strata_wins.loc['ECE', 'bald_laplace'])}/12**, **HD95 {int(strata_wins.loc['HD95', 'bald_laplace'])}/12**, and **NLL {int(strata_wins.loc['NLL', 'bald_laplace'])}/12** strata.",
        "- The cleanest baseline BALD showcases are BUSI, ISIC2018, and KvasirSEG. DRIVE is also useful if you want a boundary/calibration story instead of a pure IoU story.",
        "- For curve quantification, use `tables/victory_baseline_budget_auc.csv` for metric-wise budget AUC / budget mean, `tables/victory_baseline_budget_rank.csv` for the cleanest single-number summary across mixed higher/lower metrics, `victory_baseline_budget_wins_*.csv` for checkpoint-wise wins at steps 2 / 5 / 10 / 15 / full, and the strata tables for family/bin-level morphology evidence.",
        "",
        "## Package 2: baseline -> finetuned SAM",
        "",
        f"- Moving from `run_baseline_v3` to `run_ft_combined`, every strategy improves in absolute quality, but BALD gains the least on average IoU: **+{overall_gains.loc['bald_laplace', 'iou_gain_pp']:.1f} pp** vs **+{overall_gains.loc['entropy', 'iou_gain_pp']:.1f} pp** for Entropy and **+{overall_gains.loc['bald_mcdropout', 'iou_gain_pp']:.1f} pp** for MC Dropout.",
        f"- BALD also gets a smaller ECE reduction under combined finetuning: **{overall_gains.loc['bald_laplace', 'ece_gain_pp']:+.1f} pp** vs **{overall_gains.loc['entropy', 'ece_gain_pp']:+.1f} pp** for Entropy.",
        "- In the uncertainty-only trio, BALD is the baseline winner on 4/5 datasets, and that edge disappears after dataset-specific finetuning.",
        "- Dataset-by-dataset winner summary:",
    ]
    for group in GROUP_TO_RUN:
        sub = winner_shift[winner_shift["group"] == group]
        trio_base = sub[sub["stage"] == "baseline"]["trio_winner"].iloc[0]
        trio_ft = sub[sub["stage"] == "finetuned"]["trio_winner"].iloc[0]
        if trio_base == trio_ft:
            lines.append(f"- `{group}`: trio winner remains `{trio_ft}`.")
        else:
            lines.append(f"- `{group}`: trio winner changes from `{trio_base}` to `{trio_ft}`.")
    lines.extend([
        "",
        "## Recommended starting figures",
        "",
        "- `victory/01_baseline_scoreboard.png`",
        "- `victory/00_dominance_matrix_legend.png`",
        "- `victory/01b_budget_wins_iou.png` and `victory/01b_budget_wins_ece.png`",
        "- `victory/01c_budget_dominance_matrix.png`",
        "- `victory/01d_budget_takeover_timeline.png`",
        "- `victory/01e_budget_bump_chart.png`",
        "- `victory/02_iou_curve.png` and `victory/02_iou_curve_by_dataset.png`",
        "- `victory/02_ece_curve.png` and `victory/02_ece_curve_by_dataset.png`",
        "- `victory/03_baseline_advantage_heatmap.png`",
        "- `victory/05_strata_dominance_matrix.png`",
        "- `all_strategies/01_budget_dominance_matrix.png`, `all_strategies/02_strata_dominance_matrix.png`, `all_strategies/03_budget_takeover_timeline.png`, and `all_strategies/04_budget_bump_chart.png` for the full six-strategy view",
        "- `no_gt/01_budget_dominance_matrix.png`, `no_gt/02_strata_dominance_matrix.png`, `no_gt/03_budget_takeover_timeline.png`, and `no_gt/04_budget_bump_chart.png` for the five-strategy view without GT-based",
        "- `transition/01_trio_baseline_vs_ftcombined_curves.png`",
        "- `transition/05_ftcombined_delta_curves.png` and `transition/06_ftcombined_final_dumbbells.png`",
        "- `transition/03_dataset_specific_gain_heatmaps.png`",
    ])
    (OUT / "README.md").write_text("\n".join(lines))


def main() -> None:
    ensure_dirs()
    data = {run: load_run(run) for run in RUNS if (BASE / run / "aggregate.csv").exists()}
    victory_tables = build_victory_tables(data["run_baseline_v3"])
    transition_tables = build_transition_tables(data)
    ft_dataset_specific = stitch_dataset_specific_run(data)
    ft_combined_tables = build_run_tables(data["run_ft_combined"])
    ft_dataset_specific_tables = build_run_tables(ft_dataset_specific)
    write_tables({**victory_tables, **transition_tables})
    write_prefixed_tables("ft_combined", ft_combined_tables)
    write_prefixed_tables("ft_dataset_specific", ft_dataset_specific_tables)
    plot_victory_package(data["run_baseline_v3"], victory_tables)
    plot_transition_package(data, transition_tables)
    plot_all_strategies_package(data["run_baseline_v3"])
    plot_no_gt_package(data["run_baseline_v3"])
    plot_finetuned_bundle(data["run_ft_combined"], OUT / "ft_combined")
    plot_finetuned_bundle(ft_dataset_specific, OUT / "ft_dataset_specific")
    write_readme(victory_tables, transition_tables)
    print(f"Wrote outputs to: {OUT}")


if __name__ == "__main__":
    main()
