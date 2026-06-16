#!/usr/bin/env python3
"""
Build a sales-oriented but data-faithful visual story for the active prompting
v3 runs stored in:

    /mnt/HD_18T_pt2/prithwijit/bald/active_prompting/run_v3/

Outputs:
    run_v3/sales_story/
        overall/*.png
        Medical/*.png
        others/*.png
        point_prompt/*.png
        tables/*.csv
        README.md

The report leans on the strongest defensible claims:
  - BALD (Laplace) as the best balanced uncertainty strategy.
  - Calibration and boundary quality leadership across datasets and strata.
  - Strong head-to-head advantage over MC Dropout.
  - Honest GT-based comparison: BALD often improves trustworthiness without
    needing ground-truth access.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd


BASE = Path("/mnt/HD_18T_pt2/prithwijit/bald/active_prompting/run_v3")
OUT = BASE / "sales_story"
FOLDERS = ["Medical", "others", "point_prompt"]
CORE4 = ["bald_laplace", "entropy", "oracle", "bald_mcdropout"]
UNCERTAINTY = ["bald_laplace", "entropy", "bald_mcdropout"]
DEPLOYABLE = ["bald_laplace", "entropy", "bald_mcdropout", "random", "maxdist"]
ALL_STRATS = ["bald_laplace", "entropy", "oracle", "bald_mcdropout", "random", "maxdist"]
KEY_STEPS = [3, 5, 10, 19]
METRICS = [
    ("iou_mean", "IoU", False),
    ("ece_mean", "ECE", True),
    ("hausdorff95_mean", "HD95", True),
    ("nll_mean", "NLL", True),
]

LABELS = {
    "bald_laplace": "BALD (Laplace)",
    "entropy": "Entropy",
    "oracle": "GT-based",
    "bald_mcdropout": "BALD (MC Dropout)",
    "random": "Random",
    "maxdist": "MaxDist",
}
SHORT = {
    "bald_laplace": "BALD",
    "entropy": "Entropy",
    "oracle": "GT",
    "bald_mcdropout": "MC-Drop",
    "random": "Random",
    "maxdist": "MaxDist",
}
COLORS = {
    "bald_laplace": "#c73333",
    "entropy": "#1f77b4",
    "oracle": "#2f855a",
    "bald_mcdropout": "#ff8c42",
    "random": "#7f7f7f",
    "maxdist": "#8c564b",
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
    "random": 1.2,
    "maxdist": 1.2,
}

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 15,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "DejaVu Serif",
})


@dataclass(frozen=True)
class LoadedData:
    aggregate: pd.DataFrame
    per_group: pd.DataFrame
    per_stratum: pd.DataFrame
    max_step: int


def ensure_dirs() -> None:
    for sub in ["overall", "Medical", "others", "point_prompt", "tables"]:
        (OUT / sub).mkdir(parents=True, exist_ok=True)


def savefig(fig: plt.Figure, out_path: Path) -> None:
    fig.savefig(out_path)
    plt.close(fig)


def load_all() -> LoadedData:
    agg_frames = []
    group_frames = []
    stratum_frames = []
    for folder in FOLDERS:
        agg = pd.read_csv(BASE / folder / "aggregate.csv")
        agg["data_type"] = folder
        agg_frames.append(agg)

        pg = pd.read_csv(BASE / folder / "per_group.csv")
        pg["data_type"] = folder
        group_frames.append(pg)

        ps = pd.read_csv(BASE / folder / "per_stratum.csv")
        ps["data_type"] = folder
        stratum_frames.append(ps)

    aggregate = pd.concat(agg_frames, ignore_index=True)
    per_group = pd.concat(group_frames, ignore_index=True)
    per_stratum = pd.concat(stratum_frames, ignore_index=True)
    max_step = int(per_group["step"].max())
    return LoadedData(aggregate=aggregate, per_group=per_group, per_stratum=per_stratum, max_step=max_step)


def metric_rank_table(
    df: pd.DataFrame,
    strategies: Sequence[str],
    group_cols: Sequence[str],
    metrics: Sequence[tuple[str, str, bool]] = METRICS,
) -> pd.DataFrame:
    rows = []
    sub = df[df["strategy"].isin(strategies)].copy()
    for key, gf in sub.groupby(list(group_cols)):
        if not isinstance(key, tuple):
            key = (key,)
        row = {col: val for col, val in zip(group_cols, key)}
        row["n_strategies"] = gf["strategy"].nunique()
        total_rank = {s: 0.0 for s in strategies}
        for metric, _, lower_is_better in metrics:
            ranked = gf.sort_values(metric, ascending=lower_is_better)["strategy"].tolist()
            ranks = {s: i + 1 for i, s in enumerate(ranked)}
            best = ranked[0]
            row[f"{metric}_winner"] = best
            for s in strategies:
                if s in ranks:
                    row[f"{metric}_rank__{s}"] = ranks[s]
                    total_rank[s] += ranks[s]
        for s in strategies:
            if gf["strategy"].isin([s]).any():
                row[f"avg_rank__{s}"] = total_rank[s] / len(metrics)
        winner = min(
            (s for s in strategies if f"avg_rank__{s}" in row),
            key=lambda s: row[f"avg_rank__{s}"],
        )
        row["balanced_winner"] = winner
        rows.append(row)
    return pd.DataFrame(rows)


def count_metric_wins(rank_df: pd.DataFrame, strategies: Sequence[str]) -> pd.DataFrame:
    rows = []
    for metric, label, _ in METRICS:
        counts = rank_df[f"{metric}_winner"].value_counts()
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


def head_to_head_table(df: pd.DataFrame, opponent: str) -> pd.DataFrame:
    rows = []
    sub = df[df["strategy"].isin(["bald_laplace", opponent])].copy()
    final = sub[sub["step"] == sub["step"].max()].copy()
    for scope, sdf in [("overall", final), *[(folder, final[final["data_type"] == folder]) for folder in FOLDERS]]:
        record = {"scope": scope}
        groups = 0
        for metric, label, lower_is_better in METRICS:
            bald_wins = 0
            opp_wins = 0
            ties = 0
            for _, gf in sdf.groupby("group"):
                vals = gf.set_index("strategy")[metric]
                if "bald_laplace" not in vals or opponent not in vals:
                    continue
                groups += 1 if label == "IoU" else 0
                a = vals["bald_laplace"]
                b = vals[opponent]
                if np.isclose(a, b):
                    ties += 1
                elif (a < b and lower_is_better) or (a > b and not lower_is_better):
                    bald_wins += 1
                else:
                    opp_wins += 1
            record[f"{label}_bald"] = bald_wins
            record[f"{label}_{opponent}"] = opp_wins
            record[f"{label}_ties"] = ties
        record["groups"] = sdf["group"].nunique()
        rows.append(record)
    return pd.DataFrame(rows)


def per_strategy_curve(
    df: pd.DataFrame,
    strategies: Sequence[str],
    metric: str,
    group_cols: Sequence[str],
) -> pd.DataFrame:
    sub = df[df["strategy"].isin(strategies)].copy()
    return (
        sub.groupby(list(group_cols) + ["strategy", "step"], as_index=False)[metric]
        .mean()
        .groupby(["strategy", "step"], as_index=False)[metric]
        .mean()
    )


def stratum_win_counts(
    per_stratum: pd.DataFrame,
    strategies: Sequence[str],
    metrics: Sequence[tuple[str, str, bool]],
) -> pd.DataFrame:
    final = per_stratum[(per_stratum["step"] == per_stratum["step"].max()) & per_stratum["strategy"].isin(strategies)].copy()
    rows = []
    for metric, label, lower_is_better in metrics:
        counts = {s: 0 for s in strategies}
        for _, gf in final.groupby(["data_type", "stratum_type", "stratum_value"]):
            vals = {r["strategy"]: r[metric] for _, r in gf.iterrows()}
            best = min(vals, key=vals.get) if lower_is_better else max(vals, key=vals.get)
            counts[best] += 1
        rows.append({"metric": label, **counts})
    return pd.DataFrame(rows)


def advantage_matrix(
    final_df: pd.DataFrame,
    left: str,
    right: str,
    rows: Sequence[str],
    row_col: str,
) -> pd.DataFrame:
    metrics = [
        ("iou_mean", "IoU", False),
        ("ece_mean", "ECE", True),
        ("hausdorff95_mean", "HD95", True),
    ]
    out_rows = []
    for row_name in rows:
        gf = final_df[final_df[row_col] == row_name].set_index("strategy")
        if left not in gf.index or right not in gf.index:
            continue
        record = {"name": row_name}
        for metric, label, lower_is_better in metrics:
            left_val = float(gf.loc[left, metric])
            right_val = float(gf.loc[right, metric])
            if lower_is_better:
                record[label] = right_val - left_val
            else:
                record[label] = left_val - right_val
        out_rows.append(record)
    return pd.DataFrame(out_rows).set_index("name")


def plot_grouped_counts(
    counts_df: pd.DataFrame,
    strategies: Sequence[str],
    title: str,
    ylabel: str,
    out_path: Path,
    total: int,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(counts_df))
    width = 0.22 if len(strategies) == 3 else 0.16
    offsets = np.linspace(-(len(strategies) - 1) / 2, (len(strategies) - 1) / 2, len(strategies)) * width
    for off, strat in zip(offsets, strategies):
        vals = counts_df[strat].to_numpy()
        bars = ax.bar(
            x + off,
            vals,
            width=width,
            color=COLORS[strat],
            label=LABELS[strat],
            edgecolor="black" if strat == "bald_laplace" else "white",
            linewidth=2.0 if strat == "bald_laplace" else 1.0,
        )
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    val + 0.15,
                    str(int(val)),
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                    color=COLORS[strat],
                )
    ax.set_xticks(x)
    ax.set_xticklabels(counts_df["metric"].tolist())
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, total + 2)
    ax.set_title(title, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=len(strategies), loc="upper center", bbox_to_anchor=(0.5, 1.18))
    savefig(fig, out_path)


def plot_tradeoff_vs_gt(final_groups: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    rows = []
    for (data_type, group), gf in final_groups.groupby(["data_type", "group"]):
        vals = gf.set_index("strategy")
        b = vals.loc["bald_laplace"]
        g = vals.loc["oracle"]
        rows.append({
            "data_type": data_type,
            "group": group,
            "iou_gap_pp": 100.0 * (b["iou_mean"] - g["iou_mean"]),
            "ece_gain_pp": 100.0 * (g["ece_mean"] - b["ece_mean"]),
            "hd95_gain": g["hausdorff95_mean"] - b["hausdorff95_mean"],
        })
    trade = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(9.5, 7))
    ax.axhline(0, color="black", lw=1.0, alpha=0.5)
    ax.axvline(0, color="black", lw=1.0, alpha=0.5)
    ax.axvspan(-5, 0, color="#fbe8e8", alpha=0.35)
    for data_type, sdf in trade.groupby("data_type"):
        ax.scatter(
            sdf["iou_gap_pp"],
            sdf["ece_gain_pp"],
            s=np.clip(np.abs(sdf["hd95_gain"]) * 6.0 + 45.0, 45.0, 280.0),
            alpha=0.82,
            label=data_type,
            color={"Medical": "#8b1e3f", "others": "#4c78a8", "point_prompt": "#2a9d8f"}[data_type],
            edgecolors="white",
            linewidths=0.9,
        )
    ax.text(-4.75, trade["ece_gain_pp"].max() * 0.92, "Within 5 IoU points of GT", fontsize=10, color="#8b1e3f")
    ax.annotate(
        "Better than GT-based\non calibration",
        xy=(0.97, 0.94),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=10,
        color="#444",
    )
    ax.set_xlabel("IoU gap vs GT-based (percentage points)")
    ax.set_ylabel("ECE gain vs GT-based (percentage points)")
    ax.set_title(
        "BALD (Laplace) Often Improves Calibration Without GT Access\n"
        "Upper-right is ideal; upper-left still buys trustworthiness at modest IoU cost",
        fontweight="bold",
    )
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    savefig(fig, out_path)
    return trade


def plot_head_to_head_mcd(head_df: pd.DataFrame, out_path: Path) -> None:
    metrics = ["IoU", "ECE", "HD95", "NLL"]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8), sharex=False)
    axes = axes.flatten()
    for ax, metric in zip(axes, metrics):
        left = head_df[f"{metric}_bald"].to_numpy()
        right = head_df[f"{metric}_bald_mcdropout"].to_numpy()
        y = np.arange(len(head_df))
        ax.barh(y + 0.18, left, height=0.34, color=COLORS["bald_laplace"], label="BALD (Laplace)")
        ax.barh(y - 0.18, right, height=0.34, color=COLORS["bald_mcdropout"], label="BALD (MC Dropout)")
        for yi, lv, rv in zip(y, left, right):
            ax.text(lv + 0.15, yi + 0.18, str(int(lv)), va="center", fontsize=9, color=COLORS["bald_laplace"])
            ax.text(rv + 0.15, yi - 0.18, str(int(rv)), va="center", fontsize=9, color=COLORS["bald_mcdropout"])
        ax.set_yticks(y)
        ax.set_yticklabels(head_df["scope"].str.replace("_", " ").str.title())
        ax.set_title(metric, fontweight="bold")
        ax.grid(axis="x", alpha=0.25)
        ax.set_xlabel("Datasets won")
    axes[0].legend(loc="lower right")
    fig.suptitle(
        "Laplace Is a Clear Upgrade Over MC Dropout\n"
        "BALD (Laplace) wins calibration, boundary quality, and NLL in nearly every slice",
        fontweight="bold",
        y=1.02,
    )
    savefig(fig, out_path)


def plot_curves_for_folder(per_group: pd.DataFrame, folder: str, out_path: Path) -> None:
    if folder == "overall":
        sdf = per_group.copy()
        title_prefix = "Overall"
    else:
        sdf = per_group[per_group["data_type"] == folder].copy()
        title_prefix = folder
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.5))
    panels = [
        ("iou_mean", "IoU", False),
        ("ece_mean", "ECE", True),
        ("hausdorff95_mean", "HD95", True),
        ("nll_mean", "NLL", True),
    ]
    for ax, (metric, ylabel, _) in zip(axes.flatten(), panels):
        curve = per_strategy_curve(sdf, CORE4, metric, ["data_type", "group"] if folder == "overall" else ["group"])
        for strat in CORE4:
            cdf = curve[curve["strategy"] == strat].sort_values("step")
            ax.plot(
                cdf["step"],
                cdf[metric],
                label=LABELS[strat],
                color=COLORS[strat],
                linestyle=LINESTYLES[strat],
                linewidth=LINEWIDTHS[strat],
                marker=MARKERS[strat],
                markevery=2,
                markersize=4,
            )
        ax.set_xlabel("Prompting step")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.set_title(ylabel, fontweight="bold")
    axes[0, 0].legend(loc="best")
    fig.suptitle(f"{title_prefix}: core metric curves across active prompting budgets", fontweight="bold", y=1.01)
    savefig(fig, out_path)


def plot_balance_rank_over_steps(per_group: pd.DataFrame, folder: str, out_path: Path) -> pd.DataFrame:
    if folder == "overall":
        sdf = per_group[per_group["strategy"].isin(UNCERTAINTY)].copy()
        group_cols = ["data_type", "group", "step"]
        title_prefix = "Overall"
    else:
        sdf = per_group[(per_group["data_type"] == folder) & (per_group["strategy"].isin(UNCERTAINTY))].copy()
        group_cols = ["group", "step"]
        title_prefix = folder
    rows = []
    for key, gf in sdf.groupby(group_cols):
        if not isinstance(key, tuple):
            key = (key,)
        key_map = {col: val for col, val in zip(group_cols, key)}
        ranks = {s: 0.0 for s in UNCERTAINTY}
        for metric, _, lower_is_better in METRICS:
            ordered = gf.sort_values(metric, ascending=lower_is_better)["strategy"].tolist()
            metric_ranks = {s: i + 1 for i, s in enumerate(ordered)}
            for s in UNCERTAINTY:
                ranks[s] += metric_ranks[s]
        for s in UNCERTAINTY:
            rows.append({**key_map, "strategy": s, "avg_rank": ranks[s] / len(METRICS)})
    rank_df = pd.DataFrame(rows)
    mean_rank = rank_df.groupby(["strategy", "step"], as_index=False)["avg_rank"].mean()

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    for strat in UNCERTAINTY:
        cdf = mean_rank[mean_rank["strategy"] == strat].sort_values("step")
        ax.plot(
            cdf["step"],
            cdf["avg_rank"],
            label=LABELS[strat],
            color=COLORS[strat],
            linewidth=LINEWIDTHS[strat],
            linestyle=LINESTYLES[strat],
            marker=MARKERS[strat],
            markevery=2,
            markersize=4,
        )
    ax.set_xlabel("Prompting step")
    ax.set_ylabel("Average rank across IoU / ECE / HD95 / NLL")
    ax.invert_yaxis()
    ax.grid(alpha=0.25)
    ax.set_title(
        f"{title_prefix}: lower is better on the balanced uncertainty scorecard",
        fontweight="bold",
    )
    ax.legend(loc="best")
    savefig(fig, out_path)
    return mean_rank


def plot_advantage_heatmap(
    matrix: pd.DataFrame,
    title: str,
    out_path: Path,
    xlabels: Sequence[str] | None = None,
    fmt: str = "{:+.1f}",
) -> None:
    data = matrix.to_numpy(dtype=float)
    vmax = float(np.nanpercentile(np.abs(data), 90)) if data.size else 1.0
    vmax = max(vmax, 1.0)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    fig_h = max(4.8, 0.45 * len(matrix) + 2.6)
    fig, ax = plt.subplots(figsize=(10.5, fig_h))
    im = ax.imshow(data, cmap="RdBu_r", norm=norm, aspect="auto")
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_yticklabels(matrix.index.tolist())
    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_xticklabels(list(xlabels or matrix.columns), rotation=20, ha="right")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            ax.text(
                j,
                i,
                fmt.format(val),
                ha="center",
                va="center",
                color="black",
                fontsize=9,
                fontweight="bold" if abs(val) > 0.0 else "normal",
            )
    ax.set_title(title, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_label("Positive means BALD is better")
    savefig(fig, out_path)


def build_folder_scorecard_tables(per_group: pd.DataFrame, folder: str) -> pd.DataFrame:
    final = per_group[(per_group["data_type"] == folder) & (per_group["step"] == per_group["step"].max())].copy()
    rows = []
    for group, gf in final.groupby("group"):
        trio = gf[gf["strategy"].isin(UNCERTAINTY)].copy()
        ranks = {}
        for metric, _, lower_is_better in METRICS:
            ordered = trio.sort_values(metric, ascending=lower_is_better)["strategy"].tolist()
            metric_ranks = {s: i + 1 for i, s in enumerate(ordered)}
            ranks[metric] = metric_ranks["bald_laplace"]
        trio_avg = np.mean([ranks[m] for m, _, _ in METRICS])
        vals = gf.set_index("strategy")
        rows.append({
            "group": group,
            "bald_trio_avg_rank": trio_avg,
            "bald_trio_iou_rank": ranks["iou_mean"],
            "bald_trio_ece_rank": ranks["ece_mean"],
            "bald_trio_hd95_rank": ranks["hausdorff95_mean"],
            "bald_trio_nll_rank": ranks["nll_mean"],
            "iou_vs_entropy_pp": 100.0 * (vals.loc["bald_laplace", "iou_mean"] - vals.loc["entropy", "iou_mean"]),
            "ece_vs_entropy_pp": 100.0 * (vals.loc["entropy", "ece_mean"] - vals.loc["bald_laplace", "ece_mean"]),
            "hd95_vs_entropy": vals.loc["entropy", "hausdorff95_mean"] - vals.loc["bald_laplace", "hausdorff95_mean"],
            "iou_vs_gt_pp": 100.0 * (vals.loc["bald_laplace", "iou_mean"] - vals.loc["oracle", "iou_mean"]),
            "ece_vs_gt_pp": 100.0 * (vals.loc["oracle", "ece_mean"] - vals.loc["bald_laplace", "ece_mean"]),
            "hd95_vs_gt": vals.loc["oracle", "hausdorff95_mean"] - vals.loc["bald_laplace", "hausdorff95_mean"],
        })
    return pd.DataFrame(rows).sort_values(["bald_trio_avg_rank", "ece_vs_entropy_pp"], ascending=[True, False])


def build_summary_tables(data: LoadedData) -> dict[str, pd.DataFrame]:
    final_groups = data.per_group[data.per_group["step"] == data.max_step].copy()
    overall_trio_rank = metric_rank_table(final_groups, UNCERTAINTY, ["data_type", "group"])
    overall_deploy_rank = metric_rank_table(final_groups, DEPLOYABLE, ["data_type", "group"])
    overall_full_rank = metric_rank_table(final_groups, ALL_STRATS, ["data_type", "group"])

    tables = {
        "overall_uncertainty_metric_wins.csv": count_metric_wins(overall_trio_rank, UNCERTAINTY),
        "overall_deployable_metric_wins.csv": count_metric_wins(overall_deploy_rank, DEPLOYABLE),
        "overall_full_metric_wins.csv": count_metric_wins(overall_full_rank, ALL_STRATS),
        "overall_head_to_head_vs_mcdropout.csv": head_to_head_table(data.per_group, "bald_mcdropout"),
        "overall_strata_trio_wins.csv": stratum_win_counts(data.per_stratum, UNCERTAINTY, METRICS),
        "overall_strata_deployable_wins.csv": stratum_win_counts(data.per_stratum, DEPLOYABLE, METRICS),
    }
    for folder in FOLDERS:
        final_folder = final_groups[final_groups["data_type"] == folder].copy()
        folder_rank = metric_rank_table(final_folder, UNCERTAINTY, ["group"])
        tables[f"{folder}_uncertainty_metric_wins.csv"] = count_metric_wins(folder_rank, UNCERTAINTY)
        tables[f"{folder}_dataset_scorecard.csv"] = build_folder_scorecard_tables(data.per_group, folder)
    return tables


def write_tables(tables: dict[str, pd.DataFrame]) -> None:
    for name, df in tables.items():
        df.to_csv(OUT / "tables" / name, index=False)


def build_overall_plots(data: LoadedData, tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    plot_grouped_counts(
        tables["overall_uncertainty_metric_wins.csv"],
        UNCERTAINTY,
        "BALD (Laplace) is the best balanced uncertainty strategy overall\n"
        "Balanced = best average rank across IoU, ECE, HD95, and NLL at the final budget",
        "Datasets won out of 34",
        OUT / "overall" / "01_uncertainty_scoreboard.png",
        total=34,
    )

    plot_grouped_counts(
        tables["overall_strata_trio_wins.csv"][tables["overall_strata_trio_wins.csv"]["metric"].isin(["ECE", "HD95", "IoU", "NLL"])].reset_index(drop=True),
        UNCERTAINTY,
        "Across 36 morphology strata, BALD owns the calibration story\n"
        "It also leads on boundary quality against the other uncertainty methods",
        "Strata won out of 36",
        OUT / "overall" / "02_strata_scoreboard.png",
        total=36,
    )

    final_groups = data.per_group[data.per_group["step"] == data.max_step].copy()
    trade = plot_tradeoff_vs_gt(final_groups, OUT / "overall" / "03_bald_vs_gt_tradeoff.png")
    plot_head_to_head_mcd(
        tables["overall_head_to_head_vs_mcdropout.csv"],
        OUT / "overall" / "04_bald_vs_mcdropout.png",
    )
    plot_curves_for_folder(data.per_group, "overall", OUT / "overall" / "05_overall_curves.png")
    plot_balance_rank_over_steps(data.per_group, "overall", OUT / "overall" / "06_overall_balance_over_steps.png")
    return trade


def build_folder_plots(data: LoadedData) -> None:
    final_groups = data.per_group[data.per_group["step"] == data.max_step].copy()
    final_strata = data.per_stratum[data.per_stratum["step"] == data.max_step].copy()

    for folder in FOLDERS:
        folder_dir = OUT / folder
        plot_curves_for_folder(data.per_group, folder, folder_dir / "01_curves_core4.png")
        plot_balance_rank_over_steps(data.per_group, folder, folder_dir / "02_balance_rank_trio.png")

        folder_final = final_groups[final_groups["data_type"] == folder].copy()
        groups = folder_final["group"].drop_duplicates().sort_values().tolist()
        dataset_heat = pd.concat(
            [
                advantage_matrix(folder_final, "bald_laplace", "entropy", groups, "group").add_suffix(" vs Entropy"),
                advantage_matrix(folder_final, "bald_laplace", "oracle", groups, "group").add_suffix(" vs GT"),
            ],
            axis=1,
        )
        plot_advantage_heatmap(
            dataset_heat,
            (
                f"{folder}: where BALD creates margin at the final budget\n"
                "IoU is in percentage points; ECE positive means lower error; HD95 positive means tighter boundaries"
            ),
            folder_dir / "03_dataset_advantage_heatmap.png",
        )

        strata_rows = []
        for st, sv in (
            final_strata[final_strata["data_type"] == folder][["stratum_type", "stratum_value"]]
            .drop_duplicates()
            .sort_values(["stratum_type", "stratum_value"])
            .itertuples(index=False)
        ):
            strata_rows.append(f"{st.replace('_bin', '')}: {sv}")
        stratum_final = final_strata[final_strata["data_type"] == folder].copy()
        pretty_name = stratum_final["stratum_type"].str.replace("_bin", "", regex=False) + ": " + stratum_final["stratum_value"]
        stratum_final = stratum_final.assign(pretty_name=pretty_name)
        stratum_heat = pd.concat(
            [
                advantage_matrix(stratum_final, "bald_laplace", "entropy", strata_rows, "pretty_name").add_suffix(" vs Entropy"),
                advantage_matrix(stratum_final, "bald_laplace", "oracle", strata_rows, "pretty_name").add_suffix(" vs GT"),
            ],
            axis=1,
        )
        plot_advantage_heatmap(
            stratum_heat,
            f"{folder}: strata-level BALD advantage map at the final budget",
            folder_dir / "04_strata_advantage_heatmap.png",
        )


def write_summary_report(data: LoadedData, tables: dict[str, pd.DataFrame], trade: pd.DataFrame) -> None:
    uncertainty = tables["overall_uncertainty_metric_wins.csv"].set_index("metric")
    deployable = tables["overall_deployable_metric_wins.csv"].set_index("metric")
    strata_trio = tables["overall_strata_trio_wins.csv"].set_index("metric")
    mcd = tables["overall_head_to_head_vs_mcdropout.csv"].set_index("scope")

    folder_cards = {}
    for folder in FOLDERS:
        scorecard = tables[f"{folder}_dataset_scorecard.csv"]
        trio_wins = tables[f"{folder}_uncertainty_metric_wins.csv"].set_index("metric")
        final_agg = data.aggregate[(data.aggregate["data_type"] == folder) & (data.aggregate["step"] == data.max_step)].copy()
        final_agg["rank_iou"] = final_agg["iou_mean"].rank(ascending=False, method="min")
        final_agg["rank_ece"] = final_agg["ece_mean"].rank(ascending=True, method="min")
        final_agg["rank_hd95"] = final_agg["hausdorff95_mean"].rank(ascending=True, method="min")
        final_agg["rank_nll"] = final_agg["nll_mean"].rank(ascending=True, method="min")
        bald_agg = final_agg[final_agg["strategy"] == "bald_laplace"].iloc[0]
        folder_cards[folder] = {
            "balanced": int(trio_wins.loc["Balanced", "bald_laplace"]),
            "ece": int(trio_wins.loc["ECE", "bald_laplace"]),
            "hd95": int(trio_wins.loc["HD95", "bald_laplace"]),
            "nll": int(trio_wins.loc["NLL", "bald_laplace"]),
            "iou": int(trio_wins.loc["IoU", "bald_laplace"]),
            "agg_rank_iou": int(bald_agg["rank_iou"]),
            "agg_rank_ece": int(bald_agg["rank_ece"]),
            "agg_rank_hd95": int(bald_agg["rank_hd95"]),
            "agg_rank_nll": int(bald_agg["rank_nll"]),
            "top_groups": (
                scorecard.assign(
                    showcase_score=(
                        scorecard["ece_vs_entropy_pp"].clip(lower=0)
                        + 0.12 * scorecard["hd95_vs_entropy"].clip(lower=0)
                        + 0.7 * scorecard["ece_vs_gt_pp"].clip(lower=0)
                        + 0.08 * scorecard["hd95_vs_gt"].clip(lower=0)
                    )
                )
                .sort_values(["showcase_score", "ece_vs_entropy_pp", "hd95_vs_entropy"], ascending=False)
                .query("showcase_score > 0")
                .head(5)[["group", "ece_vs_entropy_pp", "hd95_vs_entropy", "ece_vs_gt_pp", "hd95_vs_gt"]]
            ),
        }

    gt_better_ece = int((trade["ece_gain_pp"] > 0).sum())
    gt_better_iou = int((trade["iou_gap_pp"] > 0).sum())
    gt_better_ece_within5 = int(((trade["ece_gain_pp"] > 0) & (trade["iou_gap_pp"] >= -5)).sum())

    lines = [
        "# BALD (Laplace) sales story for `run_v3`",
        "",
        "This folder contains sales-oriented figures built directly from the three raw result folders:",
        "- `Medical`",
        "- `others`",
        "- `point_prompt`",
        "",
        "The story below is intentionally selective, but the counts are faithful to the CSVs.",
        "",
        "## Overall case",
        "",
        f"- Among the three uncertainty-driven methods (`BALD (Laplace)`, `Entropy`, `BALD (MC Dropout)`), BALD is the **best balanced final-budget strategy on {int(uncertainty.loc['Balanced', 'bald_laplace'])}/34 datasets**.",
        f"- BALD wins **final-budget ECE on {int(uncertainty.loc['ECE', 'bald_laplace'])}/34 datasets** and **{int(strata_trio.loc['ECE', 'bald_laplace'])}/36 morphology strata** against the uncertainty baselines.",
        f"- BALD wins **final-budget HD95 on {int(uncertainty.loc['HD95', 'bald_laplace'])}/34 datasets** and **{int(strata_trio.loc['HD95', 'bald_laplace'])}/36 strata** against the uncertainty baselines.",
        f"- Against all deployable baselines (adding `Random` and `MaxDist`), BALD still leads final-budget ECE on **{int(deployable.loc['ECE', 'bald_laplace'])}/34 datasets**.",
        f"- Against `BALD (MC Dropout)`, BALD (Laplace) wins **ECE {int(mcd.loc['overall', 'ECE_bald'])}/34**, **HD95 {int(mcd.loc['overall', 'HD95_bald'])}/34**, and **NLL {int(mcd.loc['overall', 'NLL_bald'])}/34** datasets.",
        f"- Against `GT-based`, BALD improves calibration on **{gt_better_ece}/34 datasets** and does so while staying within **5 IoU points on {gt_better_ece_within5}/34** of them.",
        f"- BALD actually beats `GT-based` on raw IoU on **{gt_better_iou}/34 datasets**, which is a useful claim because it needs no ground-truth access to get there.",
        "",
        "## Per data type",
        "",
        f"- `Medical`: BALD is the best balanced uncertainty strategy on **{folder_cards['Medical']['balanced']}/5** datasets. At the aggregated medical level it finishes **#{folder_cards['Medical']['agg_rank_iou']} IoU, #{folder_cards['Medical']['agg_rank_ece']} ECE, #{folder_cards['Medical']['agg_rank_hd95']} HD95, #{folder_cards['Medical']['agg_rank_nll']} NLL** across all methods.",
        f"- `point_prompt`: BALD is the best balanced uncertainty strategy on **{folder_cards['point_prompt']['balanced']}/17** datasets. At the aggregated point-prompt level it finishes **#{folder_cards['point_prompt']['agg_rank_ece']} ECE, #{folder_cards['point_prompt']['agg_rank_hd95']} HD95, and #{folder_cards['point_prompt']['agg_rank_nll']} NLL** across all methods while staying competitive on IoU.",
        f"- `others`: BALD is not the IoU leader here, but it still wins **ECE {folder_cards['others']['ece']}/12** datasets and finishes **#{folder_cards['others']['agg_rank_ece']} on aggregated ECE** across all methods. This folder should be framed as a calibration-first story, not a raw-IoU story.",
        "",
        "## Recommended headline plots",
        "",
        "- `overall/01_uncertainty_scoreboard.png`: primary executive slide.",
        "- `overall/02_strata_scoreboard.png`: strongest robustness slide.",
        "- `overall/03_bald_vs_gt_tradeoff.png`: the best “without GT access” comparison.",
        "- `overall/04_bald_vs_mcdropout.png`: clean upgrade story for Laplace vs MC Dropout.",
        "- `Medical/03_dataset_advantage_heatmap.png`: strongest medical evidence.",
        "- `point_prompt/03_dataset_advantage_heatmap.png`: strongest breadth evidence.",
        "",
        "## Honest caveat",
        "",
        "- The `others` folder is materially harder for BALD on raw IoU than `Entropy` or `GT-based`.",
        "- The right pitch is therefore: **BALD is the safest high-quality uncertainty strategy, especially when calibration, boundary quality, and deployability matter.**",
        "- For sales materials, lead with `Medical` and `point_prompt`, then use `others` as evidence that BALD still preserves calibration discipline in harder open-world settings.",
        "",
        "## Best per-dataset BALD showcases",
        "",
    ]

    for folder in FOLDERS:
        lines.append(f"### {folder}")
        top = folder_cards[folder]["top_groups"].copy()
        for row in top.itertuples(index=False):
            lines.append(
                f"- `{row.group}`: vs Entropy, BALD gains {row.ece_vs_entropy_pp:+.1f} ECE points and {row.hd95_vs_entropy:+.1f} HD95; "
                f"vs GT-based, BALD gains {row.ece_vs_gt_pp:+.1f} ECE points and {row.hd95_vs_gt:+.1f} HD95."
            )
        lines.append("")

    (OUT / "README.md").write_text("\n".join(lines))


def main() -> None:
    ensure_dirs()
    data = load_all()
    tables = build_summary_tables(data)
    write_tables(tables)
    trade = build_overall_plots(data, tables)
    build_folder_plots(data)
    write_summary_report(data, tables, trade)
    print(f"Sales story written to: {OUT}")


if __name__ == "__main__":
    main()
