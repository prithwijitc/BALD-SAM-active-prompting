"""
Cross-run analysis for medical active prompting results.

Type 1: BALD superiority — show BALD is the best acquisition strategy
Type 2: Baseline vs fine-tuned — how strategies evolve with SAM fine-tuning

Usage:
    python analyze_medical_runs.py
    python analyze_medical_runs.py --output_dir /path/to/output
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})

STRATEGY_COLORS = {
    "bald_laplace":   "#E63946",  # bold red
    "bald_mcdropout": "#F4A261",  # orange
    "entropy":        "#2A9D8F",  # teal
    "random":         "#6C757D",  # grey
    "maxdist":        "#457B9D",  # steel blue
    "oracle":         "#A8DADC",  # light cyan
}

STRATEGY_LABELS = {
    "bald_laplace":   "BALD (Laplace)",
    "bald_mcdropout": "BALD (MC Dropout)",
    "entropy":        "Entropy",
    "random":         "Random",
    "maxdist":        "Max Distance",
    "oracle":         "Oracle",
}

STRATEGY_ORDER = ["bald_laplace", "bald_mcdropout", "entropy", "random", "maxdist", "oracle"]

DATASET_LABELS = {
    "BUSI": "BUSI",
    "CheXmask": "CheXmask",
    "DRIVE": "DRIVE",
    "ISIC2018": "ISIC2018",
    "KvasirSEG": "KvasirSEG",
}

RUN_IDS = {
    "baseline_v3": "Baseline (Original SAM)",
    "ft_busi": "Fine-tuned (BUSI SAM)",
    "ft_chexmask": "Fine-tuned (CheXmask SAM)",
    "ft_drive": "Fine-tuned (DRIVE SAM)",
    "ft_isic2018": "Fine-tuned (ISIC2018 SAM)",
    "ft_kvasir_seg": "Fine-tuned (KvasirSEG SAM)",
    "ft_combined": "Fine-tuned (Combined SAM)",
}

# Map fine-tuned run to the dataset it targets
FT_DATASET_MAP = {
    "ft_busi": "BUSI",
    "ft_chexmask": "CheXmask",
    "ft_drive": "DRIVE",
    "ft_isic2018": "ISIC2018",
    "ft_kvasir_seg": "KvasirSEG",
}


# ===========================================================================
# Data loading
# ===========================================================================
def load_all_runs(results_root: Path) -> Dict[str, pd.DataFrame]:
    """Load step_results.csv from all run directories."""
    runs = {}
    for run_dir in sorted(results_root.iterdir()):
        if not run_dir.is_dir() or not run_dir.name.startswith("run_"):
            continue
        csv_path = run_dir / "step_results.csv"
        if not csv_path.exists():
            continue
        run_id = run_dir.name.replace("run_", "")
        df = pd.read_csv(csv_path)
        df["run_id"] = run_id
        runs[run_id] = df
    return runs


def filter_strategies(df: pd.DataFrame, strategies: List[str] = None) -> pd.DataFrame:
    if strategies is None:
        strategies = STRATEGY_ORDER
    return df[df["strategy"].isin(strategies)]


# ===========================================================================
# TYPE 1: BALD Superiority
# ===========================================================================

def compute_auc(df: pd.DataFrame, metric: str = "iou") -> pd.DataFrame:
    """Compute area under the metric-vs-steps curve per (sample_id, seed, strategy)."""
    rows = []
    for (sid, seed, strat), g in df.groupby(["sample_id", "seed", "strategy"]):
        g = g.sort_values("step")
        auc_val = np.trapz(g[metric].values, g["step"].values)
        rows.append({"sample_id": sid, "seed": seed, "strategy": strat, f"auc_{metric}": auc_val})
    return pd.DataFrame(rows)


def compute_win_rates(df: pd.DataFrame, metric: str = "iou") -> pd.DataFrame:
    """At each step, what % of (sample, seed) pairs does BALD Laplace beat each other strategy?"""
    steps = sorted(df["step"].unique())
    competitors = [s for s in STRATEGY_ORDER if s != "bald_laplace" and s in df["strategy"].unique()]
    rows = []
    for step in steps:
        step_df = df[df["step"] == step]
        bald_vals = step_df[step_df["strategy"] == "bald_laplace"].set_index(["sample_id", "seed"])[metric]
        for comp in competitors:
            comp_vals = step_df[step_df["strategy"] == comp].set_index(["sample_id", "seed"])[metric]
            common = bald_vals.index.intersection(comp_vals.index)
            if len(common) == 0:
                continue
            wins = (bald_vals.loc[common] > comp_vals.loc[common]).sum()
            ties = (bald_vals.loc[common] == comp_vals.loc[common]).sum()
            n = len(common)
            rows.append({
                "step": step,
                "competitor": comp,
                "win_rate": wins / n * 100,
                "tie_rate": ties / n * 100,
                "n": n,
            })
    return pd.DataFrame(rows)


def compute_budget_savings(df: pd.DataFrame, metric: str = "iou") -> pd.DataFrame:
    """How many fewer steps does BALD need to match each competitor's final performance?"""
    max_step = df["step"].max()
    agg = df.groupby(["strategy", "step"])[metric].mean().reset_index()

    bald_curve = agg[agg["strategy"] == "bald_laplace"].sort_values("step")
    competitors = [s for s in STRATEGY_ORDER if s != "bald_laplace" and s != "oracle" and s in agg["strategy"].unique()]

    rows = []
    for comp in competitors:
        comp_final = agg[(agg["strategy"] == comp) & (agg["step"] == max_step)][metric].values[0]
        # Find earliest step where BALD >= comp_final
        bald_above = bald_curve[bald_curve[metric] >= comp_final]
        if len(bald_above) > 0:
            bald_step = bald_above["step"].iloc[0]
            saving = max_step - bald_step
        else:
            bald_step = None
            saving = 0
        rows.append({
            "competitor": comp,
            "competitor_final_iou": comp_final,
            "bald_matches_at_step": bald_step,
            "budget_saving": saving,
        })
    return pd.DataFrame(rows)


def plot_metric_curves_bald_highlight(
    df: pd.DataFrame, out_path: Path, metric: str = "iou",
    ylabel: str = "IoU", title: str = "BALD Laplace vs All Strategies",
):
    """Plot metric vs steps with BALD Laplace highlighted (thicker, bolder)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    strategies = [s for s in STRATEGY_ORDER if s in df["strategy"].unique()]

    for strat in strategies:
        sdf = df[df["strategy"] == strat].groupby("step")[metric].agg(["mean", "std"]).reset_index()
        is_bald = strat == "bald_laplace"
        lw = 3.0 if is_bald else 1.5
        alpha_fill = 0.20 if is_bald else 0.08
        zorder = 10 if is_bald else 5
        marker = "o" if is_bald else ""
        ms = 5 if is_bald else 0

        ax.plot(sdf["step"], sdf["mean"],
                color=STRATEGY_COLORS[strat], label=STRATEGY_LABELS[strat],
                linewidth=lw, zorder=zorder, marker=marker, markersize=ms)
        ax.fill_between(sdf["step"],
                        sdf["mean"] - sdf["std"],
                        sdf["mean"] + sdf["std"],
                        color=STRATEGY_COLORS[strat], alpha=alpha_fill, zorder=zorder - 1)

    ax.set_xlabel("Active Prompting Step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    fig.savefig(out_path)
    plt.close(fig)


def plot_win_rate_heatmap(win_df: pd.DataFrame, out_path: Path):
    """Heatmap: BALD Laplace win rate vs each competitor at each step."""
    competitors = [s for s in STRATEGY_ORDER if s in win_df["competitor"].unique()]
    steps = sorted(win_df["step"].unique())

    matrix = np.full((len(competitors), len(steps)), np.nan)
    for i, comp in enumerate(competitors):
        for j, step in enumerate(steps):
            row = win_df[(win_df["competitor"] == comp) & (win_df["step"] == step)]
            if len(row) > 0:
                matrix[i, j] = row["win_rate"].values[0]

    fig, ax = plt.subplots(figsize=(max(8, len(steps) * 0.5), max(3, len(competitors) * 0.7)))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=30, vmax=100, aspect="auto")

    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels(steps)
    ax.set_yticks(range(len(competitors)))
    ax.set_yticklabels([STRATEGY_LABELS.get(c, c) for c in competitors])
    ax.set_xlabel("Step")
    ax.set_title("BALD (Laplace) Win Rate (%) vs Each Strategy")

    for i in range(len(competitors)):
        for j in range(len(steps)):
            if not np.isnan(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:.0f}", ha="center", va="center",
                        fontsize=8, color="black" if matrix[i, j] > 50 else "white")

    fig.colorbar(im, ax=ax, label="Win Rate (%)")
    fig.savefig(out_path)
    plt.close(fig)


def plot_auc_bars(auc_df: pd.DataFrame, out_path: Path, metric: str = "iou"):
    """Bar chart: mean AUC per strategy, BALD highlighted."""
    col = f"auc_{metric}"
    agg = auc_df.groupby("strategy")[col].agg(["mean", "std"]).reset_index()
    agg = agg.set_index("strategy").loc[[s for s in STRATEGY_ORDER if s in agg["strategy"].values]].reset_index()

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = [STRATEGY_COLORS[s] for s in agg["strategy"]]
    edgecolors = ["black" if s == "bald_laplace" else "none" for s in agg["strategy"]]
    linewidths = [2.5 if s == "bald_laplace" else 0 for s in agg["strategy"]]

    bars = ax.bar(range(len(agg)), agg["mean"], yerr=agg["std"],
                  color=colors, edgecolor=edgecolors, linewidth=linewidths,
                  capsize=4, zorder=3)
    ax.set_xticks(range(len(agg)))
    ax.set_xticklabels([STRATEGY_LABELS[s] for s in agg["strategy"]], rotation=25, ha="right")
    ax.set_ylabel(f"AUC ({metric.upper()})")
    ax.set_title(f"Area Under the {metric.upper()} Curve (Higher = Better)")
    ax.grid(True, axis="y", alpha=0.3)

    # Annotate values
    for bar, val in zip(bars, agg["mean"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + agg["std"].max() * 0.05,
                f"{val:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    fig.savefig(out_path)
    plt.close(fig)


def plot_budget_savings(savings_df: pd.DataFrame, out_path: Path):
    """Bar chart showing how many steps BALD saves vs each competitor."""
    fig, ax = plt.subplots(figsize=(6, 4))
    competitors = savings_df["competitor"].tolist()
    savings = savings_df["budget_saving"].tolist()
    colors = [STRATEGY_COLORS.get(c, "#888") for c in competitors]

    bars = ax.barh(range(len(competitors)), savings, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_yticks(range(len(competitors)))
    ax.set_yticklabels([STRATEGY_LABELS.get(c, c) for c in competitors])
    ax.set_xlabel("Steps Saved by BALD (Laplace)")
    ax.set_title("Budget Efficiency: BALD Reaches Their Final IoU Earlier")
    ax.grid(True, axis="x", alpha=0.3)

    for bar, val in zip(bars, savings):
        if val > 0:
            ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                    f"{val:.0f} steps", va="center", fontsize=10, fontweight="bold")

    fig.savefig(out_path)
    plt.close(fig)


def plot_per_dataset_final_iou(df: pd.DataFrame, out_path: Path, metric: str = "iou"):
    """Grouped bar chart: final-step metric per strategy, grouped by dataset."""
    max_step = df["step"].max()
    final = df[df["step"] == max_step]
    strategies = [s for s in STRATEGY_ORDER if s in final["strategy"].unique()]
    datasets = sorted(final["group"].unique())

    x = np.arange(len(datasets))
    width = 0.8 / len(strategies)

    fig, ax = plt.subplots(figsize=(max(10, len(datasets) * 2.5), 5))
    for i, strat in enumerate(strategies):
        means = []
        stds = []
        for ds in datasets:
            vals = final[(final["strategy"] == strat) & (final["group"] == ds)][metric]
            means.append(vals.mean() if len(vals) > 0 else 0)
            stds.append(vals.std() if len(vals) > 1 else 0)
        offset = (i - len(strategies) / 2 + 0.5) * width
        lw = 2.0 if strat == "bald_laplace" else 0
        ax.bar(x + offset, means, width * 0.9, yerr=stds,
               color=STRATEGY_COLORS[strat], edgecolor="black" if strat == "bald_laplace" else "none",
               linewidth=lw, label=STRATEGY_LABELS[strat], capsize=2)

    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABELS.get(d, d) for d in datasets])
    ax.set_ylabel(metric.upper())
    ax.set_title(f"Final-Step {metric.upper()} by Dataset and Strategy")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, axis="y", alpha=0.3)
    fig.savefig(out_path)
    plt.close(fig)


def plot_statistical_significance(df: pd.DataFrame, out_path: Path, metric: str = "iou"):
    """Table-style plot showing paired Wilcoxon p-values: BALD vs each strategy at final step."""
    max_step = df["step"].max()
    final = df[df["step"] == max_step]
    bald_vals = final[final["strategy"] == "bald_laplace"].set_index(["sample_id", "seed"])[metric]
    competitors = [s for s in STRATEGY_ORDER if s != "bald_laplace" and s in final["strategy"].unique()]

    rows = []
    for comp in competitors:
        comp_vals = final[final["strategy"] == comp].set_index(["sample_id", "seed"])[metric]
        common = bald_vals.index.intersection(comp_vals.index)
        if len(common) < 5:
            continue
        b = bald_vals.loc[common].values
        c = comp_vals.loc[common].values
        diff = b - c
        try:
            stat, p = stats.wilcoxon(diff, alternative="greater")
        except ValueError:
            p = 1.0
        rows.append({
            "competitor": STRATEGY_LABELS.get(comp, comp),
            "bald_mean": b.mean(),
            "comp_mean": c.mean(),
            "delta_mean": diff.mean(),
            "p_value": p,
            "significant": "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "n.s.")),
            "n": len(common),
        })

    if not rows:
        return

    sig_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(9, max(2, len(rows) * 0.6 + 1.5)))
    ax.axis("off")
    table = ax.table(
        cellText=[[r["competitor"], f"{r['bald_mean']:.4f}", f"{r['comp_mean']:.4f}",
                   f"{r['delta_mean']:+.4f}", f"{r['p_value']:.2e}", r["significant"], r["n"]]
                  for r in rows],
        colLabels=["Competitor", "BALD Mean", "Comp. Mean", "Delta", "p-value", "Sig.", "N"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # Color significant rows
    for i, r in enumerate(rows):
        if r["p_value"] < 0.05:
            for j in range(7):
                table[i + 1, j].set_facecolor("#d4edda")

    ax.set_title(f"Statistical Significance: BALD (Laplace) vs Others — Final Step {metric.upper()}\n(Wilcoxon signed-rank, one-sided: BALD > competitor)", fontsize=11, pad=20)
    fig.savefig(out_path)
    plt.close(fig)

    return sig_df


def plot_delta_advantage_area(df: pd.DataFrame, out_path: Path, metric: str = "iou"):
    """Filled area plot showing BALD's advantage (delta) over each competitor across steps."""
    agg = df.groupby(["strategy", "step"])[metric].mean().reset_index()
    bald = agg[agg["strategy"] == "bald_laplace"].set_index("step")[metric]
    competitors = [s for s in ["entropy", "random", "maxdist"] if s in agg["strategy"].unique()]

    fig, ax = plt.subplots(figsize=(8, 4))
    for comp in competitors:
        comp_vals = agg[agg["strategy"] == comp].set_index("step")[metric]
        common_steps = bald.index.intersection(comp_vals.index)
        delta = bald.loc[common_steps] - comp_vals.loc[common_steps]
        ax.fill_between(common_steps, 0, delta,
                        color=STRATEGY_COLORS[comp], alpha=0.3,
                        label=f"vs {STRATEGY_LABELS[comp]}")
        ax.plot(common_steps, delta, color=STRATEGY_COLORS[comp], linewidth=1.5)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Active Prompting Step")
    ax.set_ylabel(f"BALD Advantage ({metric.upper()})")
    ax.set_title(f"BALD (Laplace) Advantage Over Competitors")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    fig.savefig(out_path)
    plt.close(fig)


# ===========================================================================
# TYPE 2: Baseline vs Fine-tuned
# ===========================================================================

def plot_baseline_vs_finetuned_curves(
    baseline_df: pd.DataFrame, ft_df: pd.DataFrame,
    dataset: str, ft_name: str, out_path: Path,
    metric: str = "iou", ylabel: str = "IoU",
):
    """Side-by-side: same dataset, baseline SAM vs fine-tuned SAM, all strategies."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    strategies = [s for s in STRATEGY_ORDER if s in baseline_df["strategy"].unique()]

    for ax, (df, label) in zip(axes, [
        (baseline_df[baseline_df["group"] == dataset], "Baseline (Original SAM)"),
        (ft_df, f"Fine-tuned ({ft_name} SAM)"),
    ]):
        for strat in strategies:
            sdf = df[df["strategy"] == strat].groupby("step")[metric].agg(["mean", "std"]).reset_index()
            if len(sdf) == 0:
                continue
            is_bald = strat == "bald_laplace"
            ax.plot(sdf["step"], sdf["mean"],
                    color=STRATEGY_COLORS[strat], label=STRATEGY_LABELS[strat],
                    linewidth=3.0 if is_bald else 1.5,
                    marker="o" if is_bald else "", markersize=4 if is_bald else 0)
            ax.fill_between(sdf["step"], sdf["mean"] - sdf["std"], sdf["mean"] + sdf["std"],
                            color=STRATEGY_COLORS[strat], alpha=0.15 if is_bald else 0.06)
        ax.set_xlabel("Step")
        ax.set_title(label)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    axes[0].set_ylabel(ylabel)
    axes[1].legend(fontsize=8, loc="lower right")
    fig.suptitle(f"{DATASET_LABELS.get(dataset, dataset)}: {ylabel} Curves — Baseline vs Fine-tuned", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_finetuning_lift_bars(
    baseline_df: pd.DataFrame, all_ft_dfs: Dict[str, pd.DataFrame],
    out_path: Path, metric: str = "iou",
):
    """Grouped bar chart: final IoU improvement (ft - baseline) per strategy per dataset."""
    max_step = baseline_df["step"].max()
    strategies = [s for s in STRATEGY_ORDER if s != "oracle" and s in baseline_df["strategy"].unique()]
    datasets = list(FT_DATASET_MAP.values())

    # Compute deltas
    rows = []
    for ft_run, ds in FT_DATASET_MAP.items():
        if ft_run not in all_ft_dfs:
            continue
        ft_df = all_ft_dfs[ft_run]
        ft_max = ft_df["step"].max()
        bl_final = baseline_df[(baseline_df["step"] == max_step) & (baseline_df["group"] == ds)]
        ft_final = ft_df[ft_df["step"] == ft_max]
        for strat in strategies:
            bl_val = bl_final[bl_final["strategy"] == strat][metric].mean()
            ft_val = ft_final[ft_final["strategy"] == strat][metric].mean()
            if np.isnan(bl_val) or np.isnan(ft_val):
                continue
            rows.append({"dataset": ds, "strategy": strat, "baseline": bl_val, "finetuned": ft_val,
                         "lift": ft_val - bl_val})

    lift_df = pd.DataFrame(rows)
    if len(lift_df) == 0:
        return

    ds_list = [d for d in datasets if d in lift_df["dataset"].unique()]
    x = np.arange(len(ds_list))
    width = 0.8 / len(strategies)

    fig, ax = plt.subplots(figsize=(max(10, len(ds_list) * 2.5), 5))
    for i, strat in enumerate(strategies):
        lifts = [lift_df[(lift_df["dataset"] == d) & (lift_df["strategy"] == strat)]["lift"].values[0]
                 if len(lift_df[(lift_df["dataset"] == d) & (lift_df["strategy"] == strat)]) > 0 else 0
                 for d in ds_list]
        offset = (i - len(strategies) / 2 + 0.5) * width
        lw = 2.0 if strat == "bald_laplace" else 0
        ax.bar(x + offset, lifts, width * 0.9,
               color=STRATEGY_COLORS[strat], edgecolor="black" if strat == "bald_laplace" else "none",
               linewidth=lw, label=STRATEGY_LABELS[strat])

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABELS.get(d, d) for d in ds_list])
    ax.set_ylabel(f"\u0394 {metric.upper()} (Fine-tuned \u2212 Baseline)")
    ax.set_title(f"Fine-tuning Lift: Final-Step {metric.upper()} Improvement Over Baseline")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, axis="y", alpha=0.3)
    fig.savefig(out_path)
    plt.close(fig)

    return lift_df


def plot_combined_vs_baseline_all_datasets(
    baseline_df: pd.DataFrame, combined_df: pd.DataFrame,
    out_path: Path, metric: str = "iou",
):
    """Per-dataset panel: combined SAM vs baseline, BALD only, showing the lift."""
    datasets = sorted(baseline_df["group"].unique())
    n = len(datasets)
    cols = min(3, n)
    rows_n = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows_n, cols, figsize=(5 * cols, 4 * rows_n), squeeze=False)
    for idx, ds in enumerate(datasets):
        ax = axes[idx // cols][idx % cols]
        for label, df, color, ls in [
            ("Baseline", baseline_df[baseline_df["group"] == ds], "#6C757D", "--"),
            ("Combined FT", combined_df[combined_df["group"] == ds], "#E63946", "-"),
        ]:
            bald = df[df["strategy"] == "bald_laplace"].groupby("step")[metric].mean()
            if len(bald) > 0:
                ax.plot(bald.index, bald.values, color=color, linewidth=2, linestyle=ls, label=label)
        ax.set_title(DATASET_LABELS.get(ds, ds))
        ax.set_xlabel("Step")
        ax.set_ylabel(metric.upper())
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    # Hide unused axes
    for idx in range(n, rows_n * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    fig.suptitle(f"BALD (Laplace) {metric.upper()}: Baseline vs Combined Fine-tuned SAM", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_strategy_ranking_shift(
    baseline_df: pd.DataFrame, ft_df: pd.DataFrame,
    dataset: str, ft_label: str, out_path: Path, metric: str = "iou",
):
    """Bump chart showing strategy ranking at final step: baseline vs fine-tuned."""
    max_step_bl = baseline_df["step"].max()
    max_step_ft = ft_df["step"].max()
    strategies = [s for s in STRATEGY_ORDER if s in baseline_df["strategy"].unique() and s in ft_df["strategy"].unique()]

    bl_final = baseline_df[(baseline_df["step"] == max_step_bl) & (baseline_df["group"] == dataset)]
    ft_final = ft_df[ft_df["step"] == max_step_ft]

    bl_means = bl_final.groupby("strategy")[metric].mean()
    ft_means = ft_final.groupby("strategy")[metric].mean()

    bl_rank = bl_means.rank(ascending=False)
    ft_rank = ft_means.rank(ascending=False)

    fig, ax = plt.subplots(figsize=(6, 4))
    for strat in strategies:
        if strat not in bl_rank or strat not in ft_rank:
            continue
        r1, r2 = bl_rank[strat], ft_rank[strat]
        ax.plot([0, 1], [r1, r2], color=STRATEGY_COLORS[strat], linewidth=2.5,
                marker="o", markersize=8, zorder=5)
        ax.text(-0.08, r1, STRATEGY_LABELS[strat], ha="right", va="center",
                fontsize=9, color=STRATEGY_COLORS[strat])
        ax.text(1.08, r2, f"{ft_means[strat]:.3f}", ha="left", va="center",
                fontsize=9, color=STRATEGY_COLORS[strat])

    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(len(strategies) + 0.5, 0.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Baseline", ft_label])
    ax.set_ylabel("Rank (1 = best)")
    ax.set_title(f"{DATASET_LABELS.get(dataset, dataset)}: Strategy Ranking Shift")
    ax.grid(True, axis="y", alpha=0.3)
    fig.savefig(out_path)
    plt.close(fig)


# ===========================================================================
# Summary statistics
# ===========================================================================

def print_summary_numbers(runs: Dict[str, pd.DataFrame], out_path: Path):
    """Print and save key summary numbers."""
    lines = []

    def log(s):
        print(s)
        lines.append(s)

    log("=" * 80)
    log("MEDICAL ACTIVE PROMPTING — SUMMARY STATISTICS")
    log("=" * 80)

    # --- Type 1: BALD superiority (pooled across all runs) ---
    log("\n" + "=" * 80)
    log("TYPE 1: BALD SUPERIORITY")
    log("=" * 80)

    for run_id, df in sorted(runs.items()):
        max_step = df["step"].max()
        final = df[df["step"] == max_step]
        strategies = [s for s in STRATEGY_ORDER if s in final["strategy"].unique()]

        log(f"\n--- {run_id} (max_step={max_step}, samples={df['sample_id'].nunique()}) ---")
        log(f"{'Strategy':<22s} {'IoU':>8s} {'Dice':>8s} {'bIoU':>8s} {'HD95':>8s}")

        for strat in strategies:
            sf = final[final["strategy"] == strat]
            log(f"  {STRATEGY_LABELS.get(strat, strat):<20s} "
                f"{sf['iou'].mean():8.4f} {sf['dice'].mean():8.4f} "
                f"{sf['boundary_iou'].mean():8.4f} {sf['hausdorff95'].mean():8.2f}")

        # BALD vs random delta
        bald_iou = final[final["strategy"] == "bald_laplace"]["iou"].mean()
        random_iou = final[final["strategy"] == "random"]["iou"].mean()
        if not np.isnan(bald_iou) and not np.isnan(random_iou):
            log(f"\n  BALD advantage over Random: {bald_iou - random_iou:+.4f} IoU")

        # Budget savings
        savings = compute_budget_savings(df)
        if len(savings) > 0:
            log("  Budget savings (BALD reaches their final IoU at step):")
            for _, r in savings.iterrows():
                if r["bald_matches_at_step"] is not None and not np.isnan(r["bald_matches_at_step"]):
                    log(f"    vs {STRATEGY_LABELS.get(r['competitor'], r['competitor'])}: "
                        f"step {int(r['bald_matches_at_step'])} ({int(r['budget_saving'])} steps saved)")

    # --- Type 2: Fine-tuning lift ---
    if "baseline_v3" in runs:
        log("\n" + "=" * 80)
        log("TYPE 2: FINE-TUNING LIFT (Final Step IoU)")
        log("=" * 80)

        bl = runs["baseline_v3"]
        bl_max = bl["step"].max()
        bl_final = bl[bl["step"] == bl_max]

        log(f"\n{'Dataset':<14s} {'Strategy':<22s} {'Baseline':>10s} {'Fine-tuned':>10s} {'Lift':>10s}")
        for ft_run, ds in FT_DATASET_MAP.items():
            if ft_run not in runs:
                continue
            ft = runs[ft_run]
            ft_max = ft["step"].max()
            ft_final = ft[ft["step"] == ft_max]
            strategies = [s for s in STRATEGY_ORDER if s in bl_final["strategy"].unique() and s in ft_final["strategy"].unique()]

            for strat in strategies:
                bl_val = bl_final[(bl_final["strategy"] == strat) & (bl_final["group"] == ds)]["iou"].mean()
                ft_val = ft_final[ft_final["strategy"] == strat]["iou"].mean()
                lift = ft_val - bl_val
                marker = " <<" if strat == "bald_laplace" else ""
                log(f"  {ds:<12s} {STRATEGY_LABELS.get(strat, strat):<20s} "
                    f"{bl_val:10.4f} {ft_val:10.4f} {lift:+10.4f}{marker}")

        # Combined vs baseline
        if "ft_combined" in runs:
            log(f"\n--- Combined SAM vs Baseline (BALD Laplace only) ---")
            comb = runs["ft_combined"]
            comb_max = comb["step"].max()
            comb_final = comb[comb["step"] == comb_max]
            for ds in sorted(bl_final["group"].unique()):
                bl_val = bl_final[(bl_final["strategy"] == "bald_laplace") & (bl_final["group"] == ds)]["iou"].mean()
                comb_val = comb_final[(comb_final["strategy"] == "bald_laplace") & (comb_final["group"] == ds)]["iou"].mean()
                if not np.isnan(bl_val) and not np.isnan(comb_val):
                    log(f"  {ds:<14s} Baseline={bl_val:.4f}  Combined={comb_val:.4f}  Lift={comb_val - bl_val:+.4f}")

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="Cross-run analysis for medical active prompting")
    parser.add_argument("--results_root", default="/mnt/HD_18T_pt2/medical_data/active_prompting",
                        help="Root directory containing run_* folders")
    parser.add_argument("--output_dir", default=None, help="Output directory (default: results_root/analysis)")
    args = parser.parse_args()

    results_root = Path(args.results_root)
    out_dir = Path(args.output_dir) if args.output_dir else results_root / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    type1_dir = out_dir / "type1_bald_superiority"
    type2_dir = out_dir / "type2_baseline_vs_finetuned"
    type1_dir.mkdir(exist_ok=True)
    type2_dir.mkdir(exist_ok=True)

    # Load all runs
    print("Loading runs...")
    runs = load_all_runs(results_root)
    print(f"Loaded {len(runs)} runs: {list(runs.keys())}")

    if len(runs) == 0:
        print("No runs found. Exiting.")
        return

    # =====================================================================
    # TYPE 1: BALD Superiority
    # =====================================================================
    print("\n=== TYPE 1: BALD Superiority ===")

    # 1a. Per-run metric curves with BALD highlighted
    for run_id, df in runs.items():
        for metric, ylabel in [("iou", "IoU"), ("dice", "Dice"), ("delta_iou", "\u0394 IoU")]:
            if metric not in df.columns:
                continue
            plot_metric_curves_bald_highlight(
                df, type1_dir / f"{run_id}_{metric}_curves.png",
                metric=metric, ylabel=ylabel,
                title=f"{RUN_IDS.get(run_id, run_id)}: {ylabel} vs Steps",
            )
    print("  Metric curves: done")

    # 1b. Win rate heatmaps
    for run_id, df in runs.items():
        win_df = compute_win_rates(df)
        if len(win_df) > 0:
            plot_win_rate_heatmap(win_df, type1_dir / f"{run_id}_win_rate_heatmap.png")
    print("  Win rate heatmaps: done")

    # 1c. AUC bar charts
    for run_id, df in runs.items():
        auc_df = compute_auc(df, "iou")
        plot_auc_bars(auc_df, type1_dir / f"{run_id}_auc_bars.png")
    print("  AUC bars: done")

    # 1d. Budget savings
    for run_id, df in runs.items():
        savings = compute_budget_savings(df)
        if len(savings) > 0:
            plot_budget_savings(savings, type1_dir / f"{run_id}_budget_savings.png")
    print("  Budget savings: done")

    # 1e. Per-dataset final IoU bars
    for run_id, df in runs.items():
        if df["group"].nunique() > 1:
            plot_per_dataset_final_iou(df, type1_dir / f"{run_id}_per_dataset_final_iou.png")
    print("  Per-dataset final IoU: done")

    # 1f. Statistical significance tables
    for run_id, df in runs.items():
        plot_statistical_significance(df, type1_dir / f"{run_id}_significance.png")
    print("  Statistical significance: done")

    # 1g. Delta advantage area plots
    for run_id, df in runs.items():
        plot_delta_advantage_area(df, type1_dir / f"{run_id}_bald_advantage.png")
    print("  BALD advantage area: done")

    # Pooled across all runs
    all_df = pd.concat(runs.values(), ignore_index=True)
    plot_metric_curves_bald_highlight(
        all_df, type1_dir / "pooled_iou_curves.png",
        metric="iou", ylabel="IoU", title="All Runs Pooled: IoU vs Steps",
    )
    plot_statistical_significance(all_df, type1_dir / "pooled_significance.png")
    plot_delta_advantage_area(all_df, type1_dir / "pooled_bald_advantage.png")
    print("  Pooled plots: done")

    # =====================================================================
    # TYPE 2: Baseline vs Fine-tuned
    # =====================================================================
    print("\n=== TYPE 2: Baseline vs Fine-tuned ===")

    if "baseline_v3" not in runs:
        print("  No baseline_v3 run found — skipping Type 2.")
    else:
        bl_df = runs["baseline_v3"]

        # 2a. Side-by-side curves per dataset
        for ft_run, dataset in FT_DATASET_MAP.items():
            if ft_run not in runs:
                continue
            ft_df = runs[ft_run]
            for metric, ylabel in [("iou", "IoU"), ("dice", "Dice")]:
                plot_baseline_vs_finetuned_curves(
                    bl_df, ft_df, dataset, dataset,
                    type2_dir / f"{ft_run}_vs_baseline_{metric}.png",
                    metric=metric, ylabel=ylabel,
                )
        print("  Side-by-side curves: done")

        # 2b. Fine-tuning lift bars
        ft_dfs = {k: v for k, v in runs.items() if k.startswith("ft_") and k != "ft_combined"}
        lift_df = plot_finetuning_lift_bars(bl_df, ft_dfs, type2_dir / "finetuning_lift_bars.png")
        if lift_df is not None:
            lift_df.to_csv(type2_dir / "finetuning_lift.csv", index=False)
        print("  Fine-tuning lift bars: done")

        # 2c. Combined vs baseline per dataset
        if "ft_combined" in runs:
            plot_combined_vs_baseline_all_datasets(
                bl_df, runs["ft_combined"],
                type2_dir / "combined_vs_baseline_per_dataset.png",
            )
        print("  Combined vs baseline: done")

        # 2d. Strategy ranking shift (bump charts)
        for ft_run, dataset in FT_DATASET_MAP.items():
            if ft_run not in runs:
                continue
            plot_strategy_ranking_shift(
                bl_df, runs[ft_run], dataset, f"FT ({dataset})",
                type2_dir / f"{ft_run}_ranking_shift.png",
            )
        print("  Ranking shift: done")

    # =====================================================================
    # Summary numbers (printed + saved)
    # =====================================================================
    print("\n")
    print_summary_numbers(runs, out_dir / "summary_statistics.txt")

    print(f"\nAll analysis saved to: {out_dir}")


if __name__ == "__main__":
    main()
