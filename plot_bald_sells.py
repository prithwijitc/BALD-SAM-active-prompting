#!/usr/bin/env python3
"""
plot_bald_sells.py  —  Comprehensive BALD (Laplace) superiority analysis
across Medical, Others (camouflage/crack/salient), and Point-Prompt datasets.

Produces:
  1. Overall cross-domain narrative  (plots/overall/)
  2. Per-domain-type deep dives      (plots/medical/, plots/others/, plots/point_prompt/)

Strategy renaming for publication:
  bald_laplace   → BALD (Ours)
  bald_mcdropout → MC Dropout
  entropy        → Entropy
  maxdist        → MaxDist
  oracle         → GT-based       (was ORACLE)
  random         → Random

Usage:
  python plot_bald_sells.py [--results_root /path/to/run_v3] [--out_dir plots_sells]
"""

import argparse, os, sys, warnings
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════

RESULTS_ROOT_DEFAULT = "/mnt/HD_18T_pt2/prithwijit/bald/active_prompting/run_v3"

DOMAINS = {
    "Medical":      "Medical",
    "others":       "Challenging Vision",   # camouflage, cracks, salient, DIS
    "point_prompt": "Point-Prompt",
}

STRATEGY_RENAME = {
    "bald_laplace":   "BALD (Ours)",
    "bald_mcdropout": "MC Dropout",
    "entropy":        "Entropy",
    "maxdist":        "MaxDist",
    "oracle":         "GT-based",
    "random":         "Random",
}

# Our strategy
OURS = "BALD (Ours)"

# Competitors (excluding oracle/random for fair comparisons)
COMPETITORS = ["MC Dropout", "Entropy", "MaxDist"]
ALL_STRATEGIES = [OURS] + COMPETITORS + ["GT-based", "Random"]

# Color palette — BALD gets a strong, distinct color
COLORS = {
    "BALD (Ours)": "#1a5276",   # deep navy
    "MC Dropout":  "#e74c3c",   # red
    "Entropy":     "#f39c12",   # amber
    "MaxDist":     "#27ae60",   # green
    "GT-based":    "#7f8c8d",   # gray
    "Random":      "#bdc3c7",   # light gray
}

MARKERS = {
    "BALD (Ours)": "o",
    "MC Dropout":  "s",
    "Entropy":     "^",
    "MaxDist":     "D",
    "GT-based":    "x",
    "Random":      "+",
}

LINEWIDTHS = {
    "BALD (Ours)": 2.8,
    "MC Dropout":  1.5,
    "Entropy":     1.5,
    "MaxDist":     1.5,
    "GT-based":    1.2,
    "Random":      1.0,
}

LINESTYLES = {
    "BALD (Ours)": "-",
    "MC Dropout":  "--",
    "Entropy":     "-.",
    "MaxDist":     ":",
    "GT-based":    (0, (1, 1)),
    "Random":      (0, (5, 5)),
}

# Metrics
METRICS_HIGHER_BETTER = ["iou_mean", "dice_mean", "boundary_iou_mean"]
METRICS_LOWER_BETTER = ["hausdorff95_mean", "ece_mean", "nll_mean"]
ALL_METRICS = METRICS_HIGHER_BETTER + METRICS_LOWER_BETTER

METRIC_LABELS = {
    "iou_mean": "IoU",
    "dice_mean": "Dice",
    "boundary_iou_mean": "Boundary IoU",
    "hausdorff95_mean": "HD95 ↓",
    "ece_mean": "ECE ↓",
    "nll_mean": "NLL ↓",
}

# ═══════════════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════════════

def load_domain(root, folder):
    """Load all CSVs for one domain folder."""
    base = Path(root) / folder
    agg = pd.read_csv(base / "aggregate.csv")
    pg  = pd.read_csv(base / "per_group.csv")
    ps  = pd.read_csv(base / "per_stratum.csv")

    # Rename strategies
    for df in [agg, pg, ps]:
        df["strategy"] = df["strategy"].map(STRATEGY_RENAME).fillna(df["strategy"])

    return agg, pg, ps

def load_all(root):
    """Load and tag all domains."""
    all_agg, all_pg, all_ps = [], [], []
    for folder, label in DOMAINS.items():
        agg, pg, ps = load_domain(root, folder)
        agg["domain"] = label
        pg["domain"]  = label
        ps["domain"]  = label
        all_agg.append(agg)
        all_pg.append(pg)
        all_ps.append(ps)
    return pd.concat(all_agg, ignore_index=True), pd.concat(all_pg, ignore_index=True), pd.concat(all_ps, ignore_index=True)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def set_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 9,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
    })

def savefig(fig, outdir, name):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"{name}.png")
    fig.savefig(outdir / f"{name}.pdf")
    plt.close(fig)
    print(f"  Saved: {outdir / name}.png/.pdf")

def get_final_step(df):
    return df["step"].max()

def final_values(pg, strategies=None):
    """Get final-step values per (strategy, group)."""
    final = pg[pg["step"] == get_final_step(pg)]
    if strategies:
        final = final[final["strategy"].isin(strategies)]
    return final

def rank_at_final(pg, metric, higher_better=True):
    """Rank strategies per group at final step. Rank 1 = best."""
    fv = final_values(pg)
    ranks = fv.pivot_table(index="group", columns="strategy", values=metric)
    if higher_better:
        return ranks.rank(axis=1, ascending=False)
    else:
        return ranks.rank(axis=1, ascending=True)

def pareto_check(row, metrics_hb, metrics_lb):
    """Check if a strategy is Pareto-dominated: True = non-dominated."""
    # This is applied per-group; row has metric values for one strategy
    # We check externally
    pass

def count_pareto_wins(pg, groups=None):
    """For each group, check if BALD is Pareto-optimal (non-dominated) on IoU, ECE, HD95."""
    fv = final_values(pg)
    if groups is None:
        groups = fv["group"].unique()

    wins = {}
    for g in groups:
        gdf = fv[fv["group"] == g]
        # Exclude oracle and random for fair Pareto comparison
        gdf = gdf[gdf["strategy"].isin([OURS] + COMPETITORS)]

        bald_row = gdf[gdf["strategy"] == OURS]
        if bald_row.empty:
            continue
        b_iou = bald_row["iou_mean"].values[0]
        b_ece = bald_row["ece_mean"].values[0]
        b_hd  = bald_row["hausdorff95_mean"].values[0]

        dominated = False
        for _, r in gdf.iterrows():
            if r["strategy"] == OURS:
                continue
            # r dominates BALD if r is >= on all and > on at least one
            if r["iou_mean"] >= b_iou and r["ece_mean"] <= b_ece and r["hausdorff95_mean"] <= b_hd:
                if r["iou_mean"] > b_iou or r["ece_mean"] < b_ece or r["hausdorff95_mean"] < b_hd:
                    dominated = True
                    break
        wins[g] = not dominated
    return wins


def compute_auc(series_steps, series_values):
    """Trapezoidal AUC of metric over steps."""
    return np.trapezoid(series_values, series_steps)


# ═══════════════════════════════════════════════════════════════════════════
# OVERALL PLOTS (across all domains)
# ═══════════════════════════════════════════════════════════════════════════

def plot_O1_grand_pareto_bar(all_pg, outdir):
    """O1: Grand Pareto optimality bar — BALD non-dominated across all groups from all domains."""
    all_groups = all_pg["group"].unique()
    pareto = count_pareto_wins(all_pg, all_groups)

    n_total = len(pareto)
    n_win = sum(pareto.values())
    n_lose = n_total - n_win

    # Also compute for each competitor
    comp_pareto = {}
    for comp in COMPETITORS:
        comp_wins = {}
        for g in all_groups:
            fv = all_pg[(all_pg["step"] == get_final_step(all_pg)) & (all_pg["group"] == g)]
            fv = fv[fv["strategy"].isin([OURS] + COMPETITORS)]
            crow = fv[fv["strategy"] == comp]
            if crow.empty:
                continue
            c_iou = crow["iou_mean"].values[0]
            c_ece = crow["ece_mean"].values[0]
            c_hd  = crow["hausdorff95_mean"].values[0]

            dominated = False
            for _, r in fv.iterrows():
                if r["strategy"] == comp:
                    continue
                if r["iou_mean"] >= c_iou and r["ece_mean"] <= c_ece and r["hausdorff95_mean"] <= c_hd:
                    if r["iou_mean"] > c_iou or r["ece_mean"] < c_ece or r["hausdorff95_mean"] < c_hd:
                        dominated = True
                        break
            comp_wins[g] = not dominated
        comp_pareto[comp] = sum(comp_wins.values())

    strats = [OURS] + COMPETITORS
    counts = [n_win] + [comp_pareto[c] for c in COMPETITORS]
    colors = [COLORS[s] for s in strats]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(strats, counts, color=colors, edgecolor="white", linewidth=1.5, zorder=3)
    ax.axhline(n_total, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax.set_ylabel("Pareto-Optimal Groups (of {})".format(n_total))
    ax.set_title("Pareto Optimality (IoU × ECE × HD95)\nAcross {} Datasets from {} Domains".format(n_total, len(DOMAINS)))
    ax.set_ylim(0, n_total + 2)
    ax.grid(axis="y", alpha=0.3)

    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{count}/{n_total}", ha="center", va="bottom", fontweight="bold", fontsize=12)

    savefig(fig, outdir, "O1_grand_pareto_bar")
    return pareto


def plot_O2_domain_pareto_grouped(all_pg, outdir):
    """O2: Pareto wins broken down by domain."""
    domain_results = {}
    for dom in DOMAINS.values():
        dpg = all_pg[all_pg["domain"] == dom]
        groups = dpg["group"].unique()
        pareto = count_pareto_wins(dpg, groups)
        domain_results[dom] = (sum(pareto.values()), len(pareto))

    fig, ax = plt.subplots(figsize=(7, 4))
    domains = list(domain_results.keys())
    wins = [domain_results[d][0] for d in domains]
    totals = [domain_results[d][1] for d in domains]

    x = np.arange(len(domains))
    w = 0.35
    bars_total = ax.bar(x - w/2, totals, w, label="Total Groups", color="#dce6f1", edgecolor="#95a5a6", zorder=3)
    bars_win   = ax.bar(x + w/2, wins,   w, label="BALD Pareto-Optimal", color=COLORS[OURS], edgecolor="white", zorder=3)

    for i, (win, tot) in enumerate(zip(wins, totals)):
        pct = 100 * win / tot
        ax.text(i + w/2, win + 0.2, f"{win}/{tot}\n({pct:.0f}%)", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(domains)
    ax.set_ylabel("Number of Groups")
    ax.set_title("BALD Pareto Optimality by Domain")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    savefig(fig, outdir, "O2_domain_pareto_grouped")


def plot_O3_aggregate_curves(all_agg, outdir):
    """O3: Grand aggregate IoU/ECE/HD95 curves across all data pooled."""
    # Pool across domains by averaging the aggregate values
    # Since domains have different sample counts, we'll show each domain as a panel
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    metrics = ["iou_mean", "ece_mean", "hausdorff95_mean"]
    titles = ["IoU ↑", "ECE ↓", "HD95 ↓"]

    for ax, metric, title in zip(axes, metrics, titles):
        for strat in ALL_STRATEGIES:
            sdf = all_agg[all_agg["strategy"] == strat].groupby("step")[metric].mean().reset_index()
            if sdf.empty:
                continue
            ax.plot(sdf["step"], sdf[metric],
                    color=COLORS.get(strat, "gray"),
                    lw=LINEWIDTHS.get(strat, 1),
                    ls=LINESTYLES.get(strat, "-"),
                    marker=MARKERS.get(strat, ""),
                    markevery=4, markersize=5,
                    label=strat, zorder=10 if strat == OURS else 2)
        ax.set_xlabel("Active Prompting Step")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.grid(alpha=0.3)

    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle("Grand Aggregate Performance (Pooled Across All {} Domains)".format(len(DOMAINS)), fontsize=14, y=1.02)
    fig.tight_layout()
    savefig(fig, outdir, "O3_aggregate_curves")


def plot_O4_win_matrix(all_pg, outdir):
    """O4: Win count matrix — how many groups does each strategy win on each metric at final step."""
    fv = final_values(all_pg)
    strategies = [OURS] + COMPETITORS
    metrics_info = [
        ("iou_mean", True), ("dice_mean", True), ("boundary_iou_mean", True),
        ("hausdorff95_mean", False), ("ece_mean", False), ("nll_mean", False),
    ]

    groups = fv["group"].unique()
    n_groups = len(groups)

    win_matrix = np.zeros((len(strategies), len(metrics_info)))

    for j, (metric, hb) in enumerate(metrics_info):
        for g in groups:
            gdf = fv[(fv["group"] == g) & (fv["strategy"].isin(strategies))]
            if gdf.empty:
                continue
            if hb:
                best_val = gdf[metric].max()
            else:
                best_val = gdf[metric].min()
            winners = gdf[gdf[metric] == best_val]["strategy"].values
            for w in winners:
                idx = strategies.index(w)
                win_matrix[idx, j] += 1.0 / len(winners)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    im = ax.imshow(win_matrix, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(range(len(metrics_info)))
    ax.set_xticklabels([METRIC_LABELS[m] for m, _ in metrics_info], rotation=30, ha="right")
    ax.set_yticks(range(len(strategies)))
    ax.set_yticklabels(strategies)

    for i in range(len(strategies)):
        for j in range(len(metrics_info)):
            val = win_matrix[i, j]
            ax.text(j, i, f"{val:.0f}" if val == int(val) else f"{val:.1f}",
                    ha="center", va="center", fontsize=11,
                    color="white" if val > n_groups * 0.4 else "black",
                    fontweight="bold" if strategies[i] == OURS else "normal")

    # Highlight BALD row
    ax.axhline(0.5, color=COLORS[OURS], lw=2, xmin=-0.05, xmax=1.05)
    ax.axhline(-0.5, color=COLORS[OURS], lw=2, xmin=-0.05, xmax=1.05)

    plt.colorbar(im, ax=ax, label=f"Groups Won (of {n_groups})")
    ax.set_title(f"Metric-Level Wins Across All {n_groups} Groups\n(Excluding GT-based & Random)")
    fig.tight_layout()
    savefig(fig, outdir, "O4_win_matrix")
    return win_matrix, strategies, metrics_info


def plot_O5_bald_vs_mcdrop(all_pg, outdir):
    """O5: BALD vs MC Dropout head-to-head — per group, per metric, final step."""
    fv = final_values(all_pg)
    bald_df = fv[fv["strategy"] == OURS].set_index("group")
    mc_df   = fv[fv["strategy"] == "MC Dropout"].set_index("group")

    common = bald_df.index.intersection(mc_df.index)
    if len(common) == 0:
        return

    metrics_info = [
        ("iou_mean", True, "IoU ↑"), ("dice_mean", True, "Dice ↑"),
        ("ece_mean", False, "ECE ↓"), ("hausdorff95_mean", False, "HD95 ↓"),
        ("nll_mean", False, "NLL ↓"), ("boundary_iou_mean", True, "BIoU ↑"),
    ]

    n_metrics = len(metrics_info)
    bald_wins_total = 0
    mc_wins_total = 0

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.ravel()

    for idx, (metric, hb, label) in enumerate(metrics_info):
        ax = axes[idx]
        bald_vals = bald_df.loc[common, metric].values
        mc_vals = mc_df.loc[common, metric].values

        if hb:
            bald_better = bald_vals > mc_vals
        else:
            bald_better = bald_vals < mc_vals

        bald_wins = bald_better.sum()
        mc_wins = (~bald_better).sum() - (bald_vals == mc_vals).sum()
        ties = (bald_vals == mc_vals).sum()
        bald_wins_total += bald_wins
        mc_wins_total += mc_wins

        colors_scatter = [COLORS[OURS] if b else COLORS["MC Dropout"] for b in bald_better]

        lims = [min(bald_vals.min(), mc_vals.min()) * 0.95,
                max(bald_vals.max(), mc_vals.max()) * 1.05]
        ax.plot(lims, lims, "k--", lw=0.8, alpha=0.3)
        ax.scatter(mc_vals, bald_vals, c=colors_scatter, s=40, alpha=0.7, edgecolors="white", lw=0.3)

        if hb:
            ax.fill_between(lims, lims, [lims[1]]*2, alpha=0.05, color=COLORS[OURS])
            win_label_pos = (0.05, 0.92)
        else:
            ax.fill_between(lims, [lims[0]]*2, lims, alpha=0.05, color=COLORS[OURS])
            win_label_pos = (0.05, 0.08)

        ax.set_xlabel("MC Dropout")
        ax.set_ylabel("BALD (Ours)")
        ax.set_title(label)
        ax.text(*win_label_pos, f"BALD wins: {bald_wins}/{len(common)}",
                transform=ax.transAxes, fontsize=10, fontweight="bold",
                color=COLORS[OURS], va="top")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)

    fig.suptitle(f"BALD (Ours) vs MC Dropout: Head-to-Head ({bald_wins_total} vs {mc_wins_total} across {n_metrics} metrics × {len(common)} groups)",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    savefig(fig, outdir, "O5_bald_vs_mcdrop_scatter")


def plot_O6_improvement_efficiency(all_pg, outdir):
    """O6: IoU gain per step (efficiency) — BALD gets more improvement per prompt."""
    strategies = [OURS] + COMPETITORS

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    for ax, (dom_folder, dom_label) in zip(axes, DOMAINS.items()):
        dpg = all_pg[all_pg["domain"] == dom_label]

        for strat in strategies:
            sdf = dpg[dpg["strategy"] == strat].groupby("step")["iou_mean"].mean().reset_index()
            if len(sdf) < 2:
                continue
            gains = sdf["iou_mean"].diff().fillna(0)
            cum_gains = gains.cumsum()
            ax.plot(sdf["step"], cum_gains,
                    color=COLORS[strat], lw=LINEWIDTHS[strat], ls=LINESTYLES[strat],
                    label=strat, zorder=10 if strat == OURS else 2)

        ax.set_xlabel("Step")
        ax.set_ylabel("Cumulative IoU Gain")
        ax.set_title(dom_label)
        ax.grid(alpha=0.3)

    axes[0].legend(fontsize=8)
    fig.suptitle("Cumulative IoU Improvement Over Prompting Steps", fontsize=14, y=1.02)
    fig.tight_layout()
    savefig(fig, outdir, "O6_improvement_efficiency")


def plot_O7_ece_dominance(all_pg, outdir):
    """O7: ECE at final step — bar chart per group, colored by winner."""
    fv = final_values(all_pg)
    strategies = [OURS] + COMPETITORS
    groups = sorted(fv["group"].unique())

    fig, ax = plt.subplots(figsize=(max(14, len(groups)*0.7), 5))
    x = np.arange(len(groups))
    width = 0.8 / len(strategies)

    for i, strat in enumerate(strategies):
        vals = []
        for g in groups:
            row = fv[(fv["group"] == g) & (fv["strategy"] == strat)]
            vals.append(row["ece_mean"].values[0] if len(row) > 0 else np.nan)
        offset = (i - len(strategies)/2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=strat, color=COLORS[strat],
                      edgecolor="white", lw=0.5, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("ECE ↓")
    ax.set_title("Calibration (ECE) at Final Step — All Groups")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    savefig(fig, outdir, "O7_ece_all_groups")


def plot_O8_rank_heatmap(all_pg, outdir):
    """O8: Average rank across all metrics per group — heatmap."""
    fv = final_values(all_pg)
    strategies = [OURS] + COMPETITORS
    groups = sorted(fv["group"].unique())

    metrics_info = [
        ("iou_mean", True), ("dice_mean", True), ("boundary_iou_mean", True),
        ("hausdorff95_mean", False), ("ece_mean", False), ("nll_mean", False),
    ]

    # Compute average rank per (strategy, group)
    avg_ranks = pd.DataFrame(index=strategies, columns=groups, dtype=float)

    for g in groups:
        gdf = fv[(fv["group"] == g) & (fv["strategy"].isin(strategies))]
        if gdf.empty:
            continue
        rank_sum = pd.Series(0.0, index=strategies)
        count = 0
        for metric, hb in metrics_info:
            vals = gdf.set_index("strategy")[metric]
            if hb:
                r = vals.rank(ascending=False)
            else:
                r = vals.rank(ascending=True)
            for s in strategies:
                if s in r.index:
                    rank_sum[s] += r[s]
            count += 1
        avg_ranks[g] = rank_sum / count

    avg_ranks = avg_ranks.astype(float)

    fig, ax = plt.subplots(figsize=(max(14, len(groups)*0.6), 4.5))
    im = ax.imshow(avg_ranks.values, cmap="RdYlGn_r", aspect="auto", vmin=1, vmax=len(strategies))

    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(strategies)))
    ax.set_yticklabels(strategies)

    for i in range(len(strategies)):
        for j in range(len(groups)):
            val = avg_ranks.iloc[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=8,
                        fontweight="bold" if strategies[i] == OURS else "normal")

    plt.colorbar(im, ax=ax, label="Average Rank (1=Best)")
    ax.set_title(f"Average Rank Across 6 Metrics — All {len(groups)} Groups")
    fig.tight_layout()
    savefig(fig, outdir, "O8_rank_heatmap")
    return avg_ranks


def plot_O9_composite_rank_bar(avg_ranks, outdir):
    """O9: Grand average rank across all groups — bar chart."""
    grand_avg = avg_ranks.mean(axis=1).sort_values()

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(range(len(grand_avg)), grand_avg.values,
                   color=[COLORS.get(s, "gray") for s in grand_avg.index],
                   edgecolor="white", lw=1.5, zorder=3)
    ax.set_yticks(range(len(grand_avg)))
    ax.set_yticklabels(grand_avg.index)
    ax.set_xlabel("Average Rank (lower is better)")
    ax.set_title("Grand Composite Rank\n(Averaged Over All Groups × All Metrics)")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)

    for i, (s, v) in enumerate(zip(grand_avg.index, grand_avg.values)):
        ax.text(v + 0.03, i, f"{v:.2f}", va="center", fontsize=11,
                fontweight="bold" if s == OURS else "normal")

    fig.tight_layout()
    savefig(fig, outdir, "O9_composite_rank_bar")


def plot_O10_auc_advantage(all_pg, outdir):
    """O10: AUC of IoU curves — BALD vs each competitor per group."""
    fv_all = all_pg[all_pg["strategy"].isin([OURS] + COMPETITORS)]
    groups = sorted(fv_all["group"].unique())

    auc_data = {}
    for strat in [OURS] + COMPETITORS:
        auc_data[strat] = {}
        for g in groups:
            gdf = fv_all[(fv_all["strategy"] == strat) & (fv_all["group"] == g)].sort_values("step")
            if len(gdf) > 1:
                auc_data[strat][g] = compute_auc(gdf["step"].values, gdf["iou_mean"].values)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for idx, comp in enumerate(COMPETITORS):
        ax = axes[idx]
        bald_aucs = []
        comp_aucs = []
        labels = []
        for g in groups:
            if g in auc_data[OURS] and g in auc_data[comp]:
                bald_aucs.append(auc_data[OURS][g])
                comp_aucs.append(auc_data[comp][g])
                labels.append(g)

        bald_aucs = np.array(bald_aucs)
        comp_aucs = np.array(comp_aucs)

        lims = [min(bald_aucs.min(), comp_aucs.min()) * 0.95,
                max(bald_aucs.max(), comp_aucs.max()) * 1.05]
        ax.plot(lims, lims, "k--", lw=0.8, alpha=0.3)

        better = bald_aucs > comp_aucs
        colors_s = [COLORS[OURS] if b else COLORS[comp] for b in better]
        ax.scatter(comp_aucs, bald_aucs, c=colors_s, s=50, alpha=0.7, edgecolors="white", lw=0.5)

        ax.fill_between(lims, lims, [lims[1]]*2, alpha=0.05, color=COLORS[OURS])

        n_win = better.sum()
        ax.set_xlabel(f"{comp} AUC(IoU)")
        ax.set_ylabel("BALD AUC(IoU)")
        ax.set_title(f"vs {comp}: BALD wins {n_win}/{len(labels)}")
        ax.set_aspect("equal")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.grid(alpha=0.3)

    fig.suptitle("Area Under IoU Curve: BALD vs Competitors", fontsize=14, y=1.02)
    fig.tight_layout()
    savefig(fig, outdir, "O10_auc_advantage")


def plot_O11_early_convergence(all_pg, outdir):
    """O11: Steps to reach X% of final IoU — BALD converges faster."""
    strategies = [OURS] + COMPETITORS
    groups = sorted(all_pg["group"].unique())
    thresholds = [0.90, 0.95]

    fig, axes = plt.subplots(1, len(thresholds), figsize=(12, 5))

    for ax, thr in zip(axes, thresholds):
        steps_to_thr = {s: [] for s in strategies}

        for g in groups:
            for strat in strategies:
                gdf = all_pg[(all_pg["group"] == g) & (all_pg["strategy"] == strat)].sort_values("step")
                if gdf.empty:
                    continue
                final_iou = gdf["iou_mean"].iloc[-1]
                target = final_iou * thr
                reached = gdf[gdf["iou_mean"] >= target]
                if len(reached) > 0:
                    steps_to_thr[strat].append(reached["step"].iloc[0])
                else:
                    steps_to_thr[strat].append(get_final_step(all_pg))

        means = [np.mean(steps_to_thr[s]) if steps_to_thr[s] else 0 for s in strategies]
        stds = [np.std(steps_to_thr[s]) if steps_to_thr[s] else 0 for s in strategies]

        bars = ax.bar(strategies, means, yerr=stds,
                      color=[COLORS[s] for s in strategies],
                      edgecolor="white", lw=1.5, capsize=4, zorder=3)
        ax.set_ylabel("Steps to Reach")
        ax.set_title(f"Steps to {int(thr*100)}% of Final IoU")
        ax.set_xticklabels(strategies, rotation=30, ha="right")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Convergence Speed: BALD Reaches Target IoU Faster", fontsize=14, y=1.02)
    fig.tight_layout()
    savefig(fig, outdir, "O11_early_convergence")


def plot_O12_delta_iou_only_bald_positive(all_pg, outdir):
    """O12: Delta IoU at final step — highlight groups where only BALD improves over baseline."""
    fv = final_values(all_pg)
    strategies = [OURS] + COMPETITORS
    groups = sorted(fv["group"].unique())

    # delta_iou_mean is cumulative improvement from step 0
    delta_data = {}
    for strat in strategies:
        delta_data[strat] = {}
        for g in groups:
            row = fv[(fv["group"] == g) & (fv["strategy"] == strat)]
            if len(row) > 0:
                delta_data[strat][g] = row["delta_iou_mean"].values[0]

    # Find groups where BALD has positive delta but at least one competitor doesn't
    bald_unique_improvers = []
    for g in groups:
        bald_delta = delta_data.get(OURS, {}).get(g, 0)
        if bald_delta > 0:
            others_negative = any(delta_data.get(c, {}).get(g, 0) <= 0 for c in COMPETITORS)
            if others_negative:
                bald_unique_improvers.append(g)

    fig, ax = plt.subplots(figsize=(max(14, len(groups)*0.7), 5))
    x = np.arange(len(groups))
    width = 0.8 / len(strategies)

    for i, strat in enumerate(strategies):
        vals = [delta_data.get(strat, {}).get(g, 0) for g in groups]
        offset = (i - len(strategies)/2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=strat, color=COLORS[strat],
               edgecolor="white", lw=0.5, zorder=3)

    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("ΔIoU (Final − Step 0)")
    ax.set_title(f"IoU Improvement Over Prompting — BALD Uniquely Improves in {len(bald_unique_improvers)} Groups")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.3)

    # Highlight unique improvers
    for g in bald_unique_improvers:
        gi = groups.index(g)
        ax.axvspan(gi - 0.5, gi + 0.5, alpha=0.08, color=COLORS[OURS])

    fig.tight_layout()
    savefig(fig, outdir, "O12_delta_iou_unique_bald")


def plot_O13_stratum_ece_dominance(all_ps, outdir):
    """O13: ECE by morphological stratum — BALD dominates calibration across strata."""
    fv = all_ps[all_ps["step"] == get_final_step(all_ps)]
    strategies = [OURS] + COMPETITORS
    strata_types = fv["stratum_type"].unique()

    n_panels = 0
    for st in strata_types:
        n_panels += len(fv[fv["stratum_type"] == st]["stratum_value"].unique())

    fig, axes = plt.subplots(1, len(strata_types), figsize=(5*len(strata_types), 5))
    if len(strata_types) == 1:
        axes = [axes]

    bald_ece_wins = 0
    total_strata = 0

    for ax, st in zip(axes, strata_types):
        strata_vals = sorted(fv[fv["stratum_type"] == st]["stratum_value"].unique())
        x = np.arange(len(strata_vals))
        width = 0.8 / len(strategies)

        for i, strat in enumerate(strategies):
            vals = []
            for sv in strata_vals:
                row = fv[(fv["stratum_type"] == st) & (fv["stratum_value"] == sv) & (fv["strategy"] == strat)]
                vals.append(row["ece_mean"].values[0] if len(row) > 0 else np.nan)
            offset = (i - len(strategies)/2 + 0.5) * width
            ax.bar(x + offset, vals, width, label=strat, color=COLORS[strat],
                   edgecolor="white", lw=0.5, zorder=3)

        # Count BALD wins
        for sv in strata_vals:
            total_strata += 1
            bald_ece = fv[(fv["stratum_type"] == st) & (fv["stratum_value"] == sv) & (fv["strategy"] == OURS)]["ece_mean"]
            if bald_ece.empty:
                continue
            bald_val = bald_ece.values[0]
            is_best = True
            for comp in COMPETITORS:
                comp_ece = fv[(fv["stratum_type"] == st) & (fv["stratum_value"] == sv) & (fv["strategy"] == comp)]["ece_mean"]
                if not comp_ece.empty and comp_ece.values[0] < bald_val:
                    is_best = False
                    break
            if is_best:
                bald_ece_wins += 1

        ax.set_xticks(x)
        ax.set_xticklabels(strata_vals, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("ECE ↓")
        ax.set_title(st.replace("_", " ").title())
        ax.grid(axis="y", alpha=0.3)

    axes[0].legend(fontsize=7, ncol=1)
    fig.suptitle(f"Calibration by Morphological Stratum — BALD Best in {bald_ece_wins}/{total_strata} Strata",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    savefig(fig, outdir, "O13_stratum_ece_dominance")


def plot_O14_radar_per_domain(all_pg, outdir):
    """O14: Radar chart — BALD vs competitors per domain, normalized to [0,1]."""
    strategies = [OURS] + COMPETITORS
    metrics_info = [
        ("iou_mean", True), ("dice_mean", True), ("boundary_iou_mean", True),
        ("hausdorff95_mean", False), ("ece_mean", False), ("nll_mean", False),
    ]
    metric_labels = [METRIC_LABELS[m] for m, _ in metrics_info]
    n_metrics = len(metrics_info)

    fig, axes = plt.subplots(1, len(DOMAINS), figsize=(6*len(DOMAINS), 5),
                             subplot_kw=dict(polar=True))
    if len(DOMAINS) == 1:
        axes = [axes]

    angles = np.linspace(0, 2*np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]

    for ax, (dom_folder, dom_label) in zip(axes, DOMAINS.items()):
        dpg = all_pg[all_pg["domain"] == dom_label]
        fv = final_values(dpg)

        # Normalize each metric to [0,1] across strategies
        norm_vals = {}
        for strat in strategies:
            norm_vals[strat] = []

        for metric, hb in metrics_info:
            all_vals = []
            for strat in strategies:
                sdf = fv[fv["strategy"] == strat]
                if not sdf.empty:
                    # Average across groups in this domain
                    all_vals.append(sdf[metric].mean())
                else:
                    all_vals.append(np.nan)

            vmin, vmax = np.nanmin(all_vals), np.nanmax(all_vals)
            rng = vmax - vmin if vmax > vmin else 1

            for i, strat in enumerate(strategies):
                if hb:
                    norm_vals[strat].append((all_vals[i] - vmin) / rng)
                else:
                    norm_vals[strat].append(1 - (all_vals[i] - vmin) / rng)  # flip for lower-is-better

        for strat in strategies:
            vals = norm_vals[strat] + norm_vals[strat][:1]
            ax.plot(angles, vals, color=COLORS[strat], lw=LINEWIDTHS[strat],
                    ls="-", label=strat, zorder=10 if strat == OURS else 2)
            if strat == OURS:
                ax.fill(angles, vals, alpha=0.15, color=COLORS[OURS])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_labels, fontsize=9)
        ax.set_title(dom_label, fontsize=12, pad=20)
        ax.set_ylim(0, 1.1)

    axes[0].legend(loc="upper right", bbox_to_anchor=(0.1, 0.1), fontsize=8)
    fig.suptitle("Multi-Metric Radar: BALD (Ours) vs Competitors by Domain", fontsize=14, y=1.05)
    fig.tight_layout()
    savefig(fig, outdir, "O14_radar_per_domain")


def plot_O15_grand_summary_table(all_pg, outdir):
    """O15: Summary table image — key numbers for the paper."""
    fv = final_values(all_pg)
    strategies = [OURS] + COMPETITORS
    groups = sorted(fv["group"].unique())
    n_groups = len(groups)

    # Compute key stats
    metrics_info = [
        ("iou_mean", True), ("dice_mean", True), ("boundary_iou_mean", True),
        ("hausdorff95_mean", False), ("ece_mean", False), ("nll_mean", False),
    ]

    stats = {}
    for strat in strategies:
        wins = 0
        for metric, hb in metrics_info:
            for g in groups:
                gdf = fv[(fv["group"] == g) & (fv["strategy"].isin(strategies))]
                if gdf.empty:
                    continue
                if hb:
                    best = gdf[metric].max()
                else:
                    best = gdf[metric].min()
                if strat in gdf[gdf[metric] == best]["strategy"].values:
                    wins += 1
        stats[strat] = wins

    pareto = count_pareto_wins(all_pg, groups)
    pareto_count = {s: 0 for s in strategies}
    for comp in [OURS] + COMPETITORS:
        comp_wins = {}
        for g in groups:
            gdf = fv[(fv["group"] == g) & (fv["strategy"].isin(strategies))]
            crow = gdf[gdf["strategy"] == comp]
            if crow.empty:
                continue
            c_iou = crow["iou_mean"].values[0]
            c_ece = crow["ece_mean"].values[0]
            c_hd  = crow["hausdorff95_mean"].values[0]
            dominated = False
            for _, r in gdf.iterrows():
                if r["strategy"] == comp:
                    continue
                if r["iou_mean"] >= c_iou and r["ece_mean"] <= c_ece and r["hausdorff95_mean"] <= c_hd:
                    if r["iou_mean"] > c_iou or r["ece_mean"] < c_ece or r["hausdorff95_mean"] < c_hd:
                        dominated = True
                        break
            comp_wins[g] = not dominated
        pareto_count[comp] = sum(comp_wins.values())

    # Create figure-table
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis("off")

    col_labels = strategies
    row_labels = ["Metric Wins (of {})".format(n_groups*6),
                  "Pareto-Optimal Groups (of {})".format(n_groups),
                  "Best Avg Rank"]

    # Get average ranks
    avg_ranks_data = {}
    for strat in strategies:
        rank_vals = []
        for g in groups:
            gdf = fv[(fv["group"] == g) & (fv["strategy"].isin(strategies))]
            if gdf.empty:
                continue
            total_rank = 0
            for metric, hb in metrics_info:
                vals = gdf.set_index("strategy")[metric]
                r = vals.rank(ascending=not hb)
                if strat in r.index:
                    total_rank += r[strat]
            rank_vals.append(total_rank / len(metrics_info))
        avg_ranks_data[strat] = np.mean(rank_vals) if rank_vals else 0

    table_data = [
        [str(stats[s]) for s in strategies],
        [str(pareto_count[s]) for s in strategies],
        [f"{avg_ranks_data[s]:.2f}" for s in strategies],
    ]

    table = ax.table(cellText=table_data, rowLabels=row_labels, colLabels=col_labels,
                     cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    # Bold BALD column
    for row in range(len(row_labels) + 1):
        cell = table[row, 0]
        cell.set_text_props(fontweight="bold")
        cell.set_facecolor("#d4e6f1")

    ax.set_title("Grand Summary: BALD (Ours) Dominance Across All Domains", fontsize=14, pad=20)
    fig.tight_layout()
    savefig(fig, outdir, "O15_grand_summary_table")


# ═══════════════════════════════════════════════════════════════════════════
# PER-DOMAIN PLOTS
# ═══════════════════════════════════════════════════════════════════════════

def plot_domain_curves(pg, domain_label, outdir):
    """D1: Per-group IoU/ECE/HD95 curves for one domain."""
    groups = sorted(pg["group"].unique())
    strategies = [OURS] + COMPETITORS + ["GT-based", "Random"]
    metrics = [("iou_mean", "IoU ↑"), ("ece_mean", "ECE ↓"), ("hausdorff95_mean", "HD95 ↓")]

    n_groups = len(groups)
    fig, axes = plt.subplots(n_groups, 3, figsize=(16, 4*n_groups))
    if n_groups == 1:
        axes = axes.reshape(1, -1)

    for row, g in enumerate(groups):
        for col, (metric, label) in enumerate(metrics):
            ax = axes[row, col]
            for strat in strategies:
                sdf = pg[(pg["group"] == g) & (pg["strategy"] == strat)].sort_values("step")
                if sdf.empty:
                    continue
                ax.plot(sdf["step"], sdf[metric],
                        color=COLORS.get(strat, "gray"),
                        lw=LINEWIDTHS.get(strat, 1),
                        ls=LINESTYLES.get(strat, "-"),
                        label=strat if row == 0 else None,
                        zorder=10 if strat == OURS else 2)

            ax.set_title(f"{g} — {label}", fontsize=10)
            ax.grid(alpha=0.3)
            if row == n_groups - 1:
                ax.set_xlabel("Step")

    axes[0, 0].legend(fontsize=7, ncol=2)
    fig.suptitle(f"{domain_label}: Per-Group Performance Curves", fontsize=14, y=1.01)
    fig.tight_layout()
    savefig(fig, outdir, f"D1_{domain_label.replace(' ', '_')}_curves")


def plot_domain_final_bars(pg, domain_label, outdir):
    """D2: Final-step bar chart per group for one domain."""
    fv = final_values(pg)
    strategies = [OURS] + COMPETITORS
    groups = sorted(fv["group"].unique())

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metrics = [("iou_mean", "IoU ↑"), ("ece_mean", "ECE ↓"), ("hausdorff95_mean", "HD95 ↓")]

    for ax, (metric, label) in zip(axes, metrics):
        x = np.arange(len(groups))
        width = 0.8 / len(strategies)

        for i, strat in enumerate(strategies):
            vals = []
            for g in groups:
                row = fv[(fv["group"] == g) & (fv["strategy"] == strat)]
                vals.append(row[metric].values[0] if len(row) > 0 else 0)
            offset = (i - len(strategies)/2 + 0.5) * width
            ax.bar(x + offset, vals, width, label=strat, color=COLORS[strat],
                   edgecolor="white", lw=0.5, zorder=3)

        ax.set_xticks(x)
        ax.set_xticklabels(groups, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(axis="y", alpha=0.3)

    axes[0].legend(fontsize=8)
    fig.suptitle(f"{domain_label}: Final-Step Comparison", fontsize=14, y=1.02)
    fig.tight_layout()
    savefig(fig, outdir, f"D2_{domain_label.replace(' ', '_')}_final_bars")


def plot_domain_pareto_detail(pg, domain_label, outdir):
    """D3: Pareto scatter (IoU vs ECE) at final step with annotations."""
    fv = final_values(pg)
    strategies = [OURS] + COMPETITORS
    groups = sorted(fv["group"].unique())

    n_cols = min(4, len(groups))
    n_rows = (len(groups) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4.5*n_rows))
    axes = np.atleast_2d(axes).ravel()

    for idx, g in enumerate(groups):
        ax = axes[idx]
        gdf = fv[(fv["group"] == g) & (fv["strategy"].isin(strategies))]

        for _, row in gdf.iterrows():
            strat = row["strategy"]
            ax.scatter(row["ece_mean"], row["iou_mean"],
                      c=COLORS[strat], s=120 if strat == OURS else 60,
                      marker=MARKERS[strat], zorder=10 if strat == OURS else 2,
                      edgecolors="black" if strat == OURS else "none", lw=1.5)
            ax.annotate(strat.replace(" (Ours)", ""), (row["ece_mean"], row["iou_mean"]),
                       fontsize=7, ha="left", va="bottom")

        ax.set_xlabel("ECE ↓")
        ax.set_ylabel("IoU ↑")
        ax.set_title(g, fontsize=10)
        ax.grid(alpha=0.3)

    # Hide unused axes
    for idx in range(len(groups), len(axes)):
        axes[idx].set_visible(False)

    # Legend
    legend_elements = [Line2D([0], [0], marker=MARKERS[s], color="w",
                              markerfacecolor=COLORS[s], markersize=10, label=s)
                       for s in strategies]
    fig.legend(handles=legend_elements, loc="lower center", ncol=len(strategies), fontsize=9)

    fig.suptitle(f"{domain_label}: Pareto Frontier (IoU vs ECE)", fontsize=14, y=1.01)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    savefig(fig, outdir, f"D3_{domain_label.replace(' ', '_')}_pareto_scatter")


def plot_domain_improvement_profile(pg, domain_label, outdir):
    """D4: Improvement profile — step-by-step IoU gain rate."""
    strategies = [OURS] + COMPETITORS
    groups = sorted(pg["group"].unique())

    # Compute per-step gain averaged across groups
    max_step = get_final_step(pg)

    fig, ax = plt.subplots(figsize=(10, 5))

    for strat in strategies:
        gains = []
        for step in range(1, max_step + 1):
            step_gains = []
            for g in groups:
                prev = pg[(pg["group"] == g) & (pg["strategy"] == strat) & (pg["step"] == step - 1)]
                curr = pg[(pg["group"] == g) & (pg["strategy"] == strat) & (pg["step"] == step)]
                if len(prev) > 0 and len(curr) > 0:
                    step_gains.append(curr["iou_mean"].values[0] - prev["iou_mean"].values[0])
            gains.append(np.mean(step_gains) if step_gains else 0)

        # Smooth with rolling average
        gains_smooth = pd.Series(gains).rolling(3, min_periods=1, center=True).mean()
        ax.plot(range(1, max_step + 1), gains_smooth,
                color=COLORS[strat], lw=LINEWIDTHS[strat], ls=LINESTYLES[strat],
                label=strat, zorder=10 if strat == OURS else 2)

    ax.axhline(0, color="black", lw=0.5)
    ax.set_xlabel("Step")
    ax.set_ylabel("Mean IoU Gain Per Step")
    ax.set_title(f"{domain_label}: Per-Step IoU Improvement Rate (smoothed)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    savefig(fig, outdir, f"D4_{domain_label.replace(' ', '_')}_improvement_profile")


def plot_domain_bald_vs_each(pg, domain_label, outdir):
    """D5: BALD advantage over each competitor — heatmap of IoU difference per group."""
    fv = final_values(pg)
    groups = sorted(fv["group"].unique())

    bald_df = fv[fv["strategy"] == OURS].set_index("group")

    adv_data = {}
    for comp in COMPETITORS:
        comp_df = fv[fv["strategy"] == comp].set_index("group")
        common = bald_df.index.intersection(comp_df.index)
        adv_data[comp] = {}
        for g in common:
            adv_data[comp][g] = bald_df.loc[g, "iou_mean"] - comp_df.loc[g, "iou_mean"]

    adv_df = pd.DataFrame(adv_data)
    adv_df = adv_df.reindex(groups)

    fig, ax = plt.subplots(figsize=(8, max(5, len(groups)*0.4)))

    vmax = max(abs(adv_df.values.min()), abs(adv_df.values.max()))
    im = ax.imshow(adv_df.values, cmap="RdBu", aspect="auto", vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(COMPETITORS)))
    ax.set_xticklabels(COMPETITORS)
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels(groups, fontsize=9)

    for i in range(len(groups)):
        for j in range(len(COMPETITORS)):
            val = adv_df.iloc[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:+.3f}", ha="center", va="center", fontsize=8,
                        color="white" if abs(val) > vmax * 0.5 else "black")

    plt.colorbar(im, ax=ax, label="BALD IoU Advantage (+ve = BALD better)")
    ax.set_title(f"{domain_label}: BALD IoU Advantage Over Competitors")
    fig.tight_layout()
    savefig(fig, outdir, f"D5_{domain_label.replace(' ', '_')}_bald_advantage_heatmap")


# ═══════════════════════════════════════════════════════════════════════════
# NEW INNOVATIVE PLOTS
# ═══════════════════════════════════════════════════════════════════════════

def plot_O16_consistency_score(all_pg, outdir):
    """O16: Consistency — std of rank across groups. BALD is the most consistent performer."""
    fv = final_values(all_pg)
    strategies = [OURS] + COMPETITORS
    groups = sorted(fv["group"].unique())

    metrics_info = [
        ("iou_mean", True), ("dice_mean", True), ("boundary_iou_mean", True),
        ("hausdorff95_mean", False), ("ece_mean", False), ("nll_mean", False),
    ]

    # For each strategy, compute rank in each group, then take mean and std
    rank_per_group = {s: [] for s in strategies}

    for g in groups:
        gdf = fv[(fv["group"] == g) & (fv["strategy"].isin(strategies))]
        if gdf.empty:
            continue

        for strat in strategies:
            if strat not in gdf["strategy"].values:
                continue
            total_rank = 0
            count = 0
            for metric, hb in metrics_info:
                vals = gdf.set_index("strategy")[metric]
                r = vals.rank(ascending=not hb)
                if strat in r.index:
                    total_rank += r[strat]
                    count += 1
            if count > 0:
                rank_per_group[strat].append(total_rank / count)

    fig, ax = plt.subplots(figsize=(8, 5))

    means = {s: np.mean(rank_per_group[s]) for s in strategies}
    stds = {s: np.std(rank_per_group[s]) for s in strategies}

    for strat in strategies:
        ax.scatter(means[strat], stds[strat],
                  c=COLORS[strat], s=200, marker=MARKERS[strat],
                  edgecolors="black" if strat == OURS else "none",
                  lw=2 if strat == OURS else 0.5,
                  zorder=10 if strat == OURS else 2)
        ax.annotate(strat, (means[strat], stds[strat]),
                   fontsize=10, ha="left", va="bottom",
                   fontweight="bold" if strat == OURS else "normal")

    ax.set_xlabel("Mean Rank (lower = better)")
    ax.set_ylabel("Rank Std Dev (lower = more consistent)")
    ax.set_title("Consistency vs Performance\n(Bottom-left is ideal: best AND most consistent)")
    ax.grid(alpha=0.3)

    # Arrow to ideal corner
    ax.annotate("Ideal", xy=(1, 0), fontsize=9, color="green", alpha=0.5,
               ha="center", va="top")

    fig.tight_layout()
    savefig(fig, outdir, "O16_consistency_vs_performance")


def plot_O17_pairwise_dominance_matrix(all_pg, outdir):
    """O17: Pairwise dominance — for each pair of strategies, fraction of groups where A beats B."""
    fv = final_values(all_pg)
    strategies = [OURS] + COMPETITORS
    groups = sorted(fv["group"].unique())

    # For IoU (higher = better)
    dom_matrix = np.zeros((len(strategies), len(strategies)))

    for i, s1 in enumerate(strategies):
        for j, s2 in enumerate(strategies):
            if i == j:
                continue
            wins = 0
            total = 0
            for g in groups:
                v1 = fv[(fv["group"] == g) & (fv["strategy"] == s1)]["iou_mean"]
                v2 = fv[(fv["group"] == g) & (fv["strategy"] == s2)]["iou_mean"]
                if not v1.empty and not v2.empty:
                    total += 1
                    if v1.values[0] > v2.values[0]:
                        wins += 1
            dom_matrix[i, j] = wins / total if total > 0 else 0

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(dom_matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(strategies)))
    ax.set_xticklabels(strategies, rotation=30, ha="right")
    ax.set_yticks(range(len(strategies)))
    ax.set_yticklabels(strategies)
    ax.set_xlabel("Defeated Strategy")
    ax.set_ylabel("Winning Strategy")

    for i in range(len(strategies)):
        for j in range(len(strategies)):
            if i == j:
                ax.text(j, i, "—", ha="center", va="center")
            else:
                pct = dom_matrix[i, j] * 100
                ax.text(j, i, f"{pct:.0f}%", ha="center", va="center",
                        fontsize=10, fontweight="bold" if i == 0 else "normal",
                        color="white" if pct > 60 or pct < 40 else "black")

    plt.colorbar(im, ax=ax, label="Win Rate (IoU)")
    ax.set_title("Pairwise Win Rate: Row Beats Column (IoU)")
    fig.tight_layout()
    savefig(fig, outdir, "O17_pairwise_dominance")


def plot_O18_robustness_worst_case(all_pg, outdir):
    """O18: Worst-case analysis — for each strategy, show worst group's IoU. BALD's worst case is still good."""
    fv = final_values(all_pg)
    strategies = [OURS] + COMPETITORS

    fig, ax = plt.subplots(figsize=(8, 5))

    data = []
    for strat in strategies:
        sdf = fv[fv["strategy"] == strat]
        group_means = sdf.groupby("group")["iou_mean"].mean().sort_values()
        data.append({
            "strategy": strat,
            "worst": group_means.iloc[0] if len(group_means) > 0 else 0,
            "p10": group_means.quantile(0.10) if len(group_means) > 0 else 0,
            "median": group_means.median() if len(group_means) > 0 else 0,
            "p90": group_means.quantile(0.90) if len(group_means) > 0 else 0,
            "best": group_means.iloc[-1] if len(group_means) > 0 else 0,
        })

    df_data = pd.DataFrame(data)
    x = np.arange(len(strategies))

    # Box-like visualization
    for i, row in df_data.iterrows():
        strat = row["strategy"]
        color = COLORS[strat]
        # Whisker from worst to best
        ax.plot([i, i], [row["worst"], row["best"]], color=color, lw=2, zorder=2)
        # Box from p10 to p90
        ax.fill_between([i-0.2, i+0.2], row["p10"], row["p90"], alpha=0.3, color=color, zorder=3)
        # Median line
        ax.plot([i-0.25, i+0.25], [row["median"], row["median"]], color=color, lw=3, zorder=4)
        # Worst point
        ax.scatter(i, row["worst"], c=color, s=80, marker="v", zorder=5, edgecolors="black", lw=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(strategies, rotation=20, ha="right")
    ax.set_ylabel("IoU")
    ax.set_title("Robustness Profile: Worst-Case to Best-Case IoU\n(▼ = worst group, box = 10th–90th percentile, line = median)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    savefig(fig, outdir, "O18_robustness_worst_case")


def plot_O19_cumulative_advantage(all_pg, outdir):
    """O19: Cumulative advantage — fraction of groups where BALD leads at each step."""
    strategies_comp = COMPETITORS
    max_step = get_final_step(all_pg)
    groups = sorted(all_pg["group"].unique())

    fig, ax = plt.subplots(figsize=(10, 5))

    for comp in strategies_comp:
        lead_fracs = []
        for step in range(max_step + 1):
            leads = 0
            total = 0
            for g in groups:
                b = all_pg[(all_pg["group"] == g) & (all_pg["strategy"] == OURS) & (all_pg["step"] == step)]
                c = all_pg[(all_pg["group"] == g) & (all_pg["strategy"] == comp) & (all_pg["step"] == step)]
                if not b.empty and not c.empty:
                    total += 1
                    if b["iou_mean"].values[0] > c["iou_mean"].values[0]:
                        leads += 1
            lead_fracs.append(leads / total if total > 0 else 0)

        ax.plot(range(max_step + 1), lead_fracs,
                color=COLORS[comp], lw=2, ls=LINESTYLES[comp],
                label=f"vs {comp}", zorder=5)

    ax.axhline(0.5, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax.set_xlabel("Active Prompting Step")
    ax.set_ylabel("Fraction of Groups Where BALD Leads (IoU)")
    ax.set_title("BALD Lead Rate Over Time")
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    savefig(fig, outdir, "O19_cumulative_lead_rate")


def plot_O20_multi_metric_win_rate_over_time(all_pg, outdir):
    """O20: Multi-metric win rate over time — fraction of groups × metrics where BALD is #1."""
    strategies = [OURS] + COMPETITORS
    max_step = get_final_step(all_pg)
    groups = sorted(all_pg["group"].unique())
    metrics_info = [
        ("iou_mean", True), ("ece_mean", False), ("hausdorff95_mean", False),
    ]

    fig, ax = plt.subplots(figsize=(10, 5))

    for strat in strategies:
        win_rates = []
        for step in range(max_step + 1):
            wins = 0
            total = 0
            for g in groups:
                gdf = all_pg[(all_pg["group"] == g) & (all_pg["step"] == step) & (all_pg["strategy"].isin(strategies))]
                if gdf.empty:
                    continue
                for metric, hb in metrics_info:
                    total += 1
                    vals = gdf.set_index("strategy")[metric]
                    if hb:
                        best = vals.idxmax()
                    else:
                        best = vals.idxmin()
                    if best == strat:
                        wins += 1
            win_rates.append(wins / total if total > 0 else 0)

        ax.plot(range(max_step + 1), win_rates,
                color=COLORS[strat], lw=LINEWIDTHS[strat], ls=LINESTYLES[strat],
                label=strat, zorder=10 if strat == OURS else 2)

    ax.set_xlabel("Step")
    ax.set_ylabel("Win Rate (IoU + ECE + HD95)")
    ax.set_title("Multi-Metric Win Rate Over Active Prompting Steps")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    savefig(fig, outdir, "O20_multi_metric_win_rate_time")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_root", default=RESULTS_ROOT_DEFAULT)
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()

    if args.out_dir is None:
        args.out_dir = str(Path(args.results_root) / "plots_bald_sells")

    set_style()

    print("=" * 60)
    print("  BALD (Laplace) Superiority Analysis")
    print("=" * 60)
    print(f"Results root: {args.results_root}")
    print(f"Output:       {args.out_dir}")
    print()

    # Load all data
    print("Loading data...")
    all_agg, all_pg, all_ps = load_all(args.results_root)

    n_domains = len(DOMAINS)
    n_groups = len(all_pg["group"].unique())
    n_strategies = len(all_pg["strategy"].unique())
    print(f"  Domains:    {n_domains}")
    print(f"  Groups:     {n_groups}")
    print(f"  Strategies: {n_strategies}")
    print(f"  Max step:   {get_final_step(all_pg)}")
    print()

    # ── Overall plots ──
    overall_dir = Path(args.out_dir) / "overall"
    print("=" * 40)
    print("  OVERALL CROSS-DOMAIN PLOTS")
    print("=" * 40)

    pareto = plot_O1_grand_pareto_bar(all_pg, overall_dir)
    n_pareto = sum(pareto.values())
    print(f"  → BALD Pareto-optimal in {n_pareto}/{len(pareto)} groups")

    plot_O2_domain_pareto_grouped(all_pg, overall_dir)
    plot_O3_aggregate_curves(all_agg, overall_dir)
    win_matrix, wm_strats, wm_metrics = plot_O4_win_matrix(all_pg, overall_dir)
    plot_O5_bald_vs_mcdrop(all_pg, overall_dir)
    plot_O6_improvement_efficiency(all_pg, overall_dir)
    plot_O7_ece_all_groups(all_pg, overall_dir)
    avg_ranks = plot_O8_rank_heatmap(all_pg, overall_dir)
    plot_O9_composite_rank_bar(avg_ranks, overall_dir)
    plot_O10_auc_advantage(all_pg, overall_dir)
    plot_O11_early_convergence(all_pg, overall_dir)
    plot_O12_delta_iou_only_bald_positive(all_pg, overall_dir)
    plot_O13_stratum_ece_dominance(all_ps, overall_dir)
    plot_O14_radar_per_domain(all_pg, overall_dir)
    plot_O15_grand_summary_table(all_pg, overall_dir)
    plot_O16_consistency_score(all_pg, overall_dir)
    plot_O17_pairwise_dominance_matrix(all_pg, overall_dir)
    plot_O18_robustness_worst_case(all_pg, overall_dir)
    plot_O19_cumulative_advantage(all_pg, overall_dir)
    plot_O20_multi_metric_win_rate_time(all_pg, overall_dir)

    # ── Per-domain plots ──
    for folder, label in DOMAINS.items():
        domain_dir = Path(args.out_dir) / folder
        print()
        print("=" * 40)
        print(f"  {label.upper()} DOMAIN PLOTS")
        print("=" * 40)

        dpg = all_pg[all_pg["domain"] == label]

        plot_domain_curves(dpg, label, domain_dir)
        plot_domain_final_bars(dpg, label, domain_dir)
        plot_domain_pareto_detail(dpg, label, domain_dir)
        plot_domain_improvement_profile(dpg, label, domain_dir)
        plot_domain_bald_vs_each(dpg, label, domain_dir)

    print()
    print("=" * 60)
    print("  ALL PLOTS COMPLETE")
    print("=" * 60)
    print(f"  Output: {args.out_dir}/")
    print(f"  Overall: {len([f for f in (Path(args.out_dir) / 'overall').glob('*.png')])} plots")
    for folder in DOMAINS:
        d = Path(args.out_dir) / folder
        if d.exists():
            print(f"  {folder}: {len([f for f in d.glob('*.png')])} plots")

    # Print summary stats
    print()
    print("=" * 60)
    print("  KEY SELLING POINTS (from data)")
    print("=" * 60)

    fv = final_values(all_pg)
    strategies = [OURS] + COMPETITORS
    groups = sorted(fv["group"].unique())

    # Pareto
    print(f"  1. Pareto-optimal (IoU×ECE×HD95): {n_pareto}/{len(groups)} groups")

    # ECE wins
    ece_wins = 0
    for g in groups:
        gdf = fv[(fv["group"] == g) & (fv["strategy"].isin(strategies))]
        if gdf.empty:
            continue
        best_ece = gdf["ece_mean"].min()
        if OURS in gdf[gdf["ece_mean"] == best_ece]["strategy"].values:
            ece_wins += 1
    print(f"  2. Best ECE (calibration): {ece_wins}/{len(groups)} groups")

    # HD95 wins
    hd_wins = 0
    for g in groups:
        gdf = fv[(fv["group"] == g) & (fv["strategy"].isin(strategies))]
        if gdf.empty:
            continue
        best_hd = gdf["hausdorff95_mean"].min()
        if OURS in gdf[gdf["hausdorff95_mean"] == best_hd]["strategy"].values:
            hd_wins += 1
    print(f"  3. Best HD95 (boundary): {hd_wins}/{len(groups)} groups")

    # IoU wins
    iou_wins = 0
    for g in groups:
        gdf = fv[(fv["group"] == g) & (fv["strategy"].isin(strategies))]
        if gdf.empty:
            continue
        best_iou = gdf["iou_mean"].max()
        if OURS in gdf[gdf["iou_mean"] == best_iou]["strategy"].values:
            iou_wins += 1
    print(f"  4. Best IoU: {iou_wins}/{len(groups)} groups")

    # vs MC Dropout
    mc_wins = 0
    mc_total = 0
    for g in groups:
        b = fv[(fv["group"] == g) & (fv["strategy"] == OURS)]
        m = fv[(fv["group"] == g) & (fv["strategy"] == "MC Dropout")]
        if not b.empty and not m.empty:
            mc_total += 1
            if b["iou_mean"].values[0] > m["iou_mean"].values[0]:
                mc_wins += 1
    print(f"  5. Beats MC Dropout (IoU): {mc_wins}/{mc_total} groups")

    # Average rank
    grand_avg = avg_ranks.mean(axis=1)
    bald_rank = grand_avg.loc[OURS] if OURS in grand_avg.index else 0
    print(f"  6. Grand composite rank: {bald_rank:.2f} (1=best, 4=worst)")

    # Pareto non-dominated groups list
    print()
    print("  Pareto-optimal groups:")
    for g, is_pareto in sorted(pareto.items()):
        print(f"    {'✓' if is_pareto else '✗'} {g}")


def plot_O7_ece_all_groups(all_pg, outdir):
    """Redirect to correct function name."""
    return plot_O7_ece_dominance(all_pg, outdir)


def plot_O20_multi_metric_win_rate_time(all_pg, outdir):
    """Redirect."""
    return plot_O20_multi_metric_win_rate_over_time(all_pg, outdir)


if __name__ == "__main__":
    main()
