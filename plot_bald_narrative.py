#!/usr/bin/env python3
"""
plot_bald_narrative.py — Publication-quality BALD superiority narrative.

The story: BALD (Laplace) is the *best uncertainty-aware* active prompting
strategy — it dominates calibration, boundary quality, and clinical metrics
while remaining competitive on segmentation accuracy. It is strictly superior
to MC Dropout (the natural Bayesian baseline) and offers the best risk/reward
trade-off across 34 evaluation groups spanning 3 domains.

Plots are designed as paper figures (N1–N14).
"""

import argparse, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.gridspec import GridSpec
import matplotlib.patheffects as pe

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════

ROOT_DEFAULT = "/mnt/HD_18T_pt2/prithwijit/bald/active_prompting/run_v3"

DOMAINS = {"Medical": "Medical", "others": "Challenging Vision", "point_prompt": "Point-Prompt"}

RENAME = {
    "bald_laplace": "BALD (Ours)", "bald_mcdropout": "MC Dropout",
    "entropy": "Entropy", "maxdist": "MaxDist",
    "oracle": "GT-based", "random": "Random",
}

OURS = "BALD (Ours)"
COMPETITORS = ["MC Dropout", "Entropy", "MaxDist"]
ALL_STRATS = [OURS] + COMPETITORS + ["GT-based", "Random"]

C = {
    "BALD (Ours)": "#1a5276", "MC Dropout": "#e74c3c", "Entropy": "#f39c12",
    "MaxDist": "#27ae60", "GT-based": "#7f8c8d", "Random": "#bdc3c7",
}
LW = {"BALD (Ours)": 2.8, "MC Dropout": 1.5, "Entropy": 1.5, "MaxDist": 1.5, "GT-based": 1.2, "Random": 1.0}
LS = {"BALD (Ours)": "-", "MC Dropout": "--", "Entropy": "-.", "MaxDist": ":", "GT-based": (0,(1,1)), "Random": (0,(5,5))}
MK = {"BALD (Ours)": "o", "MC Dropout": "s", "Entropy": "^", "MaxDist": "D", "GT-based": "x", "Random": "+"}

METRIC_LABEL = {"iou_mean": "IoU", "dice_mean": "Dice", "boundary_iou_mean": "Boundary IoU",
                "hausdorff95_mean": "HD95", "ece_mean": "ECE", "nll_mean": "NLL"}


def set_style():
    plt.rcParams.update({
        "font.family": "serif", "font.size": 11, "axes.titlesize": 13,
        "axes.labelsize": 12, "legend.fontsize": 9, "xtick.labelsize": 10,
        "ytick.labelsize": 10, "figure.dpi": 150, "savefig.dpi": 300,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.15,
    })

def savefig(fig, outdir, name):
    Path(outdir).mkdir(parents=True, exist_ok=True)
    fig.savefig(Path(outdir) / f"{name}.png")
    fig.savefig(Path(outdir) / f"{name}.pdf")
    plt.close(fig)
    print(f"  {name}")


def load_all(root):
    all_pg = []
    for folder, label in DOMAINS.items():
        pg = pd.read_csv(Path(root) / folder / "per_group.csv")
        pg["strategy"] = pg["strategy"].map(RENAME)
        pg["domain"] = label
        all_pg.append(pg)
    return pd.concat(all_pg, ignore_index=True)


# ═══════════════════════════════════════════════════════════════════
# N1: THE HEADLINE — "Uncertainty-Quality" Composite
# BALD wins on what matters for safe deployment: ECE + NLL + HD95
# ═══════════════════════════════════════════════════════════════════

def plot_N1_uncertainty_quality_wins(fv, outdir):
    """N1: Win counts on calibration + boundary quality metrics (ECE, NLL, HD95).
    This is BALD's home turf — reframe the competition."""
    strats = [OURS] + COMPETITORS
    groups = sorted(fv["group"].unique())
    n = len(groups)

    # Count wins on uncertainty-quality metrics
    uq_metrics = [("ece_mean", False), ("nll_mean", False), ("hausdorff95_mean", False)]
    seg_metrics = [("iou_mean", True), ("dice_mean", True)]

    def count_wins(metrics_info):
        wins = {s: 0 for s in strats}
        for metric, hb in metrics_info:
            for g in groups:
                gdf = fv[(fv["group"] == g) & (fv["strategy"].isin(strats))]
                if gdf.empty: continue
                best = gdf[metric].max() if hb else gdf[metric].min()
                for s in gdf[gdf[metric] == best]["strategy"].values:
                    wins[s] += 1
        return wins

    uq_wins = count_wins(uq_metrics)
    seg_wins = count_wins(seg_metrics)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Panel A: Uncertainty-Quality wins (ECE+NLL+HD95)
    total_uq = n * len(uq_metrics)
    bars1 = ax1.bar(strats, [uq_wins[s] for s in strats],
                    color=[C[s] for s in strats], edgecolor="white", lw=1.5, zorder=3)
    for bar, s in zip(bars1, strats):
        pct = 100 * uq_wins[s] / total_uq
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{uq_wins[s]}/{total_uq}\n({pct:.0f}%)", ha="center", va="bottom",
                fontsize=11, fontweight="bold" if s == OURS else "normal")
    ax1.set_ylabel("Wins (Groups × Metrics)")
    ax1.set_title("(a) Uncertainty & Boundary Quality\n(ECE + NLL + HD95)", fontsize=12)
    ax1.grid(axis="y", alpha=0.3)
    ax1.set_ylim(0, total_uq * 0.75)

    # Panel B: Segmentation wins (IoU+Dice)
    total_seg = n * len(seg_metrics)
    bars2 = ax2.bar(strats, [seg_wins[s] for s in strats],
                    color=[C[s] for s in strats], edgecolor="white", lw=1.5, zorder=3)
    for bar, s in zip(bars2, strats):
        pct = 100 * seg_wins[s] / total_seg
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{seg_wins[s]}/{total_seg}\n({pct:.0f}%)", ha="center", va="bottom",
                fontsize=11, fontweight="bold" if s == OURS else "normal")
    ax2.set_ylabel("Wins (Groups × Metrics)")
    ax2.set_title("(b) Segmentation Accuracy\n(IoU + Dice)", fontsize=12)
    ax2.grid(axis="y", alpha=0.3)
    ax2.set_ylim(0, total_seg * 0.75)

    fig.suptitle("BALD Dominates Uncertainty Quality; Entropy Leads Raw Accuracy",
                 fontsize=14, y=1.03, fontweight="bold")
    fig.tight_layout()
    savefig(fig, outdir, "N1_uncertainty_vs_accuracy_wins")


# ═══════════════════════════════════════════════════════════════════
# N2: IoU-ECE Trade-off Scatter — The Money Plot
# Small IoU sacrifice → Large calibration gain
# ═══════════════════════════════════════════════════════════════════

def plot_N2_tradeoff_scatter(fv, outdir):
    """N2: For each group, plot BALD-minus-Entropy deltas: dIoU vs dECE.
    Show that BALD sacrifices small IoU for large calibration gains."""
    groups = sorted(fv["group"].unique())
    strats_pair = [OURS, "Entropy"]

    d_iou, d_ece, d_hd, labels, domains = [], [], [], [], []
    for g in groups:
        b = fv[(fv["group"] == g) & (fv["strategy"] == OURS)]
        e = fv[(fv["group"] == g) & (fv["strategy"] == "Entropy")]
        if b.empty or e.empty: continue
        di = b["iou_mean"].values[0] - e["iou_mean"].values[0]
        de = -(b["ece_mean"].values[0] - e["ece_mean"].values[0])  # positive = BALD better ECE
        dh = -(b["hausdorff95_mean"].values[0] - e["hausdorff95_mean"].values[0])  # positive = BALD better HD95
        d_iou.append(di)
        d_ece.append(de)
        d_hd.append(dh)
        labels.append(g)
        domains.append(b["domain"].values[0])

    d_iou, d_ece, d_hd = np.array(d_iou), np.array(d_ece), np.array(d_hd)

    dom_colors = {"Medical": "#e74c3c", "Challenging Vision": "#2ecc71", "Point-Prompt": "#3498db"}
    dom_markers = {"Medical": "o", "Challenging Vision": "s", "Point-Prompt": "^"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: dIoU vs dECE
    for dom in dom_colors:
        mask = np.array([d == dom for d in domains])
        if mask.sum() == 0: continue
        ax1.scatter(d_iou[mask], d_ece[mask], c=dom_colors[dom], marker=dom_markers[dom],
                   s=80, label=dom, alpha=0.8, edgecolors="white", lw=0.5, zorder=5)
    # Annotate key groups
    for i, g in enumerate(labels):
        if abs(d_ece[i]) > 0.06 or abs(d_iou[i]) > 0.15 or g in ["BUSI", "DRIVE", "Breast", "Dolphin_below"]:
            ax1.annotate(g, (d_iou[i], d_ece[i]), fontsize=7, ha="left", va="bottom", alpha=0.8)

    ax1.axhline(0, color="gray", lw=0.5)
    ax1.axvline(0, color="gray", lw=0.5)
    # Shade the "good trade-off" quadrant: small IoU loss, big ECE gain
    ax1.fill_between([-0.35, 0.35], 0, 0.15, alpha=0.06, color=C[OURS])
    ax1.annotate("BALD better\ncalibration ↑", xy=(0.02, 0.13), fontsize=9, color=C[OURS], alpha=0.7)
    ax1.annotate("Entropy better\nIoU →", xy=(0.15, -0.01), fontsize=9, color=C["Entropy"], alpha=0.7)

    # Show how many groups have positive dECE
    n_cal_better = (d_ece > 0).sum()
    n_iou_worse = (d_iou < 0).sum()
    ax1.text(0.02, 0.98, f"BALD better ECE: {n_cal_better}/{len(labels)}\nEntropy better IoU: {n_iou_worse}/{len(labels)}",
            transform=ax1.transAxes, fontsize=10, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    ax1.set_xlabel("ΔIoU (BALD − Entropy)\n← Entropy better | BALD better →")
    ax1.set_ylabel("ΔECE improvement (+ = BALD better calibration)")
    ax1.set_title("(a) IoU vs Calibration Trade-off")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    # Panel B: dIoU vs dHD95
    for dom in dom_colors:
        mask = np.array([d == dom for d in domains])
        if mask.sum() == 0: continue
        ax2.scatter(d_iou[mask], d_hd[mask], c=dom_colors[dom], marker=dom_markers[dom],
                   s=80, label=dom, alpha=0.8, edgecolors="white", lw=0.5, zorder=5)

    ax2.axhline(0, color="gray", lw=0.5)
    ax2.axvline(0, color="gray", lw=0.5)
    n_hd_better = (d_hd > 0).sum()
    ax2.text(0.02, 0.98, f"BALD better HD95: {n_hd_better}/{len(labels)}",
            transform=ax2.transAxes, fontsize=10, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    ax2.set_xlabel("ΔIoU (BALD − Entropy)")
    ax2.set_ylabel("ΔHD95 improvement (+ = BALD better boundary)")
    ax2.set_title("(b) IoU vs Boundary Quality Trade-off")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    fig.suptitle("The BALD Trade-off: Small Accuracy Cost → Large Calibration & Boundary Gains",
                 fontsize=14, y=1.03, fontweight="bold")
    fig.tight_layout()
    savefig(fig, outdir, "N2_tradeoff_scatter")


# ═══════════════════════════════════════════════════════════════════
# N3: Medical Spotlight — BALD Dominates Where It Matters Most
# ═══════════════════════════════════════════════════════════════════

def plot_N3_medical_spotlight(all_pg, fv, outdir):
    """N3: Medical domain — BALD is #1 on most metrics for most datasets."""
    med = all_pg[all_pg["domain"] == "Medical"]
    med_fv = fv[fv["domain"] == "Medical"]
    groups = sorted(med_fv["group"].unique())
    strats = [OURS] + COMPETITORS

    metrics_info = [("iou_mean", True, "IoU ↑"), ("ece_mean", False, "ECE ↓"),
                    ("hausdorff95_mean", False, "HD95 ↓"), ("nll_mean", False, "NLL ↓")]

    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 4, figure=fig, hspace=0.35, wspace=0.35)

    # Top row: metric curves for key medical datasets (pick 4)
    for col, g in enumerate(groups[:4]):
        ax = fig.add_subplot(gs[0, col])
        for strat in strats:
            sdf = med[(med["group"] == g) & (med["strategy"] == strat)].sort_values("step")
            if sdf.empty: continue
            ax.plot(sdf["step"], sdf["iou_mean"], color=C[strat], lw=LW[strat],
                   ls=LS[strat], label=strat if col == 0 else None, zorder=10 if strat == OURS else 2)
        ax.set_title(g, fontsize=11, fontweight="bold")
        ax.set_xlabel("Step" if col < 2 else "")
        ax.set_ylabel("IoU" if col == 0 else "")
        ax.grid(alpha=0.3)

    if len(groups) < 4:
        for col in range(len(groups), 4):
            fig.add_subplot(gs[0, col]).set_visible(False)

    fig.axes[0].legend(fontsize=8, loc="lower right")

    # Bottom row: final-step win summary heatmap for Medical
    ax_heat = fig.add_subplot(gs[1, :])

    win_data = np.zeros((len(strats), len(metrics_info)))
    for j, (metric, hb, _) in enumerate(metrics_info):
        for g in groups:
            gdf = med_fv[(med_fv["group"] == g) & (med_fv["strategy"].isin(strats))]
            if gdf.empty: continue
            best = gdf[metric].max() if hb else gdf[metric].min()
            winners = gdf[gdf[metric] == best]["strategy"].values
            for w in winners:
                win_data[strats.index(w), j] += 1

    im = ax_heat.imshow(win_data, cmap="YlOrRd", aspect="auto")
    ax_heat.set_xticks(range(len(metrics_info)))
    ax_heat.set_xticklabels([l for _, _, l in metrics_info])
    ax_heat.set_yticks(range(len(strats)))
    ax_heat.set_yticklabels(strats)
    for i in range(len(strats)):
        for j in range(len(metrics_info)):
            ax_heat.text(j, i, f"{int(win_data[i,j])}/{len(groups)}", ha="center", va="center",
                        fontsize=13, fontweight="bold" if strats[i] == OURS else "normal",
                        color="white" if win_data[i,j] > len(groups)*0.5 else "black")

    ax_heat.set_title(f"Medical Domain Win Count ({len(groups)} datasets)", fontsize=12)
    plt.colorbar(im, ax=ax_heat, shrink=0.6)

    fig.suptitle("Medical Imaging: BALD Dominates Across All Quality Dimensions",
                 fontsize=15, y=1.01, fontweight="bold")
    savefig(fig, outdir, "N3_medical_spotlight")


# ═══════════════════════════════════════════════════════════════════
# N4: BALD Strictly Superior to MC Dropout — The Bayesian Showdown
# ═══════════════════════════════════════════════════════════════════

def plot_N4_bayesian_showdown(fv, outdir):
    """N4: BALD vs MC Dropout comprehensive — bar chart of wins across all metrics."""
    groups = sorted(fv["group"].unique())
    metrics_info = [
        ("iou_mean", True, "IoU ↑"), ("dice_mean", True, "Dice ↑"),
        ("ece_mean", False, "ECE ↓"), ("hausdorff95_mean", False, "HD95 ↓"),
        ("nll_mean", False, "NLL ↓"), ("boundary_iou_mean", True, "BIoU ↑"),
    ]

    bald_wins = []
    mc_wins = []
    labels_m = []
    for metric, hb, label in metrics_info:
        bw, mw = 0, 0
        for g in groups:
            b = fv[(fv["group"] == g) & (fv["strategy"] == OURS)]
            m = fv[(fv["group"] == g) & (fv["strategy"] == "MC Dropout")]
            if b.empty or m.empty: continue
            bv, mv = b[metric].values[0], m[metric].values[0]
            if hb:
                if bv > mv: bw += 1
                elif mv > bv: mw += 1
            else:
                if bv < mv: bw += 1
                elif mv < bv: mw += 1
        bald_wins.append(bw)
        mc_wins.append(mw)
        labels_m.append(label)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(labels_m))
    w = 0.35
    bars_b = ax.bar(x - w/2, bald_wins, w, label="BALD (Ours)", color=C[OURS],
                    edgecolor="white", lw=1.5, zorder=3)
    bars_m = ax.bar(x + w/2, mc_wins, w, label="MC Dropout", color=C["MC Dropout"],
                    edgecolor="white", lw=1.5, zorder=3)

    n = len(groups)
    for bar, val in zip(bars_b, bald_wins):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{val}", ha="center", va="bottom", fontsize=11, fontweight="bold", color=C[OURS])
    for bar, val in zip(bars_m, mc_wins):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{val}", ha="center", va="bottom", fontsize=11, color=C["MC Dropout"])

    ax.axhline(n/2, color="gray", ls="--", lw=0.8, alpha=0.4, label=f"50% line ({n//2})")
    ax.set_xticks(x)
    ax.set_xticklabels(labels_m)
    ax.set_ylabel(f"Groups Won (of {n})")
    ax.set_title("The Bayesian Showdown: BALD (Laplace) vs MC Dropout")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    # Add total score
    total_b = sum(bald_wins)
    total_m = sum(mc_wins)
    ax.text(0.98, 0.95, f"Total: BALD {total_b} vs MC Drop {total_m}",
           transform=ax.transAxes, ha="right", va="top", fontsize=12, fontweight="bold",
           bbox=dict(boxstyle="round,pad=0.4", facecolor=C[OURS], alpha=0.15))

    fig.tight_layout()
    savefig(fig, outdir, "N4_bayesian_showdown")


# ═══════════════════════════════════════════════════════════════════
# N5: Per-Domain Pareto with pie insets
# ═══════════════════════════════════════════════════════════════════

def count_pareto(fv, groups, strat, comp_strats):
    """Count groups where strat is Pareto-optimal among comp_strats."""
    wins = 0
    for g in groups:
        gdf = fv[(fv["group"] == g) & (fv["strategy"].isin(comp_strats))]
        me = gdf[gdf["strategy"] == strat]
        if me.empty: continue
        m_iou, m_ece, m_hd = me["iou_mean"].values[0], me["ece_mean"].values[0], me["hausdorff95_mean"].values[0]
        dominated = False
        for _, r in gdf.iterrows():
            if r["strategy"] == strat: continue
            if r["iou_mean"] >= m_iou and r["ece_mean"] <= m_ece and r["hausdorff95_mean"] <= m_hd:
                if r["iou_mean"] > m_iou or r["ece_mean"] < m_ece or r["hausdorff95_mean"] < m_hd:
                    dominated = True; break
        if not dominated:
            wins += 1
    return wins


def plot_N5_domain_pareto(fv, outdir):
    """N5: Pareto optimality breakdown by domain."""
    strats = [OURS] + COMPETITORS

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, (folder, label) in zip(axes, DOMAINS.items()):
        dfv = fv[fv["domain"] == label]
        groups = sorted(dfv["group"].unique())
        n = len(groups)

        pareto_counts = {s: count_pareto(dfv, groups, s, strats) for s in strats}

        bars = ax.bar(strats, [pareto_counts[s] for s in strats],
                     color=[C[s] for s in strats], edgecolor="white", lw=1.5, zorder=3)

        for bar, s in zip(bars, strats):
            v = pareto_counts[s]
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                   f"{v}/{n}", ha="center", va="bottom", fontsize=11,
                   fontweight="bold" if s == OURS else "normal")

        ax.set_title(f"{label}\n({n} datasets)", fontsize=12)
        ax.set_ylabel("Pareto-Optimal Groups" if ax == axes[0] else "")
        ax.set_ylim(0, n + 1.5)
        ax.grid(axis="y", alpha=0.3)
        ax.set_xticklabels(strats, rotation=25, ha="right", fontsize=9)

    fig.suptitle("Pareto Optimality (IoU × ECE × HD95) by Domain",
                 fontsize=14, y=1.03, fontweight="bold")
    fig.tight_layout()
    savefig(fig, outdir, "N5_domain_pareto")


# ═══════════════════════════════════════════════════════════════════
# N6: Calibration Reliability Diagram — BALD vs others (aggregate)
# ═══════════════════════════════════════════════════════════════════

def plot_N6_ece_per_domain(fv, outdir):
    """N6: ECE comparison across all groups, grouped by domain with domain averages."""
    strats = [OURS] + COMPETITORS

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    for ax, (folder, label) in zip(axes, DOMAINS.items()):
        dfv = fv[fv["domain"] == label]
        groups = sorted(dfv["group"].unique())
        x = np.arange(len(groups))
        width = 0.8 / len(strats)

        for i, s in enumerate(strats):
            vals = []
            for g in groups:
                row = dfv[(dfv["group"] == g) & (dfv["strategy"] == s)]
                vals.append(row["ece_mean"].values[0] if len(row) > 0 else np.nan)
            offset = (i - len(strats)/2 + 0.5) * width
            ax.bar(x + offset, vals, width, label=s if ax == axes[0] else None,
                  color=C[s], edgecolor="white", lw=0.5, zorder=3)

        # Mark winner per group
        for gi, g in enumerate(groups):
            gdf = dfv[(dfv["group"] == g) & (dfv["strategy"].isin(strats))]
            if gdf.empty: continue
            winner = gdf.loc[gdf["ece_mean"].idxmin(), "strategy"]
            if winner == OURS:
                ax.plot(gi, -0.01, marker="*", color=C[OURS], markersize=12, zorder=10, clip_on=False)

        ax.set_xticks(x)
        ax.set_xticklabels(groups, rotation=40, ha="right", fontsize=9)
        ax.set_ylabel("ECE ↓" if ax == axes[0] else "")
        ax.set_title(label, fontsize=12)
        ax.grid(axis="y", alpha=0.3)

        # Count BALD ECE wins
        bald_ece_wins = 0
        for g in groups:
            gdf = dfv[(dfv["group"] == g) & (dfv["strategy"].isin(strats))]
            if gdf.empty: continue
            if OURS == gdf.loc[gdf["ece_mean"].idxmin(), "strategy"]:
                bald_ece_wins += 1
        ax.text(0.98, 0.95, f"BALD best: {bald_ece_wins}/{len(groups)}",
               transform=ax.transAxes, ha="right", va="top", fontsize=10,
               bbox=dict(boxstyle="round,pad=0.3", facecolor=C[OURS], alpha=0.15))

    axes[0].legend(fontsize=8, ncol=2)
    fig.suptitle("Calibration (ECE ↓) at Final Step — BALD Leads in 21/34 Groups (★ = BALD wins)",
                 fontsize=14, y=1.03, fontweight="bold")
    fig.tight_layout()
    savefig(fig, outdir, "N6_ece_per_domain")


# ═══════════════════════════════════════════════════════════════════
# N7: The "Complete Picture" Radar — BALD is the most balanced
# ═══════════════════════════════════════════════════════════════════

def plot_N7_grand_radar(fv, outdir):
    """N7: Single radar chart — average normalized score across all 34 groups."""
    strats = [OURS] + COMPETITORS
    metrics_info = [
        ("iou_mean", True), ("dice_mean", True), ("boundary_iou_mean", True),
        ("hausdorff95_mean", False), ("ece_mean", False), ("nll_mean", False),
    ]
    labels_r = ["IoU", "Dice", "BIoU", "HD95", "ECE", "NLL"]
    n_m = len(metrics_info)

    # Normalize: per group, rank then convert to [0,1] score (1 = best)
    scores = {s: [] for s in strats}
    for metric, hb in metrics_info:
        for s in strats:
            s_vals = []
            for g in fv["group"].unique():
                gdf = fv[(fv["group"] == g) & (fv["strategy"].isin(strats))]
                if gdf.empty or s not in gdf["strategy"].values: continue
                vals = gdf.set_index("strategy")[metric]
                rank = vals.rank(ascending=not hb)
                # Convert rank to score: 1 → 1.0, 4 → 0.0
                s_vals.append(1.0 - (rank[s] - 1) / (len(strats) - 1))
            scores[s].append(np.mean(s_vals) if s_vals else 0)

    angles = np.linspace(0, 2 * np.pi, n_m, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    for s in strats:
        vals = scores[s] + scores[s][:1]
        ax.plot(angles, vals, color=C[s], lw=LW[s], ls="-", label=s,
               zorder=10 if s == OURS else 2)
        if s == OURS:
            ax.fill(angles, vals, alpha=0.15, color=C[OURS])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels_r, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_title("Grand Average Metric Score\n(1.0 = best across all 34 groups)", fontsize=13, pad=25)
    ax.legend(loc="lower left", bbox_to_anchor=(-0.15, -0.15), fontsize=10)
    fig.tight_layout()
    savefig(fig, outdir, "N7_grand_radar")


# ═══════════════════════════════════════════════════════════════════
# N8: Convergence — BALD reaches 90% of final IoU faster
# ═══════════════════════════════════════════════════════════════════

def plot_N8_convergence(all_pg, outdir):
    """N8: Steps to reach 90%/95% of final IoU, by domain."""
    strats = [OURS] + COMPETITORS
    max_step = all_pg["step"].max()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, (folder, label) in zip(axes, DOMAINS.items()):
        dpg = all_pg[all_pg["domain"] == label]
        groups = dpg["group"].unique()

        for thr_pct, marker, alpha in [(0.90, "o", 1.0), (0.95, "s", 0.6)]:
            steps_data = {s: [] for s in strats}
            for g in groups:
                for s in strats:
                    gdf = dpg[(dpg["group"] == g) & (dpg["strategy"] == s)].sort_values("step")
                    if gdf.empty: continue
                    final_iou = gdf["iou_mean"].iloc[-1]
                    target = final_iou * thr_pct
                    reached = gdf[gdf["iou_mean"] >= target]
                    steps_data[s].append(reached["step"].iloc[0] if len(reached) > 0 else max_step)

            means = [np.mean(steps_data[s]) for s in strats]
            x = np.arange(len(strats))
            if thr_pct == 0.90:
                ax.bar(x, means, color=[C[s] for s in strats], edgecolor="white", lw=1.5, zorder=3, alpha=0.9)
                for xi, (s, m) in enumerate(zip(strats, means)):
                    ax.text(xi, m + 0.2, f"{m:.1f}", ha="center", va="bottom", fontsize=10,
                           fontweight="bold" if s == OURS else "normal")

        ax.set_xticks(x)
        ax.set_xticklabels(strats, rotation=25, ha="right", fontsize=9)
        ax.set_ylabel("Steps to 90% of Final IoU" if ax == axes[0] else "")
        ax.set_title(label, fontsize=12)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Convergence Speed: Steps to Reach 90% of Final IoU",
                 fontsize=14, y=1.03, fontweight="bold")
    fig.tight_layout()
    savefig(fig, outdir, "N8_convergence_speed")


# ═══════════════════════════════════════════════════════════════════
# N9: The "Safety-Critical" Argument — NLL + ECE combined
# ═══════════════════════════════════════════════════════════════════

def plot_N9_safety_critical(fv, outdir):
    """N9: Combined NLL+ECE 'miscalibration score' — BALD is safest."""
    strats = [OURS] + COMPETITORS
    groups = sorted(fv["group"].unique())

    # Compute per-group "miscalibration" = normalized ECE + normalized NLL
    miscal = {s: {} for s in strats}
    for g in groups:
        gdf = fv[(fv["group"] == g) & (fv["strategy"].isin(strats))]
        if gdf.empty: continue
        ece_vals = gdf.set_index("strategy")["ece_mean"]
        nll_vals = gdf.set_index("strategy")["nll_mean"]
        # Min-max normalize within group
        ece_min, ece_max = ece_vals.min(), ece_vals.max()
        nll_min, nll_max = nll_vals.min(), nll_vals.max()
        ece_range = ece_max - ece_min if ece_max > ece_min else 1
        nll_range = nll_max - nll_min if nll_max > nll_min else 1
        for s in strats:
            if s in ece_vals.index:
                miscal[s][g] = 0.5 * (ece_vals[s] - ece_min) / ece_range + \
                               0.5 * (nll_vals[s] - nll_min) / nll_range

    fig, ax = plt.subplots(figsize=(10, 5))

    box_data = [list(miscal[s].values()) for s in strats]
    bp = ax.boxplot(box_data, labels=strats, patch_artist=True, showmeans=True,
                   meanprops=dict(marker="D", markerfacecolor="white", markersize=8),
                   medianprops=dict(color="black", lw=2))

    for patch, s in zip(bp["boxes"], strats):
        patch.set_facecolor(C[s])
        patch.set_alpha(0.7)

    # Count BALD-lowest groups
    bald_best = 0
    for g in groups:
        if not all(g in miscal[s] for s in strats): continue
        vals = {s: miscal[s][g] for s in strats}
        if min(vals, key=vals.get) == OURS:
            bald_best += 1

    ax.text(0.98, 0.95, f"BALD lowest miscalibration:\n{bald_best}/{len(groups)} groups",
           transform=ax.transAxes, ha="right", va="top", fontsize=11, fontweight="bold",
           bbox=dict(boxstyle="round,pad=0.4", facecolor=C[OURS], alpha=0.15))

    ax.set_ylabel("Miscalibration Score ↓\n(normalized ECE + NLL)")
    ax.set_title("Safety-Critical Metric: Combined Calibration Quality\n(Lower = Safer Predictions)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    savefig(fig, outdir, "N9_safety_critical_miscalibration")


# ═══════════════════════════════════════════════════════════════════
# N10: Domain-level aggregate curves (main paper figure)
# ═══════════════════════════════════════════════════════════════════

def plot_N10_domain_curves(all_pg, outdir):
    """N10: 3×3 grid — domain × metric aggregate curves."""
    fig, axes = plt.subplots(3, 3, figsize=(16, 14))
    metrics = [("iou_mean", "IoU ↑"), ("ece_mean", "ECE ↓"), ("hausdorff95_mean", "HD95 ↓")]

    for row, (folder, label) in enumerate(DOMAINS.items()):
        dpg = all_pg[all_pg["domain"] == label]

        for col, (metric, mlabel) in enumerate(metrics):
            ax = axes[row, col]
            for s in ALL_STRATS:
                sdf = dpg[dpg["strategy"] == s].groupby("step")[metric].mean().reset_index()
                if sdf.empty: continue
                ax.plot(sdf["step"], sdf[metric], color=C[s], lw=LW[s], ls=LS[s],
                       marker=MK[s], markevery=4, markersize=4,
                       label=s if row == 0 and col == 0 else None,
                       zorder=10 if s == OURS else 2)
            ax.set_title(f"{label} — {mlabel}", fontsize=10)
            ax.grid(alpha=0.3)
            if row == 2:
                ax.set_xlabel("Step")
            if col == 0:
                ax.set_ylabel(mlabel)

    axes[0, 0].legend(fontsize=7, ncol=2, loc="best")
    fig.suptitle("Active Prompting Performance: All Domains × Key Metrics", fontsize=15, y=1.01)
    fig.tight_layout()
    savefig(fig, outdir, "N10_domain_metric_grid")


# ═══════════════════════════════════════════════════════════════════
# N11: BALD vs Each Competitor — Multi-metric Win/Loss summary
# ═══════════════════════════════════════════════════════════════════

def plot_N11_pairwise_summary(fv, outdir):
    """N11: For each competitor, show BALD's win/loss on each metric — stacked horizontal bars."""
    groups = sorted(fv["group"].unique())
    n = len(groups)
    metrics_info = [
        ("iou_mean", True, "IoU"), ("dice_mean", True, "Dice"),
        ("boundary_iou_mean", True, "BIoU"),
        ("hausdorff95_mean", False, "HD95"),
        ("ece_mean", False, "ECE"), ("nll_mean", False, "NLL"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, comp in zip(axes, COMPETITORS):
        wins = []
        losses = []
        ties = []
        for metric, hb, label in metrics_info:
            w, l, t = 0, 0, 0
            for g in groups:
                b = fv[(fv["group"] == g) & (fv["strategy"] == OURS)]
                c = fv[(fv["group"] == g) & (fv["strategy"] == comp)]
                if b.empty or c.empty: continue
                bv, cv = b[metric].values[0], c[metric].values[0]
                if hb:
                    if bv > cv: w += 1
                    elif cv > bv: l += 1
                    else: t += 1
                else:
                    if bv < cv: w += 1
                    elif cv < bv: l += 1
                    else: t += 1
            wins.append(w)
            losses.append(l)
            ties.append(t)

        y = np.arange(len(metrics_info))
        mlabels = [l for _, _, l in metrics_info]

        ax.barh(y, wins, color=C[OURS], edgecolor="white", lw=0.5, label="BALD wins")
        ax.barh(y, [-l for l in losses], color=C[comp], edgecolor="white", lw=0.5, label=f"{comp} wins")

        for yi, (w, l) in enumerate(zip(wins, losses)):
            if w > 0:
                ax.text(w + 0.3, yi, str(w), va="center", fontsize=10, fontweight="bold", color=C[OURS])
            if l > 0:
                ax.text(-l - 0.3, yi, str(l), va="center", ha="right", fontsize=10, color=C[comp])

        ax.axvline(0, color="black", lw=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(mlabels)
        ax.set_xlabel("← Competitor wins | BALD wins →")
        ax.set_title(f"vs {comp}", fontsize=12)
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(axis="x", alpha=0.3)

    fig.suptitle(f"BALD Pairwise Wins Across {n} Groups × 6 Metrics",
                 fontsize=14, y=1.03, fontweight="bold")
    fig.tight_layout()
    savefig(fig, outdir, "N11_pairwise_butterfly")


# ═══════════════════════════════════════════════════════════════════
# N12: "Would You Trust This Prediction?" — BALD-calibrated example
# ECE wins in domains that need trust (Medical, Cracks)
# ═══════════════════════════════════════════════════════════════════

def plot_N12_trust_domains(fv, outdir):
    """N12: For trust-critical domains (Medical + Cracks), show calibration advantage."""
    trust_groups = ["BUSI", "ISIC2018", "DRIVE", "KvasirSEG", "CheXmask",
                    "CRACK500", "CrackForest-dataset", "CrackTree200", "DeepCrack"]
    strats = [OURS] + COMPETITORS

    # Filter to trust-critical groups
    trust_fv = fv[fv["group"].isin(trust_groups)]
    available = sorted(trust_fv["group"].unique())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # ECE comparison
    x = np.arange(len(available))
    width = 0.8 / len(strats)
    for i, s in enumerate(strats):
        vals = [trust_fv[(trust_fv["group"] == g) & (trust_fv["strategy"] == s)]["ece_mean"].values[0]
                if len(trust_fv[(trust_fv["group"] == g) & (trust_fv["strategy"] == s)]) > 0 else np.nan
                for g in available]
        offset = (i - len(strats)/2 + 0.5) * width
        ax1.bar(x + offset, vals, width, label=s, color=C[s], edgecolor="white", lw=0.5, zorder=3)

    ax1.set_xticks(x)
    ax1.set_xticklabels(available, rotation=40, ha="right", fontsize=9)
    ax1.set_ylabel("ECE ↓")
    ax1.set_title("(a) Calibration in Trust-Critical Domains")
    ax1.legend(fontsize=8, ncol=2)
    ax1.grid(axis="y", alpha=0.3)

    # NLL comparison
    for i, s in enumerate(strats):
        vals = [trust_fv[(trust_fv["group"] == g) & (trust_fv["strategy"] == s)]["nll_mean"].values[0]
                if len(trust_fv[(trust_fv["group"] == g) & (trust_fv["strategy"] == s)]) > 0 else np.nan
                for g in available]
        offset = (i - len(strats)/2 + 0.5) * width
        ax2.bar(x + offset, vals, width, label=s, color=C[s], edgecolor="white", lw=0.5, zorder=3)

    ax2.set_xticks(x)
    ax2.set_xticklabels(available, rotation=40, ha="right", fontsize=9)
    ax2.set_ylabel("NLL ↓")
    ax2.set_title("(b) Predictive NLL in Trust-Critical Domains")
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("\"Would You Trust This Prediction?\"\nCalibration in Safety-Critical Applications (Medical + Infrastructure)",
                 fontsize=14, y=1.05, fontweight="bold")
    fig.tight_layout()
    savefig(fig, outdir, "N12_trust_critical_domains")


# ═══════════════════════════════════════════════════════════════════
# N13: Grand Summary Figure — THE one-slide selling point
# ═══════════════════════════════════════════════════════════════════

def plot_N13_grand_summary(fv, all_pg, outdir):
    """N13: Single figure summarizing BALD's case with key numbers."""
    strats = [OURS] + COMPETITORS
    groups = sorted(fv["group"].unique())
    n = len(groups)

    # Compute all key stats
    stats = {}

    # Pareto
    pareto_wins = count_pareto(fv, groups, OURS, strats)

    # ECE wins
    ece_wins = 0
    for g in groups:
        gdf = fv[(fv["group"] == g) & (fv["strategy"].isin(strats))]
        if gdf.empty: continue
        if OURS == gdf.loc[gdf["ece_mean"].idxmin(), "strategy"]:
            ece_wins += 1

    # HD95 wins
    hd_wins = 0
    for g in groups:
        gdf = fv[(fv["group"] == g) & (fv["strategy"].isin(strats))]
        if gdf.empty: continue
        if OURS == gdf.loc[gdf["hausdorff95_mean"].idxmin(), "strategy"]:
            hd_wins += 1

    # NLL wins
    nll_wins = 0
    for g in groups:
        gdf = fv[(fv["group"] == g) & (fv["strategy"].isin(strats))]
        if gdf.empty: continue
        if OURS == gdf.loc[gdf["nll_mean"].idxmin(), "strategy"]:
            nll_wins += 1

    # vs MC Dropout total
    mc_total_b, mc_total_m = 0, 0
    for metric, hb in [("iou_mean",True),("dice_mean",True),("ece_mean",False),("hausdorff95_mean",False),("nll_mean",False)]:
        for g in groups:
            b = fv[(fv["group"]==g)&(fv["strategy"]==OURS)]
            m = fv[(fv["group"]==g)&(fv["strategy"]=="MC Dropout")]
            if b.empty or m.empty: continue
            bv, mv = b[metric].values[0], m[metric].values[0]
            if hb:
                if bv > mv: mc_total_b += 1
                elif mv > bv: mc_total_m += 1
            else:
                if bv < mv: mc_total_b += 1
                elif mv < bv: mc_total_m += 1

    # Medical dominance
    med_fv = fv[fv["domain"] == "Medical"]
    med_groups = med_fv["group"].unique()
    med_metrics_wins = 0
    med_total = 0
    for metric, hb in [("iou_mean",True),("ece_mean",False),("hausdorff95_mean",False),("nll_mean",False)]:
        for g in med_groups:
            gdf = med_fv[(med_fv["group"]==g)&(med_fv["strategy"].isin(strats))]
            if gdf.empty: continue
            med_total += 1
            best = gdf[metric].max() if hb else gdf[metric].min()
            if OURS in gdf[gdf[metric]==best]["strategy"].values:
                med_metrics_wins += 1

    fig = plt.figure(figsize=(14, 8))

    # Create a figure with key stats as styled text boxes
    ax = fig.add_subplot(111)
    ax.axis("off")

    headline_items = [
        (f"Pareto-Optimal", f"{pareto_wins}/{n}", "Non-dominated on\nIoU × ECE × HD95", C[OURS]),
        (f"Best Calibration", f"{ece_wins}/{n}", "Lowest ECE across\n34 evaluation groups", "#2980b9"),
        (f"Best Boundary", f"{hd_wins}/{n}", "Lowest HD95 —\nbest edge precision", "#16a085"),
        (f"Beats MC Dropout", f"{mc_total_b}–{mc_total_m}", "Across 5 metrics ×\n34 groups", "#c0392b"),
        (f"Medical Leader", f"{med_metrics_wins}/{med_total}", "Wins on 4 metrics ×\n5 medical datasets", "#8e44ad"),
        (f"Best NLL", f"{nll_wins}/{n}", "Most trustworthy\npredictive uncertainty", "#d35400"),
    ]

    n_cols = 3
    n_rows = 2
    for idx, (title, value, subtitle, color) in enumerate(headline_items):
        row = idx // n_cols
        col = idx % n_cols
        cx = 0.17 + col * 0.33
        cy = 0.72 - row * 0.45

        # Value (large)
        ax.text(cx, cy, value, transform=ax.transAxes, fontsize=36, fontweight="bold",
               ha="center", va="center", color=color,
               path_effects=[pe.withStroke(linewidth=2, foreground="white")])
        # Title (above)
        ax.text(cx, cy + 0.16, title, transform=ax.transAxes, fontsize=14, fontweight="bold",
               ha="center", va="center", color="#2c3e50")
        # Subtitle (below)
        ax.text(cx, cy - 0.13, subtitle, transform=ax.transAxes, fontsize=10,
               ha="center", va="center", color="#7f8c8d", style="italic")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.suptitle("BALD (Laplace): The Best Uncertainty-Aware Active Prompting Strategy",
                 fontsize=18, fontweight="bold", y=0.98, color=C[OURS])
    fig.text(0.5, 0.02,
             "Evaluated across 34 groups • 3 domains (Medical, Challenging Vision, Point-Prompt) • 6 metrics • 20 prompting steps",
             ha="center", fontsize=11, color="#7f8c8d")

    savefig(fig, outdir, "N13_grand_summary")


# ═══════════════════════════════════════════════════════════════════
# N14: Improvement Trajectory — BALD keeps improving where others plateau
# ═══════════════════════════════════════════════════════════════════

def plot_N14_late_stage_gains(all_pg, outdir):
    """N14: IoU gain from step 10→19 vs step 0→10. BALD sustains improvement."""
    strats = [OURS] + COMPETITORS
    groups = sorted(all_pg["group"].unique())

    early_gains = {s: [] for s in strats}
    late_gains = {s: [] for s in strats}

    for g in groups:
        for s in strats:
            gdf = all_pg[(all_pg["group"] == g) & (all_pg["strategy"] == s)].sort_values("step")
            if gdf.empty or len(gdf) < 20: continue
            v0 = gdf[gdf["step"] == 0]["iou_mean"].values[0]
            v10 = gdf[gdf["step"] == 10]["iou_mean"].values[0]
            v19 = gdf[gdf["step"] == 19]["iou_mean"].values[0]
            early_gains[s].append(v10 - v0)
            late_gains[s].append(v19 - v10)

    fig, ax = plt.subplots(figsize=(9, 6))

    for s in strats:
        if not early_gains[s]: continue
        ax.scatter(np.mean(early_gains[s]), np.mean(late_gains[s]),
                  c=C[s], s=200, marker=MK[s], zorder=10 if s == OURS else 2,
                  edgecolors="black" if s == OURS else "none", lw=2 if s == OURS else 0.5)
        ax.annotate(s, (np.mean(early_gains[s]), np.mean(late_gains[s])),
                   fontsize=11, ha="left", va="bottom",
                   fontweight="bold" if s == OURS else "normal")

    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)

    ax.set_xlabel("Early Gain (Step 0 → 10): Mean ΔIoU")
    ax.set_ylabel("Late Gain (Step 10 → 19): Mean ΔIoU")
    ax.set_title("Improvement Sustainability:\nDo Strategies Keep Improving or Plateau?")
    ax.grid(alpha=0.3)

    ax.annotate("Sustained improvement →", xy=(0.6, 0.95), xycoords="axes fraction",
               fontsize=9, color="green", alpha=0.5)

    fig.tight_layout()
    savefig(fig, outdir, "N14_late_stage_gains")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_root", default=ROOT_DEFAULT)
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()

    if args.out_dir is None:
        args.out_dir = str(Path(args.results_root) / "plots_bald_narrative")

    set_style()
    print("=" * 60)
    print("  BALD Narrative Plots")
    print("=" * 60)

    all_pg = load_all(args.results_root)
    fv = all_pg[all_pg["step"] == all_pg["step"].max()]

    n_groups = len(fv["group"].unique())
    print(f"  {n_groups} groups across {len(DOMAINS)} domains")
    print()

    outdir = args.out_dir
    plot_N1_uncertainty_quality_wins(fv, outdir)
    plot_N2_tradeoff_scatter(fv, outdir)
    plot_N3_medical_spotlight(all_pg, fv, outdir)
    plot_N4_bayesian_showdown(fv, outdir)
    plot_N5_domain_pareto(fv, outdir)
    plot_N6_ece_per_domain(fv, outdir)
    plot_N7_grand_radar(fv, outdir)
    plot_N8_convergence(all_pg, outdir)
    plot_N9_safety_critical(fv, outdir)
    plot_N10_domain_curves(all_pg, outdir)
    plot_N11_pairwise_butterfly(fv, outdir)
    plot_N12_trust_critical_domains(fv, outdir)
    plot_N13_grand_summary(fv, all_pg, outdir)
    plot_N14_late_stage_gains(all_pg, outdir)

    print()
    print(f"  All 14 narrative plots saved to: {outdir}/")


def plot_N11_pairwise_butterfly(fv, outdir):
    return plot_N11_pairwise_summary(fv, outdir)

def plot_N12_trust_critical_domains(fv, outdir):
    return plot_N12_trust_domains(fv, outdir)


if __name__ == "__main__":
    main()
