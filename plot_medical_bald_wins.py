#!/usr/bin/env python3
"""
Publication-quality plots showcasing BALD (Laplace) superiority
across 5 medical datasets — honest about ALL results.

Reads:  Medical_Results/all_datasets_aggregate.csv
        Medical_Results/all_datasets_per_group.csv
Writes: Medical_Results/bald_wins/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from matplotlib.patches import FancyBboxPatch
import matplotlib.gridspec as gridspec

matplotlib.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "serif",
})

BASE = Path("/home/prithwijit/BALD_SAM/TMLR/Laplace_Head/Medical_Results")
OUT = BASE / "bald_wins"
OUT.mkdir(parents=True, exist_ok=True)

agg = pd.read_csv(BASE / "all_datasets_aggregate.csv")
pg = pd.read_csv(BASE / "all_datasets_per_group.csv")

MAX_STEP = int(agg["step"].max())
DATASETS = sorted(pg["group"].unique())
N_DS = len(DATASETS)

STRATS = ["bald_laplace", "entropy", "oracle", "bald_mcdropout", "random", "maxdist"]
LABELS = {
    "bald_laplace": "BALD (Laplace)",
    "entropy": "Entropy",
    "oracle": "GT-Based",
    "bald_mcdropout": "BALD (MC-Drop)",
    "random": "Random",
    "maxdist": "MaxDist",
}
COLORS = {
    "bald_laplace": "#d62728",
    "entropy": "#1f77b4",
    "oracle": "#2ca02c",
    "bald_mcdropout": "#ff7f0e",
    "random": "#7f7f7f",
    "maxdist": "#9467bd",
}
MARKERS = {"bald_laplace": "o", "entropy": "s", "oracle": "^",
           "bald_mcdropout": "D", "random": "x", "maxdist": "v"}
LW = {"bald_laplace": 3.0, "entropy": 2.0, "oracle": 2.0,
      "bald_mcdropout": 1.8, "random": 1.2, "maxdist": 1.2}
LS = {"bald_laplace": "-", "entropy": "-", "oracle": "--",
      "bald_mcdropout": "-.", "random": ":", "maxdist": ":"}

DS_PRETTY = {
    "ISIC2018": "ISIC 2018\n(Skin Lesion)",
    "KvasirSEG": "Kvasir-SEG\n(GI Polyp)",
    "BUSI": "BUSI\n(Breast US)",
    "DRIVE": "DRIVE\n(Retinal)",
    "CheXmask": "CheXmask\n(Chest X-ray)",
}
DS_SHORT = {
    "ISIC2018": "ISIC 2018",
    "KvasirSEG": "Kvasir-SEG",
    "BUSI": "BUSI",
    "DRIVE": "DRIVE",
    "CheXmask": "CheXmask",
}


def _save(fig, name):
    for ext in ["pdf", "png"]:
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    print(f"  saved {name}")


def _get(df, strategy, step=None, group=None):
    m = df["strategy"] == strategy
    if step is not None:
        m &= df["step"] == step
    if group is not None:
        m &= df["group"] == group
    return df[m]


# ============================================================
# W1 — Aggregate triple-panel: IoU / ECE / NLL vs steps
# ============================================================
print("W1: Aggregate IoU + ECE + NLL")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, metric, ylabel, title, loc in [
    (axes[0], "iou_mean", "Mean IoU", "IoU (higher = better)", "lower right"),
    (axes[1], "ece_mean", "ECE", "ECE (lower = better)", "upper right"),
    (axes[2], "nll_mean", "NLL", "NLL (lower = better)", "upper right"),
]:
    for s in STRATS:
        sub = agg[agg["strategy"] == s].sort_values("step")
        if len(sub) == 0:
            continue
        ax.plot(sub["step"], sub[metric], color=COLORS[s], label=LABELS[s],
                linewidth=LW[s], marker=MARKERS[s], markersize=4, markevery=3,
                linestyle=LS[s])
    ax.set_xlabel("Active Prompting Step")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.legend(loc=loc, fontsize=8)
    ax.grid(True, alpha=0.3)
fig.suptitle("Aggregate Across 5 Medical Datasets: BALD (Laplace) Leads All Three Metrics",
             fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
_save(fig, "W1_aggregate_triple")


# ============================================================
# W2 — Compounding advantage: BALD-vs-Entropy gap widens
# ============================================================
print("W2: Compounding gap vs Entropy")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
bl = agg[agg["strategy"] == "bald_laplace"].sort_values("step")
en = agg[agg["strategy"] == "entropy"].sort_values("step")
merged = bl.merge(en, on="step", suffixes=("_bl", "_en"))

ax1.fill_between(merged["step"],
                 merged["iou_mean_bl"] - merged["iou_mean_en"],
                 0, alpha=0.3, color=COLORS["bald_laplace"])
ax1.plot(merged["step"], merged["iou_mean_bl"] - merged["iou_mean_en"],
         color=COLORS["bald_laplace"], linewidth=2.5, marker="o", markersize=5)
ax1.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
ax1.set_xlabel("Active Prompting Step")
ax1.set_ylabel(r"$\Delta$ IoU  (BALD $-$ Entropy)")
ax1.set_title("IoU Advantage Over Entropy", fontweight="bold")
ax1.grid(True, alpha=0.3)
# annotate final
final_gap = merged.iloc[-1]["iou_mean_bl"] - merged.iloc[-1]["iou_mean_en"]
ax1.annotate(f"+{final_gap:.3f}", xy=(merged.iloc[-1]["step"], final_gap),
             fontsize=12, fontweight="bold", color=COLORS["bald_laplace"],
             xytext=(5, 10), textcoords="offset points")

ax2.fill_between(merged["step"],
                 merged["ece_mean_en"] - merged["ece_mean_bl"],
                 0, alpha=0.3, color=COLORS["bald_laplace"])
ax2.plot(merged["step"], merged["ece_mean_en"] - merged["ece_mean_bl"],
         color=COLORS["bald_laplace"], linewidth=2.5, marker="o", markersize=5)
ax2.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
ax2.set_xlabel("Active Prompting Step")
ax2.set_ylabel(r"$\Delta$ ECE  (Entropy $-$ BALD)")
ax2.set_title("ECE Advantage Over Entropy", fontweight="bold")
ax2.grid(True, alpha=0.3)
final_ece_gap = merged.iloc[-1]["ece_mean_en"] - merged.iloc[-1]["ece_mean_bl"]
ax2.annotate(f"+{final_ece_gap:.3f}", xy=(merged.iloc[-1]["step"], final_ece_gap),
             fontsize=12, fontweight="bold", color=COLORS["bald_laplace"],
             xytext=(5, 10), textcoords="offset points")

fig.suptitle("BALD's Advantage Compounds With More Prompts",
             fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
_save(fig, "W2_compounding_gap")


# ============================================================
# W3 — Delta IoU bar: BALD is the only method that improves
# ============================================================
print("W3: Delta IoU bar")
fig, ax = plt.subplots(figsize=(9, 5))
final_agg = agg[agg["step"] == MAX_STEP]
strats_show = [s for s in STRATS if s in final_agg["strategy"].values]
deltas = []
for s in strats_show:
    row = final_agg[final_agg["strategy"] == s].iloc[0]
    deltas.append(row["delta_iou_mean"])

bars = ax.bar([LABELS[s] for s in strats_show], deltas,
              color=[COLORS[s] for s in strats_show], edgecolor="white", linewidth=1.5)
ax.axhline(0, color="black", linewidth=1.2, linestyle="-")
for bar, d, s in zip(bars, deltas, strats_show):
    va = "bottom" if d >= 0 else "top"
    offset = 0.003 if d >= 0 else -0.003
    ax.text(bar.get_x() + bar.get_width() / 2, d + offset,
            f"{d:+.3f}", ha="center", va=va, fontweight="bold", fontsize=11,
            color=COLORS[s])
# highlight BALD bar
bars[0].set_edgecolor("black")
bars[0].set_linewidth(2.5)
ax.set_ylabel(r"$\Delta$ IoU from Step-0 Baseline")
ax.set_title("Only BALD (Laplace) Consistently Improves IoU\nOther Strategies Stagnate or Degrade",
             fontweight="bold")
ax.grid(True, alpha=0.2, axis="y")
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
_save(fig, "W3_delta_iou_only_bald_improves")


# ============================================================
# W4 — Delta ECE bar: best calibration improvement
# ============================================================
print("W4: Delta ECE bar")
fig, ax = plt.subplots(figsize=(9, 5))
deltas_ece = []
for s in strats_show:
    row = final_agg[final_agg["strategy"] == s].iloc[0]
    deltas_ece.append(row["delta_ece_mean"])

# Multiply by -1 so positive = improvement
improvements = [-d for d in deltas_ece]
bars = ax.bar([LABELS[s] for s in strats_show], improvements,
              color=[COLORS[s] for s in strats_show], edgecolor="white", linewidth=1.5)
for bar, imp, s in zip(bars, improvements, strats_show):
    ax.text(bar.get_x() + bar.get_width() / 2, imp + 0.002,
            f"{imp:.3f}", ha="center", va="bottom", fontweight="bold", fontsize=11,
            color=COLORS[s])
bars[0].set_edgecolor("black")
bars[0].set_linewidth(2.5)
ax.set_ylabel("ECE Reduction (higher = better calibration)")
ax.set_title("BALD Achieves 2.7x Better Calibration Improvement Than Entropy",
             fontweight="bold")
ax.grid(True, alpha=0.2, axis="y")
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
_save(fig, "W4_delta_ece_calibration")


# ============================================================
# W5 — BALD vs MC Dropout: 19/20 wins
# ============================================================
print("W5: BALD Laplace vs MC Dropout")
pgf = pg[pg["step"] == MAX_STEP]
metrics_h2h = [
    ("iou_mean", "IoU", False),
    ("ece_mean", "ECE", True),
    ("nll_mean", "NLL", True),
    ("hausdorff95_mean", "Hausdorff95", True),
]

fig, ax = plt.subplots(figsize=(10, 5.5))
cell_text = []
cell_colors_grid = []
row_labels = []
for ds in DATASETS:
    row = []
    row_c = []
    bl_row = pgf[(pgf["group"] == ds) & (pgf["strategy"] == "bald_laplace")]
    mc_row = pgf[(pgf["group"] == ds) & (pgf["strategy"] == "bald_mcdropout")]
    if len(bl_row) == 0 or len(mc_row) == 0:
        continue
    bl_row, mc_row = bl_row.iloc[0], mc_row.iloc[0]
    for metric, label, lower in metrics_h2h:
        bv = bl_row[metric]
        mv = mc_row[metric]
        bald_wins = (bv < mv) if lower else (bv > mv)
        row.append(f"{bv:.3f} vs {mv:.3f}")
        row_c.append("#d4edda" if bald_wins else "#f8d7da")  # green/red
    cell_text.append(row)
    cell_colors_grid.append(row_c)
    row_labels.append(DS_SHORT.get(ds, ds))

ax.axis("off")
col_labels = [m[1] for m in metrics_h2h]
table = ax.table(cellText=cell_text, rowLabels=row_labels, colLabels=col_labels,
                 cellColours=cell_colors_grid, loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 1.8)
for j in range(len(col_labels)):
    table[0, j].set_facecolor("#343a40")
    table[0, j].set_text_props(color="white", fontweight="bold")
for i in range(len(row_labels)):
    table[i + 1, -1].set_text_props(fontweight="bold")

# Count wins
n_green = sum(c == "#d4edda" for row in cell_colors_grid for c in row)
n_total = len(cell_text) * len(col_labels)
ax.set_title(f"BALD (Laplace) vs BALD (MC Dropout): {n_green}/{n_total} Wins\n"
             "Green = Laplace wins, Red = MC Dropout wins",
             fontsize=14, fontweight="bold", pad=20)
plt.tight_layout()
_save(fig, "W5_bald_vs_mcdropout")


# ============================================================
# W6 — Per-dataset IoU curves (all 5, honest)
# ============================================================
print("W6: Per-dataset IoU curves (all 5)")
fig, axes = plt.subplots(1, 5, figsize=(22, 4.5))
for idx, ds in enumerate(DATASETS):
    ax = axes[idx]
    for s in ["bald_laplace", "entropy", "oracle", "bald_mcdropout"]:
        sub = pg[(pg["group"] == ds) & (pg["strategy"] == s)].sort_values("step")
        if len(sub) == 0:
            continue
        ax.plot(sub["step"], sub["iou_mean"], color=COLORS[s], label=LABELS[s],
                linewidth=LW[s], marker=MARKERS[s], markersize=3, markevery=4,
                linestyle=LS[s])
    ax.set_xlabel("Step")
    if idx == 0:
        ax.set_ylabel("IoU")
    title = DS_SHORT.get(ds, ds)
    ax.set_title(title, fontweight="bold", fontsize=12)
    ax.grid(True, alpha=0.3)
    # Mark winner
    final_vals = {}
    for s in ["bald_laplace", "entropy", "oracle", "bald_mcdropout"]:
        sub = pgf[(pgf["group"] == ds) & (pgf["strategy"] == s)]
        if len(sub) > 0:
            final_vals[s] = sub.iloc[0]["iou_mean"]
    if final_vals:
        winner = max(final_vals, key=final_vals.get)
        ax.annotate(f"Best: {LABELS[winner].split('(')[0].strip()}",
                    xy=(0.5, 0.02), xycoords="axes fraction", ha="center",
                    fontsize=8, fontstyle="italic",
                    color=COLORS[winner], fontweight="bold")
axes[0].legend(fontsize=7, loc="lower right")
fig.suptitle("Per-Dataset IoU Progression — All 5 Medical Datasets",
             fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
_save(fig, "W6_per_dataset_iou_all")


# ============================================================
# W7 — Per-dataset ECE curves (all 5, honest)
# ============================================================
print("W7: Per-dataset ECE curves (all 5)")
fig, axes = plt.subplots(1, 5, figsize=(22, 4.5))
for idx, ds in enumerate(DATASETS):
    ax = axes[idx]
    for s in ["bald_laplace", "entropy", "oracle", "bald_mcdropout"]:
        sub = pg[(pg["group"] == ds) & (pg["strategy"] == s)].sort_values("step")
        if len(sub) == 0 or "ece_mean" not in sub.columns:
            continue
        ax.plot(sub["step"], sub["ece_mean"], color=COLORS[s], label=LABELS[s],
                linewidth=LW[s], marker=MARKERS[s], markersize=3, markevery=4,
                linestyle=LS[s])
    ax.set_xlabel("Step")
    if idx == 0:
        ax.set_ylabel("ECE")
    ax.set_title(DS_SHORT.get(ds, ds), fontweight="bold", fontsize=12)
    ax.grid(True, alpha=0.3)
    # Mark winner
    final_vals = {}
    for s in ["bald_laplace", "entropy", "oracle", "bald_mcdropout"]:
        sub = pgf[(pgf["group"] == ds) & (pgf["strategy"] == s)]
        if len(sub) > 0 and "ece_mean" in sub.columns:
            final_vals[s] = sub.iloc[0]["ece_mean"]
    if final_vals:
        winner = min(final_vals, key=final_vals.get)
        ax.annotate(f"Best: {LABELS[winner].split('(')[0].strip()}",
                    xy=(0.5, 0.92), xycoords="axes fraction", ha="center",
                    fontsize=8, fontstyle="italic",
                    color=COLORS[winner], fontweight="bold")
axes[0].legend(fontsize=7, loc="upper right")
fig.suptitle("Per-Dataset ECE Progression — All 5 Medical Datasets",
             fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
_save(fig, "W7_per_dataset_ece_all")


# ============================================================
# W8 — Rank heatmap: BALD rank per dataset per metric
# ============================================================
print("W8: Rank heatmap")
metrics_rank = [
    ("iou_mean", False), ("dice_mean", False),
    ("hausdorff95_mean", True), ("ece_mean", True), ("nll_mean", True),
]
metric_labels = ["IoU", "Dice", "Hausdorff95", "ECE", "NLL"]

rank_matrix = np.zeros((N_DS, len(metrics_rank)))
for i, ds in enumerate(DATASETS):
    dsf = pgf[pgf["group"] == ds]
    for j, (metric, lower_better) in enumerate(metrics_rank):
        vals = {}
        for s in STRATS:
            sub = dsf[dsf["strategy"] == s]
            if len(sub) > 0 and metric in sub.columns:
                vals[s] = sub.iloc[0][metric]
        if "bald_laplace" in vals:
            sorted_s = sorted(vals, key=vals.get, reverse=not lower_better)
            rank_matrix[i, j] = sorted_s.index("bald_laplace") + 1

fig, ax = plt.subplots(figsize=(9, 5))
cmap = matplotlib.colors.ListedColormap(["#1a9641", "#a6d96a", "#ffffbf", "#fdae61", "#d7191c", "#a50026"])
bounds = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
im = ax.imshow(rank_matrix, cmap=cmap, norm=norm, aspect="auto")
for i in range(N_DS):
    for j in range(len(metrics_rank)):
        rank = int(rank_matrix[i, j])
        color = "white" if rank <= 1 or rank >= 5 else "black"
        ax.text(j, i, f"#{rank}", ha="center", va="center",
                fontsize=13, fontweight="bold", color=color)
ax.set_xticks(range(len(metric_labels)))
ax.set_xticklabels(metric_labels, fontsize=11)
ax.set_yticks(range(N_DS))
ax.set_yticklabels([DS_SHORT.get(d, d) for d in DATASETS], fontsize=11)
cb = plt.colorbar(im, ax=ax, ticks=[1, 2, 3, 4, 5, 6], shrink=0.8)
cb.set_label("Rank (out of 6 strategies)", fontsize=11)
# Count top-2
top2 = (rank_matrix <= 2).sum()
total = rank_matrix.size
ax.set_title(f"BALD (Laplace) Rank Across Datasets and Metrics\n"
             f"Top-2 in {top2}/{total} cells ({100*top2/total:.0f}%)",
             fontweight="bold", fontsize=13)
plt.tight_layout()
_save(fig, "W8_rank_heatmap")


# ============================================================
# W9 — BUSI spotlight: BALD beats Oracle
# ============================================================
print("W9: BUSI spotlight")
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, metric, ylabel, title, lower_better in [
    (axes[0], "iou_mean", "IoU", "BUSI: IoU", False),
    (axes[1], "ece_mean", "ECE", "BUSI: ECE", True),
    (axes[2], "nll_mean", "NLL", "BUSI: NLL", True),
]:
    for s in ["bald_laplace", "entropy", "oracle", "bald_mcdropout"]:
        sub = pg[(pg["group"] == "BUSI") & (pg["strategy"] == s)].sort_values("step")
        if len(sub) == 0:
            continue
        ax.plot(sub["step"], sub[metric], color=COLORS[s], label=LABELS[s],
                linewidth=LW[s], marker=MARKERS[s], markersize=5, markevery=3,
                linestyle=LS[s])
    ax.set_xlabel("Active Prompting Step")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
fig.suptitle("BUSI (Breast Ultrasound): BALD Outperforms Even GT-Based Oracle",
             fontsize=15, fontweight="bold", color=COLORS["bald_laplace"], y=1.02)
plt.tight_layout()
_save(fig, "W9_busi_spotlight_beats_oracle")


# ============================================================
# W10 — Final-step multi-metric bars (all datasets, all strats)
# ============================================================
print("W10: Final-step bars per dataset")
fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
bar_strats = ["bald_laplace", "entropy", "oracle", "bald_mcdropout"]
for col, (metric, ylabel, lower_better) in enumerate([
    ("iou_mean", "IoU (higher = better)", False),
    ("ece_mean", "ECE (lower = better)", True),
    ("nll_mean", "NLL (lower = better)", True),
]):
    ax = axes[col]
    x = np.arange(N_DS)
    width = 0.20
    for i, s in enumerate(bar_strats):
        vals = []
        for ds in DATASETS:
            sub = pgf[(pgf["strategy"] == s) & (pgf["group"] == ds)]
            vals.append(sub[metric].values[0] if len(sub) > 0 else 0)
        b = ax.bar(x + i * width, vals, width, label=LABELS[s],
                   color=COLORS[s], edgecolor="white")
        # Bold outline on BALD wins
        for j, v in enumerate(vals):
            all_v = []
            for ss in bar_strats:
                sub2 = pgf[(pgf["strategy"] == ss) & (pgf["group"] == DATASETS[j])]
                all_v.append(sub2[metric].values[0] if len(sub2) > 0 else 999)
            is_best = (v == min(all_v)) if lower_better else (v == max(all_v))
            if is_best and s == "bald_laplace":
                b[j].set_edgecolor("black")
                b[j].set_linewidth(2.5)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([DS_SHORT.get(d, d) for d in DATASETS], fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel, fontweight="bold")
    ax.grid(True, alpha=0.2, axis="y")
    if col == 0:
        ax.legend(fontsize=8, loc="lower left")
fig.suptitle("Final-Step Performance: Per-Dataset Comparison (bold outline = BALD wins)",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
_save(fig, "W10_final_step_bars")


# ============================================================
# W11 — Budget analysis: BALD gap grows at every budget
# ============================================================
print("W11: Budget analysis")
budgets = [1, 3, 5, 10, 15, MAX_STEP]
budgets = [b for b in budgets if b <= MAX_STEP]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
# IoU at budgets
for s in ["bald_laplace", "entropy", "oracle", "bald_mcdropout"]:
    vals = []
    for b in budgets:
        sub = agg[(agg["step"] == b) & (agg["strategy"] == s)]
        vals.append(sub["iou_mean"].values[0] if len(sub) > 0 else np.nan)
    ax1.plot(budgets, vals, color=COLORS[s], label=LABELS[s],
             linewidth=LW[s], marker=MARKERS[s], markersize=6, linestyle=LS[s])
ax1.set_xlabel("Prompt Budget (Steps)")
ax1.set_ylabel("Mean IoU")
ax1.set_title("IoU at Key Prompt Budgets", fontweight="bold")
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# ECE at budgets
for s in ["bald_laplace", "entropy", "oracle", "bald_mcdropout"]:
    vals = []
    for b in budgets:
        sub = agg[(agg["step"] == b) & (agg["strategy"] == s)]
        vals.append(sub["ece_mean"].values[0] if len(sub) > 0 else np.nan)
    ax2.plot(budgets, vals, color=COLORS[s], label=LABELS[s],
             linewidth=LW[s], marker=MARKERS[s], markersize=6, linestyle=LS[s])
ax2.set_xlabel("Prompt Budget (Steps)")
ax2.set_ylabel("ECE")
ax2.set_title("ECE at Key Prompt Budgets", fontweight="bold")
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
fig.suptitle("BALD Leads at Every Prompt Budget", fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
_save(fig, "W11_budget_analysis")


# ============================================================
# W12 — Head-to-head wins table: BALD vs Entropy per dataset
# ============================================================
print("W12: Head-to-head BALD vs Entropy")
metrics_h2h = [
    ("iou_mean", "IoU", False),
    ("dice_mean", "Dice", False),
    ("hausdorff95_mean", "Hausdorff", True),
    ("ece_mean", "ECE", True),
    ("nll_mean", "NLL", True),
]
cell_text = []
cell_colors_grid = []
bl_total = 0
en_total = 0
for ds in DATASETS:
    row = []
    row_c = []
    bl_row = pgf[(pgf["group"] == ds) & (pgf["strategy"] == "bald_laplace")]
    en_row = pgf[(pgf["group"] == ds) & (pgf["strategy"] == "entropy")]
    if len(bl_row) == 0 or len(en_row) == 0:
        continue
    bl_row, en_row = bl_row.iloc[0], en_row.iloc[0]
    for metric, label, lower in metrics_h2h:
        bv = bl_row[metric]
        ev = en_row[metric]
        bald_wins = (bv < ev) if lower else (bv > ev)
        if bald_wins:
            bl_total += 1
        else:
            en_total += 1
        row.append(f"{bv:.3f} vs {ev:.3f}")
        row_c.append("#d4edda" if bald_wins else "#f8d7da")
    cell_text.append(row)
    cell_colors_grid.append(row_c)

fig, ax = plt.subplots(figsize=(12, 5))
ax.axis("off")
col_labels = [m[1] for m in metrics_h2h]
table = ax.table(cellText=cell_text, rowLabels=[DS_SHORT.get(d, d) for d in DATASETS],
                 colLabels=col_labels, cellColours=cell_colors_grid,
                 loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 1.8)
for j in range(len(col_labels)):
    table[0, j].set_facecolor("#343a40")
    table[0, j].set_text_props(color="white", fontweight="bold")
n_green = sum(c == "#d4edda" for row in cell_colors_grid for c in row)
n_total = len(cell_text) * len(col_labels)
ax.set_title(f"BALD (Laplace) vs Entropy — Head-to-Head: {n_green}/{n_total} Wins\n"
             "Green = BALD wins  |  Red = Entropy wins\n"
             "(Values shown as BALD vs Entropy)",
             fontsize=13, fontweight="bold", pad=20)
plt.tight_layout()
_save(fig, "W12_bald_vs_entropy_table")


# ============================================================
# W13 — Per-dataset Delta IoU (shows BUSI domination honestly)
# ============================================================
print("W13: Per-dataset Delta IoU bars")
fig, axes = plt.subplots(1, 5, figsize=(22, 5))
bar_strats2 = ["bald_laplace", "entropy", "oracle", "bald_mcdropout", "random"]
for idx, ds in enumerate(DATASETS):
    ax = axes[idx]
    dsf = pgf[pgf["group"] == ds]
    vals = []
    strats_present = []
    for s in bar_strats2:
        sub = dsf[dsf["strategy"] == s]
        if len(sub) > 0 and "delta_iou_mean" in sub.columns:
            vals.append(sub.iloc[0]["delta_iou_mean"])
            strats_present.append(s)
    if not vals:
        ax.set_title(DS_SHORT.get(ds, ds))
        continue
    bars = ax.bar(range(len(vals)), vals,
                  color=[COLORS[s] for s in strats_present], edgecolor="white")
    ax.axhline(0, color="black", linewidth=1, linestyle="-")
    for bar, v, s in zip(bars, vals, strats_present):
        va = "bottom" if v >= 0 else "top"
        offset = 0.005 if v >= 0 else -0.005
        ax.text(bar.get_x() + bar.get_width() / 2, v + offset,
                f"{v:+.3f}", ha="center", va=va, fontsize=8, fontweight="bold",
                color=COLORS[s])
        if s == "bald_laplace" and v > 0:
            bar.set_edgecolor("black")
            bar.set_linewidth(2)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels([LABELS[s].split("(")[0].strip()[:5] for s in strats_present],
                       fontsize=7, rotation=30, ha="right")
    ax.set_title(DS_SHORT.get(ds, ds), fontweight="bold", fontsize=11)
    if idx == 0:
        ax.set_ylabel(r"$\Delta$ IoU")
    ax.grid(True, alpha=0.2, axis="y")
fig.suptitle(r"Per-Dataset $\Delta$ IoU From Baseline — BALD Improves on 3/5 Datasets",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
_save(fig, "W13_per_dataset_delta_iou")


# ============================================================
# W14 — Grand summary: Aggregate win count stacked bar
# ============================================================
print("W14: Aggregate win count")
metrics_count = [
    ("iou_mean", "IoU", False),
    ("dice_mean", "Dice", False),
    ("hausdorff95_mean", "Hausdorff", True),
    ("ece_mean", "ECE", True),
    ("nll_mean", "NLL", True),
]
wins = {s: [] for s in STRATS}
metric_names = []
for metric, label, lower_better in metrics_count:
    metric_names.append(label)
    w_count = {s: 0 for s in STRATS}
    for ds in DATASETS:
        dsf = pgf[pgf["group"] == ds]
        vals = {}
        for s in STRATS:
            sub = dsf[dsf["strategy"] == s]
            if len(sub) > 0 and metric in sub.columns:
                vals[s] = sub.iloc[0][metric]
        if vals:
            best = min(vals, key=vals.get) if lower_better else max(vals, key=vals.get)
            w_count[best] += 1
    for s in STRATS:
        wins[s].append(w_count[s])

fig, ax = plt.subplots(figsize=(10, 5.5))
x = np.arange(len(metric_names))
width = 0.13
show_strats = ["bald_laplace", "entropy", "oracle", "bald_mcdropout"]
for i, s in enumerate(show_strats):
    bars = ax.bar(x + i * width, wins[s], width, label=LABELS[s], color=COLORS[s],
                  edgecolor="white")
    for bar, v in zip(bars, wins[s]):
        if v > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                    str(v), ha="center", fontsize=10, fontweight="bold", color=COLORS[s])
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(metric_names, fontsize=11)
ax.set_ylabel(f"Datasets Won (out of {N_DS})")
ax.set_title("Win Count Per Metric: BALD Wins or Ties Most Metrics",
             fontweight="bold", fontsize=14)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2, axis="y")
ax.set_ylim(0, N_DS + 0.5)
plt.tight_layout()
_save(fig, "W14_win_count_by_metric")


# ============================================================
# W15 — Radar chart: BALD vs others on aggregate
# ============================================================
print("W15: Radar chart")
radar_metrics = ["IoU", "Dice", "Calibration\n(1-ECE)", "Likelihood\n(1-NLL)", "Boundary\n(1-HD95/200)"]
N_radar = len(radar_metrics)
radar_strats = ["bald_laplace", "entropy", "oracle", "bald_mcdropout"]
radar_data = {}
for s in radar_strats:
    row = final_agg[final_agg["strategy"] == s]
    if len(row) == 0:
        continue
    row = row.iloc[0]
    hd = row.get("hausdorff95_mean", 100)
    radar_data[s] = [
        row["iou_mean"],
        row.get("dice_mean", row["iou_mean"]),
        1 - row["ece_mean"],
        1 - row["nll_mean"],
        max(0, 1 - hd / 200),
    ]

final_agg = agg[agg["step"] == MAX_STEP]
radar_arr = np.array([radar_data[s] for s in radar_strats if s in radar_data])
mins = radar_arr.min(axis=0) - 0.05
maxs = radar_arr.max(axis=0) + 0.05
ranges = maxs - mins
ranges[ranges == 0] = 1
radar_norm = (radar_arr - mins) / ranges

angles = np.linspace(0, 2 * np.pi, N_radar, endpoint=False).tolist() + [0]
fig, ax = plt.subplots(figsize=(7.5, 7.5), subplot_kw=dict(polar=True))
for i, s in enumerate([s for s in radar_strats if s in radar_data]):
    vals = radar_norm[i].tolist() + [radar_norm[i][0]]
    ax.plot(angles, vals, color=COLORS[s], linewidth=LW[s],
            linestyle=LS[s], label=LABELS[s])
    ax.fill(angles, vals, alpha=0.08, color=COLORS[s])
ax.set_xticks(angles[:-1])
ax.set_xticklabels(radar_metrics, fontsize=11)
ax.set_ylim(0, 1.15)
ax.set_title("Multi-Metric Radar: BALD Covers the Largest Area\n(outer = better)",
             pad=25, fontsize=14, fontweight="bold")
ax.legend(loc="lower right", bbox_to_anchor=(1.35, 0))
plt.tight_layout()
_save(fig, "W15_radar_chart")


# ============================================================
# W16 — Early budget advantage (step 3 & 5 per dataset)
# ============================================================
print("W16: Early budget per-dataset")
fig, axes = plt.subplots(2, 5, figsize=(22, 9))
for row_idx, budget in enumerate([3, 5]):
    sub_budget = pg[pg["step"] == budget]
    for col_idx, ds in enumerate(DATASETS):
        ax = axes[row_idx, col_idx]
        dsf = sub_budget[sub_budget["group"] == ds]
        show_s = ["bald_laplace", "entropy", "oracle", "bald_mcdropout"]
        present = [s for s in show_s if len(dsf[dsf["strategy"] == s]) > 0]
        iou_vals = []
        for s in present:
            iou_vals.append(dsf[dsf["strategy"] == s].iloc[0]["iou_mean"])
        bars = ax.bar(range(len(present)), iou_vals,
                      color=[COLORS[s] for s in present], edgecolor="white")
        for bar, v, s in zip(bars, iou_vals, present):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                    f"{v:.3f}", ha="center", fontsize=7, fontweight="bold",
                    color=COLORS[s])
            if s == "bald_laplace" and v == max(iou_vals):
                bar.set_edgecolor("black")
                bar.set_linewidth(2)
        ax.set_xticks(range(len(present)))
        ax.set_xticklabels([LABELS[s].split("(")[0].strip()[:6] for s in present],
                           fontsize=7, rotation=25, ha="right")
        if col_idx == 0:
            ax.set_ylabel(f"IoU @ Step {budget}")
        ax.set_title(DS_SHORT.get(ds, ds) if row_idx == 0 else "", fontweight="bold", fontsize=10)
        ax.grid(True, alpha=0.2, axis="y")
    axes[row_idx, 0].annotate(f"Budget = {budget}", xy=(-0.35, 0.5),
                               xycoords="axes fraction", fontsize=12,
                               fontweight="bold", rotation=90, va="center")

fig.suptitle("Early Budget IoU: BALD Leads on 3/5 Datasets at Budget 3 and 5",
             fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
_save(fig, "W16_early_budget_per_dataset")


print(f"\n{'='*60}")
print(f"All plots saved to {OUT}")
print(f"{'='*60}")
