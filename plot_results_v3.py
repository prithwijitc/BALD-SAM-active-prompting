#!/usr/bin/env python3
"""Publication-quality plots for active prompting v3 results."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

matplotlib.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "serif",
})

RUN_DIR = Path("/mnt/HD_18T_pt2/prithwijit/bald/active_prompting/run_v3")
OUT_DIR = RUN_DIR / "plots"
OUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(RUN_DIR / "aggregate.csv")
pg = pd.read_csv(RUN_DIR / "per_group.csv")
max_step = df["step"].max()

# --- Style config ---
STRATS = ["bald_laplace", "entropy", "oracle", "bald_mcdropout", "random", "maxdist"]
LABELS = {
    "bald_laplace": "BALD (Laplace)",
    "entropy": "Entropy",
    "oracle": "GT-Based",
    "bald_mcdropout": "BALD (MC Dropout)",
    "random": "Random",
    "maxdist": "MaxDist",
}
COLORS = {
    "bald_laplace": "#d62728",   # red — hero
    "entropy": "#1f77b4",        # blue
    "oracle": "#2ca02c",         # green
    "bald_mcdropout": "#ff7f0e", # orange
    "random": "#7f7f7f",         # gray
    "maxdist": "#9467bd",        # purple
}
MARKERS = {
    "bald_laplace": "o",
    "entropy": "s",
    "oracle": "^",
    "bald_mcdropout": "D",
    "random": "x",
    "maxdist": "v",
}
LINEWIDTHS = {
    "bald_laplace": 2.5,
    "entropy": 2.0,
    "oracle": 2.0,
    "bald_mcdropout": 1.5,
    "random": 1.2,
    "maxdist": 1.2,
}
LINESTYLES = {
    "bald_laplace": "-",
    "entropy": "-",
    "oracle": "--",
    "bald_mcdropout": "-.",
    "random": ":",
    "maxdist": ":",
}


def savefig(fig, name):
    for ext in ["pdf", "png"]:
        fig.savefig(OUT_DIR / f"{name}.{ext}")
    print(f"  Saved {name}")
    plt.close(fig)


# ============================================================
# PLOT 1: IoU vs Steps (with std shading)
# ============================================================
print("Plot 1: IoU vs Steps")
fig, ax = plt.subplots(figsize=(8, 5))
for s in STRATS:
    sub = df[df["strategy"] == s].sort_values("step")
    ax.plot(sub["step"], sub["iou_mean"],
            color=COLORS[s], label=LABELS[s], linewidth=LINEWIDTHS[s],
            marker=MARKERS[s], markersize=5, markevery=2, linestyle=LINESTYLES[s])
    ax.fill_between(sub["step"],
                    sub["iou_mean"] - sub["iou_std"],
                    sub["iou_mean"] + sub["iou_std"],
                    alpha=0.08, color=COLORS[s])
ax.set_xlabel("Active Prompting Step")
ax.set_ylabel("IoU")
ax.set_title("Segmentation Quality (IoU) vs Active Prompting Steps")
ax.legend(loc="lower right")
ax.grid(True, alpha=0.3)
ax.set_xlim(0, max_step)
savefig(fig, "01_iou_vs_steps")

# ============================================================
# PLOT 2: ECE vs Steps
# ============================================================
print("Plot 2: ECE vs Steps")
fig, ax = plt.subplots(figsize=(8, 5))
for s in STRATS:
    sub = df[df["strategy"] == s].sort_values("step")
    ax.plot(sub["step"], sub["ece_mean"],
            color=COLORS[s], label=LABELS[s], linewidth=LINEWIDTHS[s],
            marker=MARKERS[s], markersize=5, markevery=2, linestyle=LINESTYLES[s])
ax.set_xlabel("Active Prompting Step")
ax.set_ylabel("ECE (lower is better)")
ax.set_title("Calibration Error (ECE) vs Active Prompting Steps")
ax.legend(loc="upper right")
ax.grid(True, alpha=0.3)
ax.set_xlim(0, max_step)
ax.invert_yaxis()  # lower ECE = better → visually higher = better
savefig(fig, "02_ece_vs_steps")

# ============================================================
# PLOT 3: NLL vs Steps
# ============================================================
print("Plot 3: NLL vs Steps")
fig, ax = plt.subplots(figsize=(8, 5))
for s in STRATS:
    sub = df[df["strategy"] == s].sort_values("step")
    ax.plot(sub["step"], sub["nll_mean"],
            color=COLORS[s], label=LABELS[s], linewidth=LINEWIDTHS[s],
            marker=MARKERS[s], markersize=5, markevery=2, linestyle=LINESTYLES[s])
ax.set_xlabel("Active Prompting Step")
ax.set_ylabel("NLL (lower is better)")
ax.set_title("Negative Log-Likelihood vs Active Prompting Steps")
ax.legend(loc="upper right")
ax.grid(True, alpha=0.3)
ax.set_xlim(0, max_step)
savefig(fig, "03_nll_vs_steps")

# ============================================================
# PLOT 4: Delta IoU vs Steps
# ============================================================
print("Plot 4: Delta IoU vs Steps")
fig, ax = plt.subplots(figsize=(8, 5))
for s in STRATS:
    sub = df[df["strategy"] == s].sort_values("step")
    ax.plot(sub["step"], sub["delta_iou_mean"],
            color=COLORS[s], label=LABELS[s], linewidth=LINEWIDTHS[s],
            marker=MARKERS[s], markersize=5, markevery=2, linestyle=LINESTYLES[s])
ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
ax.set_xlabel("Active Prompting Step")
ax.set_ylabel(r"$\Delta$ IoU (from step 0)")
ax.set_title("IoU Improvement Over Baseline vs Active Prompting Steps")
ax.legend(loc="lower left")
ax.grid(True, alpha=0.3)
ax.set_xlim(0, max_step)
savefig(fig, "04_delta_iou_vs_steps")

# ============================================================
# PLOT 5: Pareto — IoU vs ECE at final step (scatter)
# ============================================================
print("Plot 5: Pareto IoU vs ECE")
fig, ax = plt.subplots(figsize=(7, 5))
final = df[df["step"] == max_step]
for s in STRATS:
    row = final[final["strategy"] == s].iloc[0]
    ax.scatter(row["ece_mean"], row["iou_mean"],
               color=COLORS[s], marker=MARKERS[s], s=150, zorder=5,
               edgecolors="black", linewidth=0.8)
    offset_x, offset_y = 0.003, 0.008
    if s == "bald_mcdropout":
        offset_y = -0.015
    ax.annotate(LABELS[s], (row["ece_mean"] + offset_x, row["iou_mean"] + offset_y),
                fontsize=10, color=COLORS[s], fontweight="bold")
ax.set_xlabel("ECE (lower is better) \u2192")
ax.set_ylabel("IoU (higher is better) \u2192")
ax.set_title("Pareto Front: Segmentation Quality vs Calibration")
ax.grid(True, alpha=0.3)
# Draw ideal corner arrow
ax.annotate("", xy=(0.17, 0.63), xytext=(0.22, 0.50),
            arrowprops=dict(arrowstyle="->", color="gray", lw=1.5))
ax.text(0.175, 0.49, "ideal", color="gray", fontsize=10, ha="center")
savefig(fig, "05_pareto_iou_ece")

# ============================================================
# PLOT 6: Medical domain grouped bar chart (IoU, ECE, NLL)
# ============================================================
print("Plot 6: Medical domain bars")
medical = ["Breast", "Polyp", "Skin"]
med_final = pg[(pg["step"] == max_step) & (pg["group"].isin(medical))]
show_strats = ["bald_laplace", "entropy", "oracle", "bald_mcdropout"]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for idx, (metric, title, invert) in enumerate([
    ("iou_mean", "IoU (higher is better)", False),
    ("ece_mean", "ECE (lower is better)", True),
    ("nll_mean", "NLL (lower is better)", True),
]):
    ax = axes[idx]
    x = np.arange(len(medical))
    width = 0.18
    for i, s in enumerate(show_strats):
        vals = []
        for g in medical:
            v = med_final[(med_final["strategy"] == s) & (med_final["group"] == g)][metric].values[0]
            vals.append(v)
        bars = ax.bar(x + i * width, vals, width, label=LABELS[s], color=COLORS[s], edgecolor="white")
        # Bold outline for best per group
        for j, v in enumerate(vals):
            all_vals = [med_final[(med_final["strategy"] == ss) & (med_final["group"] == medical[j])][metric].values[0] for ss in show_strats]
            is_best = (v == min(all_vals)) if invert else (v == max(all_vals))
            if is_best:
                bars[j].set_edgecolor("black")
                bars[j].set_linewidth(2)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(medical)
    ax.set_title(title)
    ax.grid(True, alpha=0.2, axis="y")
    if idx == 0:
        ax.legend(loc="upper left", fontsize=9)

fig.suptitle("Medical / Safety-Critical Domains: BALD (Laplace) Leads", fontsize=14, fontweight="bold")
plt.tight_layout()
savefig(fig, "06_medical_domain")

# ============================================================
# PLOT 7: Head-to-head win rate (BALD vs entropy) — stacked bar
# ============================================================
print("Plot 7: Head-to-head win rates")
final_pg = pg[pg["step"] == max_step]
bl = final_pg[final_pg["strategy"] == "bald_laplace"].set_index("group")
en = final_pg[final_pg["strategy"] == "entropy"].set_index("group")

metrics_h2h = [
    ("iou_mean", "IoU", False),
    ("ece_mean", "ECE", True),
    ("nll_mean", "NLL", True),
]
bl_wins = []
en_wins = []
ties = []
for metric, label, lower_better in metrics_h2h:
    if lower_better:
        bw = (bl[metric] < en[metric]).sum()
        ew = (bl[metric] > en[metric]).sum()
    else:
        bw = (bl[metric] > en[metric]).sum()
        ew = (bl[metric] < en[metric]).sum()
    t = 17 - bw - ew
    bl_wins.append(bw)
    en_wins.append(ew)
    ties.append(t)

fig, ax = plt.subplots(figsize=(7, 4))
labels_h2h = [m[1] for m in metrics_h2h]
x = np.arange(len(labels_h2h))
w = 0.35
bars1 = ax.bar(x - w/2, bl_wins, w, label="BALD (Laplace) wins", color=COLORS["bald_laplace"])
bars2 = ax.bar(x + w/2, en_wins, w, label="Entropy wins", color=COLORS["entropy"])
for bar, val in zip(bars1, bl_wins):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, str(val),
            ha="center", fontweight="bold", color=COLORS["bald_laplace"])
for bar, val in zip(bars2, en_wins):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, str(val),
            ha="center", fontweight="bold", color=COLORS["entropy"])
ax.set_xticks(x)
ax.set_xticklabels(labels_h2h)
ax.set_ylabel("Groups Won (out of 17)")
ax.set_title("Head-to-Head: BALD (Laplace) vs Entropy Across Groups")
ax.legend()
ax.set_ylim(0, 20)
ax.axhline(8.5, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
ax.text(2.35, 8.8, "majority", color="gray", fontsize=9)
ax.grid(True, alpha=0.2, axis="y")
savefig(fig, "07_head_to_head_bald_vs_entropy")

# ============================================================
# PLOT 8: ECE heatmap across all groups (strategies as columns)
# ============================================================
print("Plot 8: ECE heatmap")
show_strats_heat = ["bald_laplace", "entropy", "oracle", "bald_mcdropout", "random", "maxdist"]
pivot_ece = final_pg.pivot_table(index="group", columns="strategy", values="ece_mean")[show_strats_heat]
pivot_ece.columns = [LABELS[s] for s in show_strats_heat]

# Sort groups by BALD advantage over entropy
bl_ece = final_pg[final_pg["strategy"]=="bald_laplace"].set_index("group")["ece_mean"]
en_ece = final_pg[final_pg["strategy"]=="entropy"].set_index("group")["ece_mean"]
sort_order = (bl_ece - en_ece).sort_values().index
pivot_ece = pivot_ece.loc[sort_order]

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(pivot_ece.values, cmap="RdYlGn_r", aspect="auto")
# Mark best per row
for i in range(len(pivot_ece)):
    best_j = pivot_ece.iloc[i].values.argmin()
    ax.text(best_j, i, f"{pivot_ece.iloc[i, best_j]:.3f}",
            ha="center", va="center", fontweight="bold", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
    for j in range(len(show_strats_heat)):
        if j != best_j:
            ax.text(j, i, f"{pivot_ece.iloc[i, j]:.3f}",
                    ha="center", va="center", fontsize=8, color="black")

ax.set_xticks(range(len(show_strats_heat)))
ax.set_xticklabels(pivot_ece.columns, rotation=30, ha="right")
ax.set_yticks(range(len(pivot_ece)))
ax.set_yticklabels(pivot_ece.index)
ax.set_title("ECE by Group (lower = better, bold = best)")
plt.colorbar(im, ax=ax, label="ECE", shrink=0.8)
plt.tight_layout()
savefig(fig, "08_ece_heatmap")

# ============================================================
# PLOT 9: Box plot of delta IoU across groups (consistency)
# ============================================================
print("Plot 9: Delta IoU box plot")
show_strats_box = ["bald_laplace", "entropy", "oracle", "bald_mcdropout", "random", "maxdist"]
fig, ax = plt.subplots(figsize=(9, 5))
box_data = []
box_colors = []
for s in show_strats_box:
    vals = final_pg[final_pg["strategy"] == s]["delta_iou_mean"].values
    box_data.append(vals)
    box_colors.append(COLORS[s])

bp = ax.boxplot(box_data, labels=[LABELS[s] for s in show_strats_box],
                patch_artist=True, showmeans=True,
                meanprops=dict(marker="D", markerfacecolor="black", markersize=6),
                medianprops=dict(color="black", linewidth=1.5))
for patch, color in zip(bp["boxes"], box_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
ax.axhline(0, color="black", linewidth=1, linestyle="--", alpha=0.5)
ax.set_ylabel(r"$\Delta$ IoU per group")
ax.set_title("Consistency of IoU Improvement Across Groups")
ax.grid(True, alpha=0.2, axis="y")
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
savefig(fig, "09_delta_iou_boxplot")

# ============================================================
# PLOT 10: Radar chart — aggregate metrics
# ============================================================
print("Plot 10: Radar chart")
show_strats_radar = ["bald_laplace", "entropy", "oracle", "bald_mcdropout"]
final_agg = df[df["step"] == max_step]

# Metrics: IoU (higher=better), 1-ECE, 1-NLL, reliability, consistency, 25th-pctl IoU
radar_labels = ["IoU", "Calibration\n(1-ECE)", "Likelihood\n(1-NLL)", "Reliability\n(% groups improved)", "Consistency\n(1-std)", "Tail Safety\n(25th pctl IoU)"]
N = len(radar_labels)

radar_data = {}
for s in show_strats_radar:
    row = final_agg[final_agg["strategy"] == s].iloc[0]
    sub_pg = final_pg[final_pg["strategy"] == s]
    reliability = (sub_pg["delta_iou_mean"] > 0).sum() / len(sub_pg)
    consistency = 1 - sub_pg["delta_iou_mean"].std()
    tail = sub_pg["iou_mean"].quantile(0.25)
    radar_data[s] = [
        row["iou_mean"],
        1 - row["ece_mean"],
        1 - row["nll_mean"],
        reliability,
        consistency,
        tail,
    ]

# Normalize each axis to [0, 1] across shown strategies
radar_arr = np.array([radar_data[s] for s in show_strats_radar])
mins = radar_arr.min(axis=0)
maxs = radar_arr.max(axis=0)
ranges = maxs - mins
ranges[ranges == 0] = 1
radar_norm = (radar_arr - mins) / ranges

angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
for i, s in enumerate(show_strats_radar):
    vals = radar_norm[i].tolist() + [radar_norm[i][0]]
    ax.plot(angles, vals, color=COLORS[s], linewidth=LINEWIDTHS[s],
            linestyle=LINESTYLES[s], label=LABELS[s])
    ax.fill(angles, vals, alpha=0.08, color=COLORS[s])
ax.set_xticks(angles[:-1])
ax.set_xticklabels(radar_labels, fontsize=10)
ax.set_ylim(0, 1.1)
ax.set_title("Multi-Metric Strategy Comparison\n(normalized, outer = better)", pad=20)
ax.legend(loc="lower right", bbox_to_anchor=(1.3, 0))
plt.tight_layout()
savefig(fig, "10_radar_chart")

# ============================================================
# PLOT 11: Per-group IoU bar chart (grouped, final step)
# ============================================================
print("Plot 11: Per-group IoU bars")
show_strats_bar = ["bald_laplace", "entropy", "oracle", "bald_mcdropout"]
groups = sorted(pg["group"].unique())

fig, ax = plt.subplots(figsize=(16, 6))
x = np.arange(len(groups))
width = 0.2
for i, s in enumerate(show_strats_bar):
    vals = [final_pg[(final_pg["strategy"] == s) & (final_pg["group"] == g)]["iou_mean"].values[0] for g in groups]
    ax.bar(x + i * width, vals, width, label=LABELS[s], color=COLORS[s], edgecolor="white")
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(groups, rotation=45, ha="right")
ax.set_ylabel("IoU")
ax.set_title("Final IoU per Group by Strategy")
ax.legend(loc="upper right")
ax.grid(True, alpha=0.2, axis="y")
plt.tight_layout()
savefig(fig, "11_per_group_iou")

# ============================================================
# PLOT 12: Combined IoU + ECE dual-axis (the "pitch plot")
# ============================================================
print("Plot 12: Dual-axis IoU + ECE")
fig, ax1 = plt.subplots(figsize=(8, 5))
ax2 = ax1.twinx()

for s in ["bald_laplace", "entropy", "oracle"]:
    sub = df[df["strategy"] == s].sort_values("step")
    ax1.plot(sub["step"], sub["iou_mean"],
             color=COLORS[s], label=f"{LABELS[s]} (IoU)", linewidth=LINEWIDTHS[s],
             linestyle="-", marker=MARKERS[s], markersize=4, markevery=3)
    ax2.plot(sub["step"], sub["ece_mean"],
             color=COLORS[s], linewidth=LINEWIDTHS[s],
             linestyle="--", alpha=0.7)

# Dummy lines for legend
ax1.plot([], [], color="gray", linestyle="-", label="IoU (solid)")
ax1.plot([], [], color="gray", linestyle="--", label="ECE (dashed)")

ax1.set_xlabel("Active Prompting Step")
ax1.set_ylabel("IoU (solid lines) \u2191")
ax2.set_ylabel("ECE (dashed lines) \u2193")
ax1.set_title("IoU vs ECE Tradeoff: BALD Achieves Best Calibration")
ax1.legend(loc="lower right", fontsize=9)
ax1.grid(True, alpha=0.2)
ax1.set_xlim(0, max_step)
savefig(fig, "12_dual_axis_iou_ece")

print(f"\nAll plots saved to {OUT_DIR}")
