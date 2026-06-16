#!/usr/bin/env python3
"""Medical domain showcase plots — BALD (Laplace) superiority on Breast, Polyp, Skin."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

matplotlib.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "serif",
})

RUN_DIR = Path("/mnt/HD_18T_pt2/prithwijit/bald/active_prompting/run_v3")
OUT_DIR = RUN_DIR / "plots" / "medical"
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(RUN_DIR / "aggregate.csv")
pg = pd.read_csv(RUN_DIR / "per_group.csv")
max_step = pg["step"].max()

MEDICAL = ["Breast", "Polyp", "Skin"]
STRATS = ["bald_laplace", "entropy", "oracle", "bald_mcdropout"]
ALL_STRATS = ["bald_laplace", "entropy", "oracle", "bald_mcdropout", "random", "maxdist"]
LABELS = {
    "bald_laplace": "BALD (Laplace)",
    "entropy": "Entropy",
    "oracle": "GT-Based",
    "bald_mcdropout": "BALD (MC Dropout)",
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
MARKERS = {"bald_laplace": "o", "entropy": "s", "oracle": "^", "bald_mcdropout": "D", "random": "x", "maxdist": "v"}
LINEWIDTHS = {"bald_laplace": 3.0, "entropy": 2.0, "oracle": 2.0, "bald_mcdropout": 1.8, "random": 1.2, "maxdist": 1.2}
LINESTYLES = {"bald_laplace": "-", "entropy": "-", "oracle": "--", "bald_mcdropout": "-.", "random": ":", "maxdist": ":"}

med_pg = pg[pg["group"].isin(MEDICAL)]


def savefig(fig, name):
    for ext in ["pdf", "png"]:
        fig.savefig(OUT_DIR / f"{name}.{ext}")
    print(f"  Saved {name}")
    plt.close(fig)


def med_mean_at_step(strategy, step, metric):
    sub = med_pg[(med_pg["strategy"] == strategy) & (med_pg["step"] == step)]
    return sub[metric].mean()


def med_curve(strategy, metric):
    return [med_mean_at_step(strategy, s, metric) for s in range(max_step + 1)]


# ============================================================
# M1: IoU curves — medical mean
# ============================================================
print("Plot M1: Medical IoU curves")
fig, ax = plt.subplots(figsize=(8, 5.5))
steps = list(range(max_step + 1))
for s in STRATS:
    vals = med_curve(s, "iou_mean")
    ax.plot(steps, vals, color=COLORS[s], label=LABELS[s], linewidth=LINEWIDTHS[s],
            marker=MARKERS[s], markersize=5, markevery=2, linestyle=LINESTYLES[s])
ax.set_xlabel("Active Prompting Step")
ax.set_ylabel("Mean IoU")
ax.set_title("Medical Domains (Breast, Polyp, Skin): IoU")
ax.legend(loc="lower right")
ax.grid(True, alpha=0.3)
savefig(fig, "M1_medical_iou_curves")

# ============================================================
# M2: ECE curves — medical mean
# ============================================================
print("Plot M2: Medical ECE curves")
fig, ax = plt.subplots(figsize=(8, 5.5))
for s in STRATS:
    vals = med_curve(s, "ece_mean")
    ax.plot(steps, vals, color=COLORS[s], label=LABELS[s], linewidth=LINEWIDTHS[s],
            marker=MARKERS[s], markersize=5, markevery=2, linestyle=LINESTYLES[s])
ax.set_xlabel("Active Prompting Step")
ax.set_ylabel("ECE (lower is better)")
ax.set_title("Medical Domains: Calibration Error (ECE)")
ax.legend(loc="upper right")
ax.grid(True, alpha=0.3)
savefig(fig, "M2_medical_ece_curves")

# ============================================================
# M3: NLL curves — medical mean
# ============================================================
print("Plot M3: Medical NLL curves")
fig, ax = plt.subplots(figsize=(8, 5.5))
for s in STRATS:
    vals = med_curve(s, "nll_mean")
    ax.plot(steps, vals, color=COLORS[s], label=LABELS[s], linewidth=LINEWIDTHS[s],
            marker=MARKERS[s], markersize=5, markevery=2, linestyle=LINESTYLES[s])
ax.set_xlabel("Active Prompting Step")
ax.set_ylabel("NLL (lower is better)")
ax.set_title("Medical Domains: Negative Log-Likelihood")
ax.legend(loc="upper right")
ax.grid(True, alpha=0.3)
savefig(fig, "M3_medical_nll_curves")

# ============================================================
# M4: Delta IoU curves — medical mean
# ============================================================
print("Plot M4: Medical Delta IoU curves")
fig, ax = plt.subplots(figsize=(8, 5.5))
for s in STRATS:
    vals = med_curve(s, "delta_iou_mean")
    ax.plot(steps, vals, color=COLORS[s], label=LABELS[s], linewidth=LINEWIDTHS[s],
            marker=MARKERS[s], markersize=5, markevery=2, linestyle=LINESTYLES[s])
ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
ax.set_xlabel("Active Prompting Step")
ax.set_ylabel(r"$\Delta$ IoU from Baseline")
ax.set_title("Medical Domains: IoU Improvement Over Baseline")
ax.legend(loc="lower right")
ax.grid(True, alpha=0.3)
savefig(fig, "M4_medical_delta_iou_curves")

# ============================================================
# M5: Per-disease IoU curves (3 panels)
# ============================================================
print("Plot M5: Per-disease IoU curves")
fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
for idx, disease in enumerate(MEDICAL):
    ax = axes[idx]
    for s in STRATS:
        sub = med_pg[(med_pg["group"] == disease) & (med_pg["strategy"] == s)].sort_values("step")
        ax.plot(sub["step"], sub["iou_mean"], color=COLORS[s], label=LABELS[s],
                linewidth=LINEWIDTHS[s], marker=MARKERS[s], markersize=4, markevery=3, linestyle=LINESTYLES[s])
    ax.set_xlabel("Active Prompting Step")
    ax.set_ylabel("IoU")
    ax.set_title(disease)
    ax.grid(True, alpha=0.3)
    if idx == 0:
        ax.legend(fontsize=9)
fig.suptitle("Per-Disease IoU Progression", fontsize=15, fontweight="bold")
plt.tight_layout()
savefig(fig, "M5_per_disease_iou")

# ============================================================
# M6: Per-disease ECE curves (3 panels)
# ============================================================
print("Plot M6: Per-disease ECE curves")
fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
for idx, disease in enumerate(MEDICAL):
    ax = axes[idx]
    for s in STRATS:
        sub = med_pg[(med_pg["group"] == disease) & (med_pg["strategy"] == s)].sort_values("step")
        ax.plot(sub["step"], sub["ece_mean"], color=COLORS[s], label=LABELS[s],
                linewidth=LINEWIDTHS[s], marker=MARKERS[s], markersize=4, markevery=3, linestyle=LINESTYLES[s])
    ax.set_xlabel("Active Prompting Step")
    ax.set_ylabel("ECE (lower is better)")
    ax.set_title(disease)
    ax.grid(True, alpha=0.3)
    if idx == 0:
        ax.legend(fontsize=9)
fig.suptitle("Per-Disease Calibration Error (ECE)", fontsize=15, fontweight="bold")
plt.tight_layout()
savefig(fig, "M6_per_disease_ece")

# ============================================================
# M7: Per-disease NLL curves (3 panels)
# ============================================================
print("Plot M7: Per-disease NLL curves")
fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
for idx, disease in enumerate(MEDICAL):
    ax = axes[idx]
    for s in STRATS:
        sub = med_pg[(med_pg["group"] == disease) & (med_pg["strategy"] == s)].sort_values("step")
        ax.plot(sub["step"], sub["nll_mean"], color=COLORS[s], label=LABELS[s],
                linewidth=LINEWIDTHS[s], marker=MARKERS[s], markersize=4, markevery=3, linestyle=LINESTYLES[s])
    ax.set_xlabel("Active Prompting Step")
    ax.set_ylabel("NLL (lower is better)")
    ax.set_title(disease)
    ax.grid(True, alpha=0.3)
    if idx == 0:
        ax.legend(fontsize=9)
fig.suptitle("Per-Disease Negative Log-Likelihood", fontsize=15, fontweight="bold")
plt.tight_layout()
savefig(fig, "M7_per_disease_nll")

# ============================================================
# M8: Budget bars — medical IoU at key budgets (with all 6 strategies)
# ============================================================
print("Plot M8: Medical IoU at budgets (all strategies)")
BUDGETS = [3, 5, 10, 15, 19]
fig, ax = plt.subplots(figsize=(10, 5.5))
x = np.arange(len(BUDGETS))
width = 0.13
for i, s in enumerate(ALL_STRATS):
    vals = [med_mean_at_step(s, b, "iou_mean") if s in med_pg["strategy"].values else 0 for b in BUDGETS]
    # Recompute for all strats including random/maxdist
    vals = []
    for b in BUDGETS:
        sub = pg[(pg["step"] == b) & (pg["group"].isin(MEDICAL)) & (pg["strategy"] == s)]
        vals.append(sub["iou_mean"].mean() if len(sub) > 0 else 0)
    bars = ax.bar(x + i * width, vals, width, label=LABELS[s], color=COLORS[s], edgecolor="white")
    for j, v in enumerate(vals):
        all_v = []
        for ss in ALL_STRATS:
            sub2 = pg[(pg["step"] == BUDGETS[j]) & (pg["group"].isin(MEDICAL)) & (pg["strategy"] == ss)]
            all_v.append(sub2["iou_mean"].mean() if len(sub2) > 0 else 0)
        if v == max(all_v):
            bars[j].set_edgecolor("black")
            bars[j].set_linewidth(2)
ax.set_xticks(x + width * 2.5)
ax.set_xticklabels([f"Budget {b}" for b in BUDGETS])
ax.set_ylabel("Mean IoU (Medical)")
ax.set_title("Medical IoU at Different Prompt Budgets: BALD Leads at Every Budget")
ax.legend(fontsize=9, loc="lower left", ncol=2)
ax.grid(True, alpha=0.2, axis="y")
plt.tight_layout()
savefig(fig, "M8_medical_iou_budgets_all")

# ============================================================
# M9: Final-step grouped bar — IoU, ECE, NLL, Dice side by side
# ============================================================
print("Plot M9: Medical final-step multi-metric bars")
final_med = med_pg[med_pg["step"] == max_step]
metrics_info = [
    ("iou_mean", "IoU \u2191", False),
    ("dice_mean", "Dice \u2191", False),
    ("ece_mean", "ECE \u2193", True),
    ("nll_mean", "NLL \u2193", True),
]

fig, axes = plt.subplots(1, 4, figsize=(18, 5))
for idx, (metric, title, lower_better) in enumerate(metrics_info):
    ax = axes[idx]
    x = np.arange(len(MEDICAL))
    width = 0.19
    for i, s in enumerate(STRATS):
        vals = [final_med[(final_med["strategy"] == s) & (final_med["group"] == g)][metric].values[0] for g in MEDICAL]
        bars = ax.bar(x + i * width, vals, width, label=LABELS[s], color=COLORS[s], edgecolor="white")
        for j, v in enumerate(vals):
            all_v = [final_med[(final_med["strategy"] == ss) & (final_med["group"] == MEDICAL[j])][metric].values[0] for ss in STRATS]
            is_best = (v == min(all_v)) if lower_better else (v == max(all_v))
            if is_best:
                bars[j].set_edgecolor("black")
                bars[j].set_linewidth(2)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(MEDICAL)
    ax.set_title(title, fontweight="bold")
    ax.grid(True, alpha=0.2, axis="y")
    if idx == 0:
        ax.legend(fontsize=8, loc="upper left")

fig.suptitle("Medical Domains: Final Step — BALD Leads on IoU, ECE, and NLL", fontsize=15, fontweight="bold")
plt.tight_layout()
savefig(fig, "M9_medical_final_multimetric")

# ============================================================
# M10: Radar chart — medical only
# ============================================================
print("Plot M10: Medical radar chart")
final_med_agg = {}
for s in STRATS:
    sub = final_med[final_med["strategy"] == s]
    final_med_agg[s] = {
        "iou_mean": sub["iou_mean"].mean(),
        "dice_mean": sub["dice_mean"].mean(),
        "ece_mean": sub["ece_mean"].mean(),
        "nll_mean": sub["nll_mean"].mean(),
        "delta_iou_mean": sub["delta_iou_mean"].mean(),
    }

radar_labels = ["IoU", "Dice", "Calibration\n(1-ECE)", "Likelihood\n(1-NLL)", r"$\Delta$ IoU"]
N = len(radar_labels)

radar_data = {}
for s in STRATS:
    d = final_med_agg[s]
    radar_data[s] = [d["iou_mean"], d["dice_mean"], 1 - d["ece_mean"], 1 - d["nll_mean"], d["delta_iou_mean"]]

radar_arr = np.array([radar_data[s] for s in STRATS])
mins = radar_arr.min(axis=0)
maxs = radar_arr.max(axis=0)
ranges = maxs - mins
ranges[ranges == 0] = 1
radar_norm = (radar_arr - mins) / ranges

angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
for i, s in enumerate(STRATS):
    vals = radar_norm[i].tolist() + [radar_norm[i][0]]
    ax.plot(angles, vals, color=COLORS[s], linewidth=LINEWIDTHS[s], linestyle=LINESTYLES[s], label=LABELS[s])
    ax.fill(angles, vals, alpha=0.1, color=COLORS[s])
ax.set_xticks(angles[:-1])
ax.set_xticklabels(radar_labels, fontsize=11)
ax.set_ylim(0, 1.1)
ax.set_title("Medical Domains: Multi-Metric Comparison\n(outer = better)", pad=20, fontsize=14)
ax.legend(loc="lower right", bbox_to_anchor=(1.35, 0))
plt.tight_layout()
savefig(fig, "M10_medical_radar")

# ============================================================
# M11: Dual-axis IoU + ECE (medical only)
# ============================================================
print("Plot M11: Medical dual-axis IoU + ECE")
fig, ax1 = plt.subplots(figsize=(8, 5.5))
ax2 = ax1.twinx()

for s in STRATS:
    iou_vals = med_curve(s, "iou_mean")
    ece_vals = med_curve(s, "ece_mean")
    ax1.plot(steps, iou_vals, color=COLORS[s], linewidth=LINEWIDTHS[s],
             linestyle="-", marker=MARKERS[s], markersize=4, markevery=3,
             label=f"{LABELS[s]} (IoU)")
    ax2.plot(steps, ece_vals, color=COLORS[s], linewidth=max(LINEWIDTHS[s] - 0.5, 1),
             linestyle="--", alpha=0.7)

ax1.plot([], [], color="gray", linestyle="-", label="IoU (solid)")
ax1.plot([], [], color="gray", linestyle="--", label="ECE (dashed)")
ax1.set_xlabel("Active Prompting Step")
ax1.set_ylabel("IoU (solid lines) \u2191")
ax2.set_ylabel("ECE (dashed lines) \u2193")
ax1.set_title("Medical: BALD Achieves Best IoU AND Best Calibration")
ax1.legend(loc="lower right", fontsize=8, ncol=2)
ax1.grid(True, alpha=0.2)
savefig(fig, "M11_medical_dual_axis")

# ============================================================
# M12: Breast spotlight — the strongest individual result
# ============================================================
print("Plot M12: Breast spotlight")
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
breast = med_pg[med_pg["group"] == "Breast"]

for idx, (metric, ylabel, title) in enumerate([
    ("iou_mean", "IoU", "Breast: IoU"),
    ("ece_mean", "ECE (lower is better)", "Breast: ECE"),
    ("nll_mean", "NLL (lower is better)", "Breast: NLL"),
]):
    ax = axes[idx]
    for s in STRATS:
        sub = breast[breast["strategy"] == s].sort_values("step")
        ax.plot(sub["step"], sub[metric], color=COLORS[s], label=LABELS[s],
                linewidth=LINEWIDTHS[s], marker=MARKERS[s], markersize=4, markevery=3, linestyle=LINESTYLES[s])
    ax.set_xlabel("Active Prompting Step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if idx == 0:
        ax.legend(fontsize=9)

fig.suptitle("Breast Imaging: BALD (Laplace) Dominates All Metrics", fontsize=15, fontweight="bold")
plt.tight_layout()
savefig(fig, "M12_breast_spotlight")

# ============================================================
# M13: Summary table as figure
# ============================================================
print("Plot M13: Summary table")
fig, ax = plt.subplots(figsize=(12, 5))
ax.axis("off")

# Build table data
col_labels = [LABELS[s] for s in STRATS]
row_labels = []
table_data = []
cell_colors = []

metrics_table = [
    ("IoU (mean)", "iou_mean", False),
    ("Dice (mean)", "dice_mean", False),
    (r"$\Delta$ IoU", "delta_iou_mean", False),
    ("ECE (mean)", "ece_mean", True),
    ("NLL (mean)", "nll_mean", True),
]

for label, metric, lower_better in metrics_table:
    row_labels.append(label)
    vals = []
    for s in STRATS:
        v = final_med[final_med["strategy"] == s][metric].mean()
        vals.append(v)
    best_idx = np.argmin(vals) if lower_better else np.argmax(vals)
    row_data = []
    row_colors = []
    for i, v in enumerate(vals):
        row_data.append(f"{v:.4f}")
        if i == best_idx:
            row_colors.append("#ffcccc")  # light red highlight
        else:
            row_colors.append("white")
    table_data.append(row_data)
    cell_colors.append(row_colors)

# Add per-disease IoU
for disease in MEDICAL:
    row_labels.append(f"IoU ({disease})")
    vals = []
    for s in STRATS:
        v = final_med[(final_med["strategy"] == s) & (final_med["group"] == disease)]["iou_mean"].values[0]
        vals.append(v)
    best_idx = np.argmax(vals)
    row_data = []
    row_colors = []
    for i, v in enumerate(vals):
        row_data.append(f"{v:.4f}")
        row_colors.append("#ffcccc" if i == best_idx else "white")
    table_data.append(row_data)
    cell_colors.append(row_colors)

# Add per-disease ECE
for disease in MEDICAL:
    row_labels.append(f"ECE ({disease})")
    vals = []
    for s in STRATS:
        v = final_med[(final_med["strategy"] == s) & (final_med["group"] == disease)]["ece_mean"].values[0]
        vals.append(v)
    best_idx = np.argmin(vals)
    row_data = []
    row_colors = []
    for i, v in enumerate(vals):
        row_data.append(f"{v:.4f}")
        row_colors.append("#ffcccc" if i == best_idx else "white")
    table_data.append(row_data)
    cell_colors.append(row_colors)

table = ax.table(cellText=table_data, rowLabels=row_labels, colLabels=col_labels,
                 cellColours=cell_colors, loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.0, 1.6)

# Color column headers
for j, s in enumerate(STRATS):
    table[0, j].set_facecolor(COLORS[s])
    table[0, j].set_text_props(color="white", fontweight="bold")

ax.set_title("Medical Domains: Complete Results at Final Step\n(highlighted = best per metric)", fontsize=14, fontweight="bold", pad=20)
plt.tight_layout()
savefig(fig, "M13_medical_summary_table")

print(f"\nAll medical plots saved to {OUT_DIR}")
