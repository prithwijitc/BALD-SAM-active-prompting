#!/usr/bin/env python3
"""Stratum-specific analysis plots for active prompting v3."""

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

ps = pd.read_csv(RUN_DIR / "per_stratum.csv")
max_step = ps["step"].max()

STRATS = ["bald_laplace", "entropy", "oracle", "bald_mcdropout"]
LABELS = {
    "bald_laplace": "BALD (Laplace)",
    "entropy": "Entropy",
    "oracle": "GT-Based",
    "bald_mcdropout": "BALD (MC Dropout)",
}
COLORS = {
    "bald_laplace": "#d62728",
    "entropy": "#1f77b4",
    "oracle": "#2ca02c",
    "bald_mcdropout": "#ff7f0e",
}
MARKERS = {"bald_laplace": "o", "entropy": "s", "oracle": "^", "bald_mcdropout": "D"}
LINEWIDTHS = {"bald_laplace": 2.5, "entropy": 2.0, "oracle": 2.0, "bald_mcdropout": 1.5}
LINESTYLES = {"bald_laplace": "-", "entropy": "-", "oracle": "--", "bald_mcdropout": "-."}

STRATUM_ORDER = {
    "object_size_bin": ["small", "medium", "large"],
    "thinness_bin": ["thin", "moderate", "compact"],
    "boundary_complexity_bin": ["complex", "moderate", "simple"],
    "contrast_bin": ["low", "medium", "high"],
}
STRATUM_TITLES = {
    "object_size_bin": "Object Size",
    "thinness_bin": "Object Thinness",
    "boundary_complexity_bin": "Boundary Complexity",
    "contrast_bin": "Image Contrast",
}


def savefig(fig, name):
    for ext in ["pdf", "png"]:
        fig.savefig(OUT_DIR / f"{name}.{ext}")
    print(f"  Saved {name}")
    plt.close(fig)


# ============================================================
# PLOT S1: ECE wins heatmap across all strata (the headline)
# ============================================================
print("Plot S1: ECE wins heatmap")
final = ps[ps["step"] == max_step]

strata_labels = []
ece_vals = {s: [] for s in STRATS}
for stype in ["object_size_bin", "thinness_bin", "boundary_complexity_bin", "contrast_bin"]:
    for sv in STRATUM_ORDER[stype]:
        strata_labels.append(f"{STRATUM_TITLES[stype]}\n{sv}")
        for s in STRATS:
            v = final[(final["stratum_type"] == stype) & (final["stratum_value"] == sv) & (final["strategy"] == s)]["ece_mean"].values[0]
            ece_vals[s].append(v)

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(strata_labels))
width = 0.19
for i, s in enumerate(STRATS):
    vals = ece_vals[s]
    bars = ax.bar(x + i * width, vals, width, label=LABELS[s], color=COLORS[s], edgecolor="white")
    # Mark best per stratum
    for j in range(len(vals)):
        all_v = [ece_vals[ss][j] for ss in STRATS]
        if vals[j] == min(all_v):
            bars[j].set_edgecolor("black")
            bars[j].set_linewidth(2)

ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(strata_labels, fontsize=8, ha="center")
ax.set_ylabel("ECE (lower is better)")
ax.set_title("ECE Across All Morphological Strata: BALD (Laplace) Wins 10/12")
ax.legend(loc="upper right", fontsize=9)
ax.grid(True, alpha=0.2, axis="y")

# Add vertical separators between stratum types
for pos in [2.5, 5.5, 8.5]:
    ax.axvline(pos + 0.1, color="gray", linewidth=0.8, linestyle="--", alpha=0.4)

plt.tight_layout()
savefig(fig, "S1_ece_all_strata")

# ============================================================
# PLOT S2: Object size — IoU + ECE side-by-side bars
# ============================================================
print("Plot S2: Object size IoU + ECE")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
stype = "object_size_bin"
bins = STRATUM_ORDER[stype]

for idx, (metric, title, invert) in enumerate([
    ("iou_mean", "IoU (higher is better)", False),
    ("ece_mean", "ECE (lower is better)", True),
]):
    ax = axes[idx]
    x = np.arange(len(bins))
    width = 0.19
    for i, s in enumerate(STRATS):
        vals = []
        for sv in bins:
            v = final[(final["stratum_type"] == stype) & (final["stratum_value"] == sv) & (final["strategy"] == s)][metric].values[0]
            vals.append(v)
        bars = ax.bar(x + i * width, vals, width, label=LABELS[s], color=COLORS[s], edgecolor="white")
        for j, v in enumerate(vals):
            all_v = [final[(final["stratum_type"] == stype) & (final["stratum_value"] == bins[j]) & (final["strategy"] == ss)][metric].values[0] for ss in STRATS]
            is_best = (v == min(all_v)) if invert else (v == max(all_v))
            if is_best:
                bars[j].set_edgecolor("black")
                bars[j].set_linewidth(2)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([b.capitalize() for b in bins])
    ax.set_title(title)
    ax.grid(True, alpha=0.2, axis="y")
    if idx == 0:
        ax.legend(fontsize=9)

fig.suptitle("Object Size Strata: BALD Leads on Small Objects (IoU) & Medium/Small (ECE)", fontsize=13, fontweight="bold")
plt.tight_layout()
savefig(fig, "S2_object_size_iou_ece")

# ============================================================
# PLOT S3: Small objects — IoU curve over steps
# ============================================================
print("Plot S3: Small objects IoU curve")
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for idx, (sv, title_suffix) in enumerate([("small", "Small Objects"), ("medium", "Medium Objects"), ("large", "Large Objects")]):
    ax = axes[idx]
    for s in STRATS:
        sub = ps[(ps["stratum_type"] == "object_size_bin") & (ps["stratum_value"] == sv) & (ps["strategy"] == s)].sort_values("step")
        ax.plot(sub["step"], sub["iou_mean"],
                color=COLORS[s], label=LABELS[s], linewidth=LINEWIDTHS[s],
                marker=MARKERS[s], markersize=4, markevery=3, linestyle=LINESTYLES[s])
    ax.set_xlabel("Active Prompting Step")
    ax.set_ylabel("IoU")
    ax.set_title(f"{title_suffix}")
    ax.grid(True, alpha=0.3)
    if idx == 0:
        ax.legend(fontsize=8, loc="lower left")

fig.suptitle("IoU by Object Size: BALD Excels on Small Objects", fontsize=14, fontweight="bold")
plt.tight_layout()
savefig(fig, "S3_object_size_iou_curves")

# ============================================================
# PLOT S4: Boundary complexity — ECE + NLL bars
# ============================================================
print("Plot S4: Boundary complexity ECE + NLL")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
stype = "boundary_complexity_bin"
bins = STRATUM_ORDER[stype]

for idx, (metric, title) in enumerate([
    ("ece_mean", "ECE (lower is better)"),
    ("nll_mean", "NLL (lower is better)"),
]):
    ax = axes[idx]
    x = np.arange(len(bins))
    width = 0.19
    for i, s in enumerate(STRATS):
        vals = []
        for sv in bins:
            v = final[(final["stratum_type"] == stype) & (final["stratum_value"] == sv) & (final["strategy"] == s)][metric].values[0]
            vals.append(v)
        bars = ax.bar(x + i * width, vals, width, label=LABELS[s], color=COLORS[s], edgecolor="white")
        for j, v in enumerate(vals):
            all_v = [final[(final["stratum_type"] == stype) & (final["stratum_value"] == bins[j]) & (final["strategy"] == ss)][metric].values[0] for ss in STRATS]
            if v == min(all_v):
                bars[j].set_edgecolor("black")
                bars[j].set_linewidth(2)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([b.capitalize() for b in bins])
    ax.set_title(title)
    ax.grid(True, alpha=0.2, axis="y")
    if idx == 0:
        ax.legend(fontsize=9)

fig.suptitle("Boundary Complexity: BALD Wins ECE Everywhere, NLL on Complex", fontsize=13, fontweight="bold")
plt.tight_layout()
savefig(fig, "S4_boundary_complexity_ece_nll")

# ============================================================
# PLOT S5: Thinness — ECE curves over steps (3 panels)
# ============================================================
print("Plot S5: Thinness ECE curves")
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for idx, sv in enumerate(["thin", "moderate", "compact"]):
    ax = axes[idx]
    for s in STRATS:
        sub = ps[(ps["stratum_type"] == "thinness_bin") & (ps["stratum_value"] == sv) & (ps["strategy"] == s)].sort_values("step")
        ax.plot(sub["step"], sub["ece_mean"],
                color=COLORS[s], label=LABELS[s], linewidth=LINEWIDTHS[s],
                marker=MARKERS[s], markersize=4, markevery=3, linestyle=LINESTYLES[s])
    ax.set_xlabel("Active Prompting Step")
    ax.set_ylabel("ECE (lower is better)")
    ax.set_title(f"Thinness: {sv.capitalize()}")
    ax.grid(True, alpha=0.3)
    if idx == 0:
        ax.legend(fontsize=8)

fig.suptitle("ECE by Object Thinness: BALD (Laplace) Leads Across All Shape Types", fontsize=14, fontweight="bold")
plt.tight_layout()
savefig(fig, "S5_thinness_ece_curves")

# ============================================================
# PLOT S6: Delta IoU for small objects (degradation comparison)
# ============================================================
print("Plot S6: Small objects delta IoU")
fig, ax = plt.subplots(figsize=(8, 5))
for s in STRATS:
    sub = ps[(ps["stratum_type"] == "object_size_bin") & (ps["stratum_value"] == "small") & (ps["strategy"] == s)].sort_values("step")
    ax.plot(sub["step"], sub["delta_iou_mean"],
            color=COLORS[s], label=LABELS[s], linewidth=LINEWIDTHS[s],
            marker=MARKERS[s], markersize=5, markevery=2, linestyle=LINESTYLES[s])
ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
ax.set_xlabel("Active Prompting Step")
ax.set_ylabel(r"$\Delta$ IoU")
ax.set_title("Small Objects: BALD Degrades Least Under Active Prompting")
ax.legend(loc="lower left")
ax.grid(True, alpha=0.3)
savefig(fig, "S6_small_objects_delta_iou")

# ============================================================
# PLOT S7: Complex boundaries — ECE + NLL curves
# ============================================================
print("Plot S7: Complex boundary ECE + NLL curves")
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for idx, (metric, ylabel, title) in enumerate([
    ("ece_mean", "ECE (lower is better)", "ECE on Complex Boundaries"),
    ("nll_mean", "NLL (lower is better)", "NLL on Complex Boundaries"),
]):
    ax = axes[idx]
    for s in STRATS:
        sub = ps[(ps["stratum_type"] == "boundary_complexity_bin") & (ps["stratum_value"] == "complex") & (ps["strategy"] == s)].sort_values("step")
        ax.plot(sub["step"], sub[metric],
                color=COLORS[s], label=LABELS[s], linewidth=LINEWIDTHS[s],
                marker=MARKERS[s], markersize=4, markevery=3, linestyle=LINESTYLES[s])
    ax.set_xlabel("Active Prompting Step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if idx == 0:
        ax.legend(fontsize=9)

fig.suptitle("Complex Boundaries: BALD Achieves Best Calibration & Likelihood", fontsize=14, fontweight="bold")
plt.tight_layout()
savefig(fig, "S7_complex_boundary_ece_nll_curves")

# ============================================================
# PLOT S8: Grand summary — 4x3 panel (stratum type x metric)
# ============================================================
print("Plot S8: Grand 4x3 panel")
stypes = ["object_size_bin", "thinness_bin", "boundary_complexity_bin", "contrast_bin"]
metrics = [("iou_mean", "IoU", False), ("ece_mean", "ECE", True), ("nll_mean", "NLL", True)]

fig, axes = plt.subplots(4, 3, figsize=(16, 18))

for row, stype in enumerate(stypes):
    bins = STRATUM_ORDER[stype]
    for col, (metric, mlabel, lower_better) in enumerate(metrics):
        ax = axes[row, col]
        x = np.arange(len(bins))
        width = 0.19
        for i, s in enumerate(STRATS):
            vals = []
            for sv in bins:
                v = final[(final["stratum_type"] == stype) & (final["stratum_value"] == sv) & (final["strategy"] == s)][metric].values[0]
                vals.append(v)
            bars = ax.bar(x + i * width, vals, width, color=COLORS[s],
                         label=LABELS[s] if (row == 0 and col == 0) else None,
                         edgecolor="white")
            for j, v in enumerate(vals):
                all_v = [final[(final["stratum_type"] == stype) & (final["stratum_value"] == bins[j]) & (final["strategy"] == ss)][metric].values[0] for ss in STRATS]
                is_best = (v == min(all_v)) if lower_better else (v == max(all_v))
                if is_best:
                    bars[j].set_edgecolor("black")
                    bars[j].set_linewidth(2)
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels([b.capitalize() for b in bins], fontsize=9)
        ax.grid(True, alpha=0.15, axis="y")

        if row == 0:
            suffix = " \u2191" if not lower_better else " \u2193"
            ax.set_title(mlabel + suffix, fontsize=13, fontweight="bold")
        if col == 0:
            ax.set_ylabel(STRATUM_TITLES[stype], fontsize=12, fontweight="bold")

fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", ncol=4, fontsize=11,
           bbox_to_anchor=(0.5, 1.01))
fig.suptitle("Performance Across All Morphological Strata\n(bold outline = best in stratum)", fontsize=15, fontweight="bold", y=1.03)
plt.tight_layout()
savefig(fig, "S8_grand_panel_all_strata")

# ============================================================
# PLOT S9: ECE win count summary bar
# ============================================================
print("Plot S9: ECE win count bar")
wins = {s: 0 for s in STRATS}
total = 0
for stype in stypes:
    for sv in STRATUM_ORDER[stype]:
        vals = {}
        for s in STRATS:
            vals[s] = final[(final["stratum_type"] == stype) & (final["stratum_value"] == sv) & (final["strategy"] == s)]["ece_mean"].values[0]
        best = min(vals, key=vals.get)
        wins[best] += 1
        total += 1

fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar([LABELS[s] for s in STRATS], [wins[s] for s in STRATS],
              color=[COLORS[s] for s in STRATS], edgecolor="white")
# Bold the winner
best_idx = np.argmax([wins[s] for s in STRATS])
bars[best_idx].set_edgecolor("black")
bars[best_idx].set_linewidth(2.5)

for bar, s in zip(bars, STRATS):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
            f"{wins[s]}/{total}", ha="center", fontweight="bold", fontsize=12, color=COLORS[s])

ax.set_ylabel("Strata Won")
ax.set_title(f"ECE Wins Across {total} Morphological Strata")
ax.set_ylim(0, 13)
ax.grid(True, alpha=0.2, axis="y")
plt.xticks(rotation=10, ha="right")
plt.tight_layout()
savefig(fig, "S9_ece_win_count")

print(f"\nAll stratum plots saved to {OUT_DIR}")
