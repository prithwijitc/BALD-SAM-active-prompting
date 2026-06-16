#!/usr/bin/env python3
"""Budget-specific analysis plots for active prompting v3."""

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

BUDGETS = [3, 5, 10, 15, 19]
MEDICAL = ["Breast", "Polyp", "Skin"]


def savefig(fig, name):
    for ext in ["pdf", "png"]:
        fig.savefig(OUT_DIR / f"{name}.{ext}")
    print(f"  Saved {name}")
    plt.close(fig)


# ============================================================
# PLOT B1: Medical IoU at each budget — BALD #1 everywhere
# ============================================================
print("Plot B1: Medical IoU at budgets")
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(BUDGETS))
width = 0.18

for i, s in enumerate(STRATS):
    vals = []
    for b in BUDGETS:
        sub = pg[(pg["step"] == b) & (pg["group"].isin(MEDICAL)) & (pg["strategy"] == s)]
        vals.append(sub["iou_mean"].mean())
    bars = ax.bar(x + i * width, vals, width, label=LABELS[s], color=COLORS[s], edgecolor="white")
    # Star the best
    for j, v in enumerate(vals):
        all_v = []
        for ss in STRATS:
            sub2 = pg[(pg["step"] == BUDGETS[j]) & (pg["group"].isin(MEDICAL)) & (pg["strategy"] == ss)]
            all_v.append(sub2["iou_mean"].mean())
        if v == max(all_v):
            ax.text(x[j] + i * width, v + 0.005, "*", ha="center", fontsize=16, fontweight="bold", color=COLORS[s])

ax.set_xticks(x + width * 1.5)
ax.set_xticklabels([f"{b}" for b in BUDGETS])
ax.set_xlabel("Prompt Budget (# active steps)")
ax.set_ylabel("Mean IoU (Medical Domains)")
ax.set_title("Medical Domain: BALD (Laplace) Leads at Every Budget")
ax.legend(loc="lower right")
ax.grid(True, alpha=0.2, axis="y")
ax.set_ylim(0.45, 0.60)
savefig(fig, "B1_medical_iou_by_budget")

# ============================================================
# PLOT B2: ECE improvement from baseline at each budget
# ============================================================
print("Plot B2: ECE improvement by budget")
fig, ax = plt.subplots(figsize=(8, 5))
for s in STRATS:
    sub = df[df["strategy"] == s].sort_values("step")
    ece0 = sub.iloc[0]["ece_mean"]
    improvements = [ece0 - sub[sub["step"] == b]["ece_mean"].values[0] for b in range(0, 20)]
    ax.plot(range(0, 20), improvements,
            color=COLORS[s], label=LABELS[s], linewidth=2.5 if s == "bald_laplace" else 1.8,
            marker=MARKERS[s], markersize=5, markevery=3)

ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
ax.set_xlabel("Active Prompting Step (Budget)")
ax.set_ylabel("ECE Reduction from Baseline (higher = better)")
ax.set_title("Calibration Improvement: BALD Dominates at Higher Budgets")
ax.legend(loc="lower right")
ax.grid(True, alpha=0.3)

# Annotate crossover
ax.annotate("BALD takes the lead", xy=(12, 0.048), fontsize=10, color=COLORS["bald_laplace"],
            fontweight="bold")
savefig(fig, "B2_ece_improvement_by_budget")

# ============================================================
# PLOT B3: ECE group wins at each budget (bar)
# ============================================================
print("Plot B3: ECE wins vs entropy by budget")
all_budgets = list(range(0, 20))
bl_wins_ece = []
en_wins_ece = []
for b in all_budgets:
    sub = pg[pg["step"] == b]
    bl = sub[sub["strategy"] == "bald_laplace"].set_index("group")["ece_mean"]
    en = sub[sub["strategy"] == "entropy"].set_index("group")["ece_mean"]
    bl_wins_ece.append((bl < en).sum())
    en_wins_ece.append((bl > en).sum())

fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(all_budgets, bl_wins_ece, color=COLORS["bald_laplace"], alpha=0.8, label="BALD (Laplace) wins")
ax.bar(all_budgets, [-v for v in en_wins_ece], color=COLORS["entropy"], alpha=0.8, label="Entropy wins")
ax.axhline(0, color="black", linewidth=0.8)
ax.axhline(8.5, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
ax.axhline(-8.5, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
ax.text(0.5, 9.0, "majority", color="gray", fontsize=9)
ax.set_xlabel("Prompt Budget (# active steps)")
ax.set_ylabel("Groups Won (out of 17)")
ax.set_title("ECE Head-to-Head: BALD vs Entropy by Budget")
ax.legend(loc="upper left")
ax.set_ylim(-14, 14)
ax.set_yticks(range(-12, 14, 3))
ax.set_yticklabels([abs(y) for y in range(-12, 14, 3)])
ax.grid(True, alpha=0.2, axis="y")
savefig(fig, "B3_ece_wins_by_budget")

# ============================================================
# PLOT B4: Majority wins (2-of-3) at each budget
# ============================================================
print("Plot B4: Majority 2-of-3 wins by budget")
majority_bl = []
majority_en = []
for b in all_budgets:
    sub = pg[pg["step"] == b]
    bl = sub[sub["strategy"] == "bald_laplace"].set_index("group")
    en = sub[sub["strategy"] == "entropy"].set_index("group")
    w_iou = (bl["iou_mean"] > en["iou_mean"]).astype(int)
    w_ece = (bl["ece_mean"] < en["ece_mean"]).astype(int)
    w_nll = (bl["nll_mean"] < en["nll_mean"]).astype(int)
    majority_bl.append((w_iou + w_ece + w_nll >= 2).sum())
    majority_en.append(17 - (w_iou + w_ece + w_nll >= 2).sum())

fig, ax = plt.subplots(figsize=(9, 5))
ax.fill_between(all_budgets, majority_bl, alpha=0.3, color=COLORS["bald_laplace"])
ax.plot(all_budgets, majority_bl, color=COLORS["bald_laplace"], linewidth=2.5,
        marker="o", markersize=6, label="BALD (Laplace) wins majority")
ax.fill_between(all_budgets, majority_en, alpha=0.3, color=COLORS["entropy"])
ax.plot(all_budgets, majority_en, color=COLORS["entropy"], linewidth=2.0,
        marker="s", markersize=5, label="Entropy wins majority")
ax.axhline(8.5, color="gray", linewidth=1, linestyle="--", alpha=0.6)
ax.text(0.5, 8.8, "majority threshold", color="gray", fontsize=9)
ax.set_xlabel("Prompt Budget (# active steps)")
ax.set_ylabel("Groups (out of 17)")
ax.set_title("2-of-3 Metric Wins (IoU + ECE + NLL): BALD Gains With Budget")
ax.legend(loc="center right")
ax.grid(True, alpha=0.2, axis="y")
ax.set_ylim(0, 17)
ax.set_yticks(range(0, 18, 2))
savefig(fig, "B4_majority_wins_by_budget")

# ============================================================
# PLOT B5: Budget-stratified bar — aggregate ECE at key budgets
# ============================================================
print("Plot B5: Aggregate ECE at key budgets")
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(BUDGETS))
width = 0.18
for i, s in enumerate(STRATS):
    vals = [df[(df["step"] == b) & (df["strategy"] == s)]["ece_mean"].values[0] for b in BUDGETS]
    bars = ax.bar(x + i * width, vals, width, label=LABELS[s], color=COLORS[s], edgecolor="white")
    for j, v in enumerate(vals):
        all_v = [df[(df["step"] == BUDGETS[j]) & (df["strategy"] == ss)]["ece_mean"].values[0] for ss in STRATS]
        if v == min(all_v):
            bars[j].set_edgecolor("black")
            bars[j].set_linewidth(2)

ax.set_xticks(x + width * 1.5)
ax.set_xticklabels([f"{b}" for b in BUDGETS])
ax.set_xlabel("Prompt Budget")
ax.set_ylabel("ECE (lower is better)")
ax.set_title("Calibration (ECE) at Different Prompt Budgets")
ax.legend(loc="upper right")
ax.grid(True, alpha=0.2, axis="y")
savefig(fig, "B5_ece_at_budgets")

# ============================================================
# PLOT B6: The "complete picture" — 2x2 panel at key budgets
# ============================================================
print("Plot B6: 2x2 panel at budget 5 and 19")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for col, budget in enumerate([5, 19]):
    # Top row: IoU
    ax = axes[0, col]
    sub = pg[pg["step"] == budget]
    pivot = sub.pivot_table(index="group", columns="strategy", values="iou_mean")
    groups = sorted(pivot.index)
    x = np.arange(len(groups))
    w = 0.2
    for i, s in enumerate(STRATS):
        vals = [pivot.loc[g, s] for g in groups]
        ax.bar(x + i * w, vals, w, color=COLORS[s], label=LABELS[s] if col == 0 else None)
    ax.set_xticks(x + w * 1.5)
    ax.set_xticklabels(groups, rotation=55, ha="right", fontsize=8)
    ax.set_ylabel("IoU")
    ax.set_title(f"IoU at Budget = {budget}")
    ax.grid(True, alpha=0.2, axis="y")
    if col == 0:
        ax.legend(fontsize=8, loc="upper right")

    # Bottom row: ECE
    ax = axes[1, col]
    pivot_ece = sub.pivot_table(index="group", columns="strategy", values="ece_mean")
    for i, s in enumerate(STRATS):
        vals = [pivot_ece.loc[g, s] for g in groups]
        ax.bar(x + i * w, vals, w, color=COLORS[s])
    ax.set_xticks(x + w * 1.5)
    ax.set_xticklabels(groups, rotation=55, ha="right", fontsize=8)
    ax.set_ylabel("ECE (lower = better)")
    ax.set_title(f"ECE at Budget = {budget}")
    ax.grid(True, alpha=0.2, axis="y")

fig.suptitle("Per-Group Performance at Low (5) vs High (19) Prompt Budget", fontsize=15, fontweight="bold")
plt.tight_layout()
savefig(fig, "B6_panel_budget_5_vs_19")

# ============================================================
# PLOT B7: Medical IoU curves over all steps
# ============================================================
print("Plot B7: Medical IoU curves")
fig, ax = plt.subplots(figsize=(8, 5))
for s in STRATS:
    med_iou = []
    for step in range(20):
        sub = pg[(pg["step"] == step) & (pg["group"].isin(MEDICAL)) & (pg["strategy"] == s)]
        med_iou.append(sub["iou_mean"].mean())
    ax.plot(range(20), med_iou,
            color=COLORS[s], label=LABELS[s],
            linewidth=2.5 if s == "bald_laplace" else 1.8,
            marker=MARKERS[s], markersize=5, markevery=2)

ax.set_xlabel("Active Prompting Step")
ax.set_ylabel("Mean IoU (Medical: Breast, Polyp, Skin)")
ax.set_title("Medical Domains: BALD (Laplace) Consistently Leads")
ax.legend(loc="lower right")
ax.grid(True, alpha=0.3)
savefig(fig, "B7_medical_iou_curve")

# ============================================================
# PLOT B8: Cumulative ECE area (bar chart)
# ============================================================
print("Plot B8: Cumulative ECE reduction")
fig, ax = plt.subplots(figsize=(6, 4))
cum_ece = {}
for s in STRATS:
    sub = df[df["strategy"] == s].sort_values("step")
    ece0 = sub.iloc[0]["ece_mean"]
    cum = sum(ece0 - row["ece_mean"] for _, row in sub.iterrows())
    cum_ece[s] = cum

bars = ax.bar([LABELS[s] for s in STRATS], [cum_ece[s] for s in STRATS],
              color=[COLORS[s] for s in STRATS], edgecolor="white")
# Bold the winner
best_idx = np.argmax([cum_ece[s] for s in STRATS])
bars[best_idx].set_edgecolor("black")
bars[best_idx].set_linewidth(2.5)
ax.set_ylabel("Cumulative ECE Reduction\n(sum across all steps)")
ax.set_title("Total Calibration Improvement")
ax.grid(True, alpha=0.2, axis="y")
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
savefig(fig, "B8_cumulative_ece_reduction")

print(f"\nAll budget plots saved to {OUT_DIR}")
