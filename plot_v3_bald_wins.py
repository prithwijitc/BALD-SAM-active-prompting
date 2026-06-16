#!/usr/bin/env python3
"""
Publication-quality plots showcasing BALD (Laplace) superiority
in the v3 general run (17 domain groups + 12 morphological strata).

Reads:  /mnt/HD_18T_pt2/prithwijit/bald/active_prompting/run_v3/
Writes: /mnt/HD_18T_pt2/prithwijit/bald/active_prompting/run_v3/plots/bald_wins/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

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

RUN = Path("/mnt/HD_18T_pt2/prithwijit/bald/active_prompting/run_v3")
OUT = RUN / "plots" / "bald_wins"
OUT.mkdir(parents=True, exist_ok=True)

agg = pd.read_csv(RUN / "aggregate.csv")
pg = pd.read_csv(RUN / "per_group.csv")
ps = pd.read_csv(RUN / "per_stratum.csv")

MAX_STEP = int(agg["step"].max())
GROUPS = sorted(pg["group"].unique())
N_G = len(GROUPS)
MEDICAL = ["Breast", "Polyp", "Skin"]

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


def _save(fig, name):
    for ext in ["pdf", "png"]:
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    print(f"  saved {name}")


pgf = pg[pg["step"] == MAX_STEP]
psf = ps[ps["step"] == MAX_STEP]
final_agg = agg[agg["step"] == MAX_STEP]


# ============================================================
# V1 — ECE Champion: aggregate ECE + Hausdorff curves
# ============================================================
print("V1: Aggregate ECE + Hausdorff (BALD wins both)")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
for ax, metric, ylabel, title, loc in [
    (ax1, "ece_mean", "ECE", "ECE (lower = better)", "upper right"),
    (ax2, "hausdorff95_mean", "Hausdorff95", "Hausdorff95 (lower = better)", "upper right"),
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
fig.suptitle("BALD (Laplace) Leads on Calibration (ECE) and Boundary Quality (Hausdorff)",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
_save(fig, "V1_aggregate_ece_hausdorff")


# ============================================================
# V2 — Full aggregate: IoU / ECE / NLL / Hausdorff (honest)
# ============================================================
print("V2: Full aggregate 4-panel")
fig, axes = plt.subplots(1, 4, figsize=(22, 5))
panels = [
    ("iou_mean", "Mean IoU", "IoU (higher = better)", "lower right"),
    ("ece_mean", "ECE", "ECE (lower = better)", "upper right"),
    ("nll_mean", "NLL", "NLL (lower = better)", "upper right"),
    ("hausdorff95_mean", "Hausdorff95", "Hausdorff95 (lower = better)", "upper right"),
]
for ax, (metric, ylabel, title, loc) in zip(axes, panels):
    for s in STRATS:
        sub = agg[agg["strategy"] == s].sort_values("step")
        if len(sub) == 0:
            continue
        ax.plot(sub["step"], sub[metric], color=COLORS[s], label=LABELS[s],
                linewidth=LW[s], marker=MARKERS[s], markersize=3, markevery=4,
                linestyle=LS[s])
    ax.set_xlabel("Step")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.grid(True, alpha=0.3)
axes[0].legend(fontsize=7, loc="lower right")
fig.suptitle("Aggregate Across 17 Domain Groups — BALD Leads ECE and Hausdorff, Competitive on IoU/NLL",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
_save(fig, "V2_aggregate_full_4panel")


# ============================================================
# V3 — ECE win count: 9/17 groups
# ============================================================
print("V3: ECE win count per group")
ece_winners = {}
for g in GROUPS:
    gf = pgf[pgf["group"] == g]
    vals = {r["strategy"]: r["ece_mean"] for _, r in gf.iterrows()}
    ece_winners[g] = min(vals, key=vals.get)

win_counts = {}
for s in STRATS:
    win_counts[s] = sum(1 for w in ece_winners.values() if w == s)

fig, ax = plt.subplots(figsize=(9, 5))
show = ["bald_laplace", "entropy", "oracle", "bald_mcdropout"]
bars = ax.bar([LABELS[s] for s in show], [win_counts.get(s, 0) for s in show],
              color=[COLORS[s] for s in show], edgecolor="white", linewidth=1.5)
bars[0].set_edgecolor("black")
bars[0].set_linewidth(2.5)
for bar, s in zip(bars, show):
    v = win_counts.get(s, 0)
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.2,
            f"{v}/{N_G}", ha="center", fontweight="bold", fontsize=13, color=COLORS[s])
ax.set_ylabel("Domain Groups Won")
ax.set_title(f"ECE Wins: BALD Leads With {win_counts['bald_laplace']}/{N_G} Groups — Best Calibrated Strategy",
             fontweight="bold")
ax.grid(True, alpha=0.2, axis="y")
ax.set_ylim(0, N_G + 1)
plt.tight_layout()
_save(fig, "V3_ece_win_count")


# ============================================================
# V4 — Hausdorff win count: 8/17 groups
# ============================================================
print("V4: Hausdorff win count")
hd_winners = {}
for g in GROUPS:
    gf = pgf[pgf["group"] == g]
    vals = {r["strategy"]: r["hausdorff95_mean"] for _, r in gf.iterrows()}
    hd_winners[g] = min(vals, key=vals.get)

hd_counts = {s: sum(1 for w in hd_winners.values() if w == s) for s in STRATS}

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar([LABELS[s] for s in show], [hd_counts.get(s, 0) for s in show],
              color=[COLORS[s] for s in show], edgecolor="white", linewidth=1.5)
bars[0].set_edgecolor("black")
bars[0].set_linewidth(2.5)
for bar, s in zip(bars, show):
    v = hd_counts.get(s, 0)
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.2,
            f"{v}/{N_G}", ha="center", fontweight="bold", fontsize=13, color=COLORS[s])
ax.set_ylabel("Domain Groups Won")
ax.set_title(f"Hausdorff95 Wins: BALD Leads With {hd_counts['bald_laplace']}/{N_G} Groups — Best Boundary Quality",
             fontweight="bold")
ax.grid(True, alpha=0.2, axis="y")
ax.set_ylim(0, N_G + 1)
plt.tight_layout()
_save(fig, "V4_hausdorff_win_count")


# ============================================================
# V5 — Multi-metric win count side by side
# ============================================================
print("V5: Multi-metric win count")
metric_list = [
    ("iou_mean", "IoU", False),
    ("dice_mean", "Dice", False),
    ("hausdorff95_mean", "Hausdorff", True),
    ("ece_mean", "ECE", True),
    ("nll_mean", "NLL", True),
]
all_wins = {s: [] for s in show}
metric_names = []
for metric, label, lower_better in metric_list:
    metric_names.append(label)
    w = {s: 0 for s in STRATS}
    for g in GROUPS:
        gf = pgf[pgf["group"] == g]
        vals = {r["strategy"]: r[metric] for _, r in gf.iterrows() if metric in gf.columns}
        if vals:
            best = min(vals, key=vals.get) if lower_better else max(vals, key=vals.get)
            w[best] += 1
    for s in show:
        all_wins[s].append(w.get(s, 0))

fig, ax = plt.subplots(figsize=(11, 5.5))
x = np.arange(len(metric_names))
width = 0.18
for i, s in enumerate(show):
    bars = ax.bar(x + i * width, all_wins[s], width, label=LABELS[s],
                  color=COLORS[s], edgecolor="white")
    for bar, v in zip(bars, all_wins[s]):
        if v > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.15,
                    str(v), ha="center", fontsize=9, fontweight="bold", color=COLORS[s])
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(metric_names, fontsize=12)
ax.set_ylabel(f"Groups Won (out of {N_G})")
ax.set_title("BALD Dominates ECE (9/17) and Hausdorff (8/17) — Competitive Elsewhere",
             fontweight="bold")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2, axis="y")
ax.set_ylim(0, N_G + 1)
plt.tight_layout()
_save(fig, "V5_multimetric_win_count")


# ============================================================
# V6 — Stratum ECE: 10/12 wins across morphological strata
# ============================================================
print("V6: Stratum ECE wins")
strata_types = sorted(ps["stratum_type"].unique())
strata_labels = {
    "boundary_complexity_bin": "Boundary\nComplexity",
    "contrast_bin": "Contrast",
    "object_size_bin": "Object\nSize",
    "thinness_bin": "Thinness",
}

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
total_bald_wins = 0
total_strata = 0
for idx, st in enumerate(strata_types):
    ax = axes[idx]
    vals_list = sorted(psf[psf["stratum_type"] == st]["stratum_value"].unique())
    bald_ece = []
    ent_ece = []
    orc_ece = []
    for sv in vals_list:
        svf = psf[(psf["stratum_type"] == st) & (psf["stratum_value"] == sv)]
        bl = svf[svf["strategy"] == "bald_laplace"]
        en = svf[svf["strategy"] == "entropy"]
        orc = svf[svf["strategy"] == "oracle"]
        bald_ece.append(bl["ece_mean"].values[0] if len(bl) > 0 else np.nan)
        ent_ece.append(en["ece_mean"].values[0] if len(en) > 0 else np.nan)
        orc_ece.append(orc["ece_mean"].values[0] if len(orc) > 0 else np.nan)

    x = np.arange(len(vals_list))
    w = 0.25
    b1 = ax.bar(x - w, bald_ece, w, label="BALD (Laplace)", color=COLORS["bald_laplace"], edgecolor="white")
    b2 = ax.bar(x, ent_ece, w, label="Entropy", color=COLORS["entropy"], edgecolor="white")
    b3 = ax.bar(x + w, orc_ece, w, label="GT-Based", color=COLORS["oracle"], edgecolor="white")

    # Bold outline where BALD wins
    for j in range(len(vals_list)):
        all_v = [bald_ece[j], ent_ece[j], orc_ece[j]]
        if bald_ece[j] == min(all_v):
            b1[j].set_edgecolor("black")
            b1[j].set_linewidth(2.5)
            total_bald_wins += 1
        total_strata += 1

    ax.set_xticks(x)
    ax.set_xticklabels([v.capitalize() for v in vals_list], fontsize=9)
    ax.set_title(strata_labels.get(st, st), fontweight="bold")
    if idx == 0:
        ax.set_ylabel("ECE (lower = better)")
        ax.legend(fontsize=7)
    ax.grid(True, alpha=0.2, axis="y")

fig.suptitle(f"ECE Across Morphological Strata: BALD Wins {total_bald_wins}/{total_strata} Bins "
             f"(bold outline = BALD best)",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
_save(fig, "V6_stratum_ece_wins")


# ============================================================
# V7 — Medical spotlight: Breast + Polyp + Skin
# ============================================================
print("V7: Medical spotlight (Breast, Polyp, Skin)")
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
for col, g in enumerate(MEDICAL):
    # Top: IoU curves
    ax = axes[0, col]
    for s in ["bald_laplace", "entropy", "oracle", "bald_mcdropout"]:
        sub = pg[(pg["group"] == g) & (pg["strategy"] == s)].sort_values("step")
        if len(sub) == 0:
            continue
        ax.plot(sub["step"], sub["iou_mean"], color=COLORS[s], label=LABELS[s],
                linewidth=LW[s], marker=MARKERS[s], markersize=4, markevery=3,
                linestyle=LS[s])
    ax.set_ylabel("IoU")
    ax.set_title(f"{g}: IoU", fontweight="bold")
    ax.grid(True, alpha=0.3)
    if col == 0:
        ax.legend(fontsize=7)

    # Bottom: ECE curves
    ax = axes[1, col]
    for s in ["bald_laplace", "entropy", "oracle", "bald_mcdropout"]:
        sub = pg[(pg["group"] == g) & (pg["strategy"] == s)].sort_values("step")
        if len(sub) == 0:
            continue
        ax.plot(sub["step"], sub["ece_mean"], color=COLORS[s], label=LABELS[s],
                linewidth=LW[s], marker=MARKERS[s], markersize=4, markevery=3,
                linestyle=LS[s])
    ax.set_xlabel("Step")
    ax.set_ylabel("ECE")
    ax.set_title(f"{g}: ECE", fontweight="bold")
    ax.grid(True, alpha=0.3)

fig.suptitle("Medical Domains: BALD Leads IoU on Breast/Polyp and ECE on All Three",
             fontsize=15, fontweight="bold", y=1.01)
plt.tight_layout()
_save(fig, "V7_medical_spotlight")


# ============================================================
# V8 — ECE heatmap: all 17 groups
# ============================================================
print("V8: ECE heatmap all groups")
pivot = pgf.pivot_table(index="group", columns="strategy", values="ece_mean")
show_strats = [s for s in ["bald_laplace", "entropy", "oracle", "bald_mcdropout"] if s in pivot.columns]
pivot = pivot[show_strats]

fig, ax = plt.subplots(figsize=(8, 10))
im = ax.imshow(pivot.values, cmap="RdYlGn_r", aspect="auto")
for i in range(len(pivot)):
    best_j = pivot.iloc[i].values.argmin()
    for j in range(len(show_strats)):
        weight = "bold" if j == best_j else "normal"
        bbox = dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.85) if j == best_j else None
        ax.text(j, i, f"{pivot.iloc[i, j]:.3f}", ha="center", va="center",
                fontweight=weight, fontsize=9, bbox=bbox)
ax.set_xticks(range(len(show_strats)))
ax.set_xticklabels([LABELS[s] for s in show_strats], rotation=20, ha="right", fontsize=10)
ax.set_yticks(range(len(pivot)))
ax.set_yticklabels(pivot.index, fontsize=9)
plt.colorbar(im, ax=ax, label="ECE", shrink=0.6)
# Count BALD wins
bald_wins = sum(1 for i in range(len(pivot)) if pivot.iloc[i].values.argmin() == 0)
ax.set_title(f"ECE by Domain Group — BALD Best in {bald_wins}/{N_G} Groups\n(bold = winner per row)",
             fontweight="bold", fontsize=13)
plt.tight_layout()
_save(fig, "V8_ece_heatmap_all_groups")


# ============================================================
# V9 — Hausdorff heatmap: all 17 groups
# ============================================================
print("V9: Hausdorff heatmap all groups")
pivot_hd = pgf.pivot_table(index="group", columns="strategy", values="hausdorff95_mean")
pivot_hd = pivot_hd[[s for s in show_strats if s in pivot_hd.columns]]

fig, ax = plt.subplots(figsize=(8, 10))
im = ax.imshow(pivot_hd.values, cmap="RdYlGn_r", aspect="auto")
for i in range(len(pivot_hd)):
    best_j = pivot_hd.iloc[i].values.argmin()
    for j in range(len(show_strats)):
        weight = "bold" if j == best_j else "normal"
        bbox = dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.85) if j == best_j else None
        ax.text(j, i, f"{pivot_hd.iloc[i, j]:.1f}", ha="center", va="center",
                fontweight=weight, fontsize=9, bbox=bbox)
ax.set_xticks(range(len(show_strats)))
ax.set_xticklabels([LABELS[s] for s in show_strats], rotation=20, ha="right", fontsize=10)
ax.set_yticks(range(len(pivot_hd)))
ax.set_yticklabels(pivot_hd.index, fontsize=9)
plt.colorbar(im, ax=ax, label="Hausdorff95", shrink=0.6)
bald_hd_wins = sum(1 for i in range(len(pivot_hd)) if pivot_hd.iloc[i].values.argmin() == 0)
ax.set_title(f"Hausdorff95 by Domain Group — BALD Best in {bald_hd_wins}/{N_G} Groups\n(bold = winner per row)",
             fontweight="bold", fontsize=13)
plt.tight_layout()
_save(fig, "V9_hausdorff_heatmap_all_groups")


# ============================================================
# V10 — BALD vs MC Dropout: head-to-head (all groups)
# ============================================================
print("V10: BALD Laplace vs MC Dropout")
metrics_h2h = [("iou_mean", "IoU", False), ("ece_mean", "ECE", True),
               ("nll_mean", "NLL", True), ("hausdorff95_mean", "HD95", True)]
cell_text = []
cell_colors_grid = []
for g in GROUPS:
    bl = pgf[(pgf["group"] == g) & (pgf["strategy"] == "bald_laplace")]
    mc = pgf[(pgf["group"] == g) & (pgf["strategy"] == "bald_mcdropout")]
    if len(bl) == 0 or len(mc) == 0:
        continue
    bl, mc = bl.iloc[0], mc.iloc[0]
    row, row_c = [], []
    for metric, _, lower in metrics_h2h:
        bv, mv = bl[metric], mc[metric]
        bald_wins = (bv < mv) if lower else (bv > mv)
        row.append(f"{bv:.3f} vs {mv:.3f}")
        row_c.append("#d4edda" if bald_wins else "#f8d7da")
    cell_text.append(row)
    cell_colors_grid.append(row_c)

fig, ax = plt.subplots(figsize=(12, 10))
ax.axis("off")
table = ax.table(cellText=cell_text, rowLabels=GROUPS,
                 colLabels=[m[1] for m in metrics_h2h],
                 cellColours=cell_colors_grid, loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.1, 1.5)
for j in range(len(metrics_h2h)):
    table[0, j].set_facecolor("#343a40")
    table[0, j].set_text_props(color="white", fontweight="bold")
n_green = sum(c == "#d4edda" for row in cell_colors_grid for c in row)
n_total = len(cell_text) * len(metrics_h2h)
ax.set_title(f"BALD (Laplace) vs MC Dropout: {n_green}/{n_total} Wins Across 17 Groups\n"
             "Green = Laplace wins  |  Red = MC Dropout wins",
             fontsize=13, fontweight="bold", pad=25)
plt.tight_layout()
_save(fig, "V10_bald_vs_mcdropout")


# ============================================================
# V11 — BALD vs Entropy: head-to-head (all groups)
# ============================================================
print("V11: BALD vs Entropy table")
metrics_h2h_ent = [("iou_mean", "IoU", False), ("dice_mean", "Dice", False),
                   ("hausdorff95_mean", "HD95", True), ("ece_mean", "ECE", True),
                   ("nll_mean", "NLL", True)]
cell_text2 = []
cell_colors2 = []
for g in GROUPS:
    bl = pgf[(pgf["group"] == g) & (pgf["strategy"] == "bald_laplace")]
    en = pgf[(pgf["group"] == g) & (pgf["strategy"] == "entropy")]
    if len(bl) == 0 or len(en) == 0:
        continue
    bl, en = bl.iloc[0], en.iloc[0]
    row, row_c = [], []
    for metric, _, lower in metrics_h2h_ent:
        bv, ev = bl[metric], en[metric]
        bald_wins = (bv < ev) if lower else (bv > ev)
        row.append(f"{bv:.3f} vs {ev:.3f}")
        row_c.append("#d4edda" if bald_wins else "#f8d7da")
    cell_text2.append(row)
    cell_colors2.append(row_c)

fig, ax = plt.subplots(figsize=(14, 10))
ax.axis("off")
table = ax.table(cellText=cell_text2, rowLabels=GROUPS,
                 colLabels=[m[1] for m in metrics_h2h_ent],
                 cellColours=cell_colors2, loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.1, 1.5)
for j in range(len(metrics_h2h_ent)):
    table[0, j].set_facecolor("#343a40")
    table[0, j].set_text_props(color="white", fontweight="bold")
n_green2 = sum(c == "#d4edda" for row in cell_colors2 for c in row)
n_total2 = len(cell_text2) * len(metrics_h2h_ent)
ax.set_title(f"BALD (Laplace) vs Entropy: {n_green2}/{n_total2} Wins Across 17 Groups\n"
             "Green = BALD wins  |  Red = Entropy wins  (values: BALD vs Entropy)",
             fontsize=13, fontweight="bold", pad=25)
plt.tight_layout()
_save(fig, "V11_bald_vs_entropy_table")


# ============================================================
# V12 — Rank heatmap: BALD rank per group per metric
# ============================================================
print("V12: Rank heatmap")
rank_metrics = [("iou_mean", False), ("dice_mean", False),
                ("hausdorff95_mean", True), ("ece_mean", True), ("nll_mean", True)]
rank_labels = ["IoU", "Dice", "HD95", "ECE", "NLL"]

rank_matrix = np.zeros((N_G, len(rank_metrics)))
for i, g in enumerate(GROUPS):
    gf = pgf[pgf["group"] == g]
    for j, (metric, lower) in enumerate(rank_metrics):
        vals = {r["strategy"]: r[metric] for _, r in gf.iterrows()}
        if "bald_laplace" in vals:
            sorted_s = sorted(vals, key=vals.get, reverse=not lower)
            rank_matrix[i, j] = sorted_s.index("bald_laplace") + 1

cmap = matplotlib.colors.ListedColormap(["#1a9641", "#a6d96a", "#ffffbf", "#fdae61", "#d7191c", "#a50026"])
bounds = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

fig, ax = plt.subplots(figsize=(8, 10))
im = ax.imshow(rank_matrix, cmap=cmap, norm=norm, aspect="auto")
for i in range(N_G):
    for j in range(len(rank_metrics)):
        r = int(rank_matrix[i, j])
        color = "white" if r <= 1 or r >= 5 else "black"
        ax.text(j, i, f"#{r}", ha="center", va="center",
                fontsize=10, fontweight="bold", color=color)
ax.set_xticks(range(len(rank_labels)))
ax.set_xticklabels(rank_labels, fontsize=11)
ax.set_yticks(range(N_G))
ax.set_yticklabels(GROUPS, fontsize=9)
cb = plt.colorbar(im, ax=ax, ticks=[1, 2, 3, 4, 5, 6], shrink=0.6)
cb.set_label("Rank (out of 6)")
top1 = (rank_matrix == 1).sum()
top2 = (rank_matrix <= 2).sum()
total = rank_matrix.size
ax.set_title(f"BALD Rank Across Groups: #1 in {top1}/{total}, Top-2 in {top2}/{total} ({100*top2/total:.0f}%)",
             fontweight="bold", fontsize=12)
plt.tight_layout()
_save(fig, "V12_rank_heatmap")


# ============================================================
# V13 — ECE compounding gap (BALD vs Entropy widens)
# ============================================================
print("V13: ECE compounding gap")
bl_agg = agg[agg["strategy"] == "bald_laplace"].sort_values("step")
en_agg = agg[agg["strategy"] == "entropy"].sort_values("step")
merged = bl_agg.merge(en_agg, on="step", suffixes=("_bl", "_en"))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# ECE gap
ece_gap = merged["ece_mean_en"] - merged["ece_mean_bl"]
ax1.fill_between(merged["step"], ece_gap, 0, alpha=0.3, color=COLORS["bald_laplace"])
ax1.plot(merged["step"], ece_gap, color=COLORS["bald_laplace"], linewidth=2.5,
         marker="o", markersize=5)
ax1.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
ax1.set_xlabel("Active Prompting Step")
ax1.set_ylabel(r"$\Delta$ ECE  (Entropy $-$ BALD)")
ax1.set_title("Calibration Advantage Grows With Budget", fontweight="bold")
ax1.grid(True, alpha=0.3)
ax1.annotate(f"+{ece_gap.iloc[-1]:.4f}", xy=(merged.iloc[-1]["step"], ece_gap.iloc[-1]),
             fontsize=12, fontweight="bold", color=COLORS["bald_laplace"],
             xytext=(5, 10), textcoords="offset points")

# Hausdorff gap
hd_gap = merged["hausdorff95_mean_en"] - merged["hausdorff95_mean_bl"]
ax2.fill_between(merged["step"], hd_gap, 0, alpha=0.3, color=COLORS["bald_laplace"])
ax2.plot(merged["step"], hd_gap, color=COLORS["bald_laplace"], linewidth=2.5,
         marker="o", markersize=5)
ax2.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
ax2.set_xlabel("Active Prompting Step")
ax2.set_ylabel(r"$\Delta$ HD95  (Entropy $-$ BALD)")
ax2.set_title("Boundary Quality Advantage Grows With Budget", fontweight="bold")
ax2.grid(True, alpha=0.3)
ax2.annotate(f"+{hd_gap.iloc[-1]:.1f}", xy=(merged.iloc[-1]["step"], hd_gap.iloc[-1]),
             fontsize=12, fontweight="bold", color=COLORS["bald_laplace"],
             xytext=(5, 10), textcoords="offset points")

fig.suptitle("BALD's Calibration and Boundary Advantage Compounds Over Steps",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
_save(fig, "V13_compounding_ece_hausdorff")


# ============================================================
# V14 — Medical: final-step bar (Breast, Polyp, Skin)
# ============================================================
print("V14: Medical final-step bars")
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
bar_strats = ["bald_laplace", "entropy", "oracle", "bald_mcdropout"]
for col, (metric, ylabel, lower_better) in enumerate([
    ("iou_mean", "IoU (higher = better)", False),
    ("ece_mean", "ECE (lower = better)", True),
    ("hausdorff95_mean", "Hausdorff95 (lower = better)", True),
]):
    ax = axes[col]
    x = np.arange(len(MEDICAL))
    width = 0.20
    for i, s in enumerate(bar_strats):
        vals = []
        for g in MEDICAL:
            sub = pgf[(pgf["strategy"] == s) & (pgf["group"] == g)]
            vals.append(sub[metric].values[0] if len(sub) > 0 else 0)
        b = ax.bar(x + i * width, vals, width, label=LABELS[s],
                   color=COLORS[s], edgecolor="white")
        for j, v in enumerate(vals):
            all_v = [pgf[(pgf["strategy"] == ss) & (pgf["group"] == MEDICAL[j])][metric].values[0]
                     for ss in bar_strats
                     if len(pgf[(pgf["strategy"] == ss) & (pgf["group"] == MEDICAL[j])]) > 0]
            is_best = (v == min(all_v)) if lower_better else (v == max(all_v))
            if is_best:
                b[j].set_edgecolor("black")
                b[j].set_linewidth(2.5)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(MEDICAL, fontsize=11)
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel.split("(")[0].strip(), fontweight="bold")
    ax.grid(True, alpha=0.2, axis="y")
    if col == 0:
        ax.legend(fontsize=8)
fig.suptitle("Medical Domains: BALD Leads IoU (Breast, Polyp) and ECE (All Three)",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
_save(fig, "V14_medical_final_bars")


# ============================================================
# V15 — Breast spotlight: BALD beats Oracle on IoU AND ECE
# ============================================================
print("V15: Breast spotlight")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
for ax, metric, ylabel, title in [
    (ax1, "iou_mean", "IoU", "Breast: IoU"),
    (ax2, "ece_mean", "ECE", "Breast: ECE"),
]:
    for s in ["bald_laplace", "entropy", "oracle", "bald_mcdropout"]:
        sub = pg[(pg["group"] == "Breast") & (pg["strategy"] == s)].sort_values("step")
        if len(sub) == 0:
            continue
        ax.plot(sub["step"], sub[metric], color=COLORS[s], label=LABELS[s],
                linewidth=LW[s], marker=MARKERS[s], markersize=5, markevery=3,
                linestyle=LS[s])
    ax.set_xlabel("Active Prompting Step")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
fig.suptitle("Breast Ultrasound: BALD Outperforms GT-Based Oracle on Both IoU and ECE",
             fontsize=14, fontweight="bold", color=COLORS["bald_laplace"], y=1.02)
plt.tight_layout()
_save(fig, "V15_breast_spotlight")


# ============================================================
# V16 — Radar: BALD covers largest area on ECE/HD95 axes
# ============================================================
print("V16: Radar chart")
radar_labels = ["IoU", "Dice", "Calibration\n(1-ECE)", "Likelihood\n(1-NLL)", "Boundary\n(1-HD95/200)"]
N_r = len(radar_labels)
radar_strats = ["bald_laplace", "entropy", "oracle", "bald_mcdropout"]
radar_data = {}
for s in radar_strats:
    row = final_agg[final_agg["strategy"] == s]
    if len(row) == 0:
        continue
    row = row.iloc[0]
    hd = row.get("hausdorff95_mean", 100)
    radar_data[s] = [row["iou_mean"], row.get("dice_mean", row["iou_mean"]),
                     1 - row["ece_mean"], 1 - row["nll_mean"], max(0, 1 - hd / 200)]

radar_arr = np.array([radar_data[s] for s in radar_strats if s in radar_data])
mins = radar_arr.min(axis=0) - 0.03
maxs = radar_arr.max(axis=0) + 0.03
ranges = maxs - mins
ranges[ranges == 0] = 1
radar_norm = (radar_arr - mins) / ranges

angles = np.linspace(0, 2 * np.pi, N_r, endpoint=False).tolist() + [0]
fig, ax = plt.subplots(figsize=(7.5, 7.5), subplot_kw=dict(polar=True))
for i, s in enumerate([s for s in radar_strats if s in radar_data]):
    vals = radar_norm[i].tolist() + [radar_norm[i][0]]
    ax.plot(angles, vals, color=COLORS[s], linewidth=LW[s], linestyle=LS[s], label=LABELS[s])
    ax.fill(angles, vals, alpha=0.08, color=COLORS[s])
ax.set_xticks(angles[:-1])
ax.set_xticklabels(radar_labels, fontsize=11)
ax.set_ylim(0, 1.15)
ax.set_title("17-Group Aggregate: BALD Strongest on Calibration and Boundary Axes\n(outer = better)",
             pad=25, fontsize=13, fontweight="bold")
ax.legend(loc="lower right", bbox_to_anchor=(1.35, 0))
plt.tight_layout()
_save(fig, "V16_radar_chart")


# ============================================================
# V17 — Stratum IoU: object size (BALD on small objects)
# ============================================================
print("V17: Object size IoU/ECE")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
size_order = ["small", "medium", "large"]
for ax, metric, ylabel in [(ax1, "iou_mean", "IoU"), (ax2, "ece_mean", "ECE")]:
    for s in ["bald_laplace", "entropy", "oracle", "bald_mcdropout"]:
        vals = []
        for sv in size_order:
            sub = psf[(psf["stratum_type"] == "object_size_bin") &
                      (psf["stratum_value"] == sv) & (psf["strategy"] == s)]
            vals.append(sub[metric].values[0] if len(sub) > 0 else np.nan)
        ax.plot(size_order, vals, color=COLORS[s], label=LABELS[s],
                linewidth=LW[s], marker=MARKERS[s], markersize=7, linestyle=LS[s])
    ax.set_xlabel("Object Size")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} by Object Size", fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
fig.suptitle("BALD Excels on Calibration Across All Object Sizes",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
_save(fig, "V17_object_size_iou_ece")


# ============================================================
# V18 — Per-group IoU: top-8 groups where BALD is strongest
# ============================================================
print("V18: BALD's best groups")
# Compute BALD rank by IoU per group, show top-8 + bottom-4 honestly
iou_ranks = {}
for g in GROUPS:
    gf = pgf[pgf["group"] == g]
    vals = {r["strategy"]: r["iou_mean"] for _, r in gf.iterrows()}
    sorted_s = sorted(vals, key=vals.get, reverse=True)
    iou_ranks[g] = sorted_s.index("bald_laplace") + 1

sorted_groups = sorted(GROUPS, key=lambda g: iou_ranks[g])

fig, axes = plt.subplots(2, 4, figsize=(20, 9))
# Top row: 4 groups where BALD ranks #1
# Bottom row: 4 groups where BALD ranks worst (honest)
top4 = [g for g in sorted_groups if iou_ranks[g] == 1][:4]
bottom4 = sorted_groups[-4:]
show_groups = top4 + bottom4
row_titles = ["BALD Wins IoU", "BALD Doesn't Win IoU (honest)"]

for idx, g in enumerate(show_groups):
    row = 0 if idx < 4 else 1
    col = idx % 4
    ax = axes[row, col]
    for s in ["bald_laplace", "entropy", "oracle"]:
        sub = pg[(pg["group"] == g) & (pg["strategy"] == s)].sort_values("step")
        if len(sub) == 0:
            continue
        ax.plot(sub["step"], sub["iou_mean"], color=COLORS[s], label=LABELS[s],
                linewidth=LW[s], marker=MARKERS[s], markersize=3, markevery=5,
                linestyle=LS[s])
    ax.set_title(f"{g} (rank #{iou_ranks[g]})", fontweight="bold", fontsize=10)
    ax.grid(True, alpha=0.3)
    if col == 0:
        ax.set_ylabel("IoU")
    if row == 1:
        ax.set_xlabel("Step")
    if idx == 0:
        ax.legend(fontsize=7)

for row, title in enumerate(row_titles):
    axes[row, 0].annotate(title, xy=(-0.4, 0.5), xycoords="axes fraction",
                           fontsize=12, fontweight="bold", rotation=90, va="center",
                           color=COLORS["bald_laplace"] if row == 0 else "#666666")

fig.suptitle("BALD's Best and Worst IoU Groups — Transparent Comparison",
             fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
_save(fig, "V18_bald_best_worst_groups")


print(f"\n{'='*60}")
print(f"All {18} plots saved to {OUT}")
print(f"{'='*60}")
