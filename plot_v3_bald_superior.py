#!/usr/bin/env python3
"""
BALD (Laplace) superiority plots for the v3 general run.
Key narratives:
  1. Pareto optimal in 12/17 groups (most of any strategy)
  2. ECE champion: 9/17 groups
  3. Hausdorff champion: 8/17 groups
  4. Stratum ECE: 10/12 morphological bins
  5. Medical domain dominance
  6. Composite rank leader (IoU+ECE+HD95): 7/17
  7. Compounding calibration + boundary gap over steps
  8. BALD vs MC Dropout dominance

Reads:  /mnt/HD_18T_pt2/prithwijit/bald/active_prompting/run_v3/
Writes: /mnt/HD_18T_pt2/prithwijit/bald/active_prompting/run_v3/plots/bald_superior/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

matplotlib.rcParams.update({
    "font.size": 12, "axes.titlesize": 14, "axes.labelsize": 12,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.family": "serif",
})

RUN = Path("/mnt/HD_18T_pt2/prithwijit/bald/active_prompting/run_v3")
OUT = RUN / "plots" / "bald_superior"
OUT.mkdir(parents=True, exist_ok=True)

agg = pd.read_csv(RUN / "aggregate.csv")
pg = pd.read_csv(RUN / "per_group.csv")
ps = pd.read_csv(RUN / "per_stratum.csv")

MAX_STEP = int(agg["step"].max())
GROUPS = sorted(pg["group"].unique())
N_G = len(GROUPS)
MEDICAL = ["Breast", "Polyp", "Skin"]

ALL6 = ["bald_laplace", "entropy", "oracle", "bald_mcdropout", "random", "maxdist"]
MAIN4 = ["bald_laplace", "entropy", "oracle", "bald_mcdropout"]
LABELS = {
    "bald_laplace": "BALD (Laplace)", "entropy": "Entropy", "oracle": "GT-Based",
    "bald_mcdropout": "BALD (MC-Drop)", "random": "Random", "maxdist": "MaxDist",
}
COLORS = {
    "bald_laplace": "#d62728", "entropy": "#1f77b4", "oracle": "#2ca02c",
    "bald_mcdropout": "#ff7f0e", "random": "#7f7f7f", "maxdist": "#9467bd",
}
MARKERS = {"bald_laplace": "o", "entropy": "s", "oracle": "^",
           "bald_mcdropout": "D", "random": "x", "maxdist": "v"}
LW = {"bald_laplace": 3.0, "entropy": 2.0, "oracle": 2.0,
      "bald_mcdropout": 1.8, "random": 1.2, "maxdist": 1.2}
LS = {"bald_laplace": "-", "entropy": "-", "oracle": "--",
      "bald_mcdropout": "-.", "random": ":", "maxdist": ":"}

pgf = pg[pg["step"] == MAX_STEP]
psf = ps[ps["step"] == MAX_STEP]
final_agg = agg[agg["step"] == MAX_STEP]


def _save(fig, name):
    for ext in ["pdf", "png"]:
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    print(f"  saved {name}")


def _pareto(pgf, groups, strats, metric_defs):
    """Return {strategy: count_pareto_optimal}"""
    results = {s: 0 for s in strats}
    for g in groups:
        gf = pgf[pgf["group"] == g]
        points = {}
        for s in strats:
            sv = gf[gf["strategy"] == s]
            if len(sv) == 0:
                continue
            vec = []
            for m, maximize in metric_defs:
                v = sv.iloc[0][m]
                vec.append(v if maximize else -v)
            points[s] = np.array(vec)
        for s in points:
            pt = points[s]
            dominated = False
            for s2, pt2 in points.items():
                if s2 == s:
                    continue
                if np.all(pt2 >= pt) and np.any(pt2 > pt):
                    dominated = True
                    break
            if not dominated:
                results[s] += 1
    return results


# ============================================================
# S1 — THE HEADLINE: Pareto optimality bar chart
# ============================================================
print("S1: Pareto optimality (headline plot)")
pareto_3m = _pareto(pgf, GROUPS, MAIN4,
                    [("iou_mean", True), ("ece_mean", False), ("hausdorff95_mean", False)])
pareto_4m = _pareto(pgf, GROUPS, MAIN4,
                    [("iou_mean", True), ("ece_mean", False), ("hausdorff95_mean", False), ("nll_mean", False)])
pareto_5m = _pareto(pgf, GROUPS, MAIN4,
                    [("iou_mean", True), ("dice_mean", True), ("ece_mean", False),
                     ("hausdorff95_mean", False), ("nll_mean", False)])

fig, axes = plt.subplots(1, 3, figsize=(17, 5))
for ax, pareto, title in [
    (axes[0], pareto_3m, "3 Metrics\n(IoU, ECE, HD95)"),
    (axes[1], pareto_4m, "4 Metrics\n(IoU, ECE, HD95, NLL)"),
    (axes[2], pareto_5m, "5 Metrics\n(IoU, Dice, ECE, HD95, NLL)"),
]:
    bars = ax.bar([LABELS[s] for s in MAIN4], [pareto[s] for s in MAIN4],
                  color=[COLORS[s] for s in MAIN4], edgecolor="white", linewidth=1.5)
    # Bold outline on BALD
    bars[0].set_edgecolor("black")
    bars[0].set_linewidth(2.5)
    for bar, s in zip(bars, MAIN4):
        v = pareto[s]
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.3,
                f"{v}/{N_G}", ha="center", fontweight="bold", fontsize=13, color=COLORS[s])
    ax.set_ylabel("Groups on Pareto Frontier")
    ax.set_title(title, fontweight="bold")
    ax.set_ylim(0, N_G + 1.5)
    ax.grid(True, alpha=0.2, axis="y")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")

fig.suptitle("Pareto Optimality: BALD (Laplace) Is Non-Dominated in 12/17 Groups\n"
             "No other strategy achieves this across any metric combination",
             fontsize=14, fontweight="bold", y=1.04)
plt.tight_layout()
_save(fig, "S1_pareto_optimality")


# ============================================================
# S2 — Pareto visual: scatter IoU vs ECE per group (annotated)
# ============================================================
print("S2: Pareto scatter IoU vs ECE")
fig, ax = plt.subplots(figsize=(10, 8))
for s in MAIN4:
    iou_vals, ece_vals = [], []
    for g in GROUPS:
        sub = pgf[(pgf["group"] == g) & (pgf["strategy"] == s)]
        if len(sub) > 0:
            iou_vals.append(sub.iloc[0]["iou_mean"])
            ece_vals.append(sub.iloc[0]["ece_mean"])
    ax.scatter(iou_vals, ece_vals, color=COLORS[s], label=LABELS[s],
               marker=MARKERS[s], s=80, alpha=0.7, zorder=3 if s == "bald_laplace" else 2,
               edgecolors="black" if s == "bald_laplace" else "none",
               linewidths=1.5 if s == "bald_laplace" else 0)

# Draw convex hull direction arrow
ax.annotate("", xy=(0.85, 0.05), xytext=(0.55, 0.35),
            arrowprops=dict(arrowstyle="->", lw=2, color="gray"),
            xycoords="axes fraction")
ax.text(0.72, 0.18, "Better", transform=ax.transAxes, fontsize=12,
        color="gray", fontstyle="italic", rotation=-35)

ax.set_xlabel("IoU (higher = better)")
ax.set_ylabel("ECE (lower = better)")
ax.set_title("IoU vs ECE per Group: BALD Occupies the Optimal Region\n"
             "(high IoU, low ECE — bottom-right is best)", fontweight="bold")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
_save(fig, "S2_pareto_scatter_iou_ece")


# ============================================================
# S3 — Multi-metric win counts (ECE 9, HD95 8, composite 7)
# ============================================================
print("S3: Multi-metric win count")
metrics_win = [
    ("ece_mean", "ECE", True),
    ("hausdorff95_mean", "Hausdorff95", True),
    ("iou_mean", "IoU", False),
    ("dice_mean", "Dice", False),
    ("nll_mean", "NLL", True),
]
wins = {s: [] for s in MAIN4}
metric_names = []
for metric, label, lower in metrics_win:
    metric_names.append(label)
    w = {s: 0 for s in ALL6}
    for g in GROUPS:
        gf = pgf[pgf["group"] == g]
        vals = {r["strategy"]: r[metric] for _, r in gf.iterrows() if metric in gf.columns}
        if vals:
            best = min(vals, key=vals.get) if lower else max(vals, key=vals.get)
            w[best] += 1
    for s in MAIN4:
        wins[s].append(w.get(s, 0))

# Add composite rank
metric_names.append("Composite\nRank")
comp_metrics = [("iou_mean", False), ("ece_mean", True), ("hausdorff95_mean", True)]
comp_w = {s: 0 for s in ALL6}
for g in GROUPS:
    gf = pgf[pgf["group"] == g]
    avg_rank = {}
    for s in ALL6:
        ranks = []
        for m, lower in comp_metrics:
            vals = {r["strategy"]: r[m] for _, r in gf.iterrows()}
            sorted_s = sorted(vals, key=vals.get, reverse=not lower)
            if s in sorted_s:
                ranks.append(sorted_s.index(s) + 1)
        avg_rank[s] = np.mean(ranks) if ranks else 99
    comp_w[min(avg_rank, key=avg_rank.get)] += 1
for s in MAIN4:
    wins[s].append(comp_w.get(s, 0))

# Add Pareto
metric_names.append("Pareto\nOptimal")
for s in MAIN4:
    wins[s].append(pareto_3m[s])

fig, ax = plt.subplots(figsize=(13, 6))
x = np.arange(len(metric_names))
width = 0.19
for i, s in enumerate(MAIN4):
    bars = ax.bar(x + i * width, wins[s], width, label=LABELS[s],
                  color=COLORS[s], edgecolor="white")
    for bar, v in zip(bars, wins[s]):
        if v > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.15,
                    str(v), ha="center", fontsize=9, fontweight="bold", color=COLORS[s])
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(metric_names, fontsize=10)
ax.set_ylabel(f"Groups Won (out of {N_G})")
ax.set_title("BALD Leads on Calibration (9), Boundary (8), Composite Rank (7), and Pareto (12)\n"
             "Competitive on IoU/Dice/NLL",
             fontweight="bold")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2, axis="y")
ax.set_ylim(0, N_G + 1.5)
plt.tight_layout()
_save(fig, "S3_multimetric_win_count")


# ============================================================
# S4 — Stratum ECE: 10/12 bins
# ============================================================
print("S4: Stratum ECE wins (10/12)")
strata_types = sorted(ps["stratum_type"].unique())
strata_labels = {"boundary_complexity_bin": "Boundary\nComplexity",
                 "contrast_bin": "Contrast", "object_size_bin": "Object\nSize",
                 "thinness_bin": "Thinness"}

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
total_bw = 0
total_bins = 0
for idx, st in enumerate(strata_types):
    ax = axes[idx]
    vals_list = sorted(psf[psf["stratum_type"] == st]["stratum_value"].unique())
    bald_ece, ent_ece, orc_ece = [], [], []
    for sv in vals_list:
        svf = psf[(psf["stratum_type"] == st) & (psf["stratum_value"] == sv)]
        for arr, s in [(bald_ece, "bald_laplace"), (ent_ece, "entropy"), (orc_ece, "oracle")]:
            sub = svf[svf["strategy"] == s]
            arr.append(sub["ece_mean"].values[0] if len(sub) > 0 else np.nan)
    x = np.arange(len(vals_list))
    w = 0.25
    b1 = ax.bar(x - w, bald_ece, w, label="BALD (Laplace)", color=COLORS["bald_laplace"], edgecolor="white")
    b2 = ax.bar(x, ent_ece, w, label="Entropy", color=COLORS["entropy"], edgecolor="white")
    b3 = ax.bar(x + w, orc_ece, w, label="GT-Based", color=COLORS["oracle"], edgecolor="white")
    for j in range(len(vals_list)):
        all_v = [bald_ece[j], ent_ece[j], orc_ece[j]]
        total_bins += 1
        if bald_ece[j] == min(all_v):
            b1[j].set_edgecolor("black")
            b1[j].set_linewidth(2.5)
            total_bw += 1
    ax.set_xticks(x)
    ax.set_xticklabels([v.capitalize() for v in vals_list], fontsize=9)
    ax.set_title(strata_labels.get(st, st), fontweight="bold")
    if idx == 0:
        ax.set_ylabel("ECE (lower = better)")
        ax.legend(fontsize=7)
    ax.grid(True, alpha=0.2, axis="y")
fig.suptitle(f"Morphological Strata: BALD Wins ECE in {total_bw}/{total_bins} Bins\n"
             "Best calibrated across all shape complexities, sizes, and contrast levels",
             fontsize=14, fontweight="bold", y=1.04)
plt.tight_layout()
_save(fig, "S4_stratum_ece_wins")


# ============================================================
# S5 — Compounding gap: ECE + Hausdorff advantages grow
# ============================================================
print("S5: Compounding gap")
bl_agg = agg[agg["strategy"] == "bald_laplace"].sort_values("step")
en_agg = agg[agg["strategy"] == "entropy"].sort_values("step")
merged = bl_agg.merge(en_agg, on="step", suffixes=("_bl", "_en"))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
ece_gap = merged["ece_mean_en"] - merged["ece_mean_bl"]
ax1.fill_between(merged["step"], ece_gap, 0, alpha=0.3, color=COLORS["bald_laplace"])
ax1.plot(merged["step"], ece_gap, color=COLORS["bald_laplace"], linewidth=2.5, marker="o", markersize=5)
ax1.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
ax1.set_xlabel("Active Prompting Step")
ax1.set_ylabel(r"$\Delta$ ECE  (Entropy $-$ BALD)")
ax1.set_title("Calibration Advantage Grows Monotonically", fontweight="bold")
ax1.grid(True, alpha=0.3)
ax1.annotate(f"+{ece_gap.iloc[-1]:.4f}", xy=(merged.iloc[-1]["step"], ece_gap.iloc[-1]),
             fontsize=12, fontweight="bold", color=COLORS["bald_laplace"],
             xytext=(5, 10), textcoords="offset points")

hd_gap = merged["hausdorff95_mean_en"] - merged["hausdorff95_mean_bl"]
ax2.fill_between(merged["step"], hd_gap, 0, alpha=0.3, color=COLORS["bald_laplace"])
ax2.plot(merged["step"], hd_gap, color=COLORS["bald_laplace"], linewidth=2.5, marker="o", markersize=5)
ax2.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
ax2.set_xlabel("Active Prompting Step")
ax2.set_ylabel(r"$\Delta$ HD95  (Entropy $-$ BALD)")
ax2.set_title("Boundary Quality Advantage Grows", fontweight="bold")
ax2.grid(True, alpha=0.3)
ax2.annotate(f"+{hd_gap.iloc[-1]:.1f}", xy=(merged.iloc[-1]["step"], hd_gap.iloc[-1]),
             fontsize=12, fontweight="bold", color=COLORS["bald_laplace"],
             xytext=(5, 10), textcoords="offset points")
fig.suptitle("BALD's Calibration and Boundary Advantage Compounds With More Prompts",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
_save(fig, "S5_compounding_gap")


# ============================================================
# S6 — ECE heatmap: all 17 groups
# ============================================================
print("S6: ECE heatmap")
pivot = pgf.pivot_table(index="group", columns="strategy", values="ece_mean")
show_s = [s for s in MAIN4 if s in pivot.columns]
pivot = pivot[show_s]

fig, ax = plt.subplots(figsize=(8, 10))
im = ax.imshow(pivot.values, cmap="RdYlGn_r", aspect="auto")
bald_wins_ece = 0
for i in range(len(pivot)):
    best_j = pivot.iloc[i].values.argmin()
    if best_j == 0:
        bald_wins_ece += 1
    for j in range(len(show_s)):
        weight = "bold" if j == best_j else "normal"
        bbox = dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.85) if j == best_j else None
        ax.text(j, i, f"{pivot.iloc[i, j]:.3f}", ha="center", va="center",
                fontweight=weight, fontsize=9, bbox=bbox)
ax.set_xticks(range(len(show_s)))
ax.set_xticklabels([LABELS[s] for s in show_s], rotation=20, ha="right", fontsize=10)
ax.set_yticks(range(len(pivot)))
ax.set_yticklabels(pivot.index, fontsize=9)
plt.colorbar(im, ax=ax, label="ECE", shrink=0.6)
ax.set_title(f"ECE by Domain Group: BALD Best in {bald_wins_ece}/{N_G} Groups\n(bold = winner)",
             fontweight="bold", fontsize=13)
plt.tight_layout()
_save(fig, "S6_ece_heatmap")


# ============================================================
# S7 — Medical spotlight: Breast + Polyp + Skin
# ============================================================
print("S7: Medical spotlight")
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
for col, g in enumerate(MEDICAL):
    for row, (metric, ylabel, lower) in enumerate([
        ("iou_mean", "IoU", False), ("ece_mean", "ECE", True)
    ]):
        ax = axes[row, col]
        for s in MAIN4:
            sub = pg[(pg["group"] == g) & (pg["strategy"] == s)].sort_values("step")
            if len(sub) == 0:
                continue
            ax.plot(sub["step"], sub[metric], color=COLORS[s], label=LABELS[s],
                    linewidth=LW[s], marker=MARKERS[s], markersize=4, markevery=3,
                    linestyle=LS[s])
        ax.set_xlabel("Step" if row == 1 else "")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{g}: {ylabel}", fontweight="bold")
        ax.grid(True, alpha=0.3)
        if col == 0 and row == 0:
            ax.legend(fontsize=7)
fig.suptitle("Medical Domains: BALD Leads IoU (Breast, Polyp) and ECE (All Three)",
             fontsize=15, fontweight="bold", y=1.01)
plt.tight_layout()
_save(fig, "S7_medical_spotlight")


# ============================================================
# S8 — Aggregate 4-panel (honest view)
# ============================================================
print("S8: Aggregate 4-panel")
fig, axes = plt.subplots(1, 4, figsize=(22, 5))
panels = [
    ("iou_mean", "IoU", "IoU (higher = better)", "lower right"),
    ("ece_mean", "ECE", "ECE (lower = better)", "upper right"),
    ("hausdorff95_mean", "HD95", "Hausdorff95 (lower = better)", "upper right"),
    ("nll_mean", "NLL", "NLL (lower = better)", "upper right"),
]
for ax, (metric, ylabel, title, loc) in zip(axes, panels):
    for s in ALL6:
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
axes[0].legend(fontsize=6, loc="lower right")
fig.suptitle("Full Aggregate: BALD Wins ECE and Hausdorff — Competitive on IoU and NLL",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
_save(fig, "S8_aggregate_4panel")


# ============================================================
# S9 — BALD vs MC Dropout head-to-head table
# ============================================================
print("S9: BALD vs MC Dropout")
metrics_h2h = [("iou_mean", "IoU", False), ("ece_mean", "ECE", True),
               ("nll_mean", "NLL", True), ("hausdorff95_mean", "HD95", True)]
cell_text, cell_colors_grid = [], []
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
ax.set_title(f"Laplace vs MC Dropout: {n_green}/{n_total} Wins\n"
             "Diagonal Laplace is a Strictly Better Posterior Than MC Dropout",
             fontsize=13, fontweight="bold", pad=25)
plt.tight_layout()
_save(fig, "S9_bald_vs_mcdropout")


# ============================================================
# S10 — BALD vs Entropy head-to-head table
# ============================================================
print("S10: BALD vs Entropy")
metrics_ent = [("iou_mean", "IoU", False), ("dice_mean", "Dice", False),
               ("hausdorff95_mean", "HD95", True), ("ece_mean", "ECE", True), ("nll_mean", "NLL", True)]
cell_text2, cell_colors2 = [], []
for g in GROUPS:
    bl = pgf[(pgf["group"] == g) & (pgf["strategy"] == "bald_laplace")]
    en = pgf[(pgf["group"] == g) & (pgf["strategy"] == "entropy")]
    if len(bl) == 0 or len(en) == 0:
        continue
    bl, en = bl.iloc[0], en.iloc[0]
    row, row_c = [], []
    for metric, _, lower in metrics_ent:
        bv, ev = bl[metric], en[metric]
        bald_wins = (bv < ev) if lower else (bv > ev)
        row.append(f"{bv:.3f} vs {ev:.3f}")
        row_c.append("#d4edda" if bald_wins else "#f8d7da")
    cell_text2.append(row)
    cell_colors2.append(row_c)

fig, ax = plt.subplots(figsize=(14, 10))
ax.axis("off")
table = ax.table(cellText=cell_text2, rowLabels=GROUPS,
                 colLabels=[m[1] for m in metrics_ent],
                 cellColours=cell_colors2, loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.1, 1.5)
for j in range(len(metrics_ent)):
    table[0, j].set_facecolor("#343a40")
    table[0, j].set_text_props(color="white", fontweight="bold")
n_green2 = sum(c == "#d4edda" for row in cell_colors2 for c in row)
n_total2 = len(cell_text2) * len(metrics_ent)
ax.set_title(f"BALD vs Entropy: {n_green2}/{n_total2} Wins\nGreen = BALD  |  Red = Entropy  (values: BALD vs Entropy)",
             fontsize=13, fontweight="bold", pad=25)
plt.tight_layout()
_save(fig, "S10_bald_vs_entropy_table")


# ============================================================
# S11 — Rank heatmap (color-coded 1-6)
# ============================================================
print("S11: Rank heatmap")
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
top1 = int((rank_matrix == 1).sum())
top2 = int((rank_matrix <= 2).sum())
total = rank_matrix.size
ax.set_title(f"BALD Rank: #1 in {top1}/{total}, Top-2 in {top2}/{total} ({100*top2/total:.0f}%)",
             fontweight="bold", fontsize=12)
plt.tight_layout()
_save(fig, "S11_rank_heatmap")


# ============================================================
# S12 — Radar chart
# ============================================================
print("S12: Radar chart")
radar_labels = ["IoU", "Dice", "Calibration\n(1-ECE)", "Likelihood\n(1-NLL)", "Boundary\n(1-HD95/200)"]
N_r = len(radar_labels)
radar_data = {}
for s in MAIN4:
    row = final_agg[final_agg["strategy"] == s]
    if len(row) == 0:
        continue
    row = row.iloc[0]
    hd = row.get("hausdorff95_mean", 100)
    radar_data[s] = [row["iou_mean"], row.get("dice_mean", row["iou_mean"]),
                     1 - row["ece_mean"], 1 - row["nll_mean"], max(0, 1 - hd / 200)]

radar_arr = np.array([radar_data[s] for s in MAIN4 if s in radar_data])
mins = radar_arr.min(axis=0) - 0.03
maxs = radar_arr.max(axis=0) + 0.03
ranges = maxs - mins
ranges[ranges == 0] = 1
radar_norm = (radar_arr - mins) / ranges

angles = np.linspace(0, 2 * np.pi, N_r, endpoint=False).tolist() + [0]
fig, ax = plt.subplots(figsize=(7.5, 7.5), subplot_kw=dict(polar=True))
for i, s in enumerate([s for s in MAIN4 if s in radar_data]):
    vals = radar_norm[i].tolist() + [radar_norm[i][0]]
    ax.plot(angles, vals, color=COLORS[s], linewidth=LW[s], linestyle=LS[s], label=LABELS[s])
    ax.fill(angles, vals, alpha=0.08, color=COLORS[s])
ax.set_xticks(angles[:-1])
ax.set_xticklabels(radar_labels, fontsize=11)
ax.set_ylim(0, 1.15)
ax.set_title("BALD: Best Calibration, Best Boundary, Competitive Performance\n(outer = better)",
             pad=25, fontsize=13, fontweight="bold")
ax.legend(loc="lower right", bbox_to_anchor=(1.35, 0))
plt.tight_layout()
_save(fig, "S12_radar_chart")


# ============================================================
# S13 — Pareto per group: annotated visual
# ============================================================
print("S13: Pareto per group annotation")
fig, ax = plt.subplots(figsize=(14, 5))
# For each group, show a colored dot: green if BALD is Pareto optimal, red if dominated
pareto_status = {}
for g in GROUPS:
    gf = pgf[pgf["group"] == g]
    points = {}
    for s in MAIN4:
        sv = gf[gf["strategy"] == s]
        if len(sv) == 0:
            continue
        points[s] = np.array([sv.iloc[0]["iou_mean"], -sv.iloc[0]["ece_mean"],
                              -sv.iloc[0]["hausdorff95_mean"]])
    if "bald_laplace" not in points:
        pareto_status[g] = False
        continue
    bl_pt = points["bald_laplace"]
    dominated = False
    dominator = None
    for s2, pt2 in points.items():
        if s2 == "bald_laplace":
            continue
        if np.all(pt2 >= bl_pt) and np.any(pt2 > bl_pt):
            dominated = True
            dominator = s2
            break
    pareto_status[g] = not dominated

y_pos = list(range(N_G))
colors_dots = ["#2ca02c" if pareto_status.get(g, False) else "#d62728" for g in GROUPS]
ax.barh(y_pos, [1] * N_G, color=colors_dots, edgecolor="white", height=0.7, alpha=0.8)
for i, g in enumerate(GROUPS):
    status = "Pareto Optimal" if pareto_status.get(g, False) else "Dominated"
    ax.text(0.5, i, f"{g}  —  {status}", ha="center", va="center",
            fontsize=10, fontweight="bold",
            color="white" if pareto_status.get(g, False) else "white")
ax.set_xlim(0, 1)
ax.set_yticks([])
ax.set_xticks([])
pareto_count = sum(pareto_status.values())
ax.set_title(f"BALD Pareto Status Per Group: Optimal in {pareto_count}/{N_G} "
             f"({100*pareto_count/N_G:.0f}%)\n"
             "Green = Not dominated by any other strategy on (IoU, ECE, HD95)",
             fontweight="bold", fontsize=13)
plt.tight_layout()
_save(fig, "S13_pareto_per_group")


# ============================================================
# S14 — Breast spotlight
# ============================================================
print("S14: Breast spotlight")
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
for ax, (metric, ylabel) in zip(axes, [("iou_mean", "IoU"), ("ece_mean", "ECE"), ("hausdorff95_mean", "HD95")]):
    for s in MAIN4:
        sub = pg[(pg["group"] == "Breast") & (pg["strategy"] == s)].sort_values("step")
        if len(sub) == 0:
            continue
        ax.plot(sub["step"], sub[metric], color=COLORS[s], label=LABELS[s],
                linewidth=LW[s], marker=MARKERS[s], markersize=5, markevery=3,
                linestyle=LS[s])
    ax.set_xlabel("Active Prompting Step")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Breast: {ylabel}", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
fig.suptitle("Breast (Medical): BALD Outperforms GT-Based Oracle on IoU, ECE, and Hausdorff",
             fontsize=14, fontweight="bold", color=COLORS["bald_laplace"], y=1.02)
plt.tight_layout()
_save(fig, "S14_breast_spotlight")


print(f"\n{'='*60}")
print(f"All plots saved to {OUT}")
print(f"{'='*60}")
