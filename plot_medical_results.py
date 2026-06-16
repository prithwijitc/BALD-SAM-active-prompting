#!/usr/bin/env python3
"""
Analysis and plotting for medical dataset active prompting results.
Reads from: /mnt/HD_18T_pt2/medical_data/active_prompting/run_v3/
Outputs to: /home/prithwijit/BALD_SAM/TMLR/Laplace_Head/Medical_Results/

Generates:
  - Aggregate curves (IoU, ECE, NLL, Delta IoU vs steps)
  - Per-dataset curves and bars
  - Cross-dataset comparison heatmaps
  - Budget analysis
  - Radar charts
  - Summary tables
"""

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

RUN_DIR = Path("/mnt/HD_18T_pt2/medical_data/active_prompting/run_v3")
OUT_DIR = Path("/home/prithwijit/BALD_SAM/TMLR/Laplace_Head/Medical_Results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

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

DATASET_NAMES = {
    "ISIC2018": "ISIC 2018\n(Skin Lesion)",
    "KvasirSEG": "Kvasir-SEG\n(Polyp)",
    "BUSI": "BUSI\n(Breast US)",
    "DRIVE": "DRIVE\n(Retinal)",
    "CheXmask": "CheXmask\n(Chest X-ray)",
}


def savefig(fig, name):
    for ext in ["pdf", "png"]:
        fig.savefig(OUT_DIR / f"{name}.{ext}")
    print(f"  Saved {name}")
    plt.close(fig)


def load_data():
    """Load aggregate and per-group CSVs. If aggregate doesn't exist, build from step_results."""
    agg_path = RUN_DIR / "aggregate.csv"
    pg_path = RUN_DIR / "per_group.csv"

    if agg_path.exists() and pg_path.exists():
        return pd.read_csv(agg_path), pd.read_csv(pg_path)

    # Fallback: build from step_results.csv
    print("aggregate.csv not found, building from step_results.csv...")
    sr = pd.read_csv(RUN_DIR / "step_results.csv")

    # Build per-group
    metrics = ["iou", "dice", "boundary_iou", "hausdorff95", "ece", "nll", "mean_bald", "mean_entropy"]
    pg_rows = []
    for (strat, step, group), g in sr.groupby(["strategy", "step", "group"]):
        row = {"strategy": strat, "step": step, "group": group}
        for m in metrics:
            if m in g.columns:
                row[f"{m}_mean"] = g[m].mean()
                row[f"{m}_std"] = g[m].std()
        # delta from step 0
        step0 = sr[(sr["strategy"] == strat) & (sr["step"] == 0) & (sr["group"] == group)]
        for m in ["iou", "dice", "ece", "nll"]:
            if m in g.columns and m in step0.columns:
                row[f"delta_{m}_mean"] = g[m].mean() - step0[m].mean()
        pg_rows.append(row)
    pg = pd.DataFrame(pg_rows)

    # Build aggregate
    agg_rows = []
    for (strat, step), g in sr.groupby(["strategy", "step"]):
        row = {"strategy": strat, "step": step}
        for m in metrics:
            if m in g.columns:
                row[f"{m}_mean"] = g[m].mean()
                row[f"{m}_std"] = g[m].std()
        step0 = sr[(sr["strategy"] == strat) & (sr["step"] == 0)]
        for m in ["iou", "dice", "ece", "nll"]:
            if m in g.columns and m in step0.columns:
                row[f"delta_{m}_mean"] = g[m].mean() - step0[m].mean()
        agg_rows.append(row)
    agg = pd.DataFrame(agg_rows)

    return agg, pg


def main():
    df, pg = load_data()
    max_step = df["step"].max()
    datasets = sorted(pg["group"].unique())
    steps = list(range(int(max_step) + 1))

    print(f"Datasets: {datasets}")
    print(f"Steps: 0 to {max_step}")
    print(f"Strategies: {df['strategy'].unique().tolist()}")

    # ============================================================
    # 1. Aggregate IoU vs Steps
    # ============================================================
    print("\nPlot 1: Aggregate IoU vs Steps")
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for s in ALL_STRATS:
        sub = df[df["strategy"] == s].sort_values("step")
        if len(sub) == 0:
            continue
        ax.plot(sub["step"], sub["iou_mean"], color=COLORS[s], label=LABELS[s],
                linewidth=LINEWIDTHS[s], marker=MARKERS[s], markersize=5, markevery=2,
                linestyle=LINESTYLES[s])
    ax.set_xlabel("Active Prompting Step")
    ax.set_ylabel("Mean IoU")
    ax.set_title("Medical Datasets: IoU vs Active Prompting Steps")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    savefig(fig, "01_aggregate_iou")

    # ============================================================
    # 2. Aggregate ECE vs Steps
    # ============================================================
    print("Plot 2: Aggregate ECE vs Steps")
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for s in ALL_STRATS:
        sub = df[df["strategy"] == s].sort_values("step")
        if len(sub) == 0:
            continue
        ax.plot(sub["step"], sub["ece_mean"], color=COLORS[s], label=LABELS[s],
                linewidth=LINEWIDTHS[s], marker=MARKERS[s], markersize=5, markevery=2,
                linestyle=LINESTYLES[s])
    ax.set_xlabel("Active Prompting Step")
    ax.set_ylabel("ECE (lower is better)")
    ax.set_title("Medical Datasets: ECE vs Active Prompting Steps")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    savefig(fig, "02_aggregate_ece")

    # ============================================================
    # 3. Aggregate NLL vs Steps
    # ============================================================
    print("Plot 3: Aggregate NLL")
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for s in ALL_STRATS:
        sub = df[df["strategy"] == s].sort_values("step")
        if len(sub) == 0:
            continue
        ax.plot(sub["step"], sub["nll_mean"], color=COLORS[s], label=LABELS[s],
                linewidth=LINEWIDTHS[s], marker=MARKERS[s], markersize=5, markevery=2,
                linestyle=LINESTYLES[s])
    ax.set_xlabel("Active Prompting Step")
    ax.set_ylabel("NLL (lower is better)")
    ax.set_title("Medical Datasets: NLL vs Active Prompting Steps")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    savefig(fig, "03_aggregate_nll")

    # ============================================================
    # 4. Per-dataset IoU curves (5 panels)
    # ============================================================
    print("Plot 4: Per-dataset IoU curves")
    n_ds = len(datasets)
    fig, axes = plt.subplots(1, n_ds, figsize=(4 * n_ds, 5))
    if n_ds == 1:
        axes = [axes]
    for idx, ds in enumerate(datasets):
        ax = axes[idx]
        for s in STRATS:
            sub = pg[(pg["group"] == ds) & (pg["strategy"] == s)].sort_values("step")
            if len(sub) == 0:
                continue
            ax.plot(sub["step"], sub["iou_mean"], color=COLORS[s], label=LABELS[s],
                    linewidth=LINEWIDTHS[s], marker=MARKERS[s], markersize=4, markevery=3,
                    linestyle=LINESTYLES[s])
        ax.set_xlabel("Step")
        ax.set_ylabel("IoU")
        title = DATASET_NAMES.get(ds, ds).replace("\n", " ")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=8)
    fig.suptitle("Per-Dataset IoU Progression", fontsize=15, fontweight="bold")
    plt.tight_layout()
    savefig(fig, "04_per_dataset_iou")

    # ============================================================
    # 5. Per-dataset ECE curves (5 panels)
    # ============================================================
    print("Plot 5: Per-dataset ECE curves")
    fig, axes = plt.subplots(1, n_ds, figsize=(4 * n_ds, 5))
    if n_ds == 1:
        axes = [axes]
    for idx, ds in enumerate(datasets):
        ax = axes[idx]
        for s in STRATS:
            sub = pg[(pg["group"] == ds) & (pg["strategy"] == s)].sort_values("step")
            if len(sub) == 0:
                continue
            ax.plot(sub["step"], sub["ece_mean"], color=COLORS[s], label=LABELS[s],
                    linewidth=LINEWIDTHS[s], marker=MARKERS[s], markersize=4, markevery=3,
                    linestyle=LINESTYLES[s])
        ax.set_xlabel("Step")
        ax.set_ylabel("ECE")
        title = DATASET_NAMES.get(ds, ds).replace("\n", " ")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=8)
    fig.suptitle("Per-Dataset ECE Progression", fontsize=15, fontweight="bold")
    plt.tight_layout()
    savefig(fig, "05_per_dataset_ece")

    # ============================================================
    # 6. Final-step bars: IoU, ECE, NLL per dataset
    # ============================================================
    print("Plot 6: Final-step multi-metric bars")
    final_pg = pg[pg["step"] == max_step]
    metrics_info = [
        ("iou_mean", "IoU \u2191", False),
        ("ece_mean", "ECE \u2193", True),
        ("nll_mean", "NLL \u2193", True),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for col, (metric, title, lower_better) in enumerate(metrics_info):
        ax = axes[col]
        x = np.arange(n_ds)
        width = 0.19
        for i, s in enumerate(STRATS):
            vals = []
            for ds in datasets:
                sub = final_pg[(final_pg["strategy"] == s) & (final_pg["group"] == ds)]
                vals.append(sub[metric].values[0] if len(sub) > 0 else 0)
            bars = ax.bar(x + i * width, vals, width, label=LABELS[s], color=COLORS[s], edgecolor="white")
            for j, v in enumerate(vals):
                all_v = []
                for ss in STRATS:
                    sub2 = final_pg[(final_pg["strategy"] == ss) & (final_pg["group"] == datasets[j])]
                    all_v.append(sub2[metric].values[0] if len(sub2) > 0 else 0)
                is_best = (v == min(all_v)) if lower_better else (v == max(all_v))
                if is_best:
                    bars[j].set_edgecolor("black")
                    bars[j].set_linewidth(2)
        ax.set_xticks(x + width * 1.5)
        ds_labels = [DATASET_NAMES.get(d, d) for d in datasets]
        ax.set_xticklabels(ds_labels, fontsize=9, ha="center")
        ax.set_title(title, fontweight="bold")
        ax.grid(True, alpha=0.2, axis="y")
        if col == 0:
            ax.legend(fontsize=8)
    fig.suptitle("Final Step: Per-Dataset Performance (bold outline = best)", fontsize=15, fontweight="bold")
    plt.tight_layout()
    savefig(fig, "06_final_multimetric_bars")

    # ============================================================
    # 7. ECE heatmap across datasets
    # ============================================================
    print("Plot 7: ECE heatmap")
    pivot_ece = final_pg.pivot_table(index="group", columns="strategy", values="ece_mean")
    show_strats = [s for s in ALL_STRATS if s in pivot_ece.columns]
    pivot_ece = pivot_ece[show_strats]
    pivot_ece.columns = [LABELS.get(s, s) for s in show_strats]

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(pivot_ece.values, cmap="RdYlGn_r", aspect="auto")
    for i in range(len(pivot_ece)):
        best_j = pivot_ece.iloc[i].values.argmin()
        for j in range(len(show_strats)):
            weight = "bold" if j == best_j else "normal"
            bbox = dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8) if j == best_j else None
            ax.text(j, i, f"{pivot_ece.iloc[i, j]:.3f}", ha="center", va="center",
                    fontweight=weight, fontsize=10, bbox=bbox)
    ax.set_xticks(range(len(show_strats)))
    ax.set_xticklabels(pivot_ece.columns, rotation=20, ha="right")
    ax.set_yticks(range(len(pivot_ece)))
    ax.set_yticklabels([DATASET_NAMES.get(d, d).replace("\n", " ") for d in pivot_ece.index])
    ax.set_title("ECE by Dataset (lower = better, bold = best)")
    plt.colorbar(im, ax=ax, label="ECE", shrink=0.8)
    plt.tight_layout()
    savefig(fig, "07_ece_heatmap")

    # ============================================================
    # 8. ECE win count bar
    # ============================================================
    print("Plot 8: ECE win count")
    wins = {s: 0 for s in show_strats}
    for ds in datasets:
        sub = final_pg[final_pg["group"] == ds]
        vals = {s: sub[sub["strategy"] == s]["ece_mean"].values[0] for s in show_strats if len(sub[sub["strategy"] == s]) > 0}
        if vals:
            best = min(vals, key=vals.get)
            wins[best] += 1
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar([LABELS[s] for s in show_strats], [wins[s] for s in show_strats],
                  color=[COLORS[s] for s in show_strats], edgecolor="white")
    best_idx = np.argmax([wins[s] for s in show_strats])
    bars[best_idx].set_edgecolor("black")
    bars[best_idx].set_linewidth(2.5)
    for bar, s in zip(bars, show_strats):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{wins[s]}/{n_ds}", ha="center", fontweight="bold", fontsize=12, color=COLORS[s])
    ax.set_ylabel("Datasets Won")
    ax.set_title(f"ECE Wins Across {n_ds} Medical Datasets")
    ax.grid(True, alpha=0.2, axis="y")
    plt.xticks(rotation=10, ha="right")
    plt.tight_layout()
    savefig(fig, "08_ece_win_count")

    # ============================================================
    # 9. Radar chart — overall medical
    # ============================================================
    print("Plot 9: Radar chart")
    final_agg = df[df["step"] == max_step]
    radar_labels = ["IoU", "Dice", "Calibration\n(1-ECE)", "Likelihood\n(1-NLL)"]
    N = len(radar_labels)
    radar_data = {}
    for s in STRATS:
        row = final_agg[final_agg["strategy"] == s]
        if len(row) == 0:
            continue
        row = row.iloc[0]
        radar_data[s] = [row["iou_mean"], row.get("dice_mean", row["iou_mean"]),
                         1 - row["ece_mean"], 1 - row["nll_mean"]]
    if radar_data:
        radar_arr = np.array([radar_data[s] for s in STRATS if s in radar_data])
        mins = radar_arr.min(axis=0)
        maxs = radar_arr.max(axis=0)
        ranges = maxs - mins
        ranges[ranges == 0] = 1
        radar_norm = (radar_arr - mins) / ranges
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist() + [0]
        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
        for i, s in enumerate([s for s in STRATS if s in radar_data]):
            vals = radar_norm[i].tolist() + [radar_norm[i][0]]
            ax.plot(angles, vals, color=COLORS[s], linewidth=LINEWIDTHS[s],
                    linestyle=LINESTYLES[s], label=LABELS[s])
            ax.fill(angles, vals, alpha=0.1, color=COLORS[s])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(radar_labels, fontsize=11)
        ax.set_ylim(0, 1.1)
        ax.set_title("Medical Datasets: Multi-Metric Comparison\n(outer = better)", pad=20)
        ax.legend(loc="lower right", bbox_to_anchor=(1.35, 0))
        plt.tight_layout()
        savefig(fig, "09_radar_chart")

    # ============================================================
    # 10. Budget analysis — IoU at key budgets
    # ============================================================
    print("Plot 10: Budget analysis")
    budgets = [3, 5, 10, 15, int(max_step)]
    budgets = [b for b in budgets if b <= max_step]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(budgets))
    width = 0.19
    for i, s in enumerate(STRATS):
        vals = []
        for b in budgets:
            sub = df[(df["step"] == b) & (df["strategy"] == s)]
            vals.append(sub["iou_mean"].values[0] if len(sub) > 0 else 0)
        ax.bar(x + i * width, vals, width, label=LABELS[s], color=COLORS[s], edgecolor="white")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([f"Budget {b}" for b in budgets])
    ax.set_ylabel("Mean IoU")
    ax.set_title("Medical IoU at Different Prompt Budgets")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, axis="y")
    plt.tight_layout()
    savefig(fig, "10_budget_iou")

    # ============================================================
    # 11. Head-to-head BALD vs Entropy
    # ============================================================
    print("Plot 11: Head-to-head")
    bl = final_pg[final_pg["strategy"] == "bald_laplace"].set_index("group")
    en = final_pg[final_pg["strategy"] == "entropy"].set_index("group")
    common = bl.index.intersection(en.index)
    if len(common) > 0:
        metrics_h2h = [("iou_mean", "IoU", False), ("ece_mean", "ECE", True), ("nll_mean", "NLL", True)]
        bl_wins_list = []
        en_wins_list = []
        for metric, label, lower_better in metrics_h2h:
            if lower_better:
                bw = (bl.loc[common, metric] < en.loc[common, metric]).sum()
            else:
                bw = (bl.loc[common, metric] > en.loc[common, metric]).sum()
            bl_wins_list.append(bw)
            en_wins_list.append(len(common) - bw)
        fig, ax = plt.subplots(figsize=(7, 4))
        labels_h2h = [m[1] for m in metrics_h2h]
        x = np.arange(len(labels_h2h))
        w = 0.35
        bars1 = ax.bar(x - w / 2, bl_wins_list, w, label="BALD (Laplace)", color=COLORS["bald_laplace"])
        bars2 = ax.bar(x + w / 2, en_wins_list, w, label="Entropy", color=COLORS["entropy"])
        for bar, val in zip(bars1, bl_wins_list):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1, str(val),
                    ha="center", fontweight="bold", color=COLORS["bald_laplace"])
        for bar, val in zip(bars2, en_wins_list):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1, str(val),
                    ha="center", fontweight="bold", color=COLORS["entropy"])
        ax.set_xticks(x)
        ax.set_xticklabels(labels_h2h)
        ax.set_ylabel(f"Datasets Won (out of {len(common)})")
        ax.set_title("Head-to-Head: BALD (Laplace) vs Entropy")
        ax.legend()
        ax.grid(True, alpha=0.2, axis="y")
        savefig(fig, "11_head_to_head")

    # ============================================================
    # 12. Summary table
    # ============================================================
    print("Plot 12: Summary table")
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis("off")
    col_labels = [LABELS[s] for s in STRATS if s in final_pg["strategy"].values]
    avail_strats = [s for s in STRATS if s in final_pg["strategy"].values]
    row_labels = []
    table_data = []
    cell_colors = []
    for ds in datasets:
        for metric, lower_better in [("iou_mean", False), ("ece_mean", True), ("nll_mean", True)]:
            label = metric.replace("_mean", "").upper()
            row_labels.append(f"{ds} {label}")
            vals = []
            for s in avail_strats:
                sub = final_pg[(final_pg["strategy"] == s) & (final_pg["group"] == ds)]
                vals.append(sub[metric].values[0] if len(sub) > 0 else float("nan"))
            best_idx = np.nanargmin(vals) if lower_better else np.nanargmax(vals)
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
    table.set_fontsize(8)
    table.scale(1.0, 1.3)
    for j, s in enumerate(avail_strats):
        table[0, j].set_facecolor(COLORS[s])
        table[0, j].set_text_props(color="white", fontweight="bold")
    ax.set_title("Medical Datasets: Full Results (highlighted = best per row)", fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    savefig(fig, "12_summary_table")

    # ============================================================
    # 13. Delta IoU curves
    # ============================================================
    print("Plot 13: Delta IoU")
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for s in ALL_STRATS:
        sub = df[df["strategy"] == s].sort_values("step")
        if len(sub) == 0 or "delta_iou_mean" not in sub.columns:
            continue
        ax.plot(sub["step"], sub["delta_iou_mean"], color=COLORS[s], label=LABELS[s],
                linewidth=LINEWIDTHS[s], marker=MARKERS[s], markersize=5, markevery=2,
                linestyle=LINESTYLES[s])
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel("Active Prompting Step")
    ax.set_ylabel(r"$\Delta$ IoU from Baseline")
    ax.set_title("Medical Datasets: IoU Improvement Over Baseline")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    savefig(fig, "13_delta_iou")

    # ============================================================
    # Per-dataset results: CSVs + individual plots
    # ============================================================
    print("\n=== Saving per-dataset results ===")
    for ds in datasets:
        ds_dir = OUT_DIR / "per_dataset" / ds
        ds_dir.mkdir(parents=True, exist_ok=True)

        # Save per-dataset CSV (all steps, all strategies)
        ds_pg = pg[pg["group"] == ds].copy()
        ds_pg.to_csv(ds_dir / f"{ds}_results.csv", index=False)
        print(f"  {ds}: saved {ds}_results.csv ({len(ds_pg)} rows)")

        # Save final-step summary CSV
        ds_final = ds_pg[ds_pg["step"] == max_step].copy()
        ds_final.to_csv(ds_dir / f"{ds}_final_summary.csv", index=False)

        # --- Per-dataset IoU curve ---
        fig, ax = plt.subplots(figsize=(8, 5.5))
        for s in ALL_STRATS:
            sub = ds_pg[ds_pg["strategy"] == s].sort_values("step")
            if len(sub) == 0:
                continue
            ax.plot(sub["step"], sub["iou_mean"], color=COLORS[s], label=LABELS[s],
                    linewidth=LINEWIDTHS[s], marker=MARKERS[s], markersize=5, markevery=2,
                    linestyle=LINESTYLES[s])
            if f"iou_std" in sub.columns:
                ax.fill_between(sub["step"],
                                sub["iou_mean"] - sub["iou_std"],
                                sub["iou_mean"] + sub["iou_std"],
                                alpha=0.1, color=COLORS[s])
        title = DATASET_NAMES.get(ds, ds).replace("\n", " ")
        ax.set_xlabel("Active Prompting Step")
        ax.set_ylabel("Mean IoU")
        ax.set_title(f"{title}: IoU vs Active Prompting Steps")
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(True, alpha=0.3)
        for ext in ["pdf", "png"]:
            fig.savefig(ds_dir / f"{ds}_iou_curve.{ext}")
        plt.close(fig)

        # --- Per-dataset ECE curve ---
        fig, ax = plt.subplots(figsize=(8, 5.5))
        for s in ALL_STRATS:
            sub = ds_pg[ds_pg["strategy"] == s].sort_values("step")
            if len(sub) == 0 or "ece_mean" not in sub.columns:
                continue
            ax.plot(sub["step"], sub["ece_mean"], color=COLORS[s], label=LABELS[s],
                    linewidth=LINEWIDTHS[s], marker=MARKERS[s], markersize=5, markevery=2,
                    linestyle=LINESTYLES[s])
        ax.set_xlabel("Active Prompting Step")
        ax.set_ylabel("ECE (lower is better)")
        ax.set_title(f"{title}: ECE vs Active Prompting Steps")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)
        for ext in ["pdf", "png"]:
            fig.savefig(ds_dir / f"{ds}_ece_curve.{ext}")
        plt.close(fig)

        # --- Per-dataset NLL curve ---
        fig, ax = plt.subplots(figsize=(8, 5.5))
        for s in ALL_STRATS:
            sub = ds_pg[ds_pg["strategy"] == s].sort_values("step")
            if len(sub) == 0 or "nll_mean" not in sub.columns:
                continue
            ax.plot(sub["step"], sub["nll_mean"], color=COLORS[s], label=LABELS[s],
                    linewidth=LINEWIDTHS[s], marker=MARKERS[s], markersize=5, markevery=2,
                    linestyle=LINESTYLES[s])
        ax.set_xlabel("Active Prompting Step")
        ax.set_ylabel("NLL (lower is better)")
        ax.set_title(f"{title}: NLL vs Active Prompting Steps")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)
        for ext in ["pdf", "png"]:
            fig.savefig(ds_dir / f"{ds}_nll_curve.{ext}")
        plt.close(fig)

        # --- Per-dataset Delta IoU curve ---
        fig, ax = plt.subplots(figsize=(8, 5.5))
        for s in ALL_STRATS:
            sub = ds_pg[ds_pg["strategy"] == s].sort_values("step")
            if len(sub) == 0 or "delta_iou_mean" not in sub.columns:
                continue
            ax.plot(sub["step"], sub["delta_iou_mean"], color=COLORS[s], label=LABELS[s],
                    linewidth=LINEWIDTHS[s], marker=MARKERS[s], markersize=5, markevery=2,
                    linestyle=LINESTYLES[s])
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_xlabel("Active Prompting Step")
        ax.set_ylabel(r"$\Delta$ IoU from Baseline")
        ax.set_title(f"{title}: IoU Improvement Over Baseline")
        ax.legend(loc="lower left", fontsize=9)
        ax.grid(True, alpha=0.3)
        for ext in ["pdf", "png"]:
            fig.savefig(ds_dir / f"{ds}_delta_iou.{ext}")
        plt.close(fig)

        # --- Per-dataset dual-axis IoU + ECE ---
        fig, ax1 = plt.subplots(figsize=(9, 5.5))
        ax2 = ax1.twinx()
        for s in STRATS:
            sub = ds_pg[ds_pg["strategy"] == s].sort_values("step")
            if len(sub) == 0:
                continue
            ax1.plot(sub["step"], sub["iou_mean"], color=COLORS[s], label=f"{LABELS[s]} IoU",
                     linewidth=LINEWIDTHS[s], linestyle="-", marker=MARKERS[s], markersize=4, markevery=3)
            if "ece_mean" in sub.columns:
                ax2.plot(sub["step"], sub["ece_mean"], color=COLORS[s],
                         linewidth=1.5, linestyle="--", alpha=0.7)
        ax1.set_xlabel("Active Prompting Step")
        ax1.set_ylabel("IoU (solid lines)")
        ax2.set_ylabel("ECE (dashed lines)")
        ax1.set_title(f"{title}: IoU (solid) + ECE (dashed)")
        ax1.legend(loc="lower left", fontsize=8)
        ax1.grid(True, alpha=0.2)
        for ext in ["pdf", "png"]:
            fig.savefig(ds_dir / f"{ds}_dual_axis.{ext}")
        plt.close(fig)

        # --- Per-dataset strategy comparison bar (final step) ---
        fig, axes_bar = plt.subplots(1, 3, figsize=(14, 4.5))
        for col, (metric, ylabel, lower_better) in enumerate([
            ("iou_mean", "IoU ↑", False),
            ("ece_mean", "ECE ↓", True),
            ("nll_mean", "NLL ↓", True),
        ]):
            ax = axes_bar[col]
            avail = [s for s in ALL_STRATS if len(ds_final[ds_final["strategy"] == s]) > 0]
            vals = []
            for s in avail:
                sub = ds_final[ds_final["strategy"] == s]
                vals.append(sub[metric].values[0] if len(sub) > 0 else 0)
            bars = ax.bar([LABELS[s] for s in avail], vals,
                          color=[COLORS[s] for s in avail], edgecolor="white")
            if vals:
                best_idx = np.argmin(vals) if lower_better else np.argmax(vals)
                bars[best_idx].set_edgecolor("black")
                bars[best_idx].set_linewidth(2.5)
            ax.set_ylabel(ylabel)
            ax.set_title(ylabel, fontweight="bold")
            ax.grid(True, alpha=0.2, axis="y")
            plt.setp(ax.get_xticklabels(), rotation=25, ha="right", fontsize=8)
        fig.suptitle(f"{title}: Final Step Comparison", fontsize=14, fontweight="bold")
        plt.tight_layout()
        for ext in ["pdf", "png"]:
            fig.savefig(ds_dir / f"{ds}_final_bars.{ext}")
        plt.close(fig)

        print(f"  {ds}: saved 6 plots (IoU, ECE, NLL, DeltaIoU, DualAxis, FinalBars)")

    # Save combined per-group CSV at top level too
    pg.to_csv(OUT_DIR / "all_datasets_per_group.csv", index=False)
    df.to_csv(OUT_DIR / "all_datasets_aggregate.csv", index=False)
    print(f"\nSaved combined CSVs to {OUT_DIR}")

    # ============================================================
    # Print text summary
    # ============================================================
    print("\n" + "=" * 60)
    print("TEXT SUMMARY")
    print("=" * 60)
    for metric, asc, label in [("iou_mean", False, "IoU"), ("ece_mean", True, "ECE"), ("nll_mean", True, "NLL")]:
        print(f"\n{label} at final step:")
        for s in ALL_STRATS:
            row = final_agg[final_agg["strategy"] == s]
            if len(row) > 0:
                v = row.iloc[0][metric]
                print(f"  {LABELS[s]:25s}  {v:.4f}")

    print(f"\nECE wins per dataset:")
    for ds in datasets:
        sub = final_pg[final_pg["group"] == ds]
        vals = {s: sub[sub["strategy"] == s]["ece_mean"].values[0] for s in show_strats if len(sub[sub["strategy"] == s]) > 0}
        if vals:
            best = min(vals, key=vals.get)
            print(f"  {ds:15s}  winner={LABELS[best]}")

    print(f"\nAll plots saved to {OUT_DIR}")
    print(f"Per-dataset results in {OUT_DIR / 'per_dataset' / '<dataset_name>'}/")


if __name__ == "__main__":
    main()
