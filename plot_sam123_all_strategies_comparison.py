#!/usr/bin/env python3
"""Compare all medical active-prompting strategies across SAM1, SAM2, and SAM3."""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


RESULTS_ROOT = Path(os.environ.get("BALD_RESULTS_ROOT", Path.cwd() / "Medical_Results")).expanduser()
ACTIVE_ROOT = Path(os.environ.get("BALD_ACTIVE_ROOT", "/mnt/HD_18T_pt2/medical_data/active_prompting")).expanduser()

OUT_DIR = RESULTS_ROOT / "sam123_all_strategies"

SAM1_RUN = Path(os.environ.get("SAM1_RUN", ACTIVE_ROOT / "run_baseline_v3")).expanduser()
SAM2_RUN = Path(os.environ.get("SAM2_RUN", ACTIVE_ROOT / "run_sam2_v1")).expanduser()
SAM2_BALD_RUN = Path(os.environ.get("SAM2_BALD_RUN", ACTIVE_ROOT / "run_sam2_v1_baldlaplace")).expanduser()
SAM3_RUN = Path(os.environ.get("SAM3_RUN", ACTIVE_ROOT / "run_sam3_v1")).expanduser()

STRATEGIES = ["bald_laplace", "bald_mcdropout", "entropy", "random", "maxdist", "oracle"]
STRATEGY_LABELS = {
    "bald_laplace": "BALD Laplace",
    "bald_mcdropout": "BALD MC Dropout",
    "entropy": "Entropy",
    "random": "Random",
    "maxdist": "Max Distance",
    "oracle": "GT Based",
}
STRATEGY_COLORS = {
    "bald_laplace": "#D1495B",
    "bald_mcdropout": "#E76F51",
    "entropy": "#3A6EA5",
    "random": "#6C757D",
    "maxdist": "#2A9D8F",
    "oracle": "#8E5EA2",
}
MODEL_LABELS = {
    "SAM1": "SAM1 baseline v3",
    "SAM2": "SAM2 v1",
    "SAM3": "SAM3 v1",
}
MODEL_COLORS = {
    "SAM1": "#3A6EA5",
    "SAM2": "#2A9D8F",
    "SAM3": "#D1495B",
}

METRIC_COLS = ["iou", "dice", "boundary_iou", "hausdorff95", "ece", "nll"]
READ_COLS = ["sample_id", "group", "seed", "strategy", "step"] + METRIC_COLS
PLOT_METRICS = [
    ("iou", "IoU", "Mean IoU"),
    ("ece", "ECE", "Mean ECE"),
    ("hausdorff95", "HD95", "Mean HD95"),
    ("nll", "NLL", "Mean NLL"),
]

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "DejaVu Sans",
    }
)


def read_meta(run_dir: Path) -> dict:
    meta_path = run_dir / "run_meta.json"
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text())


def load_csv(run_dir: Path) -> pd.DataFrame:
    csv_path = run_dir / "step_results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    return pd.read_csv(csv_path, usecols=READ_COLS)


def load_all_runs() -> pd.DataFrame:
    sam1 = load_csv(SAM1_RUN)
    sam1["model"] = "SAM1"
    sam1["model_label"] = MODEL_LABELS["SAM1"]
    sam1["source_run"] = SAM1_RUN.name
    sam1["source_mc_samples"] = read_meta(SAM1_RUN).get("mc_samples")

    sam2_rest = load_csv(SAM2_RUN)
    sam2_rest = sam2_rest[sam2_rest["strategy"] != "bald_laplace"].copy()
    sam2_rest["model"] = "SAM2"
    sam2_rest["model_label"] = MODEL_LABELS["SAM2"]
    sam2_rest["source_run"] = SAM2_RUN.name
    sam2_rest["source_mc_samples"] = read_meta(SAM2_RUN).get("mc_samples")

    sam2_bald = load_csv(SAM2_BALD_RUN)
    sam2_bald = sam2_bald[sam2_bald["strategy"] == "bald_laplace"].copy()
    sam2_bald["model"] = "SAM2"
    sam2_bald["model_label"] = MODEL_LABELS["SAM2"]
    sam2_bald["source_run"] = SAM2_BALD_RUN.name
    sam2_bald["source_mc_samples"] = read_meta(SAM2_BALD_RUN).get("mc_samples")

    sam3 = load_csv(SAM3_RUN)
    sam3["model"] = "SAM3"
    sam3["model_label"] = MODEL_LABELS["SAM3"]
    sam3["source_run"] = SAM3_RUN.name
    sam3["source_mc_samples"] = read_meta(SAM3_RUN).get("mc_samples")

    df = pd.concat([sam1, sam2_rest, sam2_bald, sam3], ignore_index=True)
    df = df[df["strategy"].isin(STRATEGIES)].copy()
    return restrict_to_common_eval(df)


def restrict_to_common_eval(df: pd.DataFrame) -> pd.DataFrame:
    key_cols = ["sample_id", "seed", "strategy", "step"]
    n_models = df["model"].nunique()
    common = (
        df[key_cols + ["model"]]
        .drop_duplicates()
        .groupby(key_cols)["model"]
        .nunique()
    )
    common = common[common == n_models].reset_index()[key_cols]
    return df.merge(common, on=key_cols, how="inner")


def summarize_steps(df: pd.DataFrame) -> pd.DataFrame:
    agg_map = {metric: ["mean", "std", "median"] for metric in METRIC_COLS}
    out = df.groupby(["model", "model_label", "strategy", "step"]).agg(agg_map)
    out.columns = [f"{metric}_{stat}" for metric, stat in out.columns]
    return out.reset_index()


def summarize_final(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, model_label, strategy), g in df.groupby(["model", "model_label", "strategy"]):
        first_step = int(g["step"].min())
        final_step = int(g["step"].max())
        first = g[g["step"] == first_step]
        final = g[g["step"] == final_step]
        curve = g.groupby("step")["iou"].mean()
        auc_vals = []
        for (_, _), sg in g.groupby(["sample_id", "seed"]):
            sg = sg.sort_values("step")
            auc_vals.append(float(np.trapezoid(sg["iou"].to_numpy(), sg["step"].to_numpy())))
        rows.append(
            {
                "model": model,
                "model_label": model_label,
                "strategy": strategy,
                "strategy_label": STRATEGY_LABELS[strategy],
                "source_run": ",".join(sorted(g["source_run"].unique())),
                "source_mc_samples": ",".join(str(x) for x in sorted(g["source_mc_samples"].dropna().unique())),
                "n_samples": int(g["sample_id"].nunique()),
                "n_rows": int(len(g)),
                "first_step": first_step,
                "final_step": final_step,
                "iou_step0_mean": first["iou"].mean(),
                "best_iou_step": int(curve.idxmax()),
                "best_iou_mean": float(curve.max()),
                "iou_final_mean": final["iou"].mean(),
                "iou_delta_mean": final["iou"].mean() - first["iou"].mean(),
                "iou_auc_mean": float(np.mean(auc_vals)),
                "dice_final_mean": final["dice"].mean(),
                "ece_final_mean": final["ece"].mean(),
                "nll_final_mean": final["nll"].mean(),
                "boundary_iou_final_mean": final["boundary_iou"].mean(),
                "hausdorff95_final_mean": final["hausdorff95"].mean(),
            }
        )
    return pd.DataFrame(rows).sort_values(["strategy", "model"])


def savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(path.with_suffix(f".{ext}"))
    plt.close(fig)


def save_separate_legends() -> None:
    legend_dir = OUT_DIR / "legends"

    model_handles = [
        Line2D([0], [0], color=MODEL_COLORS[m], linewidth=3, marker="o", markersize=6)
        for m in ["SAM1", "SAM2", "SAM3"]
    ]
    fig, ax = plt.subplots(figsize=(5.2, 0.8))
    ax.axis("off")
    ax.legend(
        model_handles,
        [MODEL_LABELS[m] for m in ["SAM1", "SAM2", "SAM3"]],
        loc="center",
        ncol=3,
        frameon=False,
    )
    savefig(fig, legend_dir / "model_legend")

    strategy_handles = [
        Line2D([0], [0], color=STRATEGY_COLORS[s], linewidth=3, marker="o", markersize=6)
        for s in STRATEGIES
    ]
    fig, ax = plt.subplots(figsize=(9.5, 0.9))
    ax.axis("off")
    ax.legend(
        strategy_handles,
        [STRATEGY_LABELS[s] for s in STRATEGIES],
        loc="center",
        ncol=6,
        frameon=False,
    )
    savefig(fig, legend_dir / "strategy_legend")


def metric_limits(step_summary: pd.DataFrame, metric: str) -> tuple[float, float]:
    mean_col = f"{metric}_mean"
    y_min = float(step_summary[mean_col].min())
    y_max = float(step_summary[mean_col].max())
    if metric in {"iou", "ece", "nll"}:
        y_min = max(0.0, y_min)
    if metric == "iou":
        y_max = min(1.0, max(0.8, y_max))
    pad = 0.05 * max(y_max - y_min, 1e-6)
    return y_min - pad, y_max + pad


def plot_individual_metric_curves_by_strategy(step_summary: pd.DataFrame) -> None:
    base_dir = OUT_DIR / "individual_metrics" / "by_strategy"
    for metric, metric_title, ylabel in PLOT_METRICS:
        mean_col = f"{metric}_mean"
        ylim = metric_limits(step_summary, metric)
        for strategy in STRATEGIES:
            fig, ax = plt.subplots(figsize=(6.6, 4.2))
            sub_s = step_summary[step_summary["strategy"] == strategy]
            for model in ["SAM1", "SAM2", "SAM3"]:
                sub = sub_s[sub_s["model"] == model].sort_values("step")
                ax.plot(
                    sub["step"],
                    sub[mean_col],
                    color=MODEL_COLORS[model],
                    linewidth=2.5,
                    marker="o",
                    markersize=4,
                )
            ax.set_title(f"{STRATEGY_LABELS[strategy]}: {metric_title}")
            ax.set_xlabel("Active prompting step")
            ax.set_ylabel(ylabel)
            ax.set_ylim(*ylim)
            ax.grid(True, alpha=0.25)
            savefig(fig, base_dir / metric / f"{strategy}_{metric}")


def plot_individual_metric_curves_by_model(step_summary: pd.DataFrame) -> None:
    base_dir = OUT_DIR / "individual_metrics" / "by_model"
    for metric, metric_title, ylabel in PLOT_METRICS:
        mean_col = f"{metric}_mean"
        ylim = metric_limits(step_summary, metric)
        for model in ["SAM1", "SAM2", "SAM3"]:
            fig, ax = plt.subplots(figsize=(6.8, 4.4))
            sub_m = step_summary[step_summary["model"] == model]
            for strategy in STRATEGIES:
                sub = sub_m[sub_m["strategy"] == strategy].sort_values("step")
                ax.plot(
                    sub["step"],
                    sub[mean_col],
                    color=STRATEGY_COLORS[strategy],
                    linewidth=2.2,
                    marker="o",
                    markersize=3.5,
                )
            ax.set_title(f"{MODEL_LABELS[model]}: {metric_title}")
            ax.set_xlabel("Active prompting step")
            ax.set_ylabel(ylabel)
            ax.set_ylim(*ylim)
            ax.grid(True, alpha=0.25)
            savefig(fig, base_dir / metric / f"{model.lower()}_{metric}")


def plot_individual_metric_final_bars(final_summary: pd.DataFrame) -> None:
    base_dir = OUT_DIR / "individual_metrics" / "final_bars"
    metric_cols = [
        ("iou", "iou_final_mean", "Final Mean IoU"),
        ("ece", "ece_final_mean", "Final Mean ECE"),
        ("hausdorff95", "hausdorff95_final_mean", "Final Mean HD95"),
        ("nll", "nll_final_mean", "Final Mean NLL"),
    ]
    models = ["SAM1", "SAM2", "SAM3"]
    x = np.arange(len(models))
    width = 0.12
    offsets = np.linspace(-2.5 * width, 2.5 * width, len(STRATEGIES))

    for metric, value_col, ylabel in metric_cols:
        fig, ax = plt.subplots(figsize=(7.4, 4.8))
        for offset, strategy in zip(offsets, STRATEGIES):
            sub = final_summary[final_summary["strategy"] == strategy].set_index("model").loc[models]
            ax.bar(
                x + offset,
                sub[value_col],
                width=width,
                color=STRATEGY_COLORS[strategy],
            )
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS[m] for m in models], rotation=12, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.grid(True, axis="y", alpha=0.25)
        savefig(fig, base_dir / f"{metric}_final_bars")


def plot_by_strategy_grid(step_summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), sharex=True, sharey=True)
    for ax, strategy in zip(axes.ravel(), STRATEGIES):
        sub_s = step_summary[step_summary["strategy"] == strategy]
        for model in ["SAM1", "SAM2", "SAM3"]:
            sub = sub_s[sub_s["model"] == model].sort_values("step")
            ax.plot(
                sub["step"],
                sub["iou_mean"],
                color=MODEL_COLORS[model],
                label=MODEL_LABELS[model],
                linewidth=2.1,
                marker="o",
                markersize=3,
            )
        ax.set_title(STRATEGY_LABELS[strategy])
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("Step")
        ax.set_ylabel("Mean IoU")
    axes[0, 0].legend(loc="best")
    fig.suptitle("Medical Active Prompting: Each Strategy Across SAM1/SAM2/SAM3", fontweight="bold")
    fig.tight_layout()
    savefig(fig, OUT_DIR / "01_by_strategy_iou_grid")


def plot_by_model_grid(step_summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), sharex=True, sharey=True)
    for ax, model in zip(axes, ["SAM1", "SAM2", "SAM3"]):
        sub_m = step_summary[step_summary["model"] == model]
        for strategy in STRATEGIES:
            sub = sub_m[sub_m["strategy"] == strategy].sort_values("step")
            ax.plot(
                sub["step"],
                sub["iou_mean"],
                color=STRATEGY_COLORS[strategy],
                label=STRATEGY_LABELS[strategy],
                linewidth=2,
                marker="o",
                markersize=3,
            )
        ax.set_title(MODEL_LABELS[model])
        ax.set_xlabel("Step")
        ax.set_ylabel("Mean IoU")
        ax.grid(True, alpha=0.25)
    axes[0].legend(loc="best")
    fig.suptitle("Medical Active Prompting: All Strategies Within Each SAM Model", fontweight="bold")
    fig.tight_layout()
    savefig(fig, OUT_DIR / "02_by_model_iou_grid")


def plot_individual_strategy_curves(step_summary: pd.DataFrame) -> None:
    out_dir = OUT_DIR / "by_strategy"
    for strategy in STRATEGIES:
        fig, ax = plt.subplots(figsize=(8, 5))
        sub_s = step_summary[step_summary["strategy"] == strategy]
        for model in ["SAM1", "SAM2", "SAM3"]:
            sub = sub_s[sub_s["model"] == model].sort_values("step")
            ax.plot(
                sub["step"],
                sub["iou_mean"],
                color=MODEL_COLORS[model],
                label=MODEL_LABELS[model],
                linewidth=2.4,
                marker="o",
                markersize=4,
            )
            ax.fill_between(
                sub["step"],
                sub["iou_mean"] - sub["iou_std"],
                sub["iou_mean"] + sub["iou_std"],
                color=MODEL_COLORS[model],
                alpha=0.08,
                linewidth=0,
            )
        ax.set_title(f"{STRATEGY_LABELS[strategy]}: SAM1 vs SAM2 vs SAM3")
        ax.set_xlabel("Active prompting step")
        ax.set_ylabel("Mean IoU")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
        savefig(fig, out_dir / f"{strategy}_iou")


def plot_individual_model_curves(step_summary: pd.DataFrame) -> None:
    out_dir = OUT_DIR / "by_model"
    for model in ["SAM1", "SAM2", "SAM3"]:
        fig, ax = plt.subplots(figsize=(8.5, 5.2))
        sub_m = step_summary[step_summary["model"] == model]
        for strategy in STRATEGIES:
            sub = sub_m[sub_m["strategy"] == strategy].sort_values("step")
            ax.plot(
                sub["step"],
                sub["iou_mean"],
                color=STRATEGY_COLORS[strategy],
                label=STRATEGY_LABELS[strategy],
                linewidth=2.2,
                marker="o",
                markersize=3.5,
            )
        ax.set_title(f"{MODEL_LABELS[model]}: all strategies")
        ax.set_xlabel("Active prompting step")
        ax.set_ylabel("Mean IoU")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
        savefig(fig, out_dir / f"{model.lower()}_all_strategies_iou")


def plot_final_heatmaps(final_summary: pd.DataFrame) -> None:
    for value_col, title, name in [
        ("iou_final_mean", "Final Mean IoU", "03_final_iou_heatmap"),
        ("best_iou_mean", "Best Mean IoU", "04_best_iou_heatmap"),
        ("iou_auc_mean", "IoU AUC", "05_iou_auc_heatmap"),
    ]:
        matrix = final_summary.pivot(index="strategy", columns="model", values=value_col).loc[STRATEGIES, ["SAM1", "SAM2", "SAM3"]]
        fig, ax = plt.subplots(figsize=(7.5, 5.2))
        im = ax.imshow(matrix.to_numpy(), cmap="viridis", aspect="auto")
        ax.set_xticks(np.arange(matrix.shape[1]))
        ax.set_xticklabels([MODEL_LABELS[m] for m in matrix.columns], rotation=15, ha="right")
        ax.set_yticks(np.arange(matrix.shape[0]))
        ax.set_yticklabels([STRATEGY_LABELS[s] for s in matrix.index])
        ax.set_title(title)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                val = matrix.iloc[i, j]
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", color="white" if val < matrix.to_numpy().mean() else "black")
        fig.colorbar(im, ax=ax, label=title)
        fig.tight_layout()
        savefig(fig, OUT_DIR / name)


def plot_grouped_final_bars(final_summary: pd.DataFrame) -> None:
    models = ["SAM1", "SAM2", "SAM3"]
    x = np.arange(len(STRATEGIES))
    width = 0.24
    fig, ax = plt.subplots(figsize=(12, 5.2))
    for idx, model in enumerate(models):
        sub = final_summary[final_summary["model"] == model].set_index("strategy").loc[STRATEGIES]
        ax.bar(
            x + (idx - 1) * width,
            sub["iou_final_mean"],
            width=width,
            label=MODEL_LABELS[model],
            color=MODEL_COLORS[model],
        )
    ax.set_xticks(x)
    ax.set_xticklabels([STRATEGY_LABELS[s] for s in STRATEGIES], rotation=20, ha="right")
    ax.set_ylabel("Final mean IoU")
    ax.set_title("Final Mean IoU by Strategy and SAM Model")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="best")
    savefig(fig, OUT_DIR / "06_final_iou_grouped_bars")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_all_runs()

    duplicate_keys = df.duplicated(["sample_id", "seed", "strategy", "step", "model"]).sum()
    if duplicate_keys:
        raise RuntimeError(f"Found {duplicate_keys} duplicate evaluation keys after merging runs.")

    step_summary = summarize_steps(df)
    final_summary = summarize_final(df)
    step_summary.to_csv(OUT_DIR / "sam123_all_strategies_step_summary.csv", index=False)
    final_summary.to_csv(OUT_DIR / "sam123_all_strategies_final_summary.csv", index=False)

    plot_by_strategy_grid(step_summary)
    plot_by_model_grid(step_summary)
    plot_individual_strategy_curves(step_summary)
    plot_individual_model_curves(step_summary)
    plot_final_heatmaps(final_summary)
    plot_grouped_final_bars(final_summary)
    save_separate_legends()
    plot_individual_metric_curves_by_strategy(step_summary)
    plot_individual_metric_curves_by_model(step_summary)
    plot_individual_metric_final_bars(final_summary)

    print(f"Wrote outputs to {OUT_DIR}")
    cols = [
        "model",
        "strategy",
        "source_run",
        "source_mc_samples",
        "iou_step0_mean",
        "best_iou_step",
        "best_iou_mean",
        "iou_final_mean",
        "iou_delta_mean",
        "iou_auc_mean",
    ]
    print(final_summary[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
