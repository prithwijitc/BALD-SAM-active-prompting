#!/usr/bin/env python3
"""Compare SAM1/SAM2/SAM3 medical BALD-Laplace active prompting runs."""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESULTS_ROOT = Path(os.environ.get("BALD_RESULTS_ROOT", Path.cwd() / "Medical_Results")).expanduser()
ACTIVE_ROOT = Path(os.environ.get("BALD_ACTIVE_ROOT", "/mnt/HD_18T_pt2/medical_data/active_prompting")).expanduser()

RUNS = [
    {
        "model": "SAM1",
        "label": "SAM1 baseline v3",
        "path": Path(os.environ.get("SAM1_RUN", ACTIVE_ROOT / "run_baseline_v3")).expanduser(),
        "color": "#3A6EA5",
    },
    {
        "model": "SAM2",
        "label": "SAM2 v1",
        "path": Path(os.environ.get("SAM2_BALD_RUN", ACTIVE_ROOT / "run_sam2_v1_baldlaplace")).expanduser(),
        "color": "#2A9D8F",
    },
    {
        "model": "SAM3",
        "label": "SAM3 v1",
        "path": Path(os.environ.get("SAM3_RUN", ACTIVE_ROOT / "run_sam3_v1")).expanduser(),
        "color": "#D1495B",
    },
]

OUT_DIR = RESULTS_ROOT / "sam123_bald_laplace"
STRATEGY = "bald_laplace"
METRICS = ["iou", "dice", "boundary_iou", "ece", "nll"]

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
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


def load_run(run: dict) -> pd.DataFrame:
    csv_path = run["path"] / "step_results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    cols = [
        "sample_id",
        "group",
        "seed",
        "strategy",
        "step",
        "iou",
        "dice",
        "boundary_iou",
        "hausdorff95",
        "ece",
        "nll",
    ]
    df = pd.read_csv(csv_path, usecols=cols)
    df = df[df["strategy"] == STRATEGY].copy()
    if df.empty:
        raise RuntimeError(f"No {STRATEGY} rows found in {csv_path}")
    df["model"] = run["model"]
    df["label"] = run["label"]
    return df


def restrict_to_common_eval(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only sample/seed/step combinations present for all models."""
    combo_cols = ["sample_id", "seed", "step"]
    counts = df[combo_cols + ["model"]].drop_duplicates().groupby(combo_cols)["model"].nunique()
    common = counts[counts == df["model"].nunique()].reset_index()[combo_cols]
    return df.merge(common, on=combo_cols, how="inner")


def summarize_steps(df: pd.DataFrame) -> pd.DataFrame:
    agg_map = {}
    for metric in ["iou", "dice", "boundary_iou", "hausdorff95", "ece", "nll"]:
        agg_map[metric] = ["mean", "std", "median"]
    out = df.groupby(["model", "label", "step"]).agg(agg_map)
    out.columns = [f"{metric}_{stat}" for metric, stat in out.columns]
    return out.reset_index()


def summarize_final(df: pd.DataFrame, meta_by_model: dict) -> pd.DataFrame:
    rows = []
    for model, g in df.groupby("model"):
        label = g["label"].iloc[0]
        final_step = int(g["step"].max())
        first = g[g["step"] == g["step"].min()]
        final = g[g["step"] == final_step]
        step_iou = g.groupby("step")["iou"].mean()
        best_iou_step = int(step_iou.idxmax())
        best_iou_mean = float(step_iou.max())
        auc_rows = []
        for (_, _), sg in g.groupby(["sample_id", "seed"]):
            sg = sg.sort_values("step")
            auc_rows.append(float(np.trapezoid(sg["iou"].to_numpy(), sg["step"].to_numpy())))

        row = {
            "model": model,
            "label": label,
            "n_samples": int(g["sample_id"].nunique()),
            "n_rows": int(len(g)),
            "first_step": int(g["step"].min()),
            "final_step": final_step,
            "budget": meta_by_model.get(model, {}).get("budget"),
            "mc_samples": meta_by_model.get(model, {}).get("mc_samples"),
            "iou_step0_mean": first["iou"].mean(),
            "best_iou_step": best_iou_step,
            "best_iou_mean": best_iou_mean,
            "iou_final_mean": final["iou"].mean(),
            "iou_delta_mean": final["iou"].mean() - first["iou"].mean(),
            "iou_auc_mean": float(np.mean(auc_rows)),
            "dice_final_mean": final["dice"].mean(),
            "boundary_iou_final_mean": final["boundary_iou"].mean(),
            "ece_final_mean": final["ece"].mean(),
            "nll_final_mean": final["nll"].mean(),
            "hausdorff95_final_mean": final["hausdorff95"].mean(),
        }
        rows.append(row)
    return pd.DataFrame(rows).sort_values("model")


def summarize_per_dataset_final(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, label, group), g in df.groupby(["model", "label", "group"]):
        final = g[g["step"] == g["step"].max()]
        first = g[g["step"] == g["step"].min()]
        rows.append(
            {
                "model": model,
                "label": label,
                "group": group,
                "n_samples": int(g["sample_id"].nunique()),
                "iou_step0_mean": first["iou"].mean(),
                "iou_final_mean": final["iou"].mean(),
                "iou_delta_mean": final["iou"].mean() - first["iou"].mean(),
                "dice_final_mean": final["dice"].mean(),
                "ece_final_mean": final["ece"].mean(),
                "nll_final_mean": final["nll"].mean(),
            }
        )
    return pd.DataFrame(rows).sort_values(["group", "model"])


def savefig(fig: plt.Figure, name: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"{name}.{ext}")
    plt.close(fig)


def plot_metric_curves(step_summary: pd.DataFrame) -> None:
    metric_info = [
        ("iou", "IoU", "Mean IoU"),
        ("dice", "Dice", "Mean Dice"),
        ("ece", "ECE", "Mean ECE"),
        ("nll", "NLL", "Mean NLL"),
    ]
    color_map = {r["model"]: r["color"] for r in RUNS}
    label_map = {r["model"]: r["label"] for r in RUNS}

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (metric, title, ylabel) in zip(axes.ravel(), metric_info):
        for model in [r["model"] for r in RUNS]:
            sub = step_summary[step_summary["model"] == model].sort_values("step")
            mean_col = f"{metric}_mean"
            std_col = f"{metric}_std"
            ax.plot(
                sub["step"],
                sub[mean_col],
                color=color_map[model],
                label=label_map[model],
                linewidth=2.2,
                marker="o",
                markersize=3.5,
            )
            ax.fill_between(
                sub["step"],
                sub[mean_col] - sub[std_col],
                sub[mean_col] + sub[std_col],
                color=color_map[model],
                alpha=0.10,
                linewidth=0,
            )
        ax.set_title(title)
        ax.set_xlabel("Active prompting step")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
    axes[0, 0].legend(loc="lower right")
    fig.suptitle("Medical BALD-Laplace: SAM1 vs SAM2 vs SAM3", fontweight="bold")
    fig.tight_layout()
    savefig(fig, "01_metric_curves")


def plot_iou_and_delta(step_summary: pd.DataFrame) -> None:
    color_map = {r["model"]: r["color"] for r in RUNS}
    label_map = {r["model"]: r["label"] for r in RUNS}

    fig, ax = plt.subplots(figsize=(8, 5))
    for model in [r["model"] for r in RUNS]:
        sub = step_summary[step_summary["model"] == model].sort_values("step")
        ax.plot(
            sub["step"],
            sub["iou_mean"],
            color=color_map[model],
            label=label_map[model],
            linewidth=2.4,
            marker="o",
            markersize=4,
        )
    ax.set_xlabel("Active prompting step")
    ax.set_ylabel("Mean IoU")
    ax.set_title("BALD-Laplace IoU Curve")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    savefig(fig, "02_iou_curve")

    fig, ax = plt.subplots(figsize=(8, 5))
    for model in [r["model"] for r in RUNS]:
        sub = step_summary[step_summary["model"] == model].sort_values("step").copy()
        sub["delta_iou"] = sub["iou_mean"] - sub["iou_mean"].iloc[0]
        ax.plot(
            sub["step"],
            sub["delta_iou"],
            color=color_map[model],
            label=label_map[model],
            linewidth=2.4,
            marker="o",
            markersize=4,
        )
    ax.axhline(0, color="#333333", linewidth=1, alpha=0.5)
    ax.set_xlabel("Active prompting step")
    ax.set_ylabel("Mean IoU change from step 0")
    ax.set_title("BALD-Laplace IoU Change")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    savefig(fig, "03_delta_iou_curve")


def plot_final_bars(final_summary: pd.DataFrame, per_dataset: pd.DataFrame) -> None:
    color_map = {r["model"]: r["color"] for r in RUNS}

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ordered = final_summary.set_index("model").loc[[r["model"] for r in RUNS]].reset_index()
    ax.bar(
        ordered["label"],
        ordered["iou_final_mean"],
        color=[color_map[m] for m in ordered["model"]],
        width=0.65,
    )
    for idx, row in ordered.iterrows():
        ax.text(idx, row["iou_final_mean"] + 0.01, f"{row['iou_final_mean']:.3f}", ha="center")
    ax.set_ylabel("Final mean IoU")
    ax.set_title("Final BALD-Laplace IoU")
    ax.set_ylim(0, max(0.8, float(ordered["iou_final_mean"].max()) + 0.08))
    ax.grid(True, axis="y", alpha=0.25)
    savefig(fig, "04_final_iou_bar")

    x = np.arange(len(ordered))
    width = 0.34
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(
        x - width / 2,
        ordered["best_iou_mean"],
        width=width,
        color=[color_map[m] for m in ordered["model"]],
        alpha=0.75,
        label="Best mean IoU",
    )
    ax.bar(
        x + width / 2,
        ordered["iou_final_mean"],
        width=width,
        color=[color_map[m] for m in ordered["model"]],
        alpha=1.0,
        label="Final mean IoU",
    )
    for idx, row in ordered.iterrows():
        ax.text(
            idx - width / 2,
            row["best_iou_mean"] + 0.01,
            f"{row['best_iou_mean']:.3f}\nstep {int(row['best_iou_step'])}",
            ha="center",
            fontsize=9,
        )
        ax.text(idx + width / 2, row["iou_final_mean"] + 0.01, f"{row['iou_final_mean']:.3f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(ordered["label"])
    ax.set_ylabel("Mean IoU")
    ax.set_title("Best vs Final BALD-Laplace IoU")
    ax.set_ylim(0, max(0.82, float(ordered[["best_iou_mean", "iou_final_mean"]].to_numpy().max()) + 0.10))
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="best")
    savefig(fig, "05_best_vs_final_iou_bar")

    datasets = sorted(per_dataset["group"].unique())
    models = [r["model"] for r in RUNS]
    x = np.arange(len(datasets))
    width = 0.24
    fig, ax = plt.subplots(figsize=(10, 5))
    for idx, model in enumerate(models):
        sub = per_dataset[per_dataset["model"] == model].set_index("group").loc[datasets]
        offsets = x + (idx - 1) * width
        ax.bar(
            offsets,
            sub["iou_final_mean"],
            width=width,
            label=sub["label"].iloc[0],
            color=color_map[model],
        )
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=20, ha="right")
    ax.set_ylabel("Final mean IoU")
    ax.set_title("Final BALD-Laplace IoU by Dataset")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="best")
    savefig(fig, "06_final_iou_by_dataset")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    frames = [load_run(run) for run in RUNS]
    df = restrict_to_common_eval(pd.concat(frames, ignore_index=True))

    meta_by_model = {run["model"]: read_meta(run["path"]) for run in RUNS}
    step_summary = summarize_steps(df)
    final_summary = summarize_final(df, meta_by_model)
    per_dataset = summarize_per_dataset_final(df)

    step_summary.to_csv(OUT_DIR / "sam123_bald_laplace_step_summary.csv", index=False)
    final_summary.to_csv(OUT_DIR / "sam123_bald_laplace_final_summary.csv", index=False)
    per_dataset.to_csv(OUT_DIR / "sam123_bald_laplace_per_dataset_final.csv", index=False)

    plot_metric_curves(step_summary)
    plot_iou_and_delta(step_summary)
    plot_final_bars(final_summary, per_dataset)

    print(f"Wrote outputs to {OUT_DIR}")
    print("")
    print(final_summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
