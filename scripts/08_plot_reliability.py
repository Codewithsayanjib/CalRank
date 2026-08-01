#!/usr/bin/env python3
"""
Generate reliability diagrams.

Per (model, dataset): 4-panel PDF showing raw, temperature, Platt, isotonic.
Per model: 2×4 grid of raw reliability diagrams across all 8 datasets.
Full paper figure: 6×8 grid (all models × all datasets, raw scores).

Outputs in data/figures/
"""
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (MODELS, DATASETS, SCORES_DIR, FIGURES_DIR, CALIBRATION_DIR,
                    ECE_BINS, CALIBRATION_SPLIT, RANDOM_SEED)
from utils import (compute_ece, reliability_bins,
                   fit_temperature, fit_platt, fit_isotonic,
                   apply_temperature, apply_platt, apply_isotonic)


# ── Drawing helpers ───────────────────────────────────────────────────────────

COLORS = {
    "raw":         "#4878cf",
    "temperature": "#e07b39",
    "platt":       "#3b9e3d",
    "isotonic":    "#8b5cf6",
}


def draw_diagram(ax, probs: np.ndarray, labels: np.ndarray,
                 title: str = "", color: str = "#4878cf", n_bins: int = ECE_BINS):
    if len(probs) == 0:
        ax.axis("off")
        return
    centers, accs, counts = reliability_bins(probs, labels, n_bins)
    valid  = ~np.isnan(accs)
    width  = 1.0 / n_bins

    ax.plot([0, 1], [0, 1], "k--", lw=0.8, zorder=3, label="Perfect")
    ax.bar(centers[valid], accs[valid], width=width * 0.85,
           align="center", alpha=0.75, color=color, zorder=2)

    ece = compute_ece(probs, labels, n_bins)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence", fontsize=7)
    ax.set_ylabel("Fraction relevant", fontsize=7)
    ax.set_title(f"{title}\nECE={ece:.3f}", fontsize=7, pad=3)
    ax.tick_params(labelsize=6)


def get_split_calibrators(judged: pd.DataFrame):
    """Fit all three calibrators on the 50 % fit split; return them."""
    labels = judged["qrel_label"].values.astype(int)
    logits = judged["ce_logit"].values
    probs  = judged["ce_prob"].values
    try:
        idx_fit, _ = train_test_split(
            np.arange(len(judged)), test_size=CALIBRATION_SPLIT,
            stratify=labels, random_state=RANDOM_SEED)
    except ValueError:
        idx_fit, _ = train_test_split(
            np.arange(len(judged)), test_size=CALIBRATION_SPLIT,
            random_state=RANDOM_SEED)
    lf, pf, yf = logits[idx_fit], probs[idx_fit], labels[idx_fit]
    T  = fit_temperature(lf, yf)
    lr = fit_platt(lf, yf) if len(np.unique(yf)) >= 2 else None
    ir = fit_isotonic(pf, yf)
    return T, lr, ir


# ── Per (model, dataset): 4-panel reliability PDF ────────────────────────────

def plot_individual(model_key: str, dataset: str):
    out_file = FIGURES_DIR / model_key / f"{dataset}.pdf"
    if out_file.exists():
        return
    path = SCORES_DIR / model_key / f"{dataset}.parquet"
    if not path.exists():
        return

    df     = pd.read_parquet(path)
    judged = df[df["qrel_label"] >= 0].reset_index(drop=True)
    if len(judged) < 10:
        return

    labels = judged["qrel_label"].values.astype(int)
    logits = judged["ce_logit"].values
    probs  = judged["ce_prob"].values

    T, lr, ir = get_split_calibrators(judged)

    treatments = {
        "raw":         probs,
        "temperature": apply_temperature(logits, T),
        "platt":       apply_platt(logits, lr) if lr is not None else np.full_like(probs, np.nan),
        "isotonic":    apply_isotonic(probs, ir),
    }

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
    fig.suptitle(f"{model_key}  /  {dataset}", fontsize=9, y=1.01)

    for ax, (name, p) in zip(axes, treatments.items()):
        draw_diagram(ax, p, labels, title=name, color=COLORS[name])

    plt.tight_layout()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
    print(f"    {out_file.name}")


# ── Per-model 2×4 grid ────────────────────────────────────────────────────────

def plot_model_grid(model_key: str, treatment: str = "raw"):
    out_file = FIGURES_DIR / model_key / f"grid_{treatment}.pdf"
    if out_file.exists():
        return

    import math
    n_ds  = len(DATASETS)
    ncols = math.ceil(math.sqrt(n_ds))
    nrows = math.ceil(n_ds / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 3.5))
    axes = np.atleast_2d(axes)
    fig.suptitle(f"{model_key}  —  {treatment}  reliability diagrams", fontsize=10)

    for idx in range(nrows * ncols):          # blank any unused cells
        if idx >= n_ds:
            axes[idx // ncols][idx % ncols].axis("off")

    for idx, dataset in enumerate(DATASETS):
        ax   = axes[idx // ncols][idx % ncols]
        path = SCORES_DIR / model_key / f"{dataset}.parquet"
        if not path.exists():
            ax.axis("off")
            ax.set_title(dataset, fontsize=7)
            continue

        df     = pd.read_parquet(path)
        judged = df[df["qrel_label"] >= 0].reset_index(drop=True)
        if len(judged) < 10:
            ax.axis("off")
            continue

        labels = judged["qrel_label"].values.astype(int)
        logits = judged["ce_logit"].values
        probs  = judged["ce_prob"].values

        if treatment == "temperature":
            T, _, _ = get_split_calibrators(judged)
            p = apply_temperature(logits, T)
        else:
            p = probs

        draw_diagram(ax, p, labels, title=dataset, color=COLORS[treatment])

    plt.tight_layout()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
    print(f"  Grid saved: {out_file.name}")


# ── Paper-level 6×8 grid (all models × all datasets, raw) ─────────────────────

def plot_full_grid():
    out_file = FIGURES_DIR / "paper_all_models_all_datasets_raw.pdf"
    if out_file.exists():
        print(f"  Full grid exists: {out_file.name}")
        return

    n_models   = len(MODELS)
    n_datasets = len(DATASETS)
    fig, axes  = plt.subplots(n_models, n_datasets, figsize=(22, 16))
    fig.suptitle("Reliability diagrams — raw CE scores — all models × all datasets", fontsize=10)

    for i, model_key in enumerate(MODELS):
        for j, dataset in enumerate(DATASETS):
            ax   = axes[i][j]
            path = SCORES_DIR / model_key / f"{dataset}.parquet"

            if j == 0:
                ax.set_ylabel(model_key, fontsize=7, labelpad=4)
            if i == 0:
                ax.set_title(dataset, fontsize=7)

            if not path.exists():
                ax.axis("off")
                continue

            df     = pd.read_parquet(path)
            judged = df[df["qrel_label"] >= 0]
            if len(judged) < 10:
                ax.axis("off")
                continue

            draw_diagram(ax, judged["ce_prob"].values,
                         judged["qrel_label"].values.astype(int), title="")

    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
    print(f"  Full grid saved: {out_file}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    for model_key in MODELS:
        print(f"\n[{model_key}]  individual diagrams …")
        for dataset in DATASETS:
            plot_individual(model_key, dataset)

        print(f"[{model_key}]  per-model grids …")
        for treatment in ["raw", "temperature"]:
            plot_model_grid(model_key, treatment)

    print("\nBuilding full paper grid (6×8) …")
    plot_full_grid()

    print("\nAll reliability diagrams saved.")


if __name__ == "__main__":
    main()
