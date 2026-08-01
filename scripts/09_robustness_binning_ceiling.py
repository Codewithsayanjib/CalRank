#!/usr/bin/env python3
"""ECE robustness checks.

(A) Binning: recompute ECE for equal-width and equal-mass binning at
    n_bins in {5, 10, 15, 20, 30}.
(B) Temperature ceiling: refit temperature with the search bound raised from 20
    to 1000 and count the pairs censored at 20.

Outputs:
  data/calibration/robustness_binning.csv
  data/calibration/robustness_temperature.csv
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.special import expit as sigmoid
from scipy.optimize import minimize_scalar
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (MODELS, DATASETS, SCORES_DIR, CALIBRATION_DIR,
                    CALIBRATION_SPLIT, RANDOM_SEED)
from utils import compute_ece


# ── Equal-mass (adaptive) ECE ────────────────────────────────────────────────
def compute_ece_equal_mass(probs, labels, n_bins=15):
    """ECE with quantile-based bin edges so each bin holds ~equal mass."""
    n = len(probs)
    if n == 0:
        return float("nan")
    order = np.argsort(probs)
    p, y = probs[order], labels[order].astype(float)
    edges = np.linspace(0, n, n_bins + 1).astype(int)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if hi <= lo:
            continue
        ece += ((hi - lo) / n) * abs(p[lo:hi].mean() - y[lo:hi].mean())
    return float(ece)


def fit_temperature_bounded(logits, labels, hi):
    """Minimise NLL of sigmoid(logit/T) over T in (0.01, hi)."""
    labels = labels.astype(float)

    def nll(T):
        pr = np.clip(sigmoid(logits / T), 1e-7, 1 - 1e-7)
        return -np.mean(labels * np.log(pr) + (1 - labels) * np.log(1 - pr))

    res = minimize_scalar(nll, bounds=(0.01, hi), method="bounded")
    return float(res.x)


def split_idx(labels):
    """Reproduce the 50/50 stratified split used in 06_calibration_fit.py."""
    try:
        return train_test_split(np.arange(len(labels)), test_size=CALIBRATION_SPLIT,
                                stratify=labels, random_state=RANDOM_SEED)
    except ValueError:
        return train_test_split(np.arange(len(labels)), test_size=CALIBRATION_SPLIT,
                                random_state=RANDOM_SEED)


BIN_GRID = [5, 10, 20, 30, 15]

def main():
    bin_rows, temp_rows = [], []

    for mk in MODELS:
        for ds in DATASETS:
            path = SCORES_DIR / mk / f"{ds}.parquet"
            if not path.exists():
                continue
            df = pd.read_parquet(path)
            judged = df[df["qrel_label"] >= 0].reset_index(drop=True)
            if len(judged) < 20:
                continue

            y = judged["qrel_label"].values.astype(int)
            lg = judged["ce_logit"].values
            pr = judged["ce_prob"].values
            prev = float(y.mean())

            # ── (A) binning sensitivity on the full judged set ───────────────
            row = {"model": mk, "dataset": ds, "prevalence": prev, "n_judged": len(y)}
            for nb in BIN_GRID:
                row[f"ece_ew_{nb}"] = compute_ece(pr, y, nb)
                row[f"ece_em_{nb}"] = compute_ece_equal_mass(pr, y, nb)
            bin_rows.append(row)

            # ── (B) temperature ceiling on the fit split ─────────────────────
            i_fit, i_test = split_idx(y)
            lf, yf = lg[i_fit], y[i_fit]
            lt, yt = lg[i_test], y[i_test]

            T20   = fit_temperature_bounded(lf, yf, 20.0)
            T1000 = fit_temperature_bounded(lf, yf, 1000.0)
            temp_rows.append({
                "model": mk, "dataset": ds, "prevalence": prev,
                "T_cap20": T20, "T_uncapped": T1000,
                "censored": bool(T20 > 19.9),
                "ece_cap20":   compute_ece(sigmoid(lt / T20), yt, 15),
                "ece_uncapped": compute_ece(sigmoid(lt / T1000), yt, 15),
            })
            print(f"  {mk:14s} {ds:18s} prev={prev:.2f}  T20={T20:6.2f}  "
                  f"T*={T1000:8.2f}  {'[CENSORED]' if T20 > 19.9 else ''}")

    bdf = pd.DataFrame(bin_rows)
    tdf = pd.DataFrame(temp_rows)
    bdf.to_csv(CALIBRATION_DIR / "robustness_binning.csv", index=False)
    tdf.to_csv(CALIBRATION_DIR / "robustness_temperature.csv", index=False)

    # ── Summaries ────────────────────────────────────────────────────────────
    clean = bdf[bdf.prevalence < 1.0]
    print("\n" + "=" * 72)
    print("(A) ECE BINNING SENSITIVITY")
    print("=" * 72)
    print(f"{'scheme':<10}{'bins':>6}{'mean ECE (all45)':>20}{'mean ECE (clean20)':>22}")
    for nb in sorted(BIN_GRID):
        print(f"{'equal-w':<10}{nb:>6}{bdf[f'ece_ew_{nb}'].mean():>20.4f}"
              f"{clean[f'ece_ew_{nb}'].mean():>22.4f}")
    for nb in sorted(BIN_GRID):
        print(f"{'equal-m':<10}{nb:>6}{bdf[f'ece_em_{nb}'].mean():>20.4f}"
              f"{clean[f'ece_em_{nb}'].mean():>22.4f}")

    # Does the qualitative conclusion (overconfidence everywhere) survive?
    cols = [f"ece_ew_{n}" for n in BIN_GRID] + [f"ece_em_{n}" for n in BIN_GRID]
    spread = bdf[cols].max(axis=1) - bdf[cols].min(axis=1)
    print(f"\nPer-pair ECE spread across all 10 schemes: "
          f"mean={spread.mean():.4f}  max={spread.max():.4f}")
    # rank stability across models
    print("\nModel ranking by mean ECE under each scheme (clean subset):")
    for c in cols:
        rank = clean.groupby("model")[c].mean().sort_values().index.tolist()
        print(f"  {c:<12} {' < '.join(rank)}")

    print("\n" + "=" * 72)
    print("(B) TEMPERATURE CEILING")
    print("=" * 72)
    nc = tdf.censored.sum()
    print(f"Pairs censored at T=20 : {nc}/{len(tdf)}")
    cen = tdf[tdf.censored]
    if len(cen):
        print(f"Their true optima      : median={cen.T_uncapped.median():.1f}  "
              f"min={cen.T_uncapped.min():.1f}  max={cen.T_uncapped.max():.1f}")
        print(f"ECE change from uncapping (censored pairs): "
              f"{(cen.ece_uncapped - cen.ece_cap20).mean():+.4f}")
    cl = tdf[tdf.prevalence < 1.0]
    print(f"\nClean subset (n={len(cl)}): censored={cl.censored.sum()}")
    print(f"  mean ECE cap20={cl.ece_cap20.mean():.4f}  "
          f"uncapped={cl.ece_uncapped.mean():.4f}  "
          f"delta={(cl.ece_uncapped - cl.ece_cap20).mean():+.4f}")
    print("\nWrote robustness_binning.csv, robustness_temperature.csv")


if __name__ == "__main__":
    main()
