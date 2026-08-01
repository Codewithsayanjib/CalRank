#!/usr/bin/env python3
"""
Compute raw calibration metrics for every (model, dataset) pair.

Outputs:
  data/calibration/raw_metrics.csv
  columns: model, dataset, n_judged, n_positive, prevalence,
           ece, ece_ci_lo, ece_ci_hi, mce, brier, auc_roc

Only judged pairs (qrel_label >= 0) are used.
Bootstrap CIs over queries (1000 resamples) are computed for ECE.
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import MODELS, DATASETS, SCORES_DIR, CALIBRATION_DIR, ECE_BINS, BOOTSTRAP_N
from utils import compute_ece, compute_mce, compute_brier, bootstrap_ece_ci


def metrics_for(df: pd.DataFrame) -> dict:
    judged = df[df["qrel_label"] >= 0].copy()
    if len(judged) == 0:
        return None
    probs  = judged["ce_prob"].values
    labels = judged["qrel_label"].values

    if labels.sum() == 0 or labels.sum() == len(labels):
        auc = float("nan")
    else:
        auc = float(roc_auc_score(labels, probs))

    ece = compute_ece(probs, labels, ECE_BINS)
    ci_lo, ci_hi = bootstrap_ece_ci(judged, n_bootstrap=BOOTSTRAP_N, n_bins=ECE_BINS)

    return {
        "n_judged":   len(judged),
        "n_positive": int(labels.sum()),
        "prevalence": float(labels.mean()),
        "ece":        ece,
        "ece_ci_lo":  ci_lo,
        "ece_ci_hi":  ci_hi,
        "mce":        compute_mce(probs, labels, ECE_BINS),
        "brier":      compute_brier(probs, labels),
        "auc_roc":    auc,
    }


def main():
    rows = []
    for model_key in MODELS:
        for dataset in DATASETS:
            path = SCORES_DIR / model_key / f"{dataset}.parquet"
            if not path.exists():
                print(f"[missing] {model_key}/{dataset}")
                continue
            print(f"  {model_key:14s}  {dataset:20s} …", end=" ", flush=True)
            df = pd.read_parquet(path)
            m  = metrics_for(df)
            if m is None:
                print("no judged pairs, skip")
                continue
            rows.append({"model": model_key, "dataset": dataset, **m})
            print(f"ECE={m['ece']:.4f} [{m['ece_ci_lo']:.4f},{m['ece_ci_hi']:.4f}]  "
                  f"Brier={m['brier']:.4f}  AUC={m['auc_roc']:.4f}")

    out = pd.DataFrame(rows)
    CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
    out_file = CALIBRATION_DIR / "raw_metrics.csv"
    out.to_csv(out_file, index=False)
    print(f"\nSaved → {out_file}")

    # Quick summary table
    print("\n=== Mean ECE per model (across datasets) ===")
    print(out.groupby("model")["ece"].mean().sort_values().to_string())


if __name__ == "__main__":
    main()
