#!/usr/bin/env python3
"""
Post-hoc calibration: fit temperature scaling, Platt scaling, and isotonic
regression on 50 % of judged pairs; evaluate ECE/MCE/Brier on the other 50 %.

Outputs:
  data/calibration/calibration_results.csv   — per-method metric improvements
  data/calibration/calibration_params.json   — fitted temperature & Platt params
                                               (dropin recipe for deployment)
"""
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (MODELS, DATASETS, SCORES_DIR, CALIBRATION_DIR,
                    ECE_BINS, CALIBRATION_SPLIT, RANDOM_SEED)
from utils import (compute_ece, compute_mce, compute_brier,
                   fit_temperature, fit_platt, fit_isotonic,
                   apply_temperature, apply_platt, apply_isotonic)


def calibrate(model_key: str, dataset: str) -> dict | None:
    path = SCORES_DIR / model_key / f"{dataset}.parquet"
    if not path.exists():
        return None

    df     = pd.read_parquet(path)
    judged = df[df["qrel_label"] >= 0].reset_index(drop=True)
    if len(judged) < 20:
        print(f"    [skip] {model_key}/{dataset}: only {len(judged)} judged pairs")
        return None

    labels = judged["qrel_label"].values.astype(int)
    logits = judged["ce_logit"].values
    probs  = judged["ce_prob"].values

    # Stratified 50/50 split; fall back to unstratified when only one class
    try:
        idx_fit, idx_test = train_test_split(
            np.arange(len(judged)),
            test_size=CALIBRATION_SPLIT,
            stratify=labels,
            random_state=RANDOM_SEED,
        )
    except ValueError:
        idx_fit, idx_test = train_test_split(
            np.arange(len(judged)),
            test_size=CALIBRATION_SPLIT,
            random_state=RANDOM_SEED,
        )

    lf, lt = logits[idx_fit], logits[idx_test]
    pf, pt = probs[idx_fit],  probs[idx_test]
    yf, yt = labels[idx_fit], labels[idx_test]

    row = {"model": model_key, "dataset": dataset}

    # ── Raw baseline (test split) ────────────────────────────────────────────
    row["raw_ece"]   = compute_ece(pt, yt, ECE_BINS)
    row["raw_mce"]   = compute_mce(pt, yt, ECE_BINS)
    row["raw_brier"] = compute_brier(pt, yt)

    # ── Temperature scaling ──────────────────────────────────────────────────
    T = fit_temperature(lf, yf)
    p_temp = apply_temperature(lt, T)
    row["temp_T"]     = T
    row["temp_ece"]   = compute_ece(p_temp, yt, ECE_BINS)
    row["temp_mce"]   = compute_mce(p_temp, yt, ECE_BINS)
    row["temp_brier"] = compute_brier(p_temp, yt)
    row["temp_delta_ece"] = row["temp_ece"] - row["raw_ece"]

    # ── Platt scaling ────────────────────────────────────────────────────────
    if len(np.unique(yf)) < 2:
        # Fit split has only one class — Platt scaling undefined, skip
        row["platt_coef"] = row["platt_intercept"] = float("nan")
        row["platt_ece"] = row["platt_mce"] = row["platt_brier"] = float("nan")
        row["platt_delta_ece"] = float("nan")
    else:
        lr = fit_platt(lf, yf)
        p_platt = apply_platt(lt, lr)
        row["platt_coef"]      = float(lr.coef_[0][0])
        row["platt_intercept"] = float(lr.intercept_[0])
        row["platt_ece"]       = compute_ece(p_platt, yt, ECE_BINS)
        row["platt_mce"]       = compute_mce(p_platt, yt, ECE_BINS)
        row["platt_brier"]     = compute_brier(p_platt, yt)
        row["platt_delta_ece"] = row["platt_ece"] - row["raw_ece"]

    # ── Isotonic regression ──────────────────────────────────────────────────
    ir = fit_isotonic(pf, yf)
    p_iso = apply_isotonic(pt, ir)
    row["iso_ece"]       = compute_ece(p_iso, yt, ECE_BINS)
    row["iso_mce"]       = compute_mce(p_iso, yt, ECE_BINS)
    row["iso_brier"]     = compute_brier(p_iso, yt)
    row["iso_delta_ece"] = row["iso_ece"] - row["raw_ece"]

    return row


def main():
    CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
    rows   = []
    params = {}   # {model: {dataset: {temperature, platt_coef, platt_intercept}}}

    for model_key in MODELS:
        params[model_key] = {}
        for dataset in DATASETS:
            print(f"  {model_key:14s}  {dataset:20s} …", end=" ", flush=True)
            row = calibrate(model_key, dataset)
            if row is None:
                continue
            rows.append(row)
            params[model_key][dataset] = {
                "temperature":     row["temp_T"],
                "platt_coef":      row["platt_coef"],
                "platt_intercept": row["platt_intercept"],
            }
            print(f"T={row['temp_T']:.3f}  "
                  f"ECE: {row['raw_ece']:.4f} → temp {row['temp_ece']:.4f} "
                  f"/ platt {row['platt_ece']:.4f} / iso {row['iso_ece']:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(CALIBRATION_DIR / "calibration_results.csv", index=False)

    with open(CALIBRATION_DIR / "calibration_params.json", "w") as f:
        json.dump(params, f, indent=2)

    print(f"\nSaved → {CALIBRATION_DIR / 'calibration_results.csv'}")
    print(f"Params → {CALIBRATION_DIR / 'calibration_params.json'}")

    # Quick improvement summary
    print("\n=== Mean ΔECE (lower = better) by calibrator ===")
    for col in ["temp_delta_ece", "platt_delta_ece", "iso_delta_ece"]:
        if col in df.columns:
            print(f"  {col:20s}: {df[col].mean():+.4f}")


if __name__ == "__main__":
    main()
