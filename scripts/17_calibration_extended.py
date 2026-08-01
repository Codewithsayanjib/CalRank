#!/usr/bin/env python3
"""
Recompute calibration for every score file present, including the TREC DL
collections and the model-scope extension models.

Discovers work by globbing data/scores/<model>/<dataset>.parquet, so it picks up
new models and new collections without configuration changes. Protocol is
identical to 06_calibration_fit.py: 50/50 label-stratified split at seed 42,
calibrators fitted on one half and evaluated on the other.

Outputs:
  data/calibration/calibration_extended.csv
  data/calibration/validity_extended.csv
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.special import expit as sigmoid
from scipy.optimize import minimize_scalar
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (MODELS, EXTRA_MODELS, SCORES_DIR, CALIBRATION_DIR,
                    CALIBRATION_SPLIT, RANDOM_SEED, ECE_BINS)
from utils import compute_ece, compute_mce, compute_brier

MIN_MINORITY, MAX_PREVALENCE = 100, 0.90


def fit_temperature(logits, labels, hi=1000.0):
    labels = labels.astype(float)

    def nll(T):
        p = np.clip(sigmoid(logits / T), 1e-7, 1 - 1e-7)
        return -np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p))

    return float(minimize_scalar(nll, bounds=(0.01, hi), method="bounded").x)


def analyse(mk, ds, path):
    df = pd.read_parquet(path)
    j = df[df["qrel_label"] >= 0].reset_index(drop=True)
    if len(j) < 20:
        return None
    y = j["qrel_label"].values.astype(int)
    lg = j["ce_logit"].values.astype(np.float64)
    pr = sigmoid(lg)

    try:
        i_f, i_t = train_test_split(np.arange(len(y)), test_size=CALIBRATION_SPLIT,
                                    stratify=y, random_state=RANDOM_SEED)
    except ValueError:
        i_f, i_t = train_test_split(np.arange(len(y)), test_size=CALIBRATION_SPLIT,
                                    random_state=RANDOM_SEED)
    lf, yf, lt, yt = lg[i_f], y[i_f], lg[i_t], y[i_t]

    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    r = {"model": mk, "dataset": ds, "n_judged": len(y),
         "prevalence": float(y.mean()), "n_pos": n_pos, "n_neg": n_neg,
         "n_minority": min(n_pos, n_neg),
         "valid": bool(min(n_pos, n_neg) >= MIN_MINORITY
                       and float(y.mean()) <= MAX_PREVALENCE),
         "raw_ece": compute_ece(sigmoid(lt), yt, ECE_BINS),
         "raw_mce": compute_mce(sigmoid(lt), yt, ECE_BINS),
         "raw_brier": compute_brier(sigmoid(lt), yt),
         "raw_ece_full": compute_ece(pr, y, ECE_BINS)}

    T = fit_temperature(lf, yf)
    r["temp_T"] = T
    r["temp_ece"] = compute_ece(sigmoid(lt / T), yt, ECE_BINS)

    if len(np.unique(yf)) < 2:
        r["platt_ece"] = np.nan
    else:
        pl = LogisticRegression(C=1e10, max_iter=10_000).fit(lf.reshape(-1, 1), yf)
        r["platt_ece"] = compute_ece(pl.predict_proba(lt.reshape(-1, 1))[:, 1], yt, ECE_BINS)

    ir = IsotonicRegression(out_of_bounds="clip").fit(sigmoid(lf), yf.astype(float))
    r["iso_ece"] = compute_ece(ir.predict(sigmoid(lt)), yt, ECE_BINS)
    return r


def main():
    rows = []
    for mk in list(MODELS) + list(EXTRA_MODELS):
        d = SCORES_DIR / mk
        if not d.exists():
            continue
        for p in sorted(d.glob("*.parquet")):
            r = analyse(mk, p.stem, p)
            if r:
                r["model_group"] = "original" if mk in MODELS else "extension"
                rows.append(r)

    df = pd.DataFrame(rows)
    df.to_csv(CALIBRATION_DIR / "calibration_extended.csv", index=False)
    print(f"{len(df)} model-dataset pairs "
          f"({df.model.nunique()} models x {df.dataset.nunique()} collections)\n")

    # ── validity diagnostic over all collections ────────────────────────────
    v = (df.groupby("dataset")
           .agg(n_judged=("n_judged", "first"), prevalence=("prevalence", "first"),
                n_neg=("n_neg", "first"), n_minority=("n_minority", "first"),
                valid=("valid", "first"))
           .sort_values("prevalence"))
    v.to_csv(CALIBRATION_DIR / "validity_extended.csv")
    print("VALIDITY DIAGNOSTIC (all collections)")
    print(f"{'dataset':<20}{'n':>8}{'prev':>7}{'n_neg':>8}{'n_min':>8}{'verdict':>10}")
    for ds, r in v.iterrows():
        print(f"{ds:<20}{int(r.n_judged):>8}{r.prevalence:>7.2f}{int(r.n_neg):>8}"
              f"{int(r.n_minority):>8}{'VALID' if r.valid else 'invalid':>10}")

    core = sorted(v[v.valid].index)
    print(f"\nCore subset ({len(core)}): {', '.join(core)}")

    # ── original models on the core subset ──────────────────────────────────
    o = df[(df.model_group == "original") & (df.dataset.isin(core))]
    print(f"\nORIGINAL MODELS, CORE SUBSET (n={len(o)})")
    print(f"  raw={o.raw_ece.mean():.4f}  temp={o.temp_ece.mean():.4f}  "
          f"platt={o.platt_ece.mean():.4f}  iso={o.iso_ece.mean():.4f}")
    print("\n  per collection:")
    print(o.groupby("dataset")[["raw_ece", "temp_ece", "platt_ece", "iso_ece"]]
           .mean().round(4).to_string())

    # ── extension models ────────────────────────────────────────────────────
    e = df[df.model_group == "extension"]
    if len(e):
        print(f"\nEXTENSION MODELS (n={len(e)})")
        print(e.groupby("model")[["raw_ece", "temp_T", "temp_ece",
                                  "platt_ece", "iso_ece"]].mean().round(4).to_string())
        ec = e[e.dataset.isin(core)]
        if len(ec):
            print(f"\n  on core subset (n={len(ec)}): raw={ec.raw_ece.mean():.4f} "
                  f"temp={ec.temp_ece.mean():.4f} platt={ec.platt_ece.mean():.4f} "
                  f"iso={ec.iso_ece.mean():.4f}")
    print("\nWrote calibration_extended.csv, validity_extended.csv")


if __name__ == "__main__":
    main()
