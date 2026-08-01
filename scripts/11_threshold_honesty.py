#!/usr/bin/env python3
"""Realised precision at a fixed accept threshold.

For each model and collection, accept pairs with p >= tau and record the precision
and coverage of the accepted set, for raw scores and after each calibrator.
Restricted to the collections that pass the validity diagnostic.

Outputs:
  data/calibration/threshold_honesty.csv
  data/calibration/risk_coverage.csv
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
from config import MODELS, SCORES_DIR, CALIBRATION_DIR, CALIBRATION_SPLIT, RANDOM_SEED

def _core():
    f = CALIBRATION_DIR / "validity_extended.csv"
    if f.exists():
        v = pd.read_csv(f)
        return sorted(v[v.valid].dataset.tolist())
    return ["dbpedia-entity", "trec-covid", "webis-touche2020"]

CLEAN = _core()      # collections passing the validity diagnostic
CORE  = CLEAN
TAUS  = [0.5, 0.6, 0.7, 0.8, 0.9]


def fit_temperature(logits, labels, hi=1000.0):
    labels = labels.astype(float)

    def nll(T):
        p = np.clip(sigmoid(logits / T), 1e-7, 1 - 1e-7)
        return -np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p))

    return float(minimize_scalar(nll, bounds=(0.01, hi), method="bounded").x)


def load(mk, ds):
    p = SCORES_DIR / mk / f"{ds}.parquet"
    if not p.exists():
        return None
    d = pd.read_parquet(p)
    j = d[d["qrel_label"] >= 0].reset_index(drop=True)
    if len(j) < 20:
        return None
    y = j["qrel_label"].values.astype(int)
    lg = j["ce_logit"].values
    try:
        f, t = train_test_split(np.arange(len(y)), test_size=CALIBRATION_SPLIT,
                                stratify=y, random_state=RANDOM_SEED)
    except ValueError:
        f, t = train_test_split(np.arange(len(y)), test_size=CALIBRATION_SPLIT,
                                random_state=RANDOM_SEED)
    return lg[f], y[f], lg[t], y[t]


MIN_ACCEPTED = 30   # below this the realised precision is too noisy to report


def realised(probs, labels, tau):
    """Precision, coverage, and support when accepting everything with p >= tau.

    Precision is returned as NaN when fewer than MIN_ACCEPTED items clear the
    threshold: with a handful of accepted items the estimate is dominated by
    sampling noise and would otherwise contaminate the averages.
    """
    m = probs >= tau
    n_acc = int(m.sum())
    if n_acc < MIN_ACCEPTED:
        return np.nan, float(m.mean()), n_acc
    return float(labels[m].mean()), float(m.mean()), n_acc


def main():
    cache = {}
    for mk in MODELS:
        for ds in CLEAN:
            s = load(mk, ds)
            if s is not None:
                cache[(mk, ds)] = s

    rows, rc_rows = [], []
    for mk in MODELS:
        for ds in CLEAN:
            if (mk, ds) not in cache:
                continue
            lf, yf, lt, yt = cache[(mk, ds)]

            T = fit_temperature(lf, yf)
            variants = {"raw": sigmoid(lt), "temp": sigmoid(lt / T)}
            try:
                pl = LogisticRegression(C=1e10, max_iter=10_000).fit(lf.reshape(-1, 1), yf)
                variants["platt"] = pl.predict_proba(lt.reshape(-1, 1))[:, 1]
            except ValueError:
                pass
            try:
                ir = IsotonicRegression(out_of_bounds="clip").fit(sigmoid(lf), yf.astype(float))
                variants["iso"] = ir.predict(sigmoid(lt))
            except ValueError:
                pass

            # temperature fitted on the other collections (leave-one-domain-out)
            pool_l = [cache[(mk, s)][0] for s in CLEAN if s != ds and (mk, s) in cache]
            pool_y = [cache[(mk, s)][1] for s in CLEAN if s != ds and (mk, s) in cache]
            if pool_l:
                T_lodo = fit_temperature(np.concatenate(pool_l), np.concatenate(pool_y))
                variants["temp_lodo"] = sigmoid(lt / T_lodo)

            for name, p in variants.items():
                for tau in TAUS:
                    prec, cov, n_acc = realised(p, yt, tau)
                    rows.append({"model": mk, "dataset": ds, "method": name,
                                 "tau": tau, "precision": prec, "coverage": cov,
                                 "n_accepted": n_acc, "n_test": len(yt),
                                 "gap": (tau - prec) if not np.isnan(prec) else np.nan,
                                 "prevalence": float(yt.mean())})
                # dense sweep for risk-coverage curves
                for tau in np.linspace(0.01, 0.99, 99):
                    prec, cov, n_acc = realised(p, yt, tau)
                    rc_rows.append({"model": mk, "dataset": ds, "method": name,
                                    "tau": float(tau), "precision": prec,
                                    "coverage": cov, "n_accepted": n_acc})

    df = pd.DataFrame(rows)
    pd.DataFrame(rc_rows).to_csv(CALIBRATION_DIR / "risk_coverage.csv", index=False)
    df.to_csv(CALIBRATION_DIR / "threshold_honesty.csv", index=False)

    core = df[df.dataset.isin(CORE)]

    print("=" * 78)
    print("THRESHOLD HONESTY  |  gap = tau - realised precision  (positive = over-promise)")
    print("=" * 78)
    print("\nMean gap by method and target threshold (core 3 balanced domains):")
    piv = core.pivot_table(index="method", columns="tau", values="gap", aggfunc="mean")
    order = [m for m in ["raw", "temp_lodo", "temp", "platt", "iso"] if m in piv.index]
    print(piv.loc[order].round(3).to_string())

    print("\nMean |gap| (absolute miscalibration of the operating point):")
    core2 = core.assign(absgap=core.gap.abs())
    piv2 = core2.pivot_table(index="method", columns="tau", values="absgap", aggfunc="mean")
    print(piv2.loc[order].round(3).to_string())
    print("\nOverall mean |gap|:")
    for m in order:
        s = core2[core2.method == m]
        print(f"  {m:<11} {s.absgap.mean():.4f}   (coverage {s.coverage.mean():.3f})")

    print("\n" + "-" * 78)
    print("Worked example: operator sets tau = 0.8 ('at least 80% likely relevant')")
    print("-" * 78)
    ex = core[core.tau == 0.8]
    t = ex.pivot_table(index="dataset", columns="method", values="precision")
    t = t[[c for c in order if c in t.columns]]
    print("Realised precision on the accepted set:")
    print(t.round(3).to_string())
    print("\nCross-domain spread of realised precision (max - min):")
    for m in order:
        if m in t.columns:
            print(f"  {m:<11} spread={t[m].max()-t[m].min():.3f}  "
                  f"mean={t[m].mean():.3f}  (target 0.800)")

    print("\n" + "-" * 78)
    print("Coverage at tau=0.8 (fraction of judged pairs accepted)")
    print("-" * 78)
    c = ex.pivot_table(index="dataset", columns="method", values="coverage")
    print(c[[x for x in order if x in c.columns]].round(3).to_string())

    print("\nWrote threshold_honesty.csv, risk_coverage.csv")


if __name__ == "__main__":
    main()
