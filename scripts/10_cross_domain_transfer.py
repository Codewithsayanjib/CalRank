#!/usr/bin/env python3
"""Cross-domain calibrator transfer.

For each model and target collection, fit calibrators three ways and evaluate each
on the target's held-out split:
  oracle  in-domain (target's own fit split)
  single  on one other collection
  lodo    on the pooled other collections (leave-one-domain-out)

Restricted to the collections that pass the validity diagnostic.

Outputs:
  data/calibration/transfer_matrix.csv
  data/calibration/transfer_summary.csv
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.special import expit as sigmoid
from scipy.optimize import minimize_scalar
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (MODELS, SCORES_DIR, CALIBRATION_DIR,
                    CALIBRATION_SPLIT, RANDOM_SEED, ECE_BINS)
from utils import compute_ece

# Collections that pass the validity diagnostic, read from the extended run so
# this stays in step with 17_calibration_extended.py rather than being hardcoded.
def _core():
    f = CALIBRATION_DIR / "validity_extended.csv"
    if f.exists():
        v = pd.read_csv(f)
        return sorted(v[v.valid].dataset.tolist())
    return ["dbpedia-entity", "trec-covid", "webis-touche2020"]

CLEAN = None  # set in main()


def fit_temperature(logits, labels, hi=1000.0):
    labels = labels.astype(float)

    def nll(T):
        p = np.clip(sigmoid(logits / T), 1e-7, 1 - 1e-7)
        return -np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p))

    return float(minimize_scalar(nll, bounds=(0.01, hi), method="bounded").x)


def load_splits(mk, ds):
    """Return (logit_fit, y_fit, logit_test, y_test) using the canonical split."""
    path = SCORES_DIR / mk / f"{ds}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    j = df[df["qrel_label"] >= 0].reset_index(drop=True)
    if len(j) < 20:
        return None
    y = j["qrel_label"].values.astype(int)
    lg = j["ce_logit"].values
    try:
        i_f, i_t = train_test_split(np.arange(len(y)), test_size=CALIBRATION_SPLIT,
                                    stratify=y, random_state=RANDOM_SEED)
    except ValueError:
        i_f, i_t = train_test_split(np.arange(len(y)), test_size=CALIBRATION_SPLIT,
                                    random_state=RANDOM_SEED)
    return lg[i_f], y[i_f], lg[i_t], y[i_t]


def main():
    global CLEAN
    CLEAN = _core()
    print(f"core subset ({len(CLEAN)}): {', '.join(CLEAN)}\n")
    cache = {}
    for mk in MODELS:
        for ds in CLEAN:
            s = load_splits(mk, ds)
            if s is not None:
                cache[(mk, ds)] = s

    rows = []
    for mk in MODELS:
        for tgt in CLEAN:
            if (mk, tgt) not in cache:
                continue
            _, _, lt, yt = cache[(mk, tgt)]

            raw = compute_ece(sigmoid(lt), yt, ECE_BINS)

            # in-domain oracle
            lf, yf, _, _ = cache[(mk, tgt)]
            T_or = fit_temperature(lf, yf)
            pl_or = LogisticRegression(C=1e10, max_iter=10_000).fit(lf.reshape(-1, 1), yf)
            rows.append({
                "model": mk, "source": tgt, "target": tgt, "regime": "oracle",
                "T": T_or,
                "ece_temp": compute_ece(sigmoid(lt / T_or), yt, ECE_BINS),
                "ece_platt": compute_ece(pl_or.predict_proba(lt.reshape(-1, 1))[:, 1], yt, ECE_BINS),
                "ece_raw": raw,
            })

            # single-source transfer
            for src in CLEAN:
                if src == tgt or (mk, src) not in cache:
                    continue
                lf_s, yf_s, _, _ = cache[(mk, src)]
                T_s = fit_temperature(lf_s, yf_s)
                try:
                    pl_s = LogisticRegression(C=1e10, max_iter=10_000).fit(lf_s.reshape(-1, 1), yf_s)
                    e_pl = compute_ece(pl_s.predict_proba(lt.reshape(-1, 1))[:, 1], yt, ECE_BINS)
                except ValueError:
                    e_pl = np.nan
                rows.append({
                    "model": mk, "source": src, "target": tgt, "regime": "single",
                    "T": T_s,
                    "ece_temp": compute_ece(sigmoid(lt / T_s), yt, ECE_BINS),
                    "ece_platt": e_pl, "ece_raw": raw,
                })

            # leave-one-domain-out pooled
            pool_l = [cache[(mk, s)][0] for s in CLEAN if s != tgt and (mk, s) in cache]
            pool_y = [cache[(mk, s)][1] for s in CLEAN if s != tgt and (mk, s) in cache]
            if pool_l:
                Lp, Yp = np.concatenate(pool_l), np.concatenate(pool_y)
                T_p = fit_temperature(Lp, Yp)
                try:
                    pl_p = LogisticRegression(C=1e10, max_iter=10_000).fit(Lp.reshape(-1, 1), Yp)
                    e_pl = compute_ece(pl_p.predict_proba(lt.reshape(-1, 1))[:, 1], yt, ECE_BINS)
                except ValueError:
                    e_pl = np.nan
                rows.append({
                    "model": mk, "source": "LODO", "target": tgt, "regime": "lodo",
                    "T": T_p,
                    "ece_temp": compute_ece(sigmoid(lt / T_p), yt, ECE_BINS),
                    "ece_platt": e_pl, "ece_raw": raw,
                })

    df = pd.DataFrame(rows)
    df.to_csv(CALIBRATION_DIR / "transfer_matrix.csv", index=False)

    print("=" * 74)
    print("CROSS-DOMAIN CALIBRATOR TRANSFER  (clean subset, 5 models x 4 domains)")
    print("=" * 74)
    summ = (df.groupby("regime")[["ece_raw", "ece_temp", "ece_platt"]]
              .agg(["mean", "std"]).round(4))
    print(summ.to_string())

    base = df[df.regime == "oracle"]
    print(f"\n{'regime':<10}{'n':>5}{'temp ECE':>12}{'platt ECE':>12}"
          f"{'vs raw':>12}{'vs oracle':>12}")
    raw_mean = df.ece_raw.mean()
    or_t = base.ece_temp.mean()
    or_p = base.ece_platt.mean()
    for r in ["oracle", "lodo", "single"]:
        s = df[df.regime == r]
        print(f"{r:<10}{len(s):>5}{s.ece_temp.mean():>12.4f}{s.ece_platt.mean():>12.4f}"
              f"{s.ece_temp.mean()-raw_mean:>+12.4f}{s.ece_temp.mean()-or_t:>+12.4f}")

    # How often does a transferred calibrator still beat doing nothing?
    for r in ["lodo", "single"]:
        s = df[df.regime == r]
        wt = (s.ece_temp < s.ece_raw).sum()
        wp = (s.ece_platt < s.ece_raw).sum()
        print(f"\n{r}: temp beats raw in {wt}/{len(s)}; "
              f"platt beats raw in {wp}/{s.ece_platt.notna().sum()}")

    # Per-target LODO detail (the deployment-relevant view)
    print("\n" + "-" * 74)
    print("LEAVE-ONE-DOMAIN-OUT, per target domain (mean over 5 models)")
    print("-" * 74)
    lo = df[df.regime == "lodo"].groupby("target")[["ece_raw", "ece_temp", "ece_platt"]].mean()
    orc = df[df.regime == "oracle"].groupby("target")[["ece_temp", "ece_platt"]].mean()
    orc.columns = ["oracle_temp", "oracle_platt"]
    print(lo.join(orc).round(4).to_string())

    # Temperature dispersion: is one global T defensible?
    print("\n" + "-" * 74)
    print("Fitted temperature by source domain (per model)")
    print("-" * 74)
    piv = (df[df.regime == "oracle"].pivot_table(index="model", columns="target", values="T"))
    piv["ratio_max_min"] = piv.max(axis=1) / piv.min(axis=1)
    print(piv.round(2).to_string())

    df.groupby("regime")[["ece_raw", "ece_temp", "ece_platt"]].mean().round(4) \
      .to_csv(CALIBRATION_DIR / "transfer_summary.csv")
    print("\nWrote transfer_matrix.csv, transfer_summary.csv")


if __name__ == "__main__":
    main()
