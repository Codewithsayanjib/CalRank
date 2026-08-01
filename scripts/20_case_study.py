#!/usr/bin/env python3
"""
Qualitative case study: query-document pairs whose accept/reject decision at a
threshold of tau = 0.8 flips after calibration.

For a chosen model on TREC DL 2019 (which carries query and passage text), fit
temperature and isotonic calibrators on the stratified fit half and apply them to
the held-out half. Identify test pairs where the raw sigmoid and the calibrated
probability fall on opposite sides of tau, and surface those where calibration
corrected the decision, i.e. the raw score accepted a non-relevant passage
(qrel = 0) that calibration then rejected.

Output: data/calibration/case_study.csv
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.special import expit as sigmoid
from scipy.optimize import minimize_scalar
from sklearn.model_selection import train_test_split
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import SCORES_DIR, DATA_DIR, CALIBRATION_DIR, CALIBRATION_SPLIT, RANDOM_SEED

TAU = 0.8
MODEL = "minilm-l6"        # a default re-ranker in LangChain/LlamaIndex
YEAR = 2019


def fit_temperature(logits, labels, hi=1000.0):
    labels = labels.astype(float)
    def nll(T):
        p = np.clip(sigmoid(logits / T), 1e-7, 1 - 1e-7)
        return -np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p))
    return float(minimize_scalar(nll, bounds=(0.01, hi), method="bounded").x)


def clip_text(t, n=90):
    t = " ".join(str(t).split())
    return t if len(t) <= n else t[: n - 1] + "…"


def main():
    # scored logits (with labels) and the text table, joined on (query_id, doc_id)
    sc = pd.read_parquet(SCORES_DIR / MODEL / f"trec-dl-{YEAR}.parquet")
    tx = pd.read_parquet(DATA_DIR / "trecdl" / f"trec-dl-{YEAR}.parquet")
    sc["query_id"] = sc["query_id"].astype(str)
    sc["doc_id"] = sc["doc_id"].astype(str)
    tx["query_id"] = tx["query_id"].astype(str)
    tx["doc_id"] = tx["doc_id"].astype(str)
    df = sc.merge(tx[["query_id", "doc_id", "query", "passage"]],
                  on=["query_id", "doc_id"], how="left")

    y = df["qrel_label"].values.astype(int)
    lg = df["ce_logit"].values.astype(np.float64)
    idx_fit, idx_test = train_test_split(
        np.arange(len(df)), test_size=CALIBRATION_SPLIT,
        stratify=y, random_state=RANDOM_SEED)

    T = fit_temperature(lg[idx_fit], y[idx_fit])
    iso = IsotonicRegression(out_of_bounds="clip").fit(sigmoid(lg[idx_fit]),
                                                       y[idx_fit].astype(float))

    test = df.iloc[idx_test].copy()
    test["p_raw"] = sigmoid(test["ce_logit"].values)
    test["p_temp"] = sigmoid(test["ce_logit"].values / T)
    test["p_iso"] = iso.predict(test["p_raw"].values)

    acc_raw = test.p_raw >= TAU
    acc_iso = test.p_iso >= TAU

    # decision flips
    n_flip = int((acc_raw != acc_iso).sum())
    # the corrective case: raw accepts, isotonic rejects, and the pair is non-relevant
    good = test[acc_raw & ~acc_iso & (test.qrel_label == 0)].copy()
    good = good.sort_values("p_raw", ascending=False)

    print(f"Model {MODEL} on TREC DL {YEAR}, tau={TAU}, T={T:.2f}")
    print(f"held-out pairs: {len(test)}   accept-decision flips (raw vs iso): {n_flip}")
    print(f"corrective flips (raw accepted a non-relevant passage that iso rejected): "
          f"{len(good)}\n")

    cols = ["query", "passage", "qrel_label", "p_raw", "p_temp", "p_iso"]
    out = good[cols].head(8).reset_index(drop=True)
    for i, r in out.iterrows():
        print(f"[{i+1}] q: {clip_text(r.query, 70)}")
        print(f"    d: {clip_text(r.passage, 95)}")
        print(f"    relevance={int(r.qrel_label)}  p_raw={r.p_raw:.2f}  "
              f"p_temp={r.p_temp:.2f}  p_iso={r.p_iso:.2f}\n")

    out.to_csv(CALIBRATION_DIR / "case_study.csv", index=False)
    # also report the reverse direction for completeness
    rescue = test[~acc_raw & acc_iso & (test.qrel_label == 1)]
    print(f"(For completeness: iso accepted a relevant passage the raw score "
          f"rejected in {len(rescue)} cases.)")
    print("\nWrote case_study.csv")


if __name__ == "__main__":
    main()
