#!/usr/bin/env python3
"""
Downstream retrieval impact of calibration.

Metrics computed before and after temperature scaling:
  - nDCG@10   (expected: no change — temperature scaling is monotone)
  - MAP@100
  - Recall@100
  - Abstention quality at p > 0.5 threshold

Outputs:
  data/calibration/downstream_analysis.csv
"""
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (MODELS, DATASETS, SCORES_DIR, CALIBRATION_DIR,
                    BEIR_DATA_DIR, DATASET_SPLITS, ABSTENTION_THRESHOLD)
from beir.datasets.data_loader import GenericDataLoader
from utils import (apply_temperature, compute_ndcg10, compute_map,
                   compute_recall_at_k, compute_abstention_quality)


def load_qrels(dataset: str) -> dict:
    """Returns raw (graded) qrels: {query_id: {doc_id: int}}."""
    data_path = BEIR_DATA_DIR / dataset
    _, _, qrels = GenericDataLoader(data_folder=str(data_path)).load(
        split=DATASET_SPLITS[dataset]
    )
    return qrels


def binarize_qrels(qrels: dict) -> dict:
    return {
        qid: {doc: (1 if r >= 1 else 0) for doc, r in docs.items()}
        for qid, docs in qrels.items()
    }


def main():
    params_file = CALIBRATION_DIR / "calibration_params.json"
    if not params_file.exists():
        sys.exit("[error] Run 06_calibration_fit.py first to produce calibration_params.json")
    with open(params_file) as f:
        calib_params = json.load(f)

    rows = []
    for model_key in MODELS:
        for dataset in DATASETS:
            path = SCORES_DIR / model_key / f"{dataset}.parquet"
            if not path.exists():
                print(f"[missing] {model_key}/{dataset}")
                continue

            print(f"  {model_key:14s}  {dataset:20s} …", end=" ", flush=True)
            df      = pd.read_parquet(path)
            qrels   = load_qrels(dataset)
            qrels_b = binarize_qrels(qrels)

            # Before calibration: rank by raw sigmoid prob (monotone with logit)
            raw_df = df[["query_id", "doc_id", "ce_prob"]].rename(columns={"ce_prob": "score"})
            ndcg_raw   = compute_ndcg10(raw_df, qrels)
            map_raw    = compute_map(raw_df, qrels_b)
            recall_raw = compute_recall_at_k(raw_df, qrels_b)
            abst_raw   = compute_abstention_quality(raw_df, qrels_b, ABSTENTION_THRESHOLD)

            # After temperature scaling
            T = calib_params.get(model_key, {}).get(dataset, {}).get("temperature")
            if T:
                cal_df = df[["query_id", "doc_id", "ce_logit"]].copy()
                cal_df["score"] = apply_temperature(df["ce_logit"].values, T)
                ndcg_cal   = compute_ndcg10(cal_df, qrels)
                map_cal    = compute_map(cal_df, qrels_b)
                recall_cal = compute_recall_at_k(cal_df, qrels_b)
                abst_cal   = compute_abstention_quality(cal_df, qrels_b, ABSTENTION_THRESHOLD)
            else:
                ndcg_cal = map_cal = recall_cal = float("nan")
                abst_cal = {"precision_at_accept": float("nan"),
                            "accept_rate": float("nan"),
                            "abstain_rate": float("nan")}

            rows.append({
                "model":   model_key,
                "dataset": dataset,
                # Ranking (should be identical before/after monotone transform)
                "ndcg10_raw":       ndcg_raw,
                "ndcg10_cal":       ndcg_cal,
                "ndcg10_delta":     ndcg_cal - ndcg_raw if T else float("nan"),
                "map_raw":          map_raw,
                "recall100_raw":    recall_raw,
                # Abstention (should change with calibration)
                "abst_prec_raw":    abst_raw["precision_at_accept"],
                "abst_prec_cal":    abst_cal["precision_at_accept"],
                "accept_rate_raw":  abst_raw["accept_rate"],
                "accept_rate_cal":  abst_cal["accept_rate"],
                "abstain_rate_raw": abst_raw["abstain_rate"],
                "abstain_rate_cal": abst_cal["abstain_rate"],
            })

            d = rows[-1]
            print(f"nDCG@10={d['ndcg10_raw']:.4f}  "
                  f"abst_prec raw={d['abst_prec_raw']:.4f} → cal={d['abst_prec_cal']:.4f}")

    out = pd.DataFrame(rows)
    out_file = CALIBRATION_DIR / "downstream_analysis.csv"
    out.to_csv(out_file, index=False)
    print(f"\nSaved → {out_file}")

    # Sanity check: nDCG delta should be near 0
    delta = out["ndcg10_delta"].dropna()
    print(f"\n=== nDCG@10 delta after temp scaling (should be ~0) ===")
    print(f"  mean={delta.mean():.6f}  max_abs={delta.abs().max():.6f}")

    print(f"\n=== Abstention precision (raw → calibrated) ===")
    print(out[["model", "dataset", "abst_prec_raw", "abst_prec_cal",
               "accept_rate_raw", "accept_rate_cal"]].to_string(index=False))


if __name__ == "__main__":
    main()
