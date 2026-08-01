#!/usr/bin/env python3
"""
Score every BM25 candidate pair with each cross-encoder model.

Outputs per (model, dataset):
  data/scores/<model>/<dataset>.parquet
  columns: query_id, doc_id, bm25_rank, ce_logit, ce_prob, qrel_label

qrel_label encoding:
  -1 = unjudged (not in qrels; excluded from calibration analysis)
   0 = judged not relevant
   1 = judged relevant (binarised from any qrel >= 1)

Usage:
  python scripts/04_ce_score.py --session 1          # small models
  python scripts/04_ce_score.py --session 2          # BGE only
  python scripts/04_ce_score.py --models minilm-l6   # single model
  python scripts/04_ce_score.py                      # all models
"""
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (MODELS, DATASETS, SESSION1_MODELS, SESSION2_MODELS,
                    BEIR_DATA_DIR, RETRIEVAL_DIR, SCORES_DIR,
                    BATCH_SIZES, TRUST_REMOTE_CODE, DATASET_SPLITS, RANDOM_SEED)
from beir.datasets.data_loader import GenericDataLoader
from sentence_transformers import CrossEncoder
from scipy.special import expit as sigmoid

np.random.seed(RANDOM_SEED)


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_retrieval(dataset: str) -> dict:
    """Returns {query_id: [[doc_id, bm25_score], ...]} ordered by BM25 rank."""
    path = RETRIEVAL_DIR / f"{dataset}.jsonl"
    results = {}
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            results[row["query_id"]] = row["hits"]
    return results


def load_beir(dataset: str):
    data_path = BEIR_DATA_DIR / dataset
    split = DATASET_SPLITS[dataset]
    corpus, queries, qrels = GenericDataLoader(data_folder=str(data_path)).load(split=split)
    corpus_text = {}
    for doc_id, doc in corpus.items():
        title = doc.get("title", "").strip()
        text  = doc.get("text",  "").strip()
        corpus_text[doc_id] = f"{title} {text}".strip() if title else (text or "[empty]")
    return corpus_text, queries, qrels


def binarize_label(raw: int) -> int:
    """Convert graded qrel to binary; -1 means unjudged."""
    if raw < 0:
        return -1
    return 1 if raw >= 1 else 0


def predict_logits(model: CrossEncoder, pairs: list, batch_size: int) -> np.ndarray:
    """Run CrossEncoder inference, returning raw logits (before sigmoid)."""
    all_logits = []
    model.model.eval()
    for i in tqdm(range(0, len(pairs), batch_size), desc="  batches", leave=False):
        batch = pairs[i : i + batch_size]
        # activation_fct=identity returns logits before any default activation
        scores = model.predict(
            batch,
            activation_fn=torch.nn.Identity(),
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        # scores may be (N,) or (N,1) depending on model head
        scores = np.asarray(scores).flatten()
        all_logits.append(scores)
    return np.concatenate(all_logits).astype(np.float32)


def score_dataset(model_key: str, dataset: str, model: CrossEncoder):
    out_file = SCORES_DIR / model_key / f"{dataset}.parquet"
    if out_file.exists():
        print(f"    [skip] {model_key}/{dataset}")
        return

    corpus_text, queries, qrels = load_beir(dataset)
    retrieval = load_retrieval(dataset)

    records = []
    for qid, hits in retrieval.items():
        query_text  = queries.get(qid, "")
        query_qrels = qrels.get(qid, {})
        for rank, (doc_id, _) in enumerate(hits, start=1):
            doc_text = corpus_text.get(doc_id, "[empty]")
            raw_rel  = query_qrels.get(doc_id, -1)
            records.append((qid, doc_id, rank, query_text, doc_text, binarize_label(raw_rel)))

    pairs = [(r[3], r[4]) for r in records]   # (query_text, doc_text)
    print(f"    Scoring {len(pairs):,} pairs for {dataset} …")

    logits = predict_logits(model, pairs, BATCH_SIZES[model_key])

    df = pd.DataFrame({
        "query_id":   [r[0] for r in records],
        "doc_id":     [r[1] for r in records],
        "bm25_rank":  np.array([r[2] for r in records], dtype=np.int16),
        "ce_logit":   logits,
        "ce_prob":    sigmoid(logits).astype(np.float32),
        "qrel_label": np.array([r[5] for r in records], dtype=np.int8),
    })

    out_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_file, index=False)
    judged = (df["qrel_label"] >= 0).sum()
    print(f"    Saved {len(df):,} rows ({judged:,} judged) → {out_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models",   nargs="+", default=None, help="model keys")
    parser.add_argument("--session",  choices=["1", "2"], default=None)
    parser.add_argument("--datasets", nargs="+", default=None)
    args = parser.parse_args()

    if args.models:
        model_keys = args.models
    elif args.session == "1":
        model_keys = SESSION1_MODELS
    elif args.session == "2":
        model_keys = SESSION2_MODELS
    else:
        model_keys = list(MODELS.keys())

    datasets = args.datasets or DATASETS
    device   = get_device()
    print(f"Device: {device}\n")

    for model_key in model_keys:
        model_name = MODELS[model_key]
        print(f"[model: {model_key}]  {model_name}")

        kwargs = {"device": device}
        if model_key in TRUST_REMOTE_CODE:
            kwargs["trust_remote_code"] = True

        model = CrossEncoder(model_name, **kwargs)

        for dataset in datasets:
            score_dataset(model_key, dataset, model)

        del model
        if device == "mps":
            torch.mps.empty_cache()
        print()

    print("CE scoring complete.")


if __name__ == "__main__":
    main()
