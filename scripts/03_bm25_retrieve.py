#!/usr/bin/env python3
"""BM25 top-100 retrieval using bm25s (pure Python, no Java)."""
import sys
import json
import random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (DATASETS, BEIR_DATA_DIR, INDEX_DIR, RETRIEVAL_DIR,
                    DATASET_SPLITS, BM25_TOP_K, QUORA_QUERY_SAMPLE, RANDOM_SEED)
from beir.datasets.data_loader import GenericDataLoader
import bm25s
from tqdm import tqdm

random.seed(RANDOM_SEED)


def retrieve(dataset: str):
    out_file = RETRIEVAL_DIR / f"{dataset}.jsonl"
    if out_file.exists():
        print(f"  [skip] {dataset}  (already at {out_file})")
        return

    # Load index and doc-id mapping
    index_dir  = INDEX_DIR / dataset / "bm25s"
    docid_file = index_dir / "docids.json"
    if not docid_file.exists():
        sys.exit(f"[error] Index not found for {dataset}. Run 02_build_bm25_index.py first.")

    retriever = bm25s.BM25.load(str(index_dir), load_corpus=False)
    with open(docid_file) as f:
        doc_ids = json.load(f)

    # Load queries
    data_path = BEIR_DATA_DIR / dataset
    split     = DATASET_SPLITS[dataset]
    _, queries, _ = GenericDataLoader(data_folder=str(data_path)).load(split=split)

    # Subsample Quora to keep compute tractable
    if dataset == "quora" and len(queries) > QUORA_QUERY_SAMPLE:
        sampled = random.sample(list(queries.keys()), QUORA_QUERY_SAMPLE)
        queries = {qid: queries[qid] for qid in sampled}
        print(f"  Quora: subsampled to {len(queries)} queries")

    print(f"  Retrieving top-{BM25_TOP_K} for {len(queries):,} queries …")

    qids   = list(queries.keys())
    qtexts = [queries[qid] for qid in qids]

    # Tokenise all queries at once, then retrieve in one batch call
    query_tokens = bm25s.tokenize(qtexts, stopwords="english", show_progress=False)
    results, scores = retriever.retrieve(query_tokens, k=BM25_TOP_K, show_progress=True)
    # results shape: (n_queries, k)  — integer indices into doc_ids
    # scores  shape: (n_queries, k)

    RETRIEVAL_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        for i, qid in enumerate(qids):
            hits = [
                [doc_ids[results[i, j]], float(scores[i, j])]
                for j in range(results.shape[1])
            ]
            f.write(json.dumps({"query_id": qid, "hits": hits}) + "\n")

    print(f"  Saved {len(qids):,} queries → {out_file}")


def main():
    for dataset in DATASETS:
        print(f"\n[{dataset}]")
        retrieve(dataset)
    print("\nBM25 retrieval complete.")


if __name__ == "__main__":
    main()
