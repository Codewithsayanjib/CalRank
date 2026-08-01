#!/usr/bin/env python3
"""
Build BM25 indices for all 8 BEIR datasets using bm25s (pure Python, no Java).

BM25 parameters k1=0.9, b=0.4 match Anserini defaults.
Index + doc-id mapping saved to data/indexes/<dataset>/bm25s/
"""
import sys
import json
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DATASETS, BEIR_DATA_DIR, INDEX_DIR, DATASET_SPLITS
from beir.datasets.data_loader import GenericDataLoader
import bm25s


def build_index(dataset: str):
    index_dir = INDEX_DIR / dataset / "bm25s"
    docid_file = index_dir / "docids.json"

    if index_dir.exists() and docid_file.exists():
        print(f"  Index exists ({index_dir}), skipping.")
        return

    index_dir.mkdir(parents=True, exist_ok=True)

    data_path = BEIR_DATA_DIR / dataset
    split     = DATASET_SPLITS[dataset]
    corpus, _, _ = GenericDataLoader(data_folder=str(data_path)).load(split=split)

    print(f"  Building index for {len(corpus):,} docs …")

    # Ordered list of doc IDs so we can map BM25 result indices back to doc IDs
    doc_ids   = list(corpus.keys())
    doc_texts = []
    for doc_id in doc_ids:
        doc = corpus[doc_id]
        title = doc.get("title", "").strip()
        text  = doc.get("text",  "").strip()
        doc_texts.append(f"{title} {text}".strip() if title else (text or " "))

    # Tokenise (lowercase + English stop words; no stemming for BM25 reproducibility)
    corpus_tokens = bm25s.tokenize(doc_texts, stopwords="english", show_progress=False)

    retriever = bm25s.BM25(k1=0.9, b=0.4)
    retriever.index(corpus_tokens, show_progress=False)
    retriever.save(str(index_dir))

    with open(docid_file, "w") as f:
        json.dump(doc_ids, f)

    print(f"  Index saved → {index_dir}  ({len(doc_ids):,} docs)")


def main():
    for dataset in DATASETS:
        print(f"\n[{dataset}]")
        build_index(dataset)
    print("\nAll BM25 indices built.")


if __name__ == "__main__":
    main()
