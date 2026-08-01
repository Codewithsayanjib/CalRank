#!/usr/bin/env python3
"""Build TREC DL 2019/2020 passage scoring inputs.

Joins the qrels with the official top-1000 pool to recover judged
(query, passage) pairs with text. The pool file is not rank-ordered, so every
judged pair in it is kept rather than a top-100 prefix.

Grading: judgments are 0-3. qrel_label uses the >= 2 (TREC) convention;
qrel_label_lenient uses >= 1 (BEIR).

Outputs:
  data/trecdl/trec-dl-2019.parquet
  data/trecdl/trec-dl-2020.parquet
"""
import gzip
import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR

TRECDL = DATA_DIR / "trecdl"


def load_qrels(path):
    """TREC qrels: qid Q0 pid grade."""
    q = {}
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) < 4:
                continue
            q[(p[0], p[2])] = int(p[3])
    return q


def build(year):
    qrels = load_qrels(TRECDL / f"{year}qrels-pass.txt")
    want = set(qrels)
    rows, seen = [], set()

    with gzip.open(TRECDL / f"msmarco-passagetest{year}-top1000.tsv.gz", "rt",
                   encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            qid, pid, query, passage = parts[0], parts[1], parts[2], parts[3]
            key = (qid, pid)
            if key in want and key not in seen:
                seen.add(key)
                g = qrels[key]
                rows.append({
                    "query_id": qid, "doc_id": pid,
                    "query": query, "passage": passage,
                    "grade": g,
                    "qrel_label": 1 if g >= 2 else 0,          # TREC convention
                    "qrel_label_lenient": 1 if g >= 1 else 0,  # BEIR convention
                })

    df = pd.DataFrame(rows)
    out = TRECDL / f"trec-dl-{year}.parquet"
    df.to_parquet(out, index=False)

    cov = len(seen) / len(want)
    print(f"\nTREC DL {year}")
    print(f"  judged pairs in qrels      : {len(want)}")
    print(f"  recovered from top-1000    : {len(df)}  ({cov:.1%} coverage)")
    print(f"  queries                    : {df.query_id.nunique()}")
    print(f"  grade distribution         : {df.grade.value_counts().sort_index().to_dict()}")
    print(f"  strict  (>=2): pos={int(df.qrel_label.sum())} "
          f"neg={int((1-df.qrel_label).sum())} prevalence={df.qrel_label.mean():.3f}")
    print(f"  lenient (>=1): pos={int(df.qrel_label_lenient.sum())} "
          f"neg={int((1-df.qrel_label_lenient).sum())} "
          f"prevalence={df.qrel_label_lenient.mean():.3f}")
    print(f"  -> {out}")
    return df


if __name__ == "__main__":
    for y in (2019, 2020):
        build(y)
