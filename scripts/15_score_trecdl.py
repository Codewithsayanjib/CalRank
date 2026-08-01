#!/usr/bin/env python3
"""
Score the TREC DL judged pairs with cross-encoder re-rankers.

Handles both the five original models and any additional models passed on the
command line (used for the model-scope extension).

  python scripts/15_score_trecdl.py                     # the five original models
  python scripts/15_score_trecdl.py --models bge-large  # a named extra model

Outputs data/scores/<model>/trec-dl-<year>.parquet with the same schema as the
BEIR score files, so every downstream analysis reads them unchanged.
"""
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import MODELS, SCORES_DIR, BATCH_SIZES, DATA_DIR, RANDOM_SEED
from sentence_transformers import CrossEncoder
from scipy.special import expit as sigmoid

try:
    from config import EXTRA_MODELS, EXTRA_BATCH_SIZES, EXTRA_TRUST_REMOTE
except ImportError:
    EXTRA_MODELS, EXTRA_BATCH_SIZES, EXTRA_TRUST_REMOTE = {}, {}, set()

TRECDL = DATA_DIR / "trecdl"
YEARS = (2019, 2020)
np.random.seed(RANDOM_SEED)


def device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def score(model, pairs, bs):
    """Score with adaptive batch reduction: MPS OOM halves the batch and retries."""
    out, i = [], 0
    model.model.eval()
    with torch.no_grad():
        pbar = tqdm(total=len(pairs), desc="   pairs", leave=False)
        while i < len(pairs):
            try:
                s = model.predict(pairs[i:i + bs], activation_fn=torch.nn.Identity(),
                                  show_progress_bar=False, convert_to_numpy=True)
            except RuntimeError as e:
                if "out of memory" not in str(e).lower() or bs <= 1:
                    raise
                bs = max(1, bs // 2)
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                pbar.write(f"    OOM -> batch {bs}")
                continue
            s = np.asarray(s, dtype=np.float64)
            if s.ndim > 1:
                s = s[:, -1] if s.shape[1] == 2 else s.ravel()
            out.append(s)
            i += bs
            pbar.update(min(bs, len(pairs) - pbar.n))
        pbar.close()
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None)
    args = ap.parse_args()

    registry = {**MODELS, **EXTRA_MODELS}
    keys = args.models if args.models else list(MODELS)
    dev = device()
    print(f"device: {dev}")

    for mk in keys:
        if mk not in registry:
            print(f"[skip] unknown model key {mk}")
            continue
        outdir = SCORES_DIR / mk
        outdir.mkdir(parents=True, exist_ok=True)
        todo = [y for y in YEARS if not (outdir / f"trec-dl-{y}.parquet").exists()]
        if not todo:
            print(f"[done] {mk} already scored")
            continue

        print(f"\n=== {mk} ({registry[mk]}) ===")
        kw = {"trust_remote_code": True} if mk in EXTRA_TRUST_REMOTE else {}
        model = CrossEncoder(registry[mk], max_length=512, device=dev, **kw)
        bs = {**BATCH_SIZES, **EXTRA_BATCH_SIZES}.get(mk, 64)

        for y in todo:
            df = pd.read_parquet(TRECDL / f"trec-dl-{y}.parquet")
            pairs = list(zip(df["query"].tolist(), df["passage"].tolist()))
            print(f"  trec-dl-{y}: {len(pairs)} pairs (batch {bs})")
            lg = score(model, pairs, bs)
            out = pd.DataFrame({
                "query_id": df.query_id, "doc_id": df.doc_id,
                "bm25_rank": -1,                     # pool membership, not a rank
                "ce_logit": lg.astype(np.float32),
                "ce_prob": sigmoid(lg).astype(np.float32),
                "qrel_label": df.qrel_label.astype(int),
                "qrel_label_lenient": df.qrel_label_lenient.astype(int),
                "grade": df.grade.astype(int),
            })
            p = outdir / f"trec-dl-{y}.parquet"
            out.to_parquet(p, index=False)
            print(f"    -> {p}  (mean prob {out.ce_prob.mean():.3f}, "
                  f"prevalence {out.qrel_label.mean():.3f})")

        del model
        if dev == "mps":
            torch.mps.empty_cache()


if __name__ == "__main__":
    main()
