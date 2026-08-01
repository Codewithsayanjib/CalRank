#!/usr/bin/env python3
"""
Score BEIR judged pairs with the model-scope extension models.

Calibration is measured on judged pairs only, so the full BM25 top-100 need not
be re-scored for these models. To guarantee that the new models are compared on
exactly the same pairs as the originals, the (query_id, doc_id) list is read
back from an existing score file rather than re-derived from the retrieval run.

  python scripts/16_score_extra_models.py                    # all extra models
  python scripts/16_score_extra_models.py --models bge-large

Outputs data/scores/<model>/<dataset>.parquet, same schema as the originals.
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
from config import (EXTRA_MODELS, EXTRA_BATCH_SIZES, EXTRA_TRUST_REMOTE,
                    BEIR_DATA_DIR, SCORES_DIR, DATASET_SPLITS, RANDOM_SEED)
from beir.datasets.data_loader import GenericDataLoader
from sentence_transformers import CrossEncoder
from scipy.special import expit as sigmoid

# The collections that survive the validity diagnostic, plus scidocs for
# continuity with the four-dataset subset reported in the paper.
TARGETS = ["dbpedia-entity", "trec-covid", "webis-touche2020", "scidocs"]
REFERENCE_MODEL = "minilm-l6"       # source of the canonical pair list
np.random.seed(RANDOM_SEED)


def device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_texts(ds):
    corpus, queries, _ = GenericDataLoader(
        data_folder=str(BEIR_DATA_DIR / ds)).load(split=DATASET_SPLITS[ds])
    ctext = {}
    for did, d in corpus.items():
        t, x = d.get("title", "").strip(), d.get("text", "").strip()
        ctext[did] = f"{t} {x}".strip() if t else (x or "[empty]")
    return ctext, queries


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
    ap.add_argument("--models", nargs="*", default=list(EXTRA_MODELS))
    args = ap.parse_args()
    dev = device()
    print(f"device: {dev}")

    for mk in args.models:
        if mk not in EXTRA_MODELS:
            print(f"[skip] unknown {mk}")
            continue
        outdir = SCORES_DIR / mk
        outdir.mkdir(parents=True, exist_ok=True)
        todo = [d for d in TARGETS if not (outdir / f"{d}.parquet").exists()]
        if not todo:
            print(f"[done] {mk}")
            continue

        print(f"\n=== {mk} ({EXTRA_MODELS[mk]}) ===")
        kw = {"trust_remote_code": True} if mk in EXTRA_TRUST_REMOTE else {}
        model = CrossEncoder(EXTRA_MODELS[mk], max_length=512, device=dev, **kw)
        bs = EXTRA_BATCH_SIZES.get(mk, 32)

        for ds in todo:
            ref = pd.read_parquet(SCORES_DIR / REFERENCE_MODEL / f"{ds}.parquet")
            ref = ref[ref.qrel_label >= 0].reset_index(drop=True)   # judged only
            ctext, queries = load_texts(ds)
            pairs = [(queries[q], ctext[d]) for q, d in zip(ref.query_id, ref.doc_id)]
            print(f"  {ds}: {len(pairs)} judged pairs (batch {bs})")
            lg = score(model, pairs, bs)
            out = pd.DataFrame({
                "query_id": ref.query_id, "doc_id": ref.doc_id,
                "bm25_rank": ref.bm25_rank,
                "ce_logit": lg.astype(np.float32),
                "ce_prob": sigmoid(lg).astype(np.float32),
                "qrel_label": ref.qrel_label.astype(int),
            })
            p = outdir / f"{ds}.parquet"
            out.to_parquet(p, index=False)
            print(f"    -> {p}  (mean prob {out.ce_prob.mean():.3f})")

        del model
        if dev == "mps":
            torch.mps.empty_cache()


if __name__ == "__main__":
    main()
