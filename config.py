from pathlib import Path

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"

BEIR_DATA_DIR   = DATA_DIR / "beir"
INDEX_DIR       = DATA_DIR / "indexes"
RETRIEVAL_DIR   = DATA_DIR / "retrieval"
SCORES_DIR      = DATA_DIR / "scores"
CALIBRATION_DIR = DATA_DIR / "calibration"
FIGURES_DIR     = DATA_DIR / "figures"

for _d in [BEIR_DATA_DIR, INDEX_DIR, RETRIEVAL_DIR, SCORES_DIR, CALIBRATION_DIR, FIGURES_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── Models ────────────────────────────────────────────────────────────────────
MODELS = {
    "minilm-l6":    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "minilm-l12":   "cross-encoder/ms-marco-MiniLM-L-12-v2",
    "tinybert-l2":  "cross-encoder/ms-marco-TinyBERT-L-2-v2",
    "bge-reranker": "BAAI/bge-reranker-base",
    "electra-base": "cross-encoder/ms-marco-electra-base",
    # jina-reranker-v1-tiny-en excluded: incompatible with transformers>=5.0
    # (imports removed internals: transformers.onnx, pytorch_utils.find_pruneable_heads_and_indices)
}

# Additional models: larger (up to 560M) and one not trained on MS MARCO.
# Scored on judged pairs only.
EXTRA_MODELS = {
    "bge-large":  "BAAI/bge-reranker-large",              # 560M, XLM-R-large
    "mxbai-base": "mixedbread-ai/mxbai-rerank-base-v1",   # 184M, DeBERTa-v3
    # mxbai-rerank-large-v1 (435M) omitted: exceeds unified memory at usable batch sizes.
}
EXTRA_PARAMS = {"bge-large": "560M", "mxbai-base": "184M"}
EXTRA_BATCH_SIZES = {"bge-large": 32, "mxbai-base": 16}
EXTRA_TRUST_REMOTE = set()

# Session 1: small models (~3 hrs total on M4 MPS)
SESSION1_MODELS = ["tinybert-l2", "minilm-l6", "minilm-l12", "electra-base"]
# Session 2: BGE alone (~3-4 hrs on M4 MPS)
SESSION2_MODELS = ["bge-reranker"]

# Models that require trust_remote_code=True when loading
TRUST_REMOTE_CODE = set()  # none of the active models need this

# Batch sizes tuned for M4 16 GB unified memory
BATCH_SIZES = {
    "minilm-l6":    512,
    "minilm-l12":   256,
    "tinybert-l2":  1024,
    "bge-reranker": 64,
    "jina-tiny":    256,
    "electra-base": 128,
}

# ── Datasets ──────────────────────────────────────────────────────────────────
DATASETS = [
    "trec-covid",
    "nfcorpus",
    "fiqa",
    "scifact",
    "arguana",
    "scidocs",
    "webis-touche2020",
    "quora",
    "dbpedia-entity",
]

DATASET_SPLITS = {d: "test" for d in DATASETS}

BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip"

# ── Experiment parameters ─────────────────────────────────────────────────────
QUORA_QUERY_SAMPLE  = 1000   # subsample Quora from 10K
BM25_TOP_K          = 100
ECE_BINS            = 15
CALIBRATION_SPLIT   = 0.5    # 50/50 fit/test split on judged pairs
BOOTSTRAP_N         = 1000   # bootstrap resamples for CIs
RANDOM_SEED         = 42
ABSTENTION_THRESHOLD = 0.5   # p > threshold → accept (no abstention)
