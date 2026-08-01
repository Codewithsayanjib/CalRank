#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# CalRank full overnight pipeline — Session 1 + Session 2 back-to-back.
# No interactive prompts after this point.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
cd "$(dirname "$0")"

# Suppress HuggingFace advisory prompts / implicit-token warnings
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1

echo "========================================================"
echo "  CalRank Full Pipeline"
echo "  Started: $(date)"
echo "========================================================"

# ── Pre-flight ────────────────────────────────────────────────────────────────
echo ""
echo "[pre-flight] Environment check …"
python scripts/00_setup_check.py

echo ""
echo "[pre-flight] Pre-downloading all 6 models to HF cache …"
python scripts/prefetch_models.py

# ── Session 1: small models ───────────────────────────────────────────────────
echo ""
echo "========================================================"
echo "  SESSION 1 — TinyBERT / MiniLM-L6 / MiniLM-L12 / Jina / ELECTRA"
echo "  Started: $(date)"
echo "========================================================"
python scripts/04_ce_score.py --session 1

echo ""
echo "  Session 1 complete: $(date)"

# ── Session 2: BGE + analysis + plots ─────────────────────────────────────────
echo ""
echo "========================================================"
echo "  SESSION 2 — BGE-reranker-base + calibration + plots"
echo "  Started: $(date)"
echo "========================================================"

echo "[1/4] BGE scoring …"
python scripts/04_ce_score.py --session 2

echo "[2/4] Raw calibration metrics (ECE / MCE / Brier / AUC + bootstrap CIs) …"
python scripts/05_calibration_metrics.py

echo "[3/4] Post-hoc calibration (temperature / Platt / isotonic) …"
python scripts/06_calibration_fit.py

echo "[4/4] Downstream analysis (nDCG@10 / MAP / abstention) …"
python scripts/07_downstream_analysis.py

echo "[5/4] Reliability diagrams …"
python scripts/08_plot_reliability.py

echo ""
echo "========================================================"
echo "  ALL DONE: $(date)"
echo ""
echo "  Results:"
echo "    data/calibration/raw_metrics.csv"
echo "    data/calibration/calibration_results.csv"
echo "    data/calibration/calibration_params.json"
echo "    data/calibration/downstream_analysis.csv"
echo "    data/figures/"
echo "========================================================"
