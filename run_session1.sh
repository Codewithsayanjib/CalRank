#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# CalRank Session 1 — Small models
# Models: TinyBERT-L2, MiniLM-L6, MiniLM-L12, Jina-tiny, ELECTRA-base
# Estimated time: ~3.5 hrs on M4 MPS
#
# Prerequisites (run once, before first session):
#   python scripts/00_setup_check.py
#   python scripts/01_download_beir.py
#   python scripts/02_build_bm25_index.py
#   python scripts/03_bm25_retrieve.py
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
cd "$(dirname "$0")"

echo "========================================================"
echo "  CalRank Session 1: Small Models"
echo "  Started: $(date)"
echo "========================================================"

python scripts/04_ce_score.py --session 1

echo ""
echo "========================================================"
echo "  Session 1 complete: $(date)"
echo "  Scores written to data/scores/"
echo "  Next: run run_session2.sh (BGE + full analysis)"
echo "========================================================"
