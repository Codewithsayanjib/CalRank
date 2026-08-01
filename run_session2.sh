#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# CalRank Session 2 — BGE reranker + calibration + downstream + plots
# Estimated time: ~4 hrs on M4 MPS (BGE alone) + ~30 min for analysis
#
# Requires Session 1 to have completed successfully.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
cd "$(dirname "$0")"

echo "========================================================"
echo "  CalRank Session 2: BGE + Analysis"
echo "  Started: $(date)"
echo "========================================================"

echo ""
echo "[1/5] BGE-reranker-base scoring (~3-4 hrs) …"
python scripts/04_ce_score.py --session 2

echo ""
echo "[2/5] Raw calibration metrics (ECE, MCE, Brier, AUC-ROC + bootstrap CIs) …"
python scripts/05_calibration_metrics.py

echo ""
echo "[3/5] Post-hoc calibration fitting (temperature, Platt, isotonic) …"
python scripts/06_calibration_fit.py

echo ""
echo "[4/5] Downstream analysis (nDCG@10, MAP, Recall@100, abstention) …"
python scripts/07_downstream_analysis.py

echo ""
echo "[5/5] Reliability diagrams (per-pair + grids + paper figure) …"
python scripts/08_plot_reliability.py

echo ""
echo "========================================================"
echo "  Session 2 complete: $(date)"
echo ""
echo "  Results:"
echo "    data/calibration/raw_metrics.csv"
echo "    data/calibration/calibration_results.csv"
echo "    data/calibration/calibration_params.json"
echo "    data/calibration/downstream_analysis.csv"
echo "    data/figures/"
echo "========================================================"
