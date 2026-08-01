#!/bin/bash
# Watchdog: if the active scoring process dies, restart from where it left off.
SCORING_PID=$1
LOG=/Users/sayanjibsdevice/Desktop/CalRank/run_full.log
cd /Users/sayanjibsdevice/Desktop/CalRank

echo "[watchdog] started at $(date), watching PID $SCORING_PID" >> "$LOG"

while kill -0 "$SCORING_PID" 2>/dev/null; do
    sleep 30
done

echo "[watchdog] PID $SCORING_PID gone at $(date) — checking completion" >> "$LOG"

N=$(find /Users/sayanjibsdevice/Desktop/CalRank/data/scores -name "*.parquet" | wc -l | tr -d ' ')
if [ "$N" -ge 40 ]; then
    echo "[watchdog] $N/40 parquets — run completed normally" >> "$LOG"
    exit 0
fi

echo "[watchdog] only $N/40 — restarting pipeline" >> "$LOG"
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
bash run_full.sh >> "$LOG" 2>&1
echo "[watchdog] pipeline finished at $(date)" >> "$LOG"
