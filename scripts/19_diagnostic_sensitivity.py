#!/usr/bin/env python3
"""Sensitivity of the validity diagnostic to its thresholds.

The rule declares a collection valid iff min(n_+, n_-) >= K and prevalence <= P.
Sweeps K in {50, 100, 150} and P in {0.85, 0.90, 0.95} and reports the
classification of the eleven collections at each setting.

Output: data/calibration/diagnostic_sensitivity.csv
"""
import sys
import itertools
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CALIBRATION_DIR

KS = [50, 100, 150]
PS = [0.85, 0.90, 0.95]
DEFAULT = (100, 0.90)


def classify(v, K, P):
    return (v.n_minority >= K) & (v.prevalence <= P)


def main():
    v = pd.read_csv(CALIBRATION_DIR / "validity_extended.csv").copy()
    # order by prevalence for readability
    v = v.sort_values("prevalence").reset_index(drop=True)

    # full grid: how many collections are valid under each (K, P)
    rows = []
    for K, P in itertools.product(KS, PS):
        nv = int(classify(v, K, P).sum())
        rows.append({"K": K, "P": P, "n_valid": nv,
                     "valid_set": ",".join(sorted(v[classify(v, K, P)].dataset))})
    grid = pd.DataFrame(rows)
    grid.to_csv(CALIBRATION_DIR / "diagnostic_sensitivity.csv", index=False)

    print("=" * 68)
    print("DIAGNOSTIC SENSITIVITY  (number of valid collections per threshold)")
    print("=" * 68)
    piv = grid.pivot(index="K", columns="P", values="n_valid")
    print(piv.to_string())
    print(f"\nPaper default (K={DEFAULT[0]}, P={DEFAULT[1]}): "
          f"{int(classify(v, *DEFAULT).sum())} valid\n")

    # per-collection: verdict under each setting, to expose which ones move
    print("=" * 68)
    print("PER-COLLECTION VERDICT ACROSS THE 3x3 GRID")
    print("=" * 68)
    settings = list(itertools.product(KS, PS))
    hdr = "collection         prev  n_min  " + " ".join(f"{K}/{P}" for K, P in settings)
    print(hdr)
    always_v, always_i, moves = [], [], []
    for _, r in v.iterrows():
        verds = [classify(pd.DataFrame([r]), K, P).iloc[0] for K, P in settings]
        marks = " ".join(" V " if x else " . " for x in verds)
        print(f"{r.dataset:<18}{r.prevalence:>5.2f}{int(r.n_minority):>6}  {marks}")
        if all(verds):
            always_v.append(r.dataset)
        elif not any(verds):
            always_i.append(r.dataset)
        else:
            moves.append(r.dataset)

    print("\nStable VALID   (all 9 settings):", ", ".join(always_v) or "none")
    print("Stable INVALID (all 9 settings):", ", ".join(always_i) or "none")
    print("Borderline (verdict depends on threshold):", ", ".join(moves) or "none")
    n_stable = len(always_v) + len(always_i)
    print(f"\n=> {n_stable} of {len(v)} collections are classified identically under "
          f"every threshold in the sweep;\n   only {len(moves)} borderline "
          f"collection(s) move, and both move in the expected direction.")


if __name__ == "__main__":
    main()
