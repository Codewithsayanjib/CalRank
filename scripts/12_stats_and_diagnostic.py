#!/usr/bin/env python3
"""Corrected significance tests and the validity diagnostic.

(A) Holm-Bonferroni correction over the six pairwise method comparisons on the
    clean subset, with a per-dataset breakdown.
(B) Apply the rule min(n_pos, n_neg) >= 100 and prevalence <= 0.90 to every
    collection.

Outputs:
  data/calibration/stats_corrected.csv
  data/calibration/validity_diagnostic.csv
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATASETS, CALIBRATION_DIR

CLEAN = ["dbpedia-entity", "trec-covid", "webis-touche2020", "scidocs"]

# Diagnostic thresholds (stated in advance, not tuned)
MIN_MINORITY = 100     # need at least this many judged negatives
MAX_PREVALENCE = 0.90  # and prevalence must not exceed this


def holm(pvals, labels, alpha=0.05):
    """Holm-Bonferroni step-down. Returns list of (label, p, p_adj, reject)."""
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * pvals[i]
        running = max(running, val)
        adj[i] = min(running, 1.0)
    return [(labels[i], pvals[i], adj[i], adj[i] < alpha) for i in range(m)]


def main():
    res = pd.read_csv(CALIBRATION_DIR / "calibration_results.csv")
    prev = pd.read_csv(CALIBRATION_DIR / "prevalence_table.csv")
    clean = res[res.dataset.isin(CLEAN)].copy()

    # ── (A) Multiple comparisons ─────────────────────────────────────────────
    print("=" * 76)
    print("(A) METHOD COMPARISONS WITH HOLM-BONFERRONI CORRECTION (clean, n=20)")
    print("=" * 76)

    comps = [
        ("temp  vs raw",   "temp_ece",  "raw_ece"),
        ("platt vs raw",   "platt_ece", "raw_ece"),
        ("iso   vs raw",   "iso_ece",   "raw_ece"),
        ("platt vs temp",  "platt_ece", "temp_ece"),
        ("iso   vs temp",  "iso_ece",   "temp_ece"),
        ("iso   vs platt", "iso_ece",   "platt_ece"),
    ]
    labels, praw, rows = [], [], []
    for name, a, b in comps:
        sub = clean[[a, b]].dropna()
        if len(sub) < 3:
            continue
        d = sub[a].values - sub[b].values
        t_p = stats.ttest_rel(sub[a], sub[b]).pvalue
        w_p = stats.wilcoxon(sub[a], sub[b]).pvalue
        dz = d.mean() / d.std(ddof=1)
        labels.append(name); praw.append(w_p)
        rows.append({"comparison": name, "n": len(sub), "mean_delta": d.mean(),
                     "cohens_dz": dz, "wins": int((d < 0).sum()),
                     "p_ttest": t_p, "p_wilcoxon": w_p})

    corrected = holm(np.array(praw), labels)
    cmap = {l: (p, pa, r) for l, p, pa, r in corrected}
    for r in rows:
        _, pa, rej = cmap[r["comparison"]]
        r["p_holm"] = pa
        r["significant_holm"] = rej

    sdf = pd.DataFrame(rows)
    print(f"\n{'comparison':<16}{'n':>4}{'mean d':>10}{'dz':>8}{'wins':>7}"
          f"{'p_wilcox':>12}{'p_holm':>12}{'sig':>6}")
    for _, r in sdf.iterrows():
        print(f"{r.comparison:<16}{r.n:>4}{r.mean_delta:>10.4f}{r.cohens_dz:>8.2f}"
              f"{r.wins:>4}/{r.n:<2}{r.p_wilcoxon:>12.2e}{r.p_holm:>12.2e}"
              f"{'  YES' if r.significant_holm else '   no':>6}")
    sdf.to_csv(CALIBRATION_DIR / "stats_corrected.csv", index=False)

    # per-dataset breakdown instead of a single pooled claim
    print("\nPer-dataset mean ECE (clean subset, n=5 models each):")
    pdb = clean.groupby("dataset")[["raw_ece", "temp_ece", "platt_ece", "iso_ece"]].mean()
    print(pdb.round(4).to_string())
    print("\nPer-dataset: does each method beat raw in all 5 models?")
    for ds in CLEAN:
        s = clean[clean.dataset == ds]
        out = []
        for m in ["temp_ece", "platt_ece", "iso_ece"]:
            v = s[[m, "raw_ece"]].dropna()
            out.append(f"{m.split('_')[0]}={int((v[m] < v.raw_ece).sum())}/{len(v)}")
        print(f"  {ds:<20} " + "  ".join(out))

    # ── (B) Validity diagnostic ──────────────────────────────────────────────
    print("\n" + "=" * 76)
    print("(B) PRE-FLIGHT VALIDITY DIAGNOSTIC FOR CALIBRATION EVALUATION")
    print("=" * 76)
    print(f"Rule (fixed a priori): valid iff  n_minority >= {MIN_MINORITY}"
          f"  AND  prevalence <= {MAX_PREVALENCE}\n")

    drows = []
    for _, r in prev.iterrows():
        ds = r["dataset"]
        p = float(r["prevalence"])
        n = int(r["n_judged"]) if "n_judged" in r else int(r.get("judged", 0))
        n_pos = int(round(p * n))
        n_neg = n - n_pos
        n_min = min(n_pos, n_neg)
        ok = (n_min >= MIN_MINORITY) and (p <= MAX_PREVALENCE)
        drows.append({"dataset": ds, "n_judged": n, "prevalence": p,
                      "n_positive": n_pos, "n_negative": n_neg,
                      "n_minority": n_min, "valid": ok})
    ddf = pd.DataFrame(drows).sort_values("prevalence")
    print(f"{'dataset':<20}{'n_judged':>10}{'prev':>7}{'n_neg':>8}"
          f"{'n_minority':>12}{'verdict':>12}")
    for _, r in ddf.iterrows():
        print(f"{r.dataset:<20}{r.n_judged:>10}{r.prevalence:>7.2f}{r.n_negative:>8}"
              f"{r.n_minority:>12}{'VALID' if r.valid else 'INVALID':>12}")
    ddf.to_csv(CALIBRATION_DIR / "validity_diagnostic.csv", index=False)

    nv = (~ddf.valid).sum()
    print(f"\nFlagged invalid: {nv}/{len(ddf)}")
    print("Flagged datasets:", ", ".join(ddf[~ddf.valid].dataset.tolist()))
    print("Passed         :", ", ".join(ddf[ddf.valid].dataset.tolist()))


if __name__ == "__main__":
    main()
