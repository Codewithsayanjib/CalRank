#!/usr/bin/env python3
"""
Figures for the transfer and threshold-honesty analyses.

Palette: Okabe-Ito subset, CVD-validated (worst pair dE 13.1 under protanopia).
Transfer regime is carried by hatch as a secondary encoding, so identity never
depends on colour alone.

Outputs (PNG @300dpi, for Overleaf):
  data/figures/paper_fig_transfer.png
  data/figures/paper_fig_threshold_honesty.png
"""
import sys
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CALIBRATION_DIR, FIGURES_DIR

GRAY, BLUE, ORANGE, GREEN = "#7F7F7F", "#0072B2", "#E69F00", "#009E73"
INK, MUTED = "#1A1A1A", "#5A5A5A"
# Ordered by prevalence so the prevalence-dependence of the methods is visible.
CORE = ["trec-dl-2020", "dbpedia-entity", "trec-dl-2019",
        "trec-covid", "webis-touche2020"]
NICE = {"trec-dl-2020": "trec-dl-20\n(0.17)",
        "dbpedia-entity": "dbpedia\n(0.35)",
        "trec-dl-2019": "trec-dl-19\n(0.35)",
        "trec-covid": "trec-covid\n(0.62)",
        "webis-touche2020": "webis-touche\n(0.78)"}
SHORT = {"trec-dl-2020": "trec-dl-20", "dbpedia-entity": "dbpedia",
         "trec-dl-2019": "trec-dl-19", "trec-covid": "trec-covid",
         "webis-touche2020": "webis-touche"}

mpl.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8,
    "axes.edgecolor": "#CCCCCC", "axes.linewidth": 0.8,
    "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
})


def style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#E8E8E8", lw=0.7, zorder=0)
    ax.set_axisbelow(True)


# ── Figure 1: cross-domain transfer ──────────────────────────────────────────
def fig_transfer():
    df = pd.read_csv(CALIBRATION_DIR / "transfer_matrix.csv")
    df = df[df.target.isin(CORE) & (df.source.isin(CORE + ["LODO"]))]

    fig, ax = plt.subplots(figsize=(7.6, 3.2))
    series = [
        ("raw",       "ece_raw",   "oracle", GRAY,   None, "Uncalibrated"),
        ("temp_lodo", "ece_temp",  "lodo",   BLUE,   "//", "Temperature (transferred)"),
        ("temp_or",   "ece_temp",  "oracle", BLUE,   None, "Temperature (in-domain)"),
        ("platt_lodo","ece_platt", "lodo",   ORANGE, "//", "Platt (transferred)"),
        ("platt_or",  "ece_platt", "oracle", ORANGE, None, "Platt (in-domain)"),
    ]
    x = np.arange(len(CORE)); w = 0.155
    for i, (_, col, regime, colr, hatch, lbl) in enumerate(series):
        vals = [df[(df.regime == regime) & (df.target == t)][col].mean() for t in CORE]
        off = (i - (len(series) - 1) / 2) * w
        ax.bar(x + off, vals, w * 0.9, label=lbl, color=colr, hatch=hatch,
               edgecolor="white", linewidth=1.2, zorder=3)
        for xi, v in zip(x + off, vals):
            dy = 0.008 + (0.026 if i % 2 else 0.0)   # stagger to avoid collisions
            ax.text(xi, v + dy, f"{v:.2f}", ha="center", va="bottom",
                    fontsize=5.6, color=MUTED, zorder=4)

    ax.set_xticks(x); ax.set_xticklabels([NICE[c] for c in CORE])
    ax.set_ylabel("Expected Calibration Error")
    ax.set_title("Transferred temperature tracks its in-domain bound; "
                 "transferred Platt does not",
                 loc="left", fontweight="bold", pad=8, fontsize=9.5)
    ax.set_ylim(0, max(0.55, ax.get_ylim()[1]))
    style(ax)
    ax.legend(frameon=False, ncol=2, loc="upper left", bbox_to_anchor=(0.0, 1.0),
              handlelength=1.6, columnspacing=1.2)
    out = FIGURES_DIR / "paper_fig_transfer.png"
    fig.savefig(out); plt.close(fig)
    print("wrote", out)


# ── Figure 2: threshold honesty ──────────────────────────────────────────────
def fig_threshold():
    df = pd.read_csv(CALIBRATION_DIR / "threshold_honesty.csv")
    df = df[df.dataset.isin(CORE)]

    # Thresholds above 0.8 are dropped: too few items clear them for the
    # realised precision to be estimated (see MIN_ACCEPTED guard in script 11).
    TAU_MAX = 0.8
    df = df[df.tau <= TAU_MAX]

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2))
    METHODS = [("raw", GRAY, "o", "Uncalibrated"),
               ("temp", BLUE, "s", "Temperature"),
               ("iso", GREEN, "^", "Isotonic")]

    # (a) realised precision vs stated target
    ax = axes[0]
    taus = sorted(df.tau.unique())
    ax.plot([0.48, 0.85], [0.48, 0.85], ls=(0, (4, 3)), color="#B0B0B0",
            lw=1.2, zorder=2, label="Perfect calibration")
    for meth, colr, mk, lbl in METHODS:
        s = df[df.method == meth].groupby("tau").precision.mean()
        ax.plot(s.index, s.values, marker=mk, color=colr, lw=2.0, ms=5.5,
                mec="white", mew=1.2, zorder=3, label=lbl)
    ax.annotate("uncalibrated:\nthreshold barely moves\nthe operating point",
                xy=(0.8, 0.603), xytext=(0.60, 0.42), fontsize=6.8, color=MUTED,
                ha="center", va="center",
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.8,
                                shrinkA=0, shrinkB=3))
    ax.set_xlabel(r"Stated threshold $\tau$")
    ax.set_ylabel("Realised precision")
    ax.set_title(r"(a) Realised precision vs stated $\tau$", loc="left", fontsize=9.5)
    ax.set_xticks(taus); ax.set_ylim(0.33, 0.95)
    style(ax)
    ax.legend(frameon=False, loc="upper left", handlelength=1.8, borderpad=0.2)

    # (b) per-domain realisation at tau = 0.8
    ax = axes[1]
    e = df[df.tau == 0.8]
    x = np.arange(len(CORE)); w = 0.25
    for i, (meth, colr, _, lbl) in enumerate(METHODS):
        vals = [e[(e.method == meth) & (e.dataset == t)].precision.mean() for t in CORE]
        off = (i - 1) * w
        ax.bar(x + off, vals, w * 0.9, color=colr, edgecolor="white",
               linewidth=1.2, zorder=3, label=lbl)
        for xi, v in zip(x + off, vals):
            ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", va="bottom",
                    fontsize=5.8, color=MUTED, zorder=4)
    # The target is carried in the legend rather than as inline text, so it
    # cannot collide with a bar or its value label.
    ax.axhline(0.8, ls=(0, (4, 3)), color="#B0B0B0", lw=1.2, zorder=2,
               label=r"target $\tau=0.8$")
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[c] for c in CORE], fontsize=7.5)
    ax.set_ylabel("Realised precision")
    ax.set_ylim(0, 1.16)
    ax.set_title(r"(b) Operating point at $\tau=0.8$", loc="left", fontsize=9.5)
    style(ax)
    h, l = ax.get_legend_handles_labels()
    o = [l.index(t) for t in ["Uncalibrated", "Temperature", "Isotonic",
                              r"target $\tau=0.8$"] if t in l]
    ax.legend([h[i] for i in o], [l[i] for i in o], frameon=False,
              loc="upper center", ncol=2, handlelength=1.4, fontsize=7,
              columnspacing=1.0, borderpad=0.2, bbox_to_anchor=(0.5, 1.03))

    fig.tight_layout(w_pad=2.4)
    out = FIGURES_DIR / "paper_fig_threshold_honesty.png"
    fig.savefig(out); plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    fig_transfer()
    fig_threshold()
