#!/usr/bin/env python3
"""
Emit LaTeX table bodies from the extended calibration results, so the manuscript
tables are generated from the data rather than transcribed by hand.

Writes:
  paper/generated/tab_appendix_full.tex   all original-model pairs, all collections
  paper/generated/tab_model_scope.tex     model-scope extension comparison
"""
import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CALIBRATION_DIR, MODELS, EXTRA_MODELS, EXTRA_PARAMS, ROOT

OUT = ROOT / "paper" / "generated"
OUT.mkdir(parents=True, exist_ok=True)

PARAMS = {"tinybert-l2": "4M", "minilm-l6": "22M", "minilm-l12": "33M",
          "electra-base": "110M", "bge-reranker": "278M", **EXTRA_PARAMS}
NICE = {"tinybert-l2": "TinyBERT-L2", "minilm-l6": "MiniLM-L6",
        "minilm-l12": "MiniLM-L12", "electra-base": "ELECTRA-base",
        "bge-reranker": "BGE-base", "bge-large": "BGE-large",
        "mxbai-base": "mxbai-base", "mxbai-large": "mxbai-large"}
TRAIN = {"bge-large": "contrastive", "mxbai-base": "non-MS MARCO",
         "mxbai-large": "non-MS MARCO"}


def fmt(v, nd=3):
    return "---" if pd.isna(v) else f"{v:.{nd}f}"


def appendix_table(df):
    d = df[df.model_group == "original"].copy()
    d = d.sort_values(["model", "prevalence"])
    lines = []
    for mk in MODELS:
        sub = d[d.model == mk]
        if sub.empty:
            continue
        for _, r in sub.iterrows():
            col = "cleangreen" if r.valid else "singleclasspink"
            T = r.temp_T
            Ts = r"$\geq$20" if T > 19.9 else f"{T:.2f}"
            lines.append(
                f"\\rowcolor{{{col}}}\n{NICE.get(mk, mk):<14} & {r.dataset:<17} & "
                f"{r.prevalence:.2f} & {fmt(r.raw_ece)} & {Ts} & {fmt(r.temp_ece)} & "
                f"{fmt(r.platt_ece)} & {fmt(r.iso_ece)} \\\\")
        lines.append("\\midrule")
    if lines and lines[-1] == "\\midrule":
        lines.pop()
    (OUT / "tab_appendix_full.tex").write_text("\n".join(lines) + "\n")
    print(f"appendix: {len(d)} rows -> tab_appendix_full.tex")


def model_scope_table(df):
    core = sorted(df[df.valid].dataset.unique())
    d = df[df.dataset.isin(core)]
    if d[d.model_group == "extension"].empty:
        print("model scope: no extension data yet, skipped")
        return
    lines = []
    for mk in list(MODELS) + list(EXTRA_MODELS):
        s = d[d.model == mk]
        if s.empty:
            continue
        grp = "extension" if mk in EXTRA_MODELS else "original"
        col = "rowlight" if grp == "original" else "cleangreen"
        note = TRAIN.get(mk, "MS MARCO")
        lines.append(
            f"\\rowcolor{{{col}}}\n{NICE.get(mk, mk):<14} & {PARAMS.get(mk,'?'):>5} & "
            f"{note:<12} & {s.raw_ece.mean():.3f} & {s.temp_T.mean():.2f} & "
            f"{s.temp_ece.mean():.3f} & {fmt(s.platt_ece.mean())} & "
            f"{s.iso_ece.mean():.3f} \\\\")
    (OUT / "tab_model_scope.tex").write_text("\n".join(lines) + "\n")
    print(f"model scope: {d.model.nunique()} models -> tab_model_scope.tex")

    e = d[d.model_group == "extension"]
    o = d[d.model_group == "original"]
    print(f"\n  original  (n={len(o)}): raw={o.raw_ece.mean():.3f} "
          f"temp={o.temp_ece.mean():.3f} platt={o.platt_ece.mean():.3f} "
          f"iso={o.iso_ece.mean():.3f}")
    print(f"  extension (n={len(e)}): raw={e.raw_ece.mean():.3f} "
          f"temp={e.temp_ece.mean():.3f} platt={e.platt_ece.mean():.3f} "
          f"iso={e.iso_ece.mean():.3f}")

    import numpy as np
    from scipy import stats
    allm = d.groupby("model").agg(params=("model", "first"), raw=("raw_ece", "mean"))
    sizes = {m: float(PARAMS[m].rstrip("M")) for m in allm.index if m in PARAMS}
    x = [sizes[m] for m in allm.index if m in sizes]
    y = [allm.loc[m, "raw"] for m in allm.index if m in sizes]
    if len(x) > 3:
        rho, p = stats.spearmanr(x, y)
        print(f"\n  size vs raw ECE across {len(x)} models: rho={rho:.2f}, p={p:.2f}")


def main():
    df = pd.read_csv(CALIBRATION_DIR / "calibration_extended.csv")
    appendix_table(df)
    model_scope_table(df)


if __name__ == "__main__":
    main()
