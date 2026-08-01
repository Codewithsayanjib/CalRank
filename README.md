# CalRank: Calibration of Cross-Encoder Re-Rankers

Code and released artifacts for the paper
**"Re-evaluating Re-rankers: A Methodological Critique of Calibration on Retrieval
Benchmarks."**

Cross-encoder re-ranker scores are increasingly consumed as probabilities by
retrieval-augmented generation pipelines, abstention layers, and cascades. This
project measures whether those scores are calibrated across seven re-rankers and
eleven collections, and shows that **most standard retrieval benchmarks cannot
measure calibration** because their judged sets contain few or no negatives. We
contribute a pre-flight validity diagnostic, adopt the deeply pooled TREC Deep
Learning collections to satisfy it, and package the corrections as a per-model
deployment recipe.

Everything runs on a laptop (Apple M4, MPS); no GPU cluster is required.

## Key findings

- Re-rankers are overconfident: mean raw ECE ≈ 0.28 on the valid subset.
- Most BEIR collections cannot measure calibration (all-positive judged sets); a
  naive pooled analysis inverts the method comparison.
- A validity diagnostic — `min(n₊, n₋) ≥ 100` and `prevalence ≤ 0.90` — flags six
  of eleven collections; its classification is stable under a 3×3 threshold sweep.
- On the five valid collections, isotonic > Platt > temperature (all comparisons
  survive Holm–Bonferroni), ranking is untouched (ΔnDCG@10 ≤ 1e-4), and calibration
  makes a stated threshold portable (spread in realised precision at τ=0.8 falls
  from 0.497 to 0.078).
- Temperature transfers across domains (87% benefit retained); Platt does not (44%).

## Repository layout

```
config.py                 experiment configuration (models, datasets, params)
utils.py                  calibration metrics, calibrator fitting, ranking metrics
scripts/                  numbered, run in order (00–20)
  00_setup_check.py        environment check
  01_download_beir.py      fetch BEIR corpora            (regenerates data/beir/)
  02_build_bm25_index.py   build BM25 indexes            (regenerates data/indexes/)
  03_bm25_retrieve.py      BM25 top-100 retrieval        (regenerates data/retrieval/)
  04_ce_score.py           cross-encoder scoring         -> data/scores/
  05_calibration_metrics.py  raw ECE/MCE/Brier + bootstrap CIs
  06_calibration_fit.py    temperature/Platt/isotonic fitting
  07_downstream_analysis.py  nDCG@10, MAP, accept rates
  08_plot_reliability.py   reliability diagrams
  09_robustness_binning_ceiling.py   binning + temperature-cap robustness
  10_cross_domain_transfer.py        leave-one-domain-out calibrator transfer
  11_threshold_honesty.py            realised precision vs stated threshold
  12_stats_and_diagnostic.py         Holm-corrected tests + validity diagnostic
  13_plot_new_figures.py             transfer + threshold figures
  14_build_trecdl.py                 build TREC DL scoring inputs
  15_score_trecdl.py                 score TREC DL with the models
  16_score_extra_models.py           score the two model-scope extension models
  17_calibration_extended.py         calibration over all 7 models x 11 collections
  18_emit_latex_tables.py            generate LaTeX table bodies
  19_diagnostic_sensitivity.py       3x3 threshold sweep of the diagnostic
  20_case_study.py                   qualitative accept/reject flips at tau=0.8
data/
  scores/<model>/<collection>.parquet   RELEASED: logit, sigmoid, qrel label per pair
  calibration/*.csv, calibration_params.json  RELEASED: metrics + fitted calibrators
  trecdl/*.parquet                      RELEASED: built TREC DL scoring inputs
  figures/*.png                         RELEASED: paper figures
  beir/, indexes/, retrieval/           NOT released (large; regenerate, see below)
paper/
  calrank_fire.tex                      FIRE conference version (ACM sigconf)
  calrank.bib                           shared bibliography
```

## Released artifacts

The scored pairs and fitted calibrators are committed so that **every analysis in
the paper reproduces without any model inference**. Each `data/scores/*.parquet`
holds, per query–document pair: `query_id`, `doc_id`, `ce_logit`, `ce_prob`, and
`qrel_label` (`-1` unjudged, `0` judged non-relevant, `1` judged relevant). Given
these, scripts 05–20 run in minutes on a CPU.

## Reproducing

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Re-analysis only (uses the released scored pairs; no model inference, minutes):
python scripts/17_calibration_extended.py
python scripts/12_stats_and_diagnostic.py
python scripts/19_diagnostic_sensitivity.py
python scripts/10_cross_domain_transfer.py
python scripts/11_threshold_honesty.py
python scripts/13_plot_new_figures.py

# Full pipeline from scratch (re-downloads corpora, re-runs scoring on M4/MPS):
python scripts/01_download_beir.py
python scripts/02_build_bm25_index.py
python scripts/03_bm25_retrieve.py
python scripts/04_ce_score.py
python scripts/14_build_trecdl.py && python scripts/15_score_trecdl.py
```

Determinism: fixed seed `42`; 50/50 label-stratified calibration split throughout.

## Data provenance and licensing

- **Code** in this repository is released under the MIT License (see `LICENSE`).
- **BEIR** collections are from the BEIR benchmark
  (https://github.com/beir-cellar/beir) under their respective licenses.
- **TREC Deep Learning** judgments and pools are from NIST / MS MARCO
  (https://microsoft.github.io/msmarco/TREC-Deep-Learning).
- The released `data/scores/*.parquet` are model scores over these third-party
  collections, provided for reproducibility of the calibration analysis.

## Citation

```bibtex
@inproceedings{sur2026calrank,
  title     = {Re-evaluating Re-rankers: A Methodological Critique of Calibration
               on Retrieval Benchmarks},
  author    = {Sur, Sayanjib and Singh, Pawan Kumar},
  booktitle = {Forum for Information Retrieval Evaluation (FIRE)},
  year      = {2026}
}
```
