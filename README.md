# Short-term Fog Forecasting and Onset Detection from Multi-station ASOS Observations in Korea

Reproducible code for a comparison of **Persistence**, **XGBoost**, **1D-CNN**
and **LSTM** on two distinct tasks over twelve KMA ASOS stations in the
central-northern Korean Peninsula, 2015–2024.

Every number, table and figure in the paper is produced by the scripts in this
repository from the raw observations. Nothing is transcribed by hand. The audit
that produced this repository is documented in
[`REPRODUCIBILITY_AUDIT.md`](REPRODUCIBILITY_AUDIT.md).

**Repository:** <https://github.com/skynmwoo/fog-forecast-repro>

---

## Study

Fog reduces horizontal visibility sharply and matters for aviation, road,
maritime and military operations. Two prediction problems are evaluated
separately, because good overall accuracy does not imply useful early warning:

**Task A — overall T+1 fog prediction.** For every hour `t`, predict whether fog
occurs at `t+1`. All four transition types (0→0, 0→1, 1→0, 1→1) are included, so
a model can score well simply by carrying existing fog forward.

**Task B — 0→1 fog onset detection.** Restricted to hours where the current state
is non-fog, predict whether fog *newly forms* at `t+1`. This is the
operationally relevant early-warning problem, and it is far harder: onset events
are ~0.6 % of the candidate set.

Fog is defined as horizontal visibility ≤ 1 km. The KMA ASOS visibility variable
is reported in units of 10 m, so the label is `visibility ≤ 100`.

**Models.** Persistence (carry the current state forward; parameter-free) and a
No-Onset baseline (never predict onset) are the reference points. XGBoost uses a
28-feature tabular representation; the 1D-CNN and LSTM consume 6-hour sequences
of 14 variables.

---

## Data

Hourly ASOS observations from the **KMA Open MET Data Portal**
(<https://data.kma.go.kr>) — *지상관측 → 종관기상관측(ASOS) → 시간자료*.

Raw observations are **not redistributed in this repository**: they are
KMA-licensed and total ~110 MB. Download them yourself as follows.

1. Register a free account at <https://data.kma.go.kr>.
2. Request hourly ASOS data for each station and each year 2015–2024.
3. Save one CSV per station-year into `data/raw/` using this naming convention:

   ```
   data/raw/<StationKoreanName>_<YYYY>.csv       e.g. 인천_2015.csv
   data/raw/<YYYY>.csv                           Taebaek only (no prefix)
   ```

   The station name / file prefix mapping lives in `configs/default.yaml`
   under `data.stations` and can be changed there.

4. The files are EUC-KR encoded by default; the loader tries
   `euc-kr → cp949 → utf-8-sig → utf-8` automatically.

**Required columns** (Korean headers, as exported by the portal):

| Column | Meaning |
|---|---|
| `일시` | observation timestamp (hourly) |
| `기온(°C)` | air temperature |
| `이슬점온도(°C)` | dew point temperature |
| `습도(%)` | relative humidity |
| `풍속(m/s)` | wind speed |
| `현지기압(hPa)` | station pressure |
| `지면온도(°C)` | ground temperature |
| `시정(10m)` | horizontal visibility, units of 10 m |

Extra columns are ignored.

**Stations (12):** Baengnyeongdo, Daegwallyeong, Paju, Incheon, Ganghwa,
Dongducheon, Chuncheon, Cheorwon, Taebaek, Sokcho, Seoul, Inje.

**Temporal split** — strictly chronological, never shuffled:

| Split | Period | Used for |
|---|---|---|
| Train | 2015–2021 | model fitting, `StandardScaler` fitting |
| Validation | 2022 | early stopping and classification-threshold selection |
| Test | 2023–2024 | reported performance only |

---

## Installation

```bash
git clone https://github.com/skynmwoo/fog-forecast-repro.git
cd fog-forecast-repro
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.11 or newer. CPU-only is sufficient — a full run takes about 30 minutes
on two cores. No paths are hard-coded; everything resolves relative to the
repository root, and all settings live in `configs/default.yaml`.

---

## Reproduction

Place the raw CSVs in `data/raw/`, then:

```bash
python scripts/reproduce_all.py
```

Or run the stages individually:

```bash
python scripts/prepare_data.py           # raw -> processed parquet + prevalence/transition CSVs
python scripts/sanity_check_models.py    # structural gate: is the CNN a CNN, is the LSTM an LSTM?
python scripts/run_experiments.py        # 12 stations x 2 tasks x 4 models
python scripts/run_shap.py               # SHAP on the reproduced XGBoost models
python scripts/significance_tests.py     # paired Wilcoxon tests across the 12 stations
python scripts/transition_analysis.py    # decompose overall skill; prevalence-gap correlation
python scripts/make_tables.py            # all tables, from the result CSVs
python scripts/make_figures.py           # all figures, from the result CSVs
python scripts/compare_with_manuscript.py
python scripts/audit_consistency.py                       # config + result files
python scripts/audit_consistency.py --manuscript <file>.docx   # optional: also check a .docx
python scripts/check_narrative_consistency.py \
    --manuscript <file>.docx --cover-letter <file>.docx        # optional: wording audit
```

Note on scope: `scripts/reproduce_all.py` runs `audit_consistency.py` **without**
a manuscript argument, so the single-command pipeline verifies configuration
values and generated result files only. Checking a finished manuscript against
those files is a separate invocation with `--manuscript`.

`scripts/sanity_check_models.py` is a hard gate. It fails — and
`reproduce_all.py` stops — unless the 1D-CNN contains real `nn.Conv1d` layers
applied along the time axis and the LSTM contains a real `nn.LSTM` whose final
hidden state feeds the classifier, with non-zero gradients reaching both and
output that changes when the time axis is reversed. This exists because an
earlier version of this study reported "1D-CNN" and "LSTM" results whose
architecture could not be verified against the manuscript.

Useful options:

```bash
python scripts/run_experiments.py --stations Seoul Inje       # subset of stations
python scripts/run_experiments.py --task onset                # one task only
python scripts/run_experiments.py --seeds 42 1337 2024 --stability   # seed-variability diagnostic
```

---

## Expected outputs

| Path | Contents |
|---|---|
| `data/processed/<Station>.parquet` | cleaned, feature-engineered hourly table with a `split` column |
| `results/metrics/station_metrics.csv` | **source of truth** — threshold, precision, recall, F1, PR-AUC, ROC-AUC, TP/FP/TN/FN, prevalence, n, per station × task × model |
| `results/metrics/station_metrics_default_threshold.csv` | XGBoost at the untuned 0.5 threshold |
| `results/metrics/thresholds.csv` | the validation-selected threshold actually applied to the test set |
| `results/metrics/fog_prevalence.csv` | per-station fog occurrence rate |
| `results/metrics/transition_counts.csv` | 0→0 / 0→1 / 1→0 / 1→1 counts |
| `results/metrics/split_summary.csv` | rows, period and positive counts per station × split |
| `results/metrics/shap_global.csv`, `shap_station.csv`, `shap_station_normalized.csv` | SHAP importances |
| `results/metrics/error_visibility_stats.csv` | visibility distribution by TP / FP / FN / TN |
| `results/metrics/stability_metrics.csv` | multi-seed diagnostic |
| `results/tables/*.csv` | every manuscript table |
| `results/tables/table_significance_tests.csv` | paired Wilcoxon signed-rank tests, Holm-corrected, with rank-biserial effect sizes |
| `results/tables/table_transition_recall.csv` | overall-task recall split into 1→1 (fog persisting) and 0→1 (fog forming), with precision |
| `results/tables/table_prevalence_gap_correlation.csv` | correlation between each model's advantage over Persistence and station fog prevalence |
| `results/metrics/transition_recall_by_station.csv`, `prevalence_gap.csv` | per-station inputs to the two analyses above |
| `results/tables/old_vs_new_comparison.csv` | previous manuscript values vs. reproduced values |
| `results/tables/consistency_audit.csv` | config ↔ result cross-check (also manuscript ↔ config ↔ result when `--manuscript` is passed) |
| `results/figures/*.png`, `*.pdf` | every manuscript figure |
| `checkpoints/<task>/xgb_<task>_<Station>.json` | trained XGBoost models (regenerated locally; **not committed** to the repository) |
| `checkpoints/<task>/predictions_<task>[_<model>]_<Station>.csv` | per-row test predictions for every model, used by `transition_analysis.py` (not committed) |
| `results/logs/run_env.json` | package versions, seeds, stations, elapsed time |

---

## Reproducibility

* **Python** 3.11; exact package versions pinned in `requirements.txt`
  (`torch==2.13.0`, `xgboost==3.2.0`, `scikit-learn==1.8.0`, `shap==0.51.0`).
* **Seed** 42 for reported results, set for Python `random`, `PYTHONHASHSEED`,
  NumPy, PyTorch and CUDA; `torch.use_deterministic_algorithms(True)` enabled.
* **Device** CPU. Results in the paper were produced on 2 CPU cores.
* **Determinism.** Persistence and XGBoost are exactly reproducible.
  1D-CNN and LSTM depend on initialisation; run the `--stability` diagnostic to
  see the spread across seeds. The paper reports the seed-42 run and states the
  observed variability; it does not average over seeds.
* **Training protocol.** BCEWithLogitsLoss with
  `pos_weight = n_negative / n_positive` from the training split; Adam at 1e-3;
  batch size 512; at most 15 epochs; early stopping when validation F1 has not
  improved for 3 epochs; the weights from the best validation-F1 epoch are
  restored. XGBoost uses `scale_pos_weight = n_negative / n_positive`, computed
  separately per station and task.
* **Threshold selection.** A 181-point grid over [0.05, 0.95] is swept on the
  validation set; the F1-maximising threshold is frozen and applied to the test
  set. The test set is never used for any selection decision.
* **Evaluation set.** All four models are scored on an identical set of test
  rows — those for which a complete 6-hour history exists — so F1 values share a
  denominator.
* **Significance testing.** Model pairs are compared with two-sided Wilcoxon
  signed-rank tests over the twelve stations (one paired observation per
  station), two-sided, n = 12. Holm correction is applied within each prediction
  task **and evaluation metric**; the three correction families are overall-task
  F1 (six comparisons), onset-task F1 (three) and onset-task PR-AUC (three).
  Effect size is the matched-pairs rank-biserial correlation. A non-significant
  result indicates that these data do not resolve the difference; it does not
  establish that two models are equivalent.

---

## Repository layout

```
configs/default.yaml      all hyperparameters, feature lists, paths, seeds
src/data/                 preprocess.py, features.py, sequences.py
src/models/               cnn1d.py, lstm.py  (+ structural assertions)
src/train/deep.py         shared training loop, seeding
src/evaluate/metrics.py   metrics and validation threshold selection
scripts/                  runnable pipeline stages
tools/                    document-generation scripts (see tools/README.md)
results/                  metrics, tables, figures, logs
reference/                previous manuscript claims + legacy result CSVs, preserved
checkpoints/              empty here; populated locally by scripts/run_experiments.py
```

### What this repository does and does not ship

* **Ships:** all source code, `configs/default.yaml`, every result CSV behind
  the reported numbers, the statistical test output, all regenerated figures,
  and the document-generation scripts in `tools/`.
* **Does not ship:** the raw KMA ASOS observations (licensed; see *Data*),
  trained model files (`checkpoints/` — regenerated by the pipeline in minutes),
  and the `.docx` manuscript templates that `tools/` operates on.

---

## Citation

If you use this code, please cite the accompanying paper. Full bibliographic
details will be added here on acceptance. Until then, please cite the
repository:

```
Woo, S.; Kim, I. Short-term fog forecasting and onset detection from
multi-station ASOS observations in Korea (software).
https://github.com/skynmwoo/fog-forecast-repro
```

## License

Code: MIT (see `LICENSE`).
Data: KMA ASOS observations are subject to the Korea Meteorological
Administration's terms of use and are not redistributed here.
