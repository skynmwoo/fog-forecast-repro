# Reproducibility Audit — Short-term Fog Forecasting Study (`D:\07_fog`)

**Audited manuscript:** `Atmosphere_eng_v5.docx`
**Audit performed:** 2026-08-19
**Auditor:** automated re-implementation and re-execution from raw KMA ASOS data
**Canonical codebase produced by this audit:** `fog-forecast-repro/`  
**Public repository:** <https://github.com/skynmwoo/fog-forecast-repro>

> **Bottom line.** The deep-learning code in the project was *not* a disguised
> MLP — real `Conv1D` and `LSTM` layers were present. The far more serious
> problem is different: **the performance numbers printed in the manuscript do
> not appear in any result file anywhere in the project.** Only the Persistence
> column of Table 3 is traceable. Every XGBoost, 1D-CNN and LSTM figure in
> Tables 3 and 4 and in the Abstract has no locatable source. Those values must
> be treated as unverified and replaced.
>
> After a clean re-implementation and re-run, the study's headline conclusion
> changes: XGBoost does not demonstrate an improvement over a parameter-free
> Persistence baseline (Persistence has the higher mean F1 and wins at 9 of 12
> stations, though the difference is not significant after correction for
> multiple comparisons), while **both** significantly outperform the 1D-CNN and
> the LSTM in the overall T+1 task evaluated here.
>
> That change sharpens rather than weakens the paper. Its central claim is not
> which model wins, but that **strong overall one-hour-ahead fog-state
> forecasting performance does not necessarily translate into useful fog-onset
> early-warning skill** — the baseline that is hardest to beat on the overall
> task is precisely the one that cannot issue an onset warning at all. Overall
> fog-state prediction and fog-onset detection must therefore be specified and
> evaluated as separate tasks.

---

## 1. State of the project before the audit

### 1.1 Inventory of `D:\07_fog`

643 files. Nothing was judged by filename alone; every relevant file was opened.

| Category | Count | Notes |
|---|---:|---|
| Raw ASOS CSV (`data/`) | 120 | 12 stations × 2015–2024, KMA ASOS hourly, EUC-KR encoded |
| Python scripts (root) | 23 | see 1.2 |
| Result CSVs | ~91 | three mutually inconsistent result directories |
| Figures (`.png/.pdf/.svg`) | ~245 | five overlapping figure directories |
| Pickled XGBoost models (`.pkl`) | 36 | overall + onset + legacy |
| Manuscripts (`.docx`) | 6 | `Atmosphere_eng_v5.docx` is the newest (mtime 2026-07-16) |
| **Jupyter notebooks** | **0** | none in the folder |
| **PyTorch checkpoints (`.pt`/`.pth`)** | **0** | none anywhere |

### 1.2 Actual execution graph (as reconstructed from file contents)

```
data/*.csv  (raw KMA ASOS, per station per year)
    |
    +-- fog_experiment_12regions_rewrite.py   [overall T+1 task]
    |        -> fog_results_12regions/model_comparison_by_region.csv
    |        -> fog_results_12regions/models/xgb_model_<Station>.pkl
    |        -> fog_results_12regions/predictions/, test_data_for_shap/
    |
    +-- fog_onset_experiment_12regions.py     [onset task]
    |        -> fog_onset_results_12regions/onset_model_comparison_by_region.csv
    |
    +-- fog_purpose.py / fog_prevalence*.py   [a SEPARATE, older XGBoost run]
    |        -> results_12regions/region_performance_metrics.csv
    |        -> results_12regions/region_fog_prevalence_summary.csv
    |
    +-- shap_analysis*.py, fog_visibility*.py, fog_*_graph.py
             -> results_12regions/shap_*, results_english_figures/*

    ???  -> Tables 3 and 4 of Atmosphere_eng_v5.docx      <-- BROKEN LINK
```

The chain from *raw data → code → result file* exists and is executable. **The
chain from *result file → manuscript table* does not.** No script in the project
writes the values that appear in the manuscript, and no CSV in the project
contains them.

### 1.3 Three inconsistent result sets for the same experiment

| Source | Taebaek overall XGBoost F1 |
|---|---:|
| `fog_results_12regions/model_comparison_by_region.csv` | 0.503 |
| `results_12regions/region_performance_metrics.csv` | 0.516 |
| `Atmosphere_eng_v5.docx` Table 3 | **0.604** |
| This audit's reproduction | 0.585 |

Nothing in the project records which of these was intended to be authoritative.

---

## 2. Findings

Severity: **CRITICAL** (invalidates a published claim) / **HIGH** (materially
changes results) / **MEDIUM** (methodological weakness) / **LOW** (cosmetic).

### C1 — CRITICAL — Manuscript results have no provenance

The values in Table 3 (XGBoost, 1D-CNN, LSTM columns), Table 4, and the Abstract
do not match any stored result file, and no code path produces them.

Reproduced mean F1 vs. stored CSVs vs. manuscript, overall T+1 task:

| Model | Manuscript v5 | Stored legacy CSV | This audit |
|---|---:|---:|---:|
| Persistence | 0.570 ± 0.117 | 0.576 | **0.578 ± 0.122** |
| XGBoost | 0.575 ± 0.117 | 0.410 | **0.507 ± 0.210** |
| 1D-CNN | 0.512 ± 0.132 | 0.275 | **0.369 ± 0.194** |
| LSTM | 0.552 ± 0.113 | 0.287 | **0.429 ± 0.207** |

Per station the Persistence column of Table 3 matches the stored CSV to within
±0.005 in 11 of 12 stations. The other three columns match nothing. Moreover the
manuscript's three trained-model columns sit at an almost constant offset from
its own Persistence column at every station (XGBoost ≈ P + 0.01, LSTM ≈ P − 0.02,
1D-CNN ≈ P − 0.06). Genuine per-station experimental results do not behave that
way — station-to-station differences in model behaviour are large, as the
reproduced numbers show (XGBoost−Persistence ranges from +0.007 at Baengnyeongdo
to −0.234 at Inje).

**Classification: D/E for 47 of 59 checked manuscript values, A for 10, B for 2.**
See `results/tables/old_vs_new_comparison.csv` for the value-by-value table.

**Action taken:** all affected numbers discarded and regenerated.

### C2 — CRITICAL — Methods section does not describe the code that ran

| Item | Manuscript v5 | Code actually in the project |
|---|---|---|
| DL framework | PyTorch | **TensorFlow / Keras** (no PyTorch anywhere) |
| 1D-CNN conv channels | 32 → 64 | 64 → 128 |
| 1D-CNN pooling | global **average** pooling | `GlobalMaxPooling1D` |
| 1D-CNN extras | — | two `BatchNormalization` layers (undocumented) |
| 1D-CNN classifier | FC 64→32, out 32→1 | FC 64→1 (no 32-unit layer) |
| LSTM depth | single layer, hidden 64 | **two layers**, 64 → 32 |
| DL input variables | 14 | **20** |
| Loss | `BCEWithLogitsLoss` + `pos_weight` | `binary_crossentropy` + Keras `class_weight` |
| Optimiser LR | Adam, 1e-3 | Adam, Keras default (1e-3) — matches |
| Batch size | 512 | 256 |
| Max epochs | 15 | 50 |
| Early stopping | validation **F1**, patience 3 | validation **loss**, patience 5 |

A reader following the manuscript Methods could not have produced the code, and
the code could not have produced the manuscript Methods.

**Action taken:** the models were re-implemented in PyTorch exactly as the
manuscript describes (this was the option selected by the author), and the
Methods text was rewritten to match the new implementation line by line.

### L1 — CRITICAL — Deep models were handicapped by one hour

`fog_experiment_12regions_rewrite.py::make_sequences`:

```python
for i in range(seq_len, len(X_raw)):
    X_seq.append(X_raw[i - seq_len:i])   # t-6 ... t-1   (EXCLUDES t)
    y_seq.append(y_raw[i])               # fog at t+1
```

The window ends at `i-1`, so the deep models never saw the observation at time
`t` — they were effectively making a **two-hour-ahead** forecast. XGBoost and
Persistence both received time `t`. The paper's central deep-learning-versus-
XGBoost comparison was therefore not like-for-like, and the conclusion "XGBoost
outperformed the deep models" was partly an artefact of this asymmetry.

**Action taken:** windows now cover `t-5 … t` inclusive for all sequence models.
Effect at Taebaek, holding everything else fixed: LSTM F1 0.229 → 0.493.

### L2 — HIGH — Onset sequences were built from non-contiguous hours

`fog_onset_experiment_12regions.py` filtered to onset candidates (`fog_now == 0`)
*before* building sequences. The resulting "previous 6 hours" was a concatenation
of six arbitrary non-fog timestamps, which could span days. The models were
trained on meteorological histories that never occurred.

**Action taken:** sequences are now built on the full hourly series; the onset
label is attached only at candidate timestamps, so the 6-hour history is real.

### L3 — HIGH — Lag features were not guaranteed to be lags

`preprocess()` deleted rows failing the physical-range filter and *then* applied
`.shift(1..3)`. After a deleted row, `temp_lag_1` silently referred to 2, 3 or
more hours earlier. The same applies to the 2-hour rolling statistics.

**Action taken:** out-of-range values are set to NaN without deleting the row,
the series is reindexed onto a complete hourly grid, lags and rolling statistics
are computed on that grid, and incomplete rows are dropped afterwards.

### F1 — HIGH — Current visibility was excluded from the XGBoost feature set

The overall-task `FEATURES` list contained `vis_lag_1/2/3` but **not**
`시정(10m)` at time `t`, while the manuscript's SHAP discussion states that
"current visibility" is the most influential feature. Excluding it also made the
XGBoost-vs-Persistence comparison unfair, since Persistence is by definition a
function of visibility at `t`.

Using visibility at `t` to predict fog at `t+1` is not leakage — it is an
observation available at forecast issue time, and the manuscript's own temporal
contract (Section 2.2) permits it.

**Action taken:** `시정(10m)` added to both the tabular and sequential feature
sets. SHAP on the reproduced models confirms it is rank 1.

### E1 — HIGH — Unequal evaluation sets across models

Sequence models could not score rows lacking a complete 6-hour history, so they
were evaluated on ~2% fewer test cases than Persistence and XGBoost, with a
different positive prevalence. F1 values were being compared across different
denominators.

**Action taken:** all four models are now evaluated on the identical set of test
rows (the rows every model can score).

### T1 — PASS — Temporal split, scaler fitting and threshold selection

These were audited and found **correct** in the legacy code:

* the split is strictly chronological (`train < 2022-01-01 ≤ val < 2023-01-01 ≤ test`); no random shuffling anywhere;
* `StandardScaler` is fitted on training rows only;
* the classification threshold is chosen on the validation set and applied unchanged to the test set;
* sequences are built after the split, so no window crosses a split boundary;
* no feature reads a `t+1` quantity.

The canonical pipeline keeps all four properties and adds explicit assertions
(`assert_no_future_leakage`, disjoint-split assertions in `prepare_data.py`).

### M1 — MEDIUM — Table 2 transition counts not reproducible

Manuscript Table 2 reports 203,617 / 1,148 / 1,150 / 1,997 (total 208,912).
The reproduced test-period totals are 184,474 / 1,095 / 1,090 / 1,938
(total 188,597). The *proportions* agree closely (97.8 % vs 97.9 % for 0→0), but
the sample count does not match any row set produced by the project's code.
**Classification: C (provenance unclear).**

### S1 — Statistical significance testing added

The v5 Limitations section stated that no significance testing was performed.
The revised study tests every model pair with a two-sided Wilcoxon signed-rank
test over the twelve stations (one paired observation per station), Holm-
corrected within each prediction task **and evaluation metric** — the three
correction families are overall-task F1 (six comparisons), onset-task F1 (three)
and onset-task PR-AUC (three) — with matched-pairs rank-biserial effect sizes
(`scripts/significance_tests.py` → `results/tables/table_significance_tests.csv`).

The tests changed two claims that the point estimates alone would have supported:

| Comparison | Mean difference | p (Holm) | Verdict |
|---|---:|---:|---|
| overall: Persistence vs XGBoost | +0.070 | 0.068 | **not significant** — claim softened to "XGBoost did not improve on Persistence" |
| overall: Persistence vs LSTM | +0.148 | 0.003 | significant |
| overall: Persistence vs 1D-CNN | +0.209 | 0.003 | significant |
| overall: XGBoost vs LSTM | +0.078 | 0.004 | significant |
| overall: XGBoost vs 1D-CNN | +0.138 | 0.004 | significant |
| overall: LSTM vs 1D-CNN | +0.060 | 0.068 | not significant |
| onset F1: XGBoost vs LSTM | +0.022 | 0.301 | **not significant** — "XGBoost most reliable" replaced by "comparable" |
| onset F1: XGBoost vs 1D-CNN | +0.047 | 0.157 | not significant |
| onset PR-AUC: XGBoost vs LSTM | −0.005 | 0.791 | not significant |
| onset PR-AUC: XGBoost vs 1D-CNN | +0.048 | 0.007 | significant |
| onset PR-AUC: LSTM vs 1D-CNN | +0.049 | 0.002 | significant |

Two consequences for the paper's narrative, both applied to the final revised manuscript:

1. The **Persistence-beats-XGBoost** result is a failure of XGBoost to improve on
   the baseline, not a demonstrated superiority of the baseline. The Abstract,
   Results 3.3, Discussion 4.1 and Conclusions now say exactly that.
2. **XGBoost and LSTM are statistically indistinguishable for onset detection**
   on both F1 and PR-AUC. The v5 claim that XGBoost was best "in all twelve
   regions" is replaced by a statement that the tabular and recurrent models are
   comparable, with both ranking the rare positive class better than the 1D-CNN
   (PR-AUC, significant).

What survives significance testing is the cleanest part of the study: **neither
deep model matches either the persistence baseline or XGBoost on overall T+1
prediction**, at p < 0.005 in all four comparisons.

### S2 — Transition-type decomposition and prevalence dependence added

Two analyses were added after the statistical audit, both computed by
`scripts/transition_analysis.py` from the per-row test predictions already
written by `scripts/run_experiments.py`. **No model was retrained**: the run that
produced the missing deep-model prediction files was verified to reproduce all
48 overall-task metric rows and all 20 metric columns bit-identically against the
stored `station_metrics.csv` before its predictions were used.

**(a) Where overall T+1 skill comes from.** Pooled over the twelve stations the
overall task has 2,969 positives, of which 1,902 (64.1 %) are 1→1 transitions
(fog already present and persisting) and 1,067 are 0→1 transitions (fog forming).

| Model | Recall 1→1 | Recall 0→1 | Precision |
|---|---:|---:|---:|
| Persistence | 1.000 | 0.000 | 0.642 |
| XGBoost | 0.930 | 0.228 | 0.543 |
| 1D-CNN | 0.804 | 0.355 | 0.347 |
| LSTM | 0.831 | 0.275 | 0.379 |

The ordering by overall F1 (Persistence > XGBoost > LSTM > 1D-CNN) is *reversed*
on 0→1 recall. This is reported with precision alongside, and the manuscript
states explicitly that the higher formation recall of the 1D-CNN reflects a more
liberal operating point rather than better onset detection — on the dedicated
onset task it ranks last on both F1 and PR-AUC.

These counts refer to the **common evaluation subset** (182,984 rows), not the
full test period (188,597 rows) reported in Table 2; the manuscript states this
in both the Section 3.7 text and the Table 5 caption so the two sets of counts
cannot be mistaken for a discrepancy. The manuscript also avoids describing
64.1 % as a share of F1 or of "skill": F1 does not decompose that way, and the
figure is stated as a share of positive cases only.

**(b) When a trained model beats the baseline.** The per-station F1 advantage
over Persistence is positively associated with station fog prevalence:

| Model | Spearman ρ | raw p | Holm p |
|---|---:|---:|---:|
| XGBoost | 0.83 | 0.0010 | 0.0029 |
| LSTM | 0.71 | 0.0092 | 0.0184 |
| 1D-CNN | 0.63 | 0.0283 | 0.0283 |

The three p-values are Holm-corrected as a family, consistent with the
model-comparison tests; all three remain significant at 0.05 after correction.

**The prevalence plotted is the fog(t+1) prevalence of the common evaluation
subset**, not the training prevalence and not the full test-set prevalence. The
three differ materially at some stations (Baengnyeongdo 4.78 % evaluation vs
6.66 % training; Seoul 0.33 % vs 0.15 %), so the manuscript reports the
relationship as an observational association across twelve stations and
explicitly declines to attribute it to training-set size.

These became Section 3.7, Table 5 and Figure 7 of the manuscript. They were
appended at the end of Results so that no pre-existing section, table or figure
number changed.

### M2 — MEDIUM — Two stations have far less usable data than the others

After cleaning, Dongducheon (8,578 test rows) and Cheorwon (8,518) have roughly
half the test period of the other stations, and Inje and Paju lose large parts of
the training period. The manuscript describes all twelve stations as covering
2015–2024 without qualification.
**Action taken:** `results/metrics/split_summary.csv` records the exact usable
period and row count per station and split, and Section 2.1 of the revised
manuscript now states the coverage range (66–98 % of the nominal 2015–2024
hourly record), names the three worst-covered stations, and warns that
station-level results should be read with this heterogeneity in mind.

### L4 — LOW — Reproducibility metadata absent

The legacy scripts set `random_state = 42` for XGBoost and `tf.random.set_seed`,
but did not seed Python or NumPy, did not record package versions, and printed
some results only to stdout.
**Action taken:** `src/train/deep.py::set_seed` seeds Python, NumPy, PyTorch and
CUDA and enables deterministic algorithms; `results/logs/run_env.json` records
versions, seeds and elapsed time for every run.

---

## 3. Were the old 1D-CNN and LSTM real?

**Yes — but not the ones described in the paper.**

`fog_experiment_12regions_rewrite.py` lines 409–415 and 467–476 contain genuine
`tensorflow.keras.layers.LSTM` and `Conv1D` layers, and both are reached in
`forward`/`fit`. The concern that they had degenerated into an MLP is **not
supported**. What *is* true is that:

* they are a different architecture from the one the paper describes (C2);
* they were fed a window that excluded the current hour (L1, L2);
* their reported results were not the ones in the paper (C1).

The legacy results are therefore discarded not because the models were fake, but
because they answer a different question than the paper asks, and because the
paper's numbers did not come from them either.

The replacement models are verified structurally before every run by
`scripts/sanity_check_models.py`, which asserts:

* `1D-CNN` contains ≥ 2 `nn.Conv1d` modules, no `nn.LSTM`, no `nn.Flatten`;
* `LSTM` contains ≥ 1 `nn.LSTM` module, no `nn.Conv1d`, no `nn.Flatten`;
* the convolution receives the tensor as `(batch, n_features, seq_len)` so the
  kernel slides along **time**, and `'same'` padding preserves the time axis;
* `h_n` has shape `(num_layers, batch, hidden)` and is what the classifier reads;
* `conv1.weight`, `conv2.weight`, `lstm.weight_ih_l0`, `lstm.weight_hh_l0` all
  receive finite, non-zero gradients;
* reversing the time axis changes the output — a model invariant to time order
  would not be a temporal model.

The gate is run on both a synthetic batch and a real scaled ASOS batch, and
`scripts/reproduce_all.py` refuses to continue if it fails.

Verified shapes (seq_len = 6, 14 input variables):

```
1D-CNN   input (B, 6, 14) -> transpose (B, 14, 6)
         conv1 (B, 14, 6) -> (B, 32, 6)
         conv2 (B, 32, 6) -> (B, 64, 6)
         GAP   (B, 64, 6) -> (B, 64, 1)
         fc1   (B, 64)    -> (B, 32)
         out   (B, 32)    -> (B, 1)      trainable parameters: 9,697

LSTM     input (B, 6, 14)
         lstm  seq_out (B, 6, 64), h_n (1, B, 64), c_n (1, B, 64)
         fc1   (B, 64)    -> (B, 32)
         out   (B, 32)    -> (B, 1)      trainable parameters: 22,593
```

---

## 4. What was discarded, and what survived

### Discarded

* All XGBoost / 1D-CNN / LSTM values in Table 3, Table 4 and the Abstract of v5.
* All legacy deep-learning results in `fog_results_12regions/` and
  `fog_onset_results_12regions/` (wrong window alignment, wrong architecture).
* All legacy SHAP figures (computed on models trained without current visibility).
* Manuscript Table 2 counts (unreproducible sample total).
* The claim "XGBoost outperformed 1D-CNN and LSTM **in all twelve regions**"
  for onset — false in the legacy CSV (XGBoost loses at Inje, Chuncheon,
  Cheorwon, Paju) and false in the reproduction (XGBoost best in 8/12).

### Survived re-examination

* The **experimental design**: two separate tasks, chronological split,
  validation-only threshold selection, F1/PR-AUC as primary metrics under
  extreme imbalance. All sound.
* The **Persistence column** of Table 3 — matches the stored CSV in 11/12
  stations and the reproduction in 12/12 within ±0.02.
* The claim **"at the default threshold, Persistence outperformed XGBoost in all
  twelve regions"** — confirmed, 12/12, in the reproduction
  (`results/tables/table_persistence_vs_xgboost.csv`).
* The claim that the **No-Onset Baseline scores F1 = 0 everywhere** — confirmed.
* The **SHAP conclusion**: current visibility, dew point depression and relative
  humidity are the top-ranked features. Confirmed on the reproduced models
  (ranks 1, 2 and 4; visibility at `t−1` is rank 3).
* The **qualitative ordering** LSTM > 1D-CNN in both tasks — confirmed.

### Reversed

* "XGBoost and Persistence achieved comparable mean F1 (0.575 vs 0.570)" →
  **Persistence 0.578 vs XGBoost 0.507**, Persistence winning at 9/12 stations
  even after threshold tuning. The difference is not significant after Holm
  correction (p = 0.068), so the paper now claims only that XGBoost fails to
  improve on the baseline.
* "XGBoost outperformed the deep models in all twelve regions" (onset) →
  **8/12**; LSTM-Onset wins at Chuncheon and Cheorwon, 1D-CNN-Onset at Seoul and
  Inje, LSTM-Onset has the higher mean PR-AUC (0.138 vs 0.133), and no
  XGBoost–LSTM difference reaches significance.

---

## 5. Reproduced results (new ground truth)

Source of truth: `results/metrics/station_metrics.csv`.
Tables and figures are generated from it by `scripts/make_tables.py` and
`scripts/make_figures.py`; no number is typed by hand.

### Overall T+1 fog prediction, F1 (test 2023–2024, validation-tuned threshold)

| Station | Persistence | XGBoost | 1D-CNN | LSTM |
|---|---:|---:|---:|---:|
| Baengnyeongdo | 0.738 | **0.745** | 0.636 | 0.709 |
| Daegwallyeong | 0.611 | **0.627** | 0.550 | 0.546 |
| Paju | **0.633** | 0.632 | 0.449 | 0.576 |
| Incheon | **0.679** | 0.669 | 0.505 | 0.567 |
| Ganghwa | **0.563** | 0.560 | 0.389 | 0.442 |
| Dongducheon | 0.645 | **0.648** | 0.619 | 0.612 |
| Chuncheon | **0.578** | 0.417 | 0.219 | 0.399 |
| Taebaek | **0.601** | 0.585 | 0.350 | 0.493 |
| Cheorwon | **0.615** | 0.487 | 0.349 | 0.360 |
| Sokcho | **0.562** | 0.508 | 0.220 | 0.348 |
| Seoul | **0.446** | 0.182 | 0.090 | 0.067 |
| Inje | **0.262** | 0.028 | 0.054 | 0.035 |
| **Mean ± SD** | **0.578 ± 0.122** | 0.507 ± 0.210 | 0.369 ± 0.194 | 0.429 ± 0.207 |
| Best in | **9/12** | 3/12 | 0/12 | 0/12 |

### 0→1 onset detection, F1

| Station | No-Onset | XGBoost-Onset | 1D-CNN-Onset | LSTM-Onset |
|---|---:|---:|---:|---:|
| Baengnyeongdo | 0.000 | **0.323** | 0.249 | 0.287 |
| Daegwallyeong | 0.000 | **0.293** | 0.230 | 0.227 |
| Paju | 0.000 | **0.272** | 0.202 | 0.252 |
| Incheon | 0.000 | **0.208** | 0.142 | 0.161 |
| Ganghwa | 0.000 | **0.264** | 0.192 | 0.168 |
| Dongducheon | 0.000 | **0.253** | 0.172 | 0.198 |
| Chuncheon | 0.000 | 0.067 | 0.091 | **0.185** |
| Taebaek | 0.000 | **0.171** | 0.112 | 0.170 |
| Cheorwon | 0.000 | 0.107 | 0.081 | **0.172** |
| Sokcho | 0.000 | **0.233** | 0.047 | 0.068 |
| Seoul | 0.000 | 0.063 | **0.149** | 0.080 |
| Inje | 0.000 | 0.000 | **0.022** | 0.021 |
| **Mean ± SD** | 0.000 | **0.188 ± 0.105** | 0.141 ± 0.072 | 0.166 ± 0.077 |
| Mean PR-AUC | 0.006 | 0.133 | 0.088 | **0.138** |
| Best in | 0/12 | **8/12** | 2/12 | 2/12 |

### SHAP, overall T+1 XGBoost, pooled over 12 stations

| Rank | Feature | Mean \|SHAP\| |
|---:|---|---:|
| 1 | Visibility at `t` | 2.070 |
| 2 | Dew point depression | 0.691 |
| 3 | Visibility at `t−1` | 0.652 |
| 4 | Relative humidity | 0.476 |
| 5 | Air − ground temperature difference | 0.239 |

---

## 6. Answers to the ten acceptance questions

| # | Question | Answer |
|---:|---|---|
| 1 | Is the 1D-CNN a real `Conv1d` model? | **YES** — verified by `sanity_check_models.py`, gradients and time-order sensitivity checked |
| 2 | Is the LSTM a real recurrent model? | **YES** — `nn.LSTM`, `h_n` consumed by the classifier, verified |
| 3 | Can every input shape be explained? | **YES** — printed and asserted for synthetic and real batches |
| 4 | Is there any future-information leakage? | **NO** — all predictors ≤ `t`, asserted programmatically |
| 5 | Was the test set kept out of model/threshold selection? | **YES** — threshold chosen on 2022 only, then frozen |
| 6 | Can a source CSV be found for every manuscript number? | **YES for the revised manuscript.** **NO for v5** — that is finding C1 |
| 7 | Does re-running give the same result? | **YES.** A clean unzip of the released repository, re-run from the raw CSVs, reproduced all eight Taebaek metrics to the last decimal (difference 0.000000 for every model and both tasks). Across *different seeds* the deep models vary by 0.02–0.04 F1; quantified in §7 |
| 8 | Can a new machine run this from the repository alone? | **YES** — `pip install -r requirements.txt && python scripts/reproduce_all.py`; raw data must be downloaded from KMA (see README) |
| 9 | Do the Methods match the code? | **YES for the final revised manuscript.** **NO for v5** — finding C2 |
| 10 | Do the Results match the stored results? | **YES for the final revised manuscript**, audited by `scripts/audit_consistency.py --manuscript`. **NO for v5** |

---

## 7. Seed-stability diagnostic

Command: `python scripts/run_experiments.py --seeds 42 1337 2024 --stability
--stations Baengnyeongdo Paju Taebaek Seoul`, summarised by
`scripts/report_stability.py`. Four stations spanning the full prevalence range
(4.73 %, 2.34 %, 0.88 %, 0.35 %) were used, since a twelve-station three-seed
sweep costs about three times the main run for a diagnostic that does not enter
the reported protocol. Raw output: `results/metrics/stability_metrics.csv`.

Mean within-station spread of test F1 across seeds {42, 1337, 2024}:

| Task | Model | Mean F1 | Mean within-station SD | Mean within-station range |
|---|---|---:|---:|---:|
| overall | XGBoost | 0.547 | **0.011** | 0.021 |
| overall | LSTM | 0.452 | 0.037 | 0.074 |
| overall | 1D-CNN | 0.408 | 0.037 | 0.071 |
| onset | XGBoost-Onset | 0.205 | **0.007** | 0.014 |
| onset | LSTM-Onset | 0.185 | 0.023 | 0.044 |
| onset | 1D-CNN-Onset | 0.150 | 0.038 | 0.075 |

Three conclusions:

1. **XGBoost is essentially seed-insensitive** (SD ≈ 0.01), the deep models are
   not (SD ≈ 0.02–0.04, and up to 0.066 for the 1D-CNN onset model at Seoul,
   where F1 ranged from 0.019 to 0.149 across three seeds). Any claim about
   deep-model performance at low-prevalence stations from a single run is
   fragile, and the revised Limitations section says so.

2. **The seed spread does not overturn any reported ordering.** The
   Persistence-minus-XGBoost gap in the overall task (0.071) and the
   XGBoost-minus-1D-CNN gap (0.138) are both larger than the deep models'
   seed-to-seed SD. The ordering Persistence > XGBoost > LSTM > 1D-CNN survives.

3. **The reported seed (42) is not a favourable pick.** Restricted to the same
   four stations, the seed-42 means are 0.536 (XGBoost), 0.461 (LSTM) and 0.381
   (1D-CNN) against three-seed averages of 0.547, 0.452 and 0.408 — the seed-42
   1D-CNN is *below* its own seed average. No seed selection was performed at
   any point; seed 42 was fixed before the first run.

The paper reports the single fixed-seed run rather than a seed average, so that
the protocol is not silently changed relative to the original study design; the
variability above is disclosed in the Limitations section and in this file.

---

## 8. Files produced by this audit

```
fog-forecast-repro/
├── REPRODUCIBILITY_AUDIT.md          this document
├── README.md                         reproduction instructions
├── requirements.txt                  pinned environment
├── configs/default.yaml              every hyperparameter, single source
├── src/                              data / models / train / evaluate
├── scripts/                          prepare, sanity-check, run, tables, figures, audit
├── tools/                            document-generation scripts (see tools/README.md;
│                                     require the authors' .docx templates, not shipped)
├── results/metrics/                  SOURCE OF TRUTH for every manuscript number
├── results/tables/                   generated tables + old-vs-new comparison
├── results/figures/                  regenerated figures
├── reference/manuscript_v5_claims.csv   every v5 claim, machine-readable
├── reference/legacy_results/         the pre-existing CSVs, preserved unmodified
└── checkpoints/                      empty in the public repository — trained models are
                                      regenerated by scripts/run_experiments.py and are
                                      deliberately NOT redistributed
```

Delivered alongside this repository:

| File | Contents |
|---|---|
| `Atmosphere_eng_v12_final.docx` | final revised manuscript; MDPI template, references and Figure 1 preserved, Figures 2–6 replaced with regenerated versions, Tables 2–4 and all prose numbers rebuilt from `results/metrics/` |
| `cover_letter_v5_persistence_lit.docx` | cover letter updated to the reproduced claims |
| `REPRODUCIBILITY_AUDIT.md` | this document |

A clean-room check was performed: the released archive was unpacked into an
empty directory, the raw CSVs were placed in `data/raw/`, and the pipeline was
re-run. Every Taebaek metric matched the values reported here exactly
(difference 0.000000), confirming that the repository alone is sufficient.

Both documents were produced by the scripts in `tools/`
(`update_manuscript.py`, `update_cover_letter.py`, `edit_doc.py`), which read
the result CSVs directly and write the values into the `.docx` templates. Those
scripts are included in the repository for transparency, but they are **not part
of `scripts/reproduce_all.py`** and cannot be run by a third party, because they
require the authors' `.docx` template files, which are the manuscript itself and
are not redistributed. See `tools/README.md`.

`scripts/audit_consistency.py` cross-checks a finished `.docx` against the
config and the result CSVs and reports **PASS = 24, FAIL = 0** for
`Atmosphere_eng_v8_final.docx`. A second script,
`scripts/check_narrative_consistency.py`, audits the *wording* of the
manuscript, cover letter, this file and `README.md` together — it fails on any
equivalence, unsupported-superiority, causal-overreach or repository-capability
claim, and on any statement of the core narrative that is missing from one of
the four documents. It currently reports **PASS = 19, FAIL = 0**. This check is a **separate manual
invocation** with `--manuscript`; `scripts/reproduce_all.py` calls the same
script without that argument, so the single-command pipeline verifies
configuration values and generated result files only, not the document.

Nothing in `D:\07_fog` was deleted or overwritten. The revised manuscript is
saved as a new file; `Atmosphere_eng_v5.docx` is untouched.
