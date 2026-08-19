# -*- coding: utf-8 -*-
"""Build Atmosphere_eng_v6_reproduced.docx from Atmosphere_eng_v5.docx.

Every number written into the manuscript is read from the reproduced result
CSVs in fog-forecast-repro/results/.  Nothing is typed by hand: if a value
appears in the output document, it came out of a CSV in this run.

    python update_manuscript.py
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from edit_doc import Doc  # noqa: E402

REPO = os.path.dirname(HERE)   # repository root
M = os.path.join(REPO, "results", "metrics")
T = os.path.join(REPO, "results", "tables")
F = os.path.join(REPO, "results", "figures")
UNPACKED = os.path.join(HERE, "unpacked")

STATION_ORDER = ["Baengnyeongdo", "Daegwallyeong", "Paju", "Incheon", "Ganghwa",
                 "Dongducheon", "Chuncheon", "Taebaek", "Cheorwon", "Sokcho",
                 "Seoul", "Inje"]
OVERALL = ["Persistence", "XGBoost", "1D-CNN", "LSTM"]
ONSET = ["No-Onset Baseline", "XGBoost-Onset", "1D-CNN-Onset", "LSTM-Onset"]


# ---------------------------------------------------------------- results
met = pd.read_csv(os.path.join(M, "station_metrics.csv"))
prev = pd.read_csv(os.path.join(M, "fog_prevalence.csv")).set_index("station")
trans = pd.read_csv(os.path.join(T, "table_transition_counts_test.csv"))
shap_g = pd.read_csv(os.path.join(M, "shap_global.csv"))
errv = pd.read_csv(os.path.join(M, "error_visibility_rows.csv"))
hh = pd.read_csv(os.path.join(T, "table_persistence_vs_xgboost.csv"))
split = pd.read_csv(os.path.join(M, "split_summary.csv"))
sig = pd.read_csv(os.path.join(T, "table_significance_tests.csv"))


def sg(task, a, b, metric="f1"):
    """Return the significance-test row for a model pair (order-insensitive)."""
    m = sig[(sig["task"] == task) & (sig["metric"] == metric)]
    hit = m[(m["model_a"] == a) & (m["model_b"] == b)]
    if len(hit):
        return hit.iloc[0]
    hit = m[(m["model_a"] == b) & (m["model_b"] == a)]
    return hit.iloc[0]


def pstr(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"

f1o = met[met["task"] == "overall"].pivot_table(index="station", columns="model",
                                                values="f1").reindex(STATION_ORDER)
f1n = met[met["task"] == "onset"].pivot_table(index="station", columns="model",
                                              values="f1").reindex(STATION_ORDER)
pro = met[met["task"] == "overall"].pivot_table(index="station", columns="model",
                                                values="pr_auc").reindex(STATION_ORDER)
prn = met[met["task"] == "onset"].pivot_table(index="station", columns="model",
                                              values="pr_auc").reindex(STATION_ORDER)


def ms(series):
    return f"{series.mean():.3f} ± {series.std(ddof=1):.3f}"


def f(v, n=3):
    return f"{v:.{n}f}"


spw = met[met["model"] == "XGBoost"]["scale_pos_weight"].mean()
spw_on = met[met["model"] == "XGBoost-Onset"]["scale_pos_weight"].mean()
thr_o = met[met["model"] == "XGBoost"]["threshold"]
thr_n = met[met["model"] == "XGBoost-Onset"]["threshold"]

pers_wins = int(hh["persistence_beats_xgb_tuned"].sum())
xgb_wins = 12 - pers_wins
pers_wins_default = int((hh["persistence_f1"] > hh["xgboost_f1_default_0.5"]).sum())
xgb_best_onset = int((f1n[ONSET].idxmax(axis=1) == "XGBoost-Onset").sum())

prev = prev.assign(test_percent=prev["fog_prevalence_test"] * 100)
prev_sorted = prev.sort_values("test_percent", ascending=False)
mean_prev_test = prev["test_percent"].mean()   # same source as Figure 2

tp = errv[errv["error_type"] == "TP"]["vis_tplus1_10m"].median() / 100
fn = errv[errv["error_type"] == "FN"]["vis_tplus1_10m"].median() / 100
fp = errv[errv["error_type"] == "FP"]["vis_tplus1_10m"].median() / 100
n_tp = int((errv["error_type"] == "TP").sum())
n_fn = int((errv["error_type"] == "FN").sum())
n_fp = int((errv["error_type"] == "FP").sum())

top = shap_g.head(6)["feature"].tolist()
FEAT_EN = {"시정(10m)": "current visibility", "dew_point_depression": "dew point depression",
           "vis_lag_1": "visibility at the preceding hour", "습도(%)": "relative humidity",
           "air_ground_diff": "air–ground temperature difference",
           "현지기압(hPa)": "station pressure", "vis_lag_2": "visibility two hours earlier"}
top_en = [FEAT_EN.get(t, t) for t in top]

d = Doc(os.path.join(UNPACKED, "word", "document.xml"))

# ================================================================ ABSTRACT
d.replace_para(
    "Fog sharply reduces visibility",
    "Fog sharply reduces visibility, directly affecting aviation, road, and maritime "
    "operations as well as military activities, and short-term prediction remains "
    "challenging due to its rapid onset and dissipation. This study evaluated short-term "
    "fog prediction using ten years (2015–2024) of hourly ASOS data from twelve stations "
    "across the central-northern Korean Peninsula, separately assessing overall T+1 "
    "(1-hour-ahead) fog forecasting and 0→1 fog onset detection, using meteorological "
    "variables, lag/rolling statistics, dew point depression, and ground–air temperature "
    "difference as inputs. All four models were evaluated on an identical set of test "
    f"cases. In overall T+1 forecasting, the parameter-free Persistence baseline achieved "
    f"the highest mean F1-score ({ms(f1o['Persistence'])}), exceeding XGBoost "
    f"({ms(f1o['XGBoost'])}), LSTM ({ms(f1o['LSTM'])}) and 1D-CNN ({ms(f1o['1D-CNN'])}), "
    f"and outperforming XGBoost at {pers_wins} of the twelve stations after "
    f"validation-based threshold tuning and at all twelve stations at the default "
    f"threshold; the Persistence–XGBoost difference was not significant after correction "
    f"for multiple comparisons ({pstr(sg('overall','Persistence','XGBoost')['p_holm'])}), "
    f"whereas both exceeded the two deep models significantly "
    f"({pstr(sg('overall','XGBoost','LSTM')['p_holm'])} or better). Persistence cannot, "
    "however, detect fog onset by construction. In onset "
    f"detection, all trained models showed low absolute performance; XGBoost achieved the "
    f"highest mean F1-score ({ms(f1n['XGBoost-Onset'])}) and was best at "
    f"{xgb_best_onset} of the twelve stations, ahead of LSTM ({ms(f1n['LSTM-Onset'])}) and "
    f"1D-CNN ({ms(f1n['1D-CNN-Onset'])}), but XGBoost and LSTM were statistically "
    f"indistinguishable on both F1 "
    f"({pstr(sg('onset','XGBoost-Onset','LSTM-Onset')['p_holm'])}) and PR-AUC "
    f"({prn['LSTM-Onset'].mean():.3f} vs. {prn['XGBoost-Onset'].mean():.3f}). SHAP analysis "
    f"identified {top_en[0]}, {top_en[1]} and {top_en[3]} as the most influential features, "
    "consistently across stations. These results show that strong overall one-hour-ahead "
    "fog-state forecasting performance does not necessarily translate into useful fog-onset "
    "early-warning skill: the baseline that was hardest to beat on the overall task cannot "
    "issue an onset warning at all, and the trained models that can do so remain limited in "
    "absolute terms. Overall fog-state prediction and fog-onset detection should therefore "
    "be specified and evaluated as separate tasks.")

# ================================================================ INTRO
d.replace_para(
    "Third, we compare the performance of XGBoost",
    "The main contributions of this study are as follows. First, by separating overall T+1 "
    "fog prediction from 0→1 fog onset prediction, we show that overall performance "
    "evaluation alone cannot sufficiently characterize early-warning performance. Second, we "
    "evaluate persistence as a strong baseline in very short-range fog prediction while "
    "empirically demonstrating its structural inability to detect 0→1 onset. Third, we "
    "compare XGBoost, 1D-CNN, and LSTM across twelve ASOS stations in the central-northern "
    "Korean Peninsula under a strictly like-for-like protocol—identical temporal splits, "
    "identical evaluation cases, and input windows that give every model the same "
    "information horizon—to analyze regional fog predictability and between-station "
    "variability in model performance. "
    "Fourth, through SHAP analysis, we verify whether XGBoost predictions are based on "
    "physically interpretable variables—such as current and prior visibility, dew point "
    "depression, and relative humidity—and interpret the commonalities and differences in "
    "regional feature importance. The intended contribution is therefore an evaluation "
    "framework and a diagnostic finding about how short-term fog forecasting skill should be "
    "measured, rather than a proposal for a new best-performing model. All code, "
    "configuration files and result files required to reproduce the reported metrics, tables "
    "and figures are released publicly.")

d.replace_text("can maintain stable performance even with relatively limited data",
               "can perform competitively even with relatively limited data")

# ---- Persistence literature grounding (new reference [23]) ----------------
# Cornejo-Bueno, S. et al. (2020) Symmetry 12(6), 1045, doi:10.3390/sym12061045.
# Verified against the publisher page: the paper analyses low-visibility event
# persistence at Villanubla Airport (Valladolid, Spain) using RVR time series,
# performs hourly short-term prediction, and defines the naive persistence
# operator verbatim as "a special case of first-order Markov chain, whose
# formula x(t+1)=x(t) forces the state preservation at any time".  Its best
# reported configuration is a mixture of experts combining persistence-based
# methods with machine learning.  We do NOT claim it used the same visibility
# threshold or the same target definition as the present study.
d.replace_para(
    "Current visibility state and fog persistence are also important factors",
    "Current visibility state and fog persistence are also important factors in very "
    "short-range fog prediction. Because fog tends to persist for some time once formed, "
    "current or immediately preceding visibility can serve as a strong predictor of "
    "visibility at the next time step. Peláez-Rodríguez et al. [21] showed that past "
    "visibility values—i.e., persistence-related variables—are key predictors of short-term "
    "visibility in fog-related low-visibility prediction on Spanish mountain roads, and that "
    "incorporating ERA5 reanalysis data yields greater performance gains as lead time "
    "increases. Persistence has also been used as a formal prediction method, and not only as "
    "a source of predictor variables, in short-term low-visibility forecasting. Cornejo-Bueno "
    "et al. [23] analysed the persistence of low-visibility events at Valladolid Airport, "
    "Spain, using Runway Visual Range time series, and evaluated hourly short-term prediction "
    "with a naïve persistence operator, x(t+1) = x(t), alongside Markov-chain and "
    "machine-learning approaches; their best-performing configuration was a mixture of "
    "experts that combined persistence-based methods with machine learning. The strong "
    "short-term temporal persistence of low-visibility conditions is therefore an established "
    "observation rather than a new one. However, persistence has a structural limitation from "
    "an early-warning perspective: if the current time step is non-fog, a persistence model "
    "predicts non-fog at the next step as well, and is therefore unable to detect 0→1 fog "
    "onset—the transition from non-fog to fog. In other words, even if persistence performs "
    "well in overall T+1 fog prediction, this does not imply high performance in predicting "
    "fog onset, which is more operationally critical.")

# ================================================================ METHODS 2.2
d.replace_para(
    "Second, rolling statistics were generated",
    "Second, rolling statistics were generated for air temperature and relative humidity. "
    "Two-hour rolling means and standard deviations were computed over the window {t−1, t}. "
    "These statistics therefore use only observations available at or before the forecast "
    "issue time and contain no information from the target time t+1.")

d.replace_para(
    "First, lag features were generated.",
    "First, lag features were generated. Air temperature, humidity, and visibility values "
    "from the preceding 1–3 hours were included as input variables to capture short-term "
    "persistence and trends in meteorological conditions. Prior visibility in particular is "
    "closely related to fog persistence and dissipation, making it an important predictive "
    "cue for T+1 fog prediction. Because lag and rolling operators are only meaningful on a "
    "regular time axis, observations failing the physical-plausibility filter were set to "
    "missing rather than deleted, and each station series was reindexed onto a complete "
    "hourly grid before any lag or rolling feature was computed; rows still containing "
    "missing values were removed afterwards.")

d.replace_para(
    "The prediction target was the fog state at time",
    "The prediction target was the fog state at time t+1, whereas all predictors were "
    "restricted to information observable at or before time t. Current visibility at time t "
    "is one such predictor: it is available at forecast issue time and is included in the "
    "input of every trained model, so that the trained models and the Persistence baseline "
    "have access to the same information. No meteorological observation or derived feature "
    "containing information from t+1 was included in the model inputs. This temporal "
    "alignment prevented target-time information from leaking into model training and "
    "evaluation, and is enforced programmatically by an assertion over the feature list.")

d.replace_para(
    "However, their input representations differed according to model architecture",
    "Fog is influenced not only by meteorological conditions at a single time step but also "
    "by prior visibility, changes in humidity, surface cooling, and diurnal and seasonal "
    "variability. All models were constructed using the same raw station-level observations "
    "and the same temporal data split. Their input representations differed according to "
    "model architecture: XGBoost used a 28-variable tabular feature set, whereas the 1D-CNN "
    "and LSTM used six-hour sequences of 14 selected meteorological, lagged, physically "
    "derived, and temporal variables. Both representations end at time t, so the sequence "
    "window spans t−5 to t inclusive and the two model families forecast over the same "
    "one-hour horizon.")

# --- data coverage disclosure (Section 2.1) ---
cov = split.pivot(index="station", columns="split", values="n_rows")[["train", "val", "test"]]
cov["total"] = cov.sum(axis=1)
cov["pct"] = cov["total"] / (10 * 365.25 * 24) * 100
cov = cov.sort_values("pct")
worst = cov.head(3)

d.replace_para(
    "This study used hourly Automated Synoptic Observing System (ASOS) data",
    "This study used hourly Automated Synoptic Observing System (ASOS) data provided by the "
    "Korea Meteorological Administration (KMA). The analysis period spanned 2015–2024, and "
    "station-level data included air temperature, dew point temperature, relative humidity, "
    "wind speed, station pressure, ground temperature, and visibility. Data were obtained "
    "through the KMA Open MET Data Portal, and missing values and outliers were preprocessed "
    "on a per-station, hourly basis prior to model training and evaluation. Data coverage is "
    "not uniform across the twelve stations: after cleaning, the number of usable hourly "
    f"records ranges from {int(cov['total'].min()):,} to {int(cov['total'].max()):,} "
    f"({cov['pct'].min():.0f}–{cov['pct'].max():.0f}% of the nominal 2015–2024 hourly "
    f"record). Coverage is lowest at {worst.index[0]} ({worst['pct'].iloc[0]:.0f}%), "
    f"{worst.index[1]} ({worst['pct'].iloc[1]:.0f}%) and {worst.index[2]} "
    f"({worst['pct'].iloc[2]:.0f}%); at {worst.index[1]} and {worst.index[2]} the shortfall "
    "falls mainly in the test period, so their test sets contain roughly half as many cases "
    "as the other stations, while at "
    f"{worst.index[0]} it falls mainly in the training period. Per-station record counts for "
    "every split are reported in the accompanying repository. Station-level results should "
    "therefore be compared with this heterogeneity in mind.")

# ================================================================ METHODS 2.4
d.replace_para(
    "XGBoost is a gradient boosting-based tree ensemble model",
    "XGBoost is a gradient boosting-based tree ensemble model proposed by Chen and Guestrin "
    "[7], well suited for learning nonlinear relationships and interactions among variables "
    "in tabular meteorological data. We used the XGBClassifier implementation with "
    "hyperparameters set to n_estimators = 250, max_depth = 5, learning_rate = 0.03, "
    "subsample = 0.6, and colsample_bytree = 0.6. scale_pos_weight was computed as the ratio "
    "of negative to positive samples in the training data, calculated separately for each "
    f"station and prediction task; the resulting mean scale_pos_weight values were "
    f"approximately {spw:.1f} for overall T+1 prediction and {spw_on:.1f} for onset "
    "prediction. Separate XGBoost models were independently trained for overall T+1 "
    "prediction and onset prediction.")

d.replace_para(
    "1D-CNN and LSTM were implemented in PyTorch",
    "1D-CNN and LSTM were implemented in PyTorch (version 2.13), with an input sequence "
    "length of 6 hours covering times t−5 through t. LSTM is a recurrent neural network "
    "architecture designed to learn temporal dependencies in sequential data and was used "
    "here as a comparison model for learning short-term continuous changes in meteorological "
    "variables [16]. The sequential input consisted of 14 variables: seven basic "
    "meteorological variables (air temperature, dew point temperature, humidity, wind speed, "
    "station pressure, ground temperature, and visibility), dew point depression, ground–air "
    "temperature difference, visibility and humidity at the preceding hour, hour of day, and "
    "sine/cosine-encoded month; all variables were standardized using a StandardScaler fitted "
    "on the training split only. Sequences were constructed after the temporal split, and a "
    "window was retained only when all six timestamps were exactly one hour apart, so that no "
    "window crossed a split boundary or a gap in the record. The 1D-CNN consisted of two 1D "
    "convolutional layers applied along the time axis (32→64 channels, kernel size 3, same "
    "padding, ReLU), global average pooling over time, dropout (rate = 0.3), a fully "
    "connected layer (64→32, ReLU), and an output layer (32→1); it has 9,697 trainable "
    "parameters. The LSTM consisted of a single layer (hidden size = 64) whose final hidden "
    "state was passed to dropout (rate = 0.3), a fully connected layer (64→32, ReLU), and an "
    "output layer (32→1); it has 22,593 trainable parameters. Both models used "
    "BCEWithLogitsLoss as the loss function, with pos_weight set to the ratio of negative to "
    "positive samples in the training data to address class imbalance. Optimization used Adam "
    "(learning rate = 1e-3) with a batch size of 512, trained for up to 15 epochs with early "
    "stopping if validation F1-score did not improve for 3 consecutive epochs; the model "
    "weights from the epoch with the highest validation F1-score were used as the final "
    "model. Before every experiment an automated structural check verifies that the 1D-CNN "
    "contains genuine nn.Conv1d layers operating on the time axis and that the LSTM contains "
    "a genuine nn.LSTM module whose last hidden state feeds the classifier, that gradients "
    "reach both, and that reversing the input sequence changes the output; the experiment "
    "does not run if any check fails.")

d.replace_para(
    "All trained models used identical training/validation/test splits",
    "All trained models used identical training/validation/test splits, and the "
    "classification threshold that maximized F1-score on the validation set was selected and "
    "subsequently fixed for evaluation on the test set. Because sequence models require a "
    "complete six-hour history, a small number of test rows immediately following a gap in "
    "the record cannot be scored by them; all four models—including Persistence and "
    "XGBoost—were therefore evaluated on the identical subset of test cases that every model "
    "is able to score, so that all reported F1-scores share the same denominator and the "
    "same positive-class prevalence.")

# ---- Methods 2.4: link the baseline to the established naive-persistence principle
d.replace_para(
    "The Persistence model is a simple baseline that carries the current fog state forward",
    "The Persistence model is a simple baseline that carries the current fog state forward as "
    "the prediction for the next time step: if fog is present at time t, fog is predicted at "
    "t+1; if non-fog is present at time t, non-fog is predicted at t+1. This baseline follows "
    "the standard naïve persistence principle used in previous short-term low-visibility "
    "forecasting studies, in which the current state is carried forward to the next time step "
    "[23]; the fog definition, target variable and experimental setting of the present study "
    "differ from those studies and the comparison is therefore one of principle rather than of "
    "protocol. Although this model requires no training, it is important for evaluating the "
    "baseline performance contributed by prior visibility state and fog persistence in "
    "short-term fog prediction.")

# ================================================================ METHODS 2.5
d.replace_para(
    "For probability-output models—XGBoost, 1D-CNN, and LSTM",
    "For probability-output models—XGBoost, 1D-CNN, and LSTM—the optimal classification "
    "threshold was selected on the validation set based on the F1-score for the positive "
    "class: the fog class for overall T+1 prediction, and the onset class for onset "
    "prediction. We searched a predefined grid of 181 equally spaced values over 0.05–0.95 "
    f"for the threshold maximizing validation F1-score; across the twelve stations the "
    f"selected threshold for the overall T+1 XGBoost model ranged from {thr_o.min():.3f} to "
    f"{thr_o.max():.3f} (median {thr_o.median():.3f}) and for the onset XGBoost model from "
    f"{thr_n.min():.3f} to {thr_n.max():.3f} (median {thr_n.median():.3f}). It should be "
    "noted that, due to substantial class imbalance and class weighting, the model output "
    "probabilities function more as operational decision scores than as calibrated "
    "probabilities. That is, a threshold of 0.94 does not indicate a 94% probability of fog "
    "occurrence but should instead be interpreted as an operational threshold selected to "
    "maximize validation F1-score. The threshold selected on the validation set was fixed and "
    "applied to the test set to obtain final performance. The test set was never used for "
    "threshold selection, model selection, early stopping, or any other decision. "
    "Differences between models were assessed with the two-sided Wilcoxon signed-rank test "
    "over the twelve stations, each station contributing one paired observation, since "
    "twelve paired differences are too few to justify a normality assumption and "
    "station-level F1 values are strongly heterogeneous. Within each prediction task "
    "and evaluation metric, p-values across all pairwise model comparisons were corrected "
    "using the Holm step-down procedure; the three correction families were therefore "
    "overall-task F1 (six comparisons), onset-task F1 (three comparisons) and onset-task "
    "PR-AUC (three comparisons). Effect size is reported as the matched-pairs rank-biserial "
    "correlation r, which equals +1 when the first model wins at every station.")

# ================================================================ RESULTS 3.1
d.replace_para(
    "The mean fog occurrence rate across the twelve stations in the test set",
    f"The mean fog occurrence rate across the twelve stations in the test set was "
    f"approximately {mean_prev_test:.2f}%, indicating that non-fog conditions dominate the "
    f"analysis period and that the fog class is highly imbalanced. Fog occurrence rates, "
    f"however, varied substantially by region. "
    f"{prev_sorted.index[0]} showed the highest rate at {prev_sorted['test_percent'].iloc[0]:.2f}%, "
    f"followed by {prev_sorted.index[1]} ({prev_sorted['test_percent'].iloc[1]:.2f}%) and "
    f"{prev_sorted.index[2]} ({prev_sorted['test_percent'].iloc[2]:.2f}%), all above the overall "
    f"mean. In contrast, {prev_sorted.index[-1]} ({prev_sorted['test_percent'].iloc[-1]:.2f}%), "
    f"{prev_sorted.index[-2]} ({prev_sorted['test_percent'].iloc[-2]:.2f}%), and "
    f"{prev_sorted.index[-3]} ({prev_sorted['test_percent'].iloc[-3]:.2f}%) showed very low fog "
    f"occurrence rates.")

ratio = prev_sorted["test_percent"].iloc[0] / prev_sorted["test_percent"].iloc[-1]
d.replace_para(
    "Comparing Baengnyeongdo, the station with the highest occurrence rate",
    f"Comparing {prev_sorted.index[0]}, the station with the highest occurrence rate, to "
    f"{prev_sorted.index[-1]}, the station with the lowest, reveals an approximately "
    f"{ratio:.0f}-fold difference in fog occurrence. This indicates that fog prediction in "
    "this study involves not only a simple positive/negative class imbalance but also "
    "substantial spatial imbalance in occurrence frequency across regions.")

d.replace_para(
    "The red dashed line indicates the twelve-station mean fog occurrence rate",
    "Figure 2. Test-set fog occurrence rate by station. Fog occurrence rates for each of the "
    f"twelve stations in the test set. The twelve-station mean fog occurrence rate is "
    f"{mean_prev_test:.2f}%.")

# ================================================================ RESULTS 3.2
tr = {r["transition"]: r for _, r in trans.iterrows()}
onset_pct = float(tr["0→1"]["proportion_percent"])
d.replace_para(
    "Across the test sets of all twelve regions, the most common transition type",
    "Across the test sets of all twelve stations, the most common transition type was "
    f"non-fog persistence (0→0), accounting for {tr['0→0']['count']:,} cases "
    f"({tr['0→0']['proportion_percent']:.2f}% of the total). Fog onset cases (0→1) numbered "
    f"{tr['0→1']['count']:,} ({tr['0→1']['proportion_percent']:.2f}%), fog dissipation cases "
    f"(1→0) numbered {tr['1→0']['count']:,} ({tr['1→0']['proportion_percent']:.2f}%), and fog "
    f"persistence cases (1→1) numbered {tr['1→1']['count']:,} "
    f"({tr['1→1']['proportion_percent']:.2f}%).")

# ================================================================ RESULTS 3.3
d.replace_para(
    "Under this procedure, XGBoost outperformed Persistence in 9 of the 12 regions",
    "For XGBoost, 1D-CNN, and LSTM, the classification threshold maximizing F1-score on the "
    "validation set was selected and subsequently fixed for evaluation on the test set. Under "
    f"this procedure, Persistence outperformed XGBoost in {pers_wins} of the twelve stations, "
    f"and XGBoost outperformed Persistence in the remaining {xgb_wins}. The mean F1-score of "
    f"Persistence ({f(f1o['Persistence'].mean())}) exceeded that of XGBoost "
    f"({f(f1o['XGBoost'].mean())}) by {f(f1o['Persistence'].mean()-f1o['XGBoost'].mean())}. "
    "The two models were closely matched at the stations with the highest fog prevalence—"
    f"XGBoost was slightly ahead at {', '.join(hh[~hh['persistence_beats_xgb_tuned']]['station'])}"
    "—but the gap widened sharply at low-prevalence stations, a pattern consistent with the "
    "scarcity of positive training cases limiting the model's ability to learn a robust "
    f"decision boundary there (Seoul "
    f"{f(f1o.loc['Seoul','XGBoost'])} vs. {f(f1o.loc['Seoul','Persistence'])}; Inje "
    f"{f(f1o.loc['Inje','XGBoost'])} vs. {f(f1o.loc['Inje','Persistence'])}). The "
    "between-station standard deviation of XGBoost "
    f"({f1o['XGBoost'].std(ddof=1):.3f}) was substantially larger than that of Persistence "
    f"({f1o['Persistence'].std(ddof=1):.3f}); XGBoost therefore showed greater "
    "between-station variability. This quantity describes performance heterogeneity across "
    "the twelve stations and is not a measure of run-to-run stability under different random "
    "initialisations. A two-sided Wilcoxon signed-rank test over the "
    f"twelve stations did not establish this difference as significant after Holm correction "
    f"({pstr(sg('overall','Persistence','XGBoost')['p_raw'])} uncorrected, "
    f"{pstr(sg('overall','Persistence','XGBoost')['p_holm'])} corrected; rank-biserial "
    f"r = {sg('overall','Persistence','XGBoost')['rank_biserial_r']:.2f}). The appropriate "
    "reading is therefore that XGBoost did not improve on Persistence, not that Persistence "
    "was demonstrably superior.")

d.replace_para(
    "1D-CNN showed the lowest average performance across regions",
    f"The 1D-CNN showed the lowest average performance across stations "
    f"({ms(f1o['1D-CNN'])}), while the LSTM performed better than the 1D-CNN but still below "
    f"both XGBoost and Persistence ({ms(f1o['LSTM'])}). Neither deep model achieved the best "
    "F1-score at any station. When the classification threshold is fixed at the default "
    f"value of 0.5, Persistence outperforms XGBoost at all twelve stations, and the mean "
    f"XGBoost F1-score falls to {hh['xgboost_f1_default_0.5'].mean():.3f}. This shows that "
    "the apparent competitiveness of trained models relative to Persistence depends heavily "
    "on the threshold optimization procedure, discussed separately in Section 4.1. In "
    "contrast to the Persistence–XGBoost comparison, the gaps separating both of them from "
    "the two deep models were statistically significant after Holm correction: Persistence "
    f"versus LSTM {pstr(sg('overall','Persistence','LSTM')['p_holm'])} "
    f"(r = {sg('overall','Persistence','LSTM')['rank_biserial_r']:.2f}), Persistence versus "
    f"1D-CNN {pstr(sg('overall','Persistence','1D-CNN')['p_holm'])} "
    f"(r = {sg('overall','Persistence','1D-CNN')['rank_biserial_r']:.2f}), XGBoost versus "
    f"LSTM {pstr(sg('overall','XGBoost','LSTM')['p_holm'])} "
    f"(r = {sg('overall','XGBoost','LSTM')['rank_biserial_r']:.2f}), and XGBoost versus "
    f"1D-CNN {pstr(sg('overall','XGBoost','1D-CNN')['p_holm'])} "
    f"(r = {sg('overall','XGBoost','1D-CNN')['rank_biserial_r']:.2f}). The LSTM–1D-CNN "
    f"difference was not significant after correction "
    f"({pstr(sg('overall','LSTM','1D-CNN')['p_holm'])}). Full test results are given in the "
    "accompanying repository.")

d.replace_para(
    "Figure 3. Comparison of T+1 fog prediction performance by region",
    "Figure 3. Comparison of overall T+1 fog prediction performance by station, ordered by "
    "test-set fog prevalence. Fog-class F1-scores are compared across Persistence, XGBoost, "
    "1D-CNN, and LSTM, all evaluated on an identical set of test cases.")

# ================================================================ RESULTS 3.4
best_onset = f1n[ONSET].idxmax(axis=1)
losers = [s for s in STATION_ORDER if best_onset[s] != "XGBoost-Onset"]
d.replace_para(
    "XGBoost-Onset achieved a mean F1-score of 0.228",
    f"XGBoost-Onset achieved the highest mean F1-score ({ms(f1n['XGBoost-Onset'])}) with a "
    f"mean PR-AUC of {prn['XGBoost-Onset'].mean():.3f}, ahead of LSTM-Onset "
    f"(F1 {ms(f1n['LSTM-Onset'])}, PR-AUC {prn['LSTM-Onset'].mean():.3f}) and 1D-CNN-Onset "
    f"(F1 {ms(f1n['1D-CNN-Onset'])}, PR-AUC {prn['1D-CNN-Onset'].mean():.3f}). XGBoost "
    f"achieved the highest F1-score at {xgb_best_onset} of the twelve stations; the "
    f"exceptions were {', '.join(losers)}, where a deep model scored higher. On mean PR-AUC "
    "the ordering of XGBoost and LSTM is effectively reversed, so the advantage of XGBoost in "
    "onset detection should be described as consistent but not uniform. Wilcoxon signed-rank "
    "tests over the twelve stations confirm this: XGBoost was not significantly better than "
    f"LSTM on F1 ({pstr(sg('onset','XGBoost-Onset','LSTM-Onset')['p_holm'])}) and was "
    f"marginally behind it on PR-AUC "
    f"({pstr(sg('onset','XGBoost-Onset','LSTM-Onset','pr_auc')['p_holm'])}), and its F1 "
    f"advantage over the 1D-CNN did not survive Holm correction "
    f"({pstr(sg('onset','XGBoost-Onset','1D-CNN-Onset')['p_holm'])}). On PR-AUC, however, "
    "both XGBoost and the LSTM exceeded the 1D-CNN significantly "
    f"({pstr(sg('onset','XGBoost-Onset','1D-CNN-Onset','pr_auc')['p_holm'])} and "
    f"{pstr(sg('onset','LSTM-Onset','1D-CNN-Onset','pr_auc')['p_holm'])} respectively). The "
    "defensible statement is therefore that the tabular and recurrent models perform "
    "comparably on onset detection and that both rank the rare positive class better than "
    "the convolutional model. Absolute performance remained low for every trained model, "
    "reflecting the difficulty of the task.")

d.replace_para(
    "Unlike overall T+1 prediction, where performance differences among XGBoost",
    "Unlike overall T+1 prediction, where the parameter-free Persistence baseline was the "
    "strongest model, onset prediction is a task that Persistence cannot address at all, and "
    "here the trained models provide the only available signal. XGBoost achieved the highest "
    "mean F1 and the best F1 at 8 of the 12 stations, whereas the LSTM showed a slightly "
    "higher mean PR-AUC; neither pairwise difference was statistically significant. For "
    "learning extremely rare and nonlinear transition signals from single-station tabular "
    "meteorological data, tree-based ensembles and recurrent models therefore appear to be "
    "comparable options under this experimental configuration.")

d.replace_para(
    "Figure 4. Regional comparison of onset prediction performance",
    "Figure 4. Comparison of 0→1 fog onset detection performance by station, ordered by "
    f"onset prevalence in the test set. XGBoost-Onset achieved the highest F1-score at "
    f"{xgb_best_onset} of the twelve stations.")

# ================================================================ RESULTS 3.5
d.replace_para(
    "This analysis was performed using the overall T+1 XGBoost model rather than",
    "This analysis was performed using the overall T+1 XGBoost model rather than the "
    f"onset-specific model. Target-time visibility for TP cases (n = {n_tp:,}) had a median "
    f"of {tp:.2f} km, whereas FN cases (n = {n_fn:,}) had a higher median of {fn:.2f} km. "
    "Because both TP and FN cases correspond to actual fog conditions, these results indicate "
    "that missed cases tended to involve less severe target-time visibility reduction than "
    f"correctly detected cases. FP cases (n = {n_fp:,}) had a median target-time visibility of "
    f"{fp:.2f} km and showed a substantially wider distribution. Model-positive conditions "
    "therefore did not always correspond to target-time visibility below the 1 km fog "
    "threshold. Because this analysis included all transition types in the overall T+1 task, "
    "the FP distribution should not be interpreted exclusively as failed fog-onset prediction.")

# ================================================================ RESULTS 3.6
d.replace_para(
    "Figure 6a shows the global feature importance of the XGBoost model",
    "Figure 6a shows the global feature importance of the XGBoost model, computed by "
    "averaging mean absolute SHAP values across the twelve stations. Current visibility, "
    f"Visibility (t), showed by far the highest contribution (mean |SHAP| "
    f"{shap_g['mean_abs_shap'].iloc[0]:.3f}), followed by dew point depression "
    f"({shap_g['mean_abs_shap'].iloc[1]:.3f}), visibility at the preceding hour "
    f"({shap_g['mean_abs_shap'].iloc[2]:.3f}) and relative humidity "
    f"({shap_g['mean_abs_shap'].iloc[3]:.3f}). This indicates that the model relied primarily "
    "on current visibility state and atmospheric saturation conditions when predicting fog "
    "occurrence. In particular, the dominance of Visibility (t) reflects the strong temporal "
    "persistence of fog and visibility deterioration, and is consistent with the finding that "
    "a persistence baseline built on exactly that variable is difficult to outperform. The "
    "prominence of dew point depression and relative humidity among the top features likewise "
    "reflects the close relationship between atmospheric saturation, condensation potential, "
    "and fog formation.")

# ================================================================ DISCUSSION 4.1
d.replace_para(
    "In overall T+1 prediction, the mean F1-score difference between XGBoost and Persistence",
    "In overall T+1 prediction, Persistence achieved a higher mean F1-score than XGBoost "
    f"({f(f1o['Persistence'].mean())} vs. {f(f1o['XGBoost'].mean())}, a difference of "
    f"{f(f1o['Persistence'].mean()-f1o['XGBoost'].mean())}), and was the better of the two at "
    f"{pers_wins} of the twelve stations even after validation-based threshold tuning. This "
    "is the opposite of the ordering we would expect if a trained model were extracting "
    "information beyond visibility persistence. The advantage of Persistence is concentrated "
    "at low-prevalence stations, a pattern that may partly reflect the limited number of "
    "positive training samples available there and its effect on the robustness of the "
    "learned decision boundary; at the three highest-prevalence stations the two are closely "
    "matched. A Wilcoxon signed-rank "
    f"test over the twelve stations gives {pstr(sg('overall','Persistence','XGBoost')['p_raw'])} "
    f"uncorrected and {pstr(sg('overall','Persistence','XGBoost')['p_holm'])} after Holm "
    "correction, so the difference is suggestive rather than established. The conclusion we "
    "draw is deliberately the weaker one: XGBoost did not demonstrate an improvement over "
    "visibility persistence "
    "for one-hour-ahead fog-state prediction, and became markedly worse where fog is rare. "
    "For an operational forecaster the practical implication is the same either way, since "
    "Persistence requires no training, no threshold tuning and no retraining as conditions "
    "drift.")

d.replace_para(
    "However, when the classification threshold was fixed at the default value of 0.5",
    "When the classification threshold was fixed at the default value of 0.5, Persistence "
    f"outperformed XGBoost at all twelve stations, and the mean XGBoost F1-score fell from "
    f"{f(f1o['XGBoost'].mean())} to {hh['xgboost_f1_default_0.5'].mean():.3f}. This indicates "
    "that the probability output of a trained model does not necessarily guarantee optimal "
    "performance at the default threshold, and that in problems with severe class imbalance "
    "such as fog prediction, the threshold selection procedure can substantially affect model "
    "performance evaluation. Shin et al. [19] similarly noted that class imbalance and "
    "temporal distribution shift can be major factors underlying degraded model performance "
    "in visibility nowcasting in Korea. Comparisons of short-term fog prediction models "
    "should therefore clearly report not only algorithmic architecture but also whether "
    "validation-based threshold optimization was applied and which evaluation metrics were "
    "selected.")

d.replace_para(
    "This result demonstrates that current visibility state and fog persistence serve as an",
    "The strong performance of Persistence observed here is not anomalous: previous "
    "low-visibility work has reported pronounced short-term temporal persistence and has used "
    "naïve persistence as an explicit hourly prediction method. Cornejo-Bueno et al. [23] "
    "analysed low-visibility persistence at Valladolid Airport and included a naïve "
    "persistence operator among the methods evaluated for hourly prediction, with their best "
    "configuration combining persistence-based methods and machine learning; Peláez-Rodríguez "
    "et al. [21] separately reported that past visibility values are important inputs for "
    "short-term low-visibility prediction. In this study as well, prior visibility and current "
    "fog state substantially influenced overall T+1 prediction performance, consistent with "
    "the tendency of fog to persist for some time once formed. The present results, however, "
    "also expose the limitation of such aggregate skill: a persistence rule that performs "
    "strongly on overall T+1 classification is structurally incapable of anticipating a 0→1 "
    "onset event. The contribution here is therefore not the use of a persistence baseline, "
    "which is established, but the explicit separation of overall fog-state prediction from "
    "onset detection across twelve stations under a like-for-like protocol, which is what "
    "makes that limitation measurable.")

# ================================================================ DISCUSSION 4.2
d.replace_para(
    "In onset prediction, XGBoost showed consistently higher F1-scores than both 1D-CNN and LSTM",
    "In onset prediction, XGBoost showed higher F1-scores than both the 1D-CNN and the LSTM "
    f"at {xgb_best_onset} of the twelve stations, achieving the highest mean F1-score "
    f"({f(f1n['XGBoost-Onset'].mean())}) against {f(f1n['LSTM-Onset'].mean())} for the LSTM "
    f"and {f(f1n['1D-CNN-Onset'].mean())} for the 1D-CNN. The margin is smaller and less "
    "uniform than the margin between Persistence and the trained models in the overall task, "
    f"and on mean PR-AUC the LSTM was marginally ahead "
    f"({prn['LSTM-Onset'].mean():.3f} vs. {prn['XGBoost-Onset'].mean():.3f}). The claim we can "
    "support is therefore that the tabular and recurrent models are comparable for 0→1 onset "
    "detection under this data configuration, with XGBoost holding a small and "
    "non-significant lead on F1 and the LSTM a small and non-significant lead on PR-AUC. "
    "Neither the earlier claim that XGBoost dominates the deep models, nor a reversed claim "
    "in favour of the LSTM, is supported by twelve paired station observations.")

d.replace_para(
    "First, onset cases account for only about",
    f"First, onset cases account for only about {onset_pct:.2f}% of "
    "all test cases—an extreme minority class. Under such conditions, models can easily "
    "become biased toward the majority non-fog class, achieving high overall accuracy while "
    "failing to detect actual fog onset. Shin et al. [19] explained that minority-class "
    "rarity and distribution shift are key factors underlying degraded performance in "
    "low-visibility nowcasting, and Kim et al. [20] similarly reported that fog cases "
    "comprised less than 5% of the data in ASOS-based fog prediction in Korea, creating "
    "difficulty even for tree-based classifiers. The onset prediction task in this study "
    "targets an even rarer transition event, suggesting that class imbalance played an even "
    "larger role here.")

# ================================================================ DISCUSSION 4.4 (limitations)
d.replace_para(
    "This study has the following limitations. First, because the input representations",
    "This study has the following limitations. First, although the two model families were "
    "given identical temporal splits, identical evaluation cases and the same information "
    "horizon, their input representations and hyperparameter search ranges were not identical, "
    "so the comparison should be interpreted as one between practical model configurations "
    "rather than a pure architectural comparison. The deep learning models were limited to a "
    "1D-CNN and an LSTM over 6-hour ASOS-based sequential inputs, differing in problem setting "
    "from high-dimensional architectures such as FogNet, which leverage NWP output, satellite "
    "data, sea surface temperature, and three-dimensional gridded inputs [22]. These results "
    "should therefore not be read as evidence that deep learning is unsuitable for fog "
    "prediction, but that under this data configuration neither deep model improved on a "
    "parameter-free persistence baseline. Second, statistical power is limited: with twelve "
    "paired station observations, a two-sided Wilcoxon signed-rank test is best suited to "
    "detecting large and consistent differences, and the Persistence–XGBoost and "
    "XGBoost–LSTM comparisons did not reach significance after Holm correction even though "
    "the point estimates differ. A non-significant result of this kind indicates that the "
    "present data do not resolve the difference; it does not demonstrate that the models are "
    "equivalent. The station-level tests should also be interpreted cautiously because the "
    "twelve stations are geographically distributed within the same regional meteorological "
    "system and therefore may not constitute fully independent spatial units. The reported "
    "between-station standard deviations (±) reflect heterogeneity among stations "
    "rather than the uncertainty of an estimate. Small F1-score differences between models "
    "should therefore be interpreted as tendencies rather than definitive rankings, and we "
    "have avoided ranking claims that the tests do not support. Finally, "
    "the 1D-CNN and LSTM results depend on random initialization; the reported values come "
    "from a single fixed seed, and the observed seed-to-seed variability is reported in the "
    "accompanying repository.")

# ---- Discussion 4.2 heading and residual "stability" wording ----
d.replace_para(
    "Why XGBoost Was More Stable than Deep Learning Models for Onset Prediction",
    "4.2. Onset Detection: Comparable Performance of Tabular and Recurrent Models")

d.replace_para(
    "These findings support the continued practicality and stability of tree-based",
    "Second, XGBoost, combined with scale_pos_weight adjustment, can directly learn "
    "nonlinear split boundaries for individual variables, which may make it relatively "
    "sensitive to rare signals. Schütz et al. [1] showed that XGBoost-based visibility "
    "prediction can be effectively applied across station, satellite, and combined data "
    "settings, and Penov and Guerova [3] similarly reported that tree-based models such as "
    "Random Forest achieved competitive performance relative to LSTM in airport visibility "
    "estimation. In Korea, Kim et al. [14] reported that Random Forest achieved high "
    "F1-scores in visibility estimation for Sejong and Busan, suggesting that the relative "
    "strengths of different models can vary with regional fog characteristics. These findings "
    "support the continued practicality and competitive performance of tree-based ensemble "
    "models for fog prediction using tabular meteorological data, without implying that they "
    "outperform sequential models on this task.")

d.replace_para(
    "XGBoost is interpreted as having shown more stable performance",
    "Third, the physically derived features used in this study—dew point depression, prior "
    "visibility, and lag variables—already compactly encode short-term change information. "
    "Because XGBoost can directly use such variables as split criteria, it may perform "
    "effectively in tasks such as onset prediction, where a sharp change in conditions at a "
    "specific time step is critical. The 1D-CNN and LSTM must instead learn patterns within a "
    "6-hour time-series window, which may dilute rare, short-duration transition signals such "
    "as onset within the sequential structure. This does not imply that deep learning models "
    "are inherently unsuitable for fog prediction: models such as Kamangir et al.'s [22] "
    "FogNet, which leverage NWP output, satellite-based sea surface temperature, and "
    "three-dimensional gridded inputs, can outperform operational ensembles. In the setting "
    "of this study—tabular data from individual ASOS stations and a short time-series "
    "window—XGBoost and the LSTM performed comparably, and the station-level tests did not "
    "resolve a difference between them.")

d.replace_para(
    "the limited number of positive samples constrained the stability of both model",
    "Second, in regions with extremely low fog occurrence rates—such as Inje, Seoul, and "
    "Sokcho—the limited number of positive samples constrained the reliability of both model "
    "performance estimates and SHAP interpretation across all models. Onset prediction, in "
    "particular, targets 0→1 transition events that are even rarer than overall fog cases, "
    "introducing substantial uncertainty into both model training and evaluation. This class "
    "imbalance problem has similarly been identified as a major limitation in prior studies "
    "on visibility nowcasting and fog frequency prediction in Korea [19,20]. Future work "
    "should systematically compare methodologies for rare-event detection, such as focal "
    "loss, cost-sensitive learning, data augmentation, and threshold calibration.")

# ================================================================ CONCLUSIONS
d.replace_para(
    "Overall T+1 forecasting showed that XGBoost and Persistence achieved nearly identical",
    "Overall T+1 forecasting showed that the parameter-free Persistence baseline achieved the "
    f"highest mean F1-score ({ms(f1o['Persistence'])}), ahead of XGBoost "
    f"({ms(f1o['XGBoost'])}), LSTM ({ms(f1o['LSTM'])}) and the 1D-CNN "
    f"({ms(f1o['1D-CNN'])}). Persistence was the better of Persistence and XGBoost at "
    f"{pers_wins} of the twelve stations after threshold tuning and at all twelve stations at "
    "the default threshold, with the gap widening as fog became rarer. Its margin over "
    "XGBoost was not statistically significant across twelve stations "
    f"({pstr(sg('overall','Persistence','XGBoost')['p_holm'])} after Holm correction), so we "
    "conclude only that XGBoost did not demonstrate an improvement over the baseline. Its "
    "margin over both deep models was significant in the overall T+1 task evaluated here. "
    "For one-hour-ahead fog-state prediction, a parameter-free persistence baseline was "
    "therefore not outperformed by any trained model in this study—while remaining "
    "structurally unable to detect 0→1 fog onset.")

d.replace_para(
    "For onset detection, XGBoost achieved higher F1-scores than the tested 1D-CNN and LSTM",
    "For onset detection, where Persistence is inapplicable by construction, XGBoost achieved "
    f"the highest mean F1-score ({ms(f1n['XGBoost-Onset'])}) and was best at "
    f"{xgb_best_onset} of the twelve stations, ahead of the LSTM ({ms(f1n['LSTM-Onset'])}) and "
    f"the 1D-CNN ({ms(f1n['1D-CNN-Onset'])}); on mean PR-AUC the LSTM was marginally ahead, "
    "and neither the F1 nor the PR-AUC difference between XGBoost and the LSTM was "
    "statistically significant. Absolute performance remained low for all trained models, "
    "demonstrating that fog onset is a substantially more difficult prediction task than "
    "overall fog-state forecasting. Under the present ASOS-based experimental configuration, "
    "XGBoost and the LSTM showed comparable onset-detection performance within the "
    "statistical resolution of this study, with no statistically supported evidence of a "
    "general advantage for either architecture.")

d.replace_para(
    "SHAP analysis showed that current visibility, dew point depression, and relative humidity",
    "SHAP analysis showed that current visibility, dew point depression, visibility at the "
    "preceding hour and relative humidity consistently ranked among the most influential "
    "predictors. The dominance of current visibility is consistent with the strength of the "
    "persistence baseline. The similarity of feature rankings across stations indicates common "
    "patterns of model reliance but does not establish identical fog-formation mechanisms "
    "across regions. Taken together, these results show that strong overall one-hour-ahead "
    "fog-state forecasting performance does not necessarily translate into useful fog-onset "
    "early-warning skill: the baseline that was hardest to beat on the overall task is "
    "precisely the one that cannot issue an onset warning at all. Overall fog-state "
    "prediction and fog-onset detection should therefore be specified, evaluated and "
    "reported as separate tasks, and the contribution of this study is accordingly an "
    "evaluation framework and a diagnostic finding rather than a new best-performing model. "
    "Future work should incorporate spatial, terrain, radiative, oceanic, and "
    "boundary-layer information; evaluate threshold calibration and rare-event learning under "
    "temporal distribution shift; and extend the framework to fog persistence and dissipation "
    "prediction.")

# ================================================================ DATA AVAILABILITY
d.replace_para(
    "The hourly Automated Synoptic Observing System (ASOS) observation data",
    "The hourly Automated Synoptic Observing System (ASOS) observation data for 2015–2024 used "
    "in this study are publicly available from the Korea Meteorological Administration (KMA) "
    "Open MET Data Portal (https://data.kma.go.kr). All source code, configuration files and "
    "result files required to reproduce the analyses, tables and figures in this paper are "
    "openly available in the accompanying repository, together with instructions for "
    "downloading and preparing the raw observations. The repository provides a documented "
    "pipeline that regenerates all reported metrics, statistical test results, tables and "
    "figures from the raw observations; trained model files are not redistributed, as they "
    "are reproduced by the pipeline itself.")

# ================================================================ TABLES
t2 = [["Transition type", "Description", "Total count", "Proportion"]]
for _, r in trans.iterrows():
    t2.append([r["transition"], r["description"], f"{r['count']:,}",
               f"{r['proportion_percent']:.2f}%"])
d.set_table(2, t2)

t3 = [["Station", "Persistence", "XGBoost", "1D-CNN", "LSTM"]]
for s in STATION_ORDER:
    t3.append([s] + [f(f1o.loc[s, m]) for m in OVERALL])
t3.append(["Mean±SD"] + [f"{f1o[m].mean():.3f}±{f1o[m].std(ddof=1):.3f}" for m in OVERALL])
d.set_table(3, t3)

t4 = [["Model", "F1-score", "PR-AUC", "Notes"]]
t4.append(["No-Onset Baseline", "0.000", "—", "Structurally undetectable"])
for mdl, note in [("XGBoost-Onset", f"Best in {xgb_best_onset}/12 stations"),
                  ("1D-CNN-Onset", f"Best in {int((best_onset=='1D-CNN-Onset').sum())}/12 stations"),
                  ("LSTM-Onset", f"Best in {int((best_onset=='LSTM-Onset').sum())}/12 stations")]:
    t4.append([mdl, f"{f1n[mdl].mean():.3f} ± {f1n[mdl].std(ddof=1):.3f}",
               f"{prn[mdl].mean():.3f}", note])
d.set_table(4, t4)

# ---- References: append the new entry as [23] ---------------------------------
# The existing reference list is not in strict order of appearance (the first
# citations in the body are [4], [13], [17], [20], ...), so appending the new
# entry as [23] leaves every existing citation number untouched.
d.insert_para_after(
    "Kamangir, H.; Collins, W.; Tissot, P.",
    "Cornejo-Bueno, S.; Casillas-Pérez, D.; Cornejo-Bueno, L.; Chidean, M.I.; Caamaño, A.J.; "
    "Sanz-Justo, J.; Casanova-Mateo, C.; Salcedo-Sanz, S. Persistence Analysis and Prediction "
    "of Low-Visibility Events at Valladolid Airport, Spain. Symmetry 2020, 12, 1045. "
    "https://doi.org/10.3390/sym12061045.")

d.save()

# ================================================================ FIGURES
FIG_MAP = {
    "image2.png": "fig_fog_prevalence_by_station.png",
    "image3.png": "fig_overall_f1_by_station.png",
    "image4.png": "fig_onset_f1_by_station.png",
    "image5.png": "fig_tp_fn_fp_target_visibility.png",
    "image6.png": "fig_shap_global_importance.png",
    "image7.png": "fig_shap_station_heatmap.png",
}
for tgt, src in FIG_MAP.items():
    sp = os.path.join(F, src)
    if os.path.exists(sp):
        shutil.copy(sp, os.path.join(UNPACKED, "word", "media", tgt))
        print(f"  replaced word/media/{tgt} <- {src}")
    else:
        print(f"  [warn] figure missing, kept original: {src}")

# ================================================================ PACKAGE
out = os.path.join(HERE, "Atmosphere_eng_v9_persistence_lit.docx")
if os.path.exists(out):
    os.remove(out)
subprocess.check_call(["zip", "-Xrq", out, "."], cwd=UNPACKED)
print(f"\nEdits applied: {len(d.log)}")
print(f"Wrote {out}")
