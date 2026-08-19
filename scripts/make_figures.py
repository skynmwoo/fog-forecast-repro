# -*- coding: utf-8 -*-
"""PHASE 12: regenerate every manuscript figure from results/metrics/*.csv.

    python scripts/make_figures.py

All figures are English-labelled and are pure functions of the result CSVs.
No figure reuses a pre-existing image file.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# Colour-blind-safe qualitative palette (Okabe-Ito)
COLORS = {
    "Persistence": "#000000", "XGBoost": "#0072B2",
    "1D-CNN": "#E69F00", "LSTM": "#009E73",
    "No-Onset Baseline": "#000000", "XGBoost-Onset": "#0072B2",
    "1D-CNN-Onset": "#E69F00", "LSTM-Onset": "#009E73",
}
OVERALL = ["Persistence", "XGBoost", "1D-CNN", "LSTM"]
ONSET = ["No-Onset Baseline", "XGBoost-Onset", "1D-CNN-Onset", "LSTM-Onset"]

# Korean ASOS column names -> English labels used in every figure.
FEATURE_EN = {
    "시정(10m)": "Visibility (t)",
    "기온(°C)": "Air temperature",
    "이슬점온도(°C)": "Dew point temperature",
    "습도(%)": "Relative humidity",
    "풍속(m/s)": "Wind speed",
    "현지기압(hPa)": "Station pressure",
    "지면온도(°C)": "Ground temperature",
    "dew_point_depression": "Dew point depression",
    "air_ground_diff": "Air − ground temp. diff.",
    "ground_air_diff": "Ground − air temp. diff.",
    "ground_air_diff_1h": "1-h change of ground−air diff.",
    "rel_humid": "Relative humidity (fraction)",
    "dpd_ratio": "Dew point depression ratio",
    "hour": "Hour of day",
    "weekday": "Day of week",
    "is_weekend": "Weekend indicator",
    "month_sin": "Month (sine)",
    "month_cos": "Month (cosine)",
    "temp_lag_1": "Air temperature (t−1)",
    "temp_lag_2": "Air temperature (t−2)",
    "temp_lag_3": "Air temperature (t−3)",
    "humidity_lag_1": "Relative humidity (t−1)",
    "humidity_lag_2": "Relative humidity (t−2)",
    "humidity_lag_3": "Relative humidity (t−3)",
    "vis_lag_1": "Visibility (t−1)",
    "vis_lag_2": "Visibility (t−2)",
    "vis_lag_3": "Visibility (t−3)",
    "temp_roll_mean_2h": "2-h rolling mean temperature",
    "humid_roll_std_2h": "2-h rolling SD humidity",
}


def en(name):
    return FEATURE_EN.get(name, name)


def _save(fig, outdir, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(outdir, f"{name}.{ext}"))
    plt.close(fig)
    print(f"  wrote {name}.png / .pdf")


def fig_prevalence(m, out):
    d = pd.read_csv(os.path.join(m, "fog_prevalence.csv"))
    d["test_percent"] = d["fog_prevalence_test"] * 100
    d = d.sort_values("test_percent", ascending=False)
    mean_pct = d["test_percent"].mean()
    fig, ax = plt.subplots(figsize=(8, 4.4))
    bars = ax.bar(d["station"], d["test_percent"], color="#4C72B0", edgecolor="black", lw=0.5)
    for b, v in zip(bars, d["test_percent"]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.06, f"{v:.2f}", ha="center", fontsize=8)
    ax.axhline(mean_pct, color="#D55E00", ls="--", lw=1.3,
               label=f"12-station mean: {mean_pct:.2f}%")
    ax.set_ylabel("Fog occurrence rate at t+1 (%)")
    ax.set_xlabel("ASOS station")
    ax.set_title("Test-set fog occurrence rate by station (2023–2024, visibility ≤ 1 km)")
    ax.legend(frameon=False, fontsize=9)
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
    _save(fig, out, "fig_fog_prevalence_by_station")


def fig_model_comparison(df, out, task, models, title, name):
    p = df[(df["task"] == task) & (df["model"].isin(models))].pivot_table(
        index="station", columns="model", values="f1")
    prev = df[(df["task"] == task)].groupby("station")["positive_prevalence_test"].first()
    order = prev.sort_values(ascending=False).index
    p = p.reindex(index=order, columns=models)

    fig, ax = plt.subplots(figsize=(11, 5.0))
    x = np.arange(len(p))
    w = 0.2
    for i, mdl in enumerate(models):
        ax.bar(x + (i - 1.5) * w, p[mdl], w, label=mdl, color=COLORS[mdl],
               edgecolor="black", lw=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}\n({prev[s]*100:.2f}%)" for s in p.index],
                       fontsize=8, rotation=30, ha="right")
    ax.set_ylabel("F1-score (test, 2023–2024)")
    ax.set_xlabel("ASOS station (positive-class prevalence in test set)")
    ax.set_title(title)
    ax.legend(ncol=4, fontsize=9, frameon=False)
    _save(fig, out, name)


def fig_scatter_prevalence(df, out):
    d = df[(df["task"] == "overall") & (df["model"].isin(OVERALL))]
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for mdl in OVERALL:
        s = d[d["model"] == mdl]
        ax.scatter(s["positive_prevalence_test"] * 100, s["f1"], label=mdl,
                   color=COLORS[mdl], s=42, edgecolor="black", lw=0.4)
    ax.set_xscale("log")
    ax.set_xlabel("Fog prevalence in test set (%, log scale)")
    ax.set_ylabel("F1-score")
    ax.set_title("Overall T+1 performance vs. station fog prevalence")
    ax.legend(frameon=False, fontsize=9)
    _save(fig, out, "fig_overall_f1_vs_prevalence")


def fig_prevalence_gap(m, out):
    """Trained-model advantage over Persistence as a function of fog prevalence."""
    p = os.path.join(m, "prevalence_gap.csv")
    if not os.path.exists(p):
        return
    d = pd.read_csv(p)
    t = os.path.join(ROOT, "results", "tables", "table_prevalence_gap_correlation.csv")
    corr = pd.read_csv(t).set_index("model") if os.path.exists(t) else None

    fig, ax = plt.subplots(figsize=(7, 4.8))
    x = d["common_eval_fog_prevalence_percent"].to_numpy()
    for mdl in ("XGBoost", "LSTM", "1D-CNN"):
        col = f"{mdl}_minus_Persistence_f1"
        if col not in d:
            continue
        y = d[col].to_numpy()
        lab = mdl
        if corr is not None and mdl in corr.index:
            lab = (rf"{mdl}  (Spearman $\rho$ = {corr.loc[mdl, 'spearman_rho']:.2f}, "
                   rf"Holm $p$ = {corr.loc[mdl, 'spearman_p_holm']:.3f})")
        ax.scatter(x, y, s=46, color=COLORS[mdl], edgecolor="black", lw=0.4, label=lab, zorder=3)
        b = np.polyfit(np.log(x), y, 1)
        xs = np.linspace(x.min(), x.max(), 100)
        ax.plot(xs, np.polyval(b, np.log(xs)), color=COLORS[mdl], lw=1.2, alpha=0.55, zorder=2)

    ax.axhline(0, color="black", lw=1.2, ls="--", zorder=1)
    ax.set_xscale("log")
    ax.set_xlabel("fog(t+1) prevalence in the common evaluation subset (%, log scale)")
    ax.set_ylabel("F1 advantage over Persistence")
    ax.set_title("Trained-model advantage over Persistence vs. station fog prevalence",
                 fontsize=11)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    ax.text(0.02, 0.96, "above 0: trained model better\nbelow 0: Persistence better",
            transform=ax.transAxes, va="top", fontsize=8, color="#555555")
    _save(fig, out, "fig_prevalence_gap")


def fig_transition_recall(m, out):
    """Recall on 1->1 (fog persisting) vs 0->1 (fog forming), pooled over stations."""
    p = os.path.join(ROOT, "results", "tables", "table_transition_recall.csv")
    if not os.path.exists(p):
        return
    d = pd.read_csv(p)
    order = [x for x in OVERALL if x in set(d["model"])]
    d = d.set_index("model").reindex(order).reset_index()

    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    xi = np.arange(len(d))
    w = 0.36
    n11 = int(d["n_positive_1to1"].iloc[0])
    n01 = int(d["n_positive_0to1"].iloc[0])
    ax.bar(xi - w / 2, d["recall_1to1"], w, color="#4C72B0", edgecolor="black", lw=0.4,
           label=rf"1$\rightarrow$1  fog persisting  (n = {n11:,})")
    ax.bar(xi + w / 2, d["recall_0to1"], w, color="#D55E00", edgecolor="black", lw=0.4,
           label=rf"0$\rightarrow$1  fog forming  (n = {n01:,})")
    for i, (a, b) in enumerate(zip(d["recall_1to1"], d["recall_0to1"])):
        ax.text(i - w / 2, a + 0.015, f"{a:.2f}", ha="center", fontsize=8.5)
        ax.text(i + w / 2, b + 0.015, f"{b:.2f}", ha="center", fontsize=8.5)
    ax.set_xticks(xi)
    ax.set_xticklabels([f"{m}\nprecision {p:.2f}" for m, p in zip(d["model"], d["precision"])],
                       fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Recall on the overall T+1 task")
    ax.set_title("Where overall T+1 skill comes from: fog persistence vs. fog formation")
    ax.legend(frameon=False, fontsize=9)
    _save(fig, out, "fig_transition_recall")


def fig_shap_global(m, out):
    p = os.path.join(m, "shap_global.csv")
    if not os.path.exists(p):
        return
    d = pd.read_csv(p).head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh([en(f) for f in d["feature"]], d["mean_abs_shap"], color="#0072B2", edgecolor="black", lw=0.4)
    ax.set_xlabel("Mean |SHAP| value (pooled across 12 stations)")
    ax.set_title("Global feature importance — overall T+1 XGBoost")
    _save(fig, out, "fig_shap_global_importance")


def fig_shap_heatmap(m, out):
    p = os.path.join(m, "shap_station_normalized.csv")
    if not os.path.exists(p):
        return
    d = pd.read_csv(p)
    top = (d.groupby("feature")["normalized"].mean()
             .sort_values(ascending=False).head(12).index.tolist())
    piv = d[d["feature"].isin(top)].pivot(index="feature", columns="station",
                                          values="normalized").reindex(top)
    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(piv.values, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(piv.shape[1]))
    ax.set_xticklabels(piv.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(piv.shape[0]))
    ax.set_yticklabels([en(f) for f in piv.index], fontsize=8)
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="Station-normalized mean |SHAP| (column sums to 1)")
    ax.set_title("Station-normalized SHAP importance")
    _save(fig, out, "fig_shap_station_heatmap")


def fig_error_visibility_box(m, out):
    """Observed target-time visibility for TP / FN / FP, pooled over stations."""
    p = os.path.join(m, "error_visibility_rows.csv")
    if not os.path.exists(p):
        return
    d = pd.read_csv(p)
    d["vis_km"] = d["vis_tplus1_10m"] / 100.0
    order = ["TP", "FN", "FP"]
    labels = {"TP": "TP\nCorrect detection", "FN": "FN\nMiss", "FP": "FP\nFalse alarm"}
    colors = {"TP": "#2E6DA4", "FN": "#7FB3D5", "FP": "#C6DBEF"}

    fig, ax = plt.subplots(figsize=(7, 5))
    data = [d[d["error_type"] == c]["vis_km"].to_numpy() for c in order]
    bp = ax.boxplot(data, patch_artist=True, showfliers=False, widths=0.5,
                    medianprops=dict(color="black", lw=2))
    for patch, c in zip(bp["boxes"], order):
        patch.set_facecolor(colors[c])
        patch.set_edgecolor("black")
    ax.axhline(1.0, color="#D55E00", ls="--", lw=1.5, label="Fog threshold: 1 km")
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels([f"{labels[c]}\n(n={len(d[d['error_type']==c]):,})" for c in order])
    ax.set_ylabel("Observed visibility at target time t+1 (km)")
    ax.set_ylim(0, 7)
    ax.set_title("Target-time visibility by prediction outcome — overall T+1 XGBoost")
    ax.legend(frameon=False, loc="upper left")
    _save(fig, out, "fig_tp_fn_fp_target_visibility")


def fig_error_visibility(m, out):
    p = os.path.join(m, "error_visibility_stats.csv")
    if not os.path.exists(p):
        return
    d = pd.read_csv(p)
    d = d[d["error_type"].isin(["TP", "FP", "FN"])]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    cats = ["TP", "FP", "FN"]
    x = np.arange(len(d["station"].unique()))
    stations = sorted(d["station"].unique())
    w = 0.26
    cmap = {"TP": "#009E73", "FP": "#E69F00", "FN": "#D55E00"}
    for i, c in enumerate(cats):
        s = d[d["error_type"] == c].set_index("station").reindex(stations)
        ax.bar(x + (i - 1) * w, s["vis_now_median"] / 100.0, w, label=c,
               color=cmap[c], edgecolor="black", lw=0.4)
    ax.axhline(1.0, color="black", ls="--", lw=1, label="fog threshold (1 km)")
    ax.set_xticks(x)
    ax.set_xticklabels(stations, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Median current visibility at time t (km)")
    ax.set_title("Current visibility by prediction outcome — overall T+1 XGBoost")
    ax.legend(frameon=False, fontsize=9, ncol=4)
    _save(fig, out, "fig_tp_fp_fn_visibility")


def main() -> int:
    cfg = yaml.safe_load(open(os.path.join(ROOT, "configs", "default.yaml"), encoding="utf-8"))
    m = os.path.join(ROOT, cfg["paths"]["results_dir"], "metrics")
    out = os.path.join(ROOT, cfg["paths"]["results_dir"], "figures")
    os.makedirs(out, exist_ok=True)
    df = pd.read_csv(os.path.join(m, "station_metrics.csv"))

    fig_prevalence(m, out)
    fig_model_comparison(df, out, "overall", OVERALL,
                         "Overall T+1 fog prediction — F1 by station",
                         "fig_overall_f1_by_station")
    fig_model_comparison(df, out, "onset", ONSET,
                         "0→1 fog onset detection — F1 by station",
                         "fig_onset_f1_by_station")
    fig_scatter_prevalence(df, out)
    fig_prevalence_gap(m, out)
    fig_transition_recall(m, out)
    fig_shap_global(m, out)
    fig_shap_heatmap(m, out)
    fig_error_visibility(m, out)
    fig_error_visibility_box(m, out)
    print(f"\nFigures written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
