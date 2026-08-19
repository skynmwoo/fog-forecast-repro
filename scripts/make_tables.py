# -*- coding: utf-8 -*-
"""PHASE 12: build every manuscript table directly from results/metrics/*.csv.

    python scripts/make_tables.py

No number is ever typed by hand: each table is a pure function of the
result CSVs written by scripts/run_experiments.py and scripts/prepare_data.py.
"""
from __future__ import annotations

import os
import sys

import pandas as pd
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

STATION_ORDER = ["Baengnyeongdo", "Daegwallyeong", "Paju", "Incheon", "Ganghwa",
                 "Dongducheon", "Chuncheon", "Taebaek", "Cheorwon", "Sokcho",
                 "Seoul", "Inje"]
OVERALL_MODELS = ["Persistence", "XGBoost", "1D-CNN", "LSTM"]
ONSET_MODELS = ["No-Onset Baseline", "XGBoost-Onset", "1D-CNN-Onset", "LSTM-Onset"]


def _pivot(df, task, models, metric):
    sub = df[(df["task"] == task) & (df["model"].isin(models))]
    p = sub.pivot_table(index="station", columns="model", values=metric, aggfunc="mean")
    p = p.reindex(index=[s for s in STATION_ORDER if s in p.index], columns=models)
    return p


def _with_mean_sd(p, decimals=3):
    out = p.round(decimals).astype(object)
    out.loc["Mean"] = p.mean().round(decimals)
    out.loc["SD"] = p.std(ddof=1).round(decimals)
    out.loc["Mean±SD"] = [f"{m:.3f}±{s:.3f}" for m, s in zip(p.mean(), p.std(ddof=1))]
    return out


def main() -> int:
    cfg = yaml.safe_load(open(os.path.join(ROOT, "configs", "default.yaml"), encoding="utf-8"))
    m = os.path.join(ROOT, cfg["paths"]["results_dir"], "metrics")
    t = os.path.join(ROOT, cfg["paths"]["results_dir"], "tables")
    os.makedirs(t, exist_ok=True)

    df = pd.read_csv(os.path.join(m, "station_metrics.csv"))

    # ---- Table: transition counts (pooled over all stations, test period) ----
    tr = pd.read_csv(os.path.join(m, "transition_counts.csv"))
    for scope in ("test", "all"):
        s = tr[tr["scope"] == scope]
        tot = s[["n_00", "n_01", "n_10", "n_11"]].sum()
        grand = int(tot.sum())
        tbl = pd.DataFrame({
            "transition": ["0→0", "0→1", "1→0", "1→1"],
            "description": ["Non-fog persistence", "Fog onset",
                            "Fog dissipation", "Fog persistence"],
            "count": [int(tot["n_00"]), int(tot["n_01"]), int(tot["n_10"]), int(tot["n_11"])],
        })
        tbl["proportion_percent"] = (tbl["count"] / grand * 100).round(2)
        tbl.to_csv(os.path.join(t, f"table_transition_counts_{scope}.csv"), index=False)

    # ---- Table: per-station fog prevalence ----
    prev = pd.read_csv(os.path.join(m, "fog_prevalence.csv"))
    prev.sort_values("fog_ratio", ascending=False).to_csv(
        os.path.join(t, "table_fog_prevalence.csv"), index=False)

    # ---- Table 3: overall T+1 F1 by station ----
    f1_overall = _pivot(df, "overall", OVERALL_MODELS, "f1")
    _with_mean_sd(f1_overall).to_csv(os.path.join(t, "table_overall_f1_by_station.csv"))

    # full metric table for the overall task
    (df[(df["task"] == "overall") & (df["model"].isin(OVERALL_MODELS))]
       .sort_values(["station", "model"])
       .to_csv(os.path.join(t, "table_overall_full_metrics.csv"), index=False))

    # ---- Table 4: onset summary ----
    f1_onset = _pivot(df, "onset", ONSET_MODELS, "f1")
    _with_mean_sd(f1_onset).to_csv(os.path.join(t, "table_onset_f1_by_station.csv"))
    pr_onset = _pivot(df, "onset", ONSET_MODELS, "pr_auc")
    _with_mean_sd(pr_onset).to_csv(os.path.join(t, "table_onset_prauc_by_station.csv"))

    summary = pd.DataFrame({
        "model": ONSET_MODELS,
        "f1_mean": [f1_onset[c].mean() for c in ONSET_MODELS],
        "f1_sd": [f1_onset[c].std(ddof=1) for c in ONSET_MODELS],
        "pr_auc_mean": [pr_onset[c].mean() for c in ONSET_MODELS],
        "best_in_n_stations": [int((f1_onset.idxmax(axis=1) == c).sum()) for c in ONSET_MODELS],
        "n_stations": len(f1_onset),
    }).round(4)
    summary.to_csv(os.path.join(t, "table_onset_summary.csv"), index=False)

    # ---- overall summary + head-to-head counts ----
    osum = pd.DataFrame({
        "model": OVERALL_MODELS,
        "f1_mean": [f1_overall[c].mean() for c in OVERALL_MODELS],
        "f1_sd": [f1_overall[c].std(ddof=1) for c in OVERALL_MODELS],
        "precision_mean": [_pivot(df, "overall", OVERALL_MODELS, "precision")[c].mean()
                           for c in OVERALL_MODELS],
        "recall_mean": [_pivot(df, "overall", OVERALL_MODELS, "recall")[c].mean()
                        for c in OVERALL_MODELS],
        "pr_auc_mean": [_pivot(df, "overall", OVERALL_MODELS, "pr_auc")[c].mean()
                        for c in OVERALL_MODELS],
        "best_in_n_stations": [int((f1_overall.idxmax(axis=1) == c).sum())
                               for c in OVERALL_MODELS],
        "n_stations": len(f1_overall),
    }).round(4)
    osum.to_csv(os.path.join(t, "table_overall_summary.csv"), index=False)

    # XGBoost vs Persistence head-to-head, tuned and default threshold
    hh = pd.DataFrame({
        "station": f1_overall.index,
        "persistence_f1": f1_overall["Persistence"].values,
        "xgboost_f1_tuned": f1_overall["XGBoost"].values,
    })
    dpath = os.path.join(m, "station_metrics_default_threshold.csv")
    if os.path.exists(dpath):
        dd = pd.read_csv(dpath)
        dd = dd[dd["task"] == "overall"].set_index("station")["f1"]
        hh["xgboost_f1_default_0.5"] = [dd.get(s) for s in hh["station"]]
    hh["persistence_beats_xgb_tuned"] = hh["persistence_f1"] > hh["xgboost_f1_tuned"]
    hh.round(4).to_csv(os.path.join(t, "table_persistence_vs_xgboost.csv"), index=False)

    # ---- thresholds actually used ----
    (df[["station", "task", "model", "threshold", "val_f1"]]
       .dropna(subset=["threshold"])
       .to_csv(os.path.join(m, "thresholds.csv"), index=False))

    print("Overall T+1 mean F1:")
    print(osum[["model", "f1_mean", "f1_sd", "best_in_n_stations"]].to_string(index=False))
    print("\nOnset mean F1:")
    print(summary[["model", "f1_mean", "f1_sd", "pr_auc_mean", "best_in_n_stations"]]
          .to_string(index=False))
    print(f"\nTables written to {t}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
