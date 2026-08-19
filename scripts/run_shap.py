# -*- coding: utf-8 -*-
"""PHASE 11: SHAP analysis of the reproduced overall-T+1 XGBoost models.

    python scripts/run_shap.py [--sample 20000]

Uses the checkpoints written by scripts/run_experiments.py, so the SHAP values
describe exactly the models whose metrics appear in the manuscript tables.

Outputs (results/metrics/):
    shap_station.csv          mean |SHAP| per station and feature
    shap_global.csv           mean |SHAP| pooled over stations (global ranking)
    shap_station_normalized.csv   per-station values scaled to sum to 1
    error_visibility_stats.csv    visibility distribution for TP / FP / FN / TN
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import shap
import yaml
from xgboost import XGBClassifier

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=20000,
                    help="max test rows per station used for SHAP (0 = all)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(os.path.join(ROOT, "configs", "default.yaml"), encoding="utf-8"))
    feats = cfg["features"]["xgb"]
    proc = os.path.join(ROOT, cfg["paths"]["processed_dir"])
    ck = os.path.join(ROOT, cfg["paths"]["checkpoints_dir"], "overall")
    m = os.path.join(ROOT, cfg["paths"]["results_dir"], "metrics")
    rng = np.random.default_rng(args.seed)

    station_rows, err_rows, row_rows = [], [], []

    for station in cfg["data"]["stations"]:
        mpath = os.path.join(ck, f"xgb_overall_{station}.json")
        ppath = os.path.join(ck, f"predictions_overall_{station}.csv")
        if not (os.path.exists(mpath) and os.path.exists(ppath)):
            print(f"[skip] {station}: checkpoint missing — run scripts/run_experiments.py first")
            continue

        model = XGBClassifier()
        model.load_model(mpath)

        df = pd.read_parquet(os.path.join(proc, f"{station}.parquet"))
        test = df[df["split"] == "test"].reset_index(drop=True)
        pred = pd.read_csv(ppath, parse_dates=["일시"])
        test = test.merge(pred[["일시", "prob", "pred"]], on="일시", how="inner")

        X = test[feats]
        if args.sample and len(X) > args.sample:
            sel = rng.choice(len(X), args.sample, replace=False)
            Xs = X.iloc[sel]
        else:
            Xs = X

        expl = shap.TreeExplainer(model)
        sv = expl.shap_values(Xs)
        mean_abs = np.abs(sv).mean(axis=0)
        for f, v in zip(feats, mean_abs):
            station_rows.append({"station": station, "feature": f, "mean_abs_shap": float(v)})

        # ---- TP / FP / FN / TN visibility distributions ----
        y = test["fog"].to_numpy(int)
        p = test["pred"].to_numpy(int)
        cat = np.select([(y == 1) & (p == 1), (y == 0) & (p == 1), (y == 1) & (p == 0)],
                        ["TP", "FP", "FN"], default="TN")
        test["error_type"] = cat
        for name, grp in test.groupby("error_type"):
            err_rows.append({
                "station": station, "error_type": name, "n": int(len(grp)),
                "vis_now_mean": float(grp["시정(10m)"].mean()),
                "vis_now_median": float(grp["시정(10m)"].median()),
                "vis_now_p25": float(grp["시정(10m)"].quantile(0.25)),
                "vis_now_p75": float(grp["시정(10m)"].quantile(0.75)),
                "vis_tplus1_median": float(grp["target_vis_tplus1"].median()),
                "rh_mean": float(grp["습도(%)"].mean()),
                "dpd_mean": float(grp["dew_point_depression"].mean()),
            })
        keep = test[test["error_type"].isin(["TP", "FP", "FN"])]
        row_rows.append(keep[["error_type", "시정(10m)", "target_vis_tplus1"]]
                        .assign(station=station))
        print(f"  {station}: SHAP on {len(Xs):,} rows")

    st = pd.DataFrame(station_rows)
    st.to_csv(os.path.join(m, "shap_station.csv"), index=False)

    glob = (st.groupby("feature")["mean_abs_shap"].mean()
              .sort_values(ascending=False).reset_index())
    glob["rank"] = np.arange(1, len(glob) + 1)
    glob.to_csv(os.path.join(m, "shap_global.csv"), index=False)

    norm = st.copy()
    norm["normalized"] = norm.groupby("station")["mean_abs_shap"].transform(lambda s: s / s.sum())
    norm.to_csv(os.path.join(m, "shap_station_normalized.csv"), index=False)

    pd.DataFrame(err_rows).to_csv(os.path.join(m, "error_visibility_stats.csv"), index=False)
    if row_rows:
        pd.concat(row_rows, ignore_index=True).rename(
            columns={"시정(10m)": "vis_now_10m", "target_vis_tplus1": "vis_tplus1_10m"}
        ).to_csv(os.path.join(m, "error_visibility_rows.csv"), index=False)

    print("\nGlobal mean |SHAP| ranking (top 10):")
    print(glob.head(10).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
