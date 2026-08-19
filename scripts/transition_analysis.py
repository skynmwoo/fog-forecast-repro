# -*- coding: utf-8 -*-
"""Decompose overall T+1 skill by transition type, and relate the trained-model
advantage over Persistence to station fog prevalence.

    python scripts/transition_analysis.py

Both analyses reuse the per-row test predictions written by
scripts/run_experiments.py; no model is retrained and no metric is recomputed
from scratch.  They answer two questions the headline F1 table cannot:

  (1) Where does overall T+1 skill actually come from?  Positives in the overall
      task are a mixture of 1->1 (fog already present and persisting) and 0->1
      (fog newly forming).  A model can score well on the overall task by
      catching only the former.

  (2) When does a trained model beat the parameter-free baseline?  The twelve
      stations span a wide range of fog prevalence, so the sign and size of the
      trained-minus-Persistence gap can be related to prevalence.

      PREVALENCE DEFINITION.  The prevalence used here is the fraction of
      fog(t+1) positives in the COMMON EVALUATION SUBSET of the test period
      (2023-2024) -- the rows every model can score.  It is neither the training
      prevalence nor the full test-set prevalence, and it differs from the
      training prevalence at several stations.  The relationship reported is an
      association across stations, not evidence about training-set size.

Outputs (results/metrics/ and results/tables/):
    transition_recall_by_station.csv   per station x model x transition type
    table_transition_recall.csv        pooled recall by model and transition type
    prevalence_gap.csv                 per-station gap vs prevalence
    table_prevalence_gap_correlation.csv   Spearman / Pearson(log) with p-values
"""
from __future__ import annotations

import glob
import os
import re
import sys

import numpy as np
import pandas as pd
import yaml
from scipy.stats import pearsonr, spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M = os.path.join(ROOT, "results", "metrics")
T = os.path.join(ROOT, "results", "tables")
CK = os.path.join(ROOT, "checkpoints", "overall")

OVERALL = ["Persistence", "XGBoost", "1D-CNN", "LSTM"]


def _transition(cur: np.ndarray, nxt: np.ndarray) -> np.ndarray:
    return np.select(
        [(cur == 0) & (nxt == 0), (cur == 0) & (nxt == 1),
         (cur == 1) & (nxt == 0), (cur == 1) & (nxt == 1)],
        ["0->0", "0->1", "1->0", "1->1"], default="?")


def _load_predictions(station: str) -> dict[str, pd.DataFrame]:
    """Return {model: dataframe} of stored per-row test predictions."""
    out: dict[str, pd.DataFrame] = {}

    xgb = os.path.join(CK, f"predictions_overall_{station}.csv")
    if os.path.exists(xgb):
        out["XGBoost"] = pd.read_csv(xgb, parse_dates=["일시"])

    for path in glob.glob(os.path.join(CK, f"predictions_overall_*_{station}.csv")):
        m = re.match(rf"predictions_overall_(.+)_{re.escape(station)}\.csv",
                     os.path.basename(path))
        if m:
            out[m.group(1)] = pd.read_csv(path, parse_dates=["일시"])
    return out


def main() -> int:
    cfg = yaml.safe_load(open(os.path.join(ROOT, "configs", "default.yaml"), encoding="utf-8"))
    stations = list(cfg["data"]["stations"])

    rows = []
    missing = []
    for station in stations:
        preds = _load_predictions(station)
        if "XGBoost" not in preds:
            missing.append(station)
            continue
        base = preds["XGBoost"]
        trans = _transition(base["fog_now"].to_numpy(int), base["fog"].to_numpy(int))

        # Persistence is defined analytically: it predicts fog(t+1) = fog(t).
        frames = {"Persistence": base.assign(pred=base["fog_now"].astype(int))}
        frames.update(preds)

        for model in OVERALL:
            if model not in frames:
                continue
            df = frames[model]
            if len(df) != len(base) or not (df["일시"].to_numpy() == base["일시"].to_numpy()).all():
                raise ValueError(f"{station}/{model}: prediction rows do not align with XGBoost")
            pred = df["pred"].to_numpy(int)
            nxt = base["fog"].to_numpy(int)
            for tt in ("0->1", "1->1"):
                sel = trans == tt
                n = int(sel.sum())
                hit = int((pred[sel] == 1).sum())
                rows.append({"station": station, "model": model, "transition": tt,
                             "n_positive": n, "n_detected": hit,
                             "recall": hit / n if n else np.nan})
            # False alarms, so that recall cannot be read in isolation.
            neg = nxt == 0
            rows.append({"station": station, "model": model, "transition": "false_alarm",
                         "n_positive": int(neg.sum()),
                         "n_detected": int((pred[neg] == 1).sum()), "recall": np.nan})

    if missing:
        print(f"[warn] no stored predictions for: {', '.join(missing)}")
    if not rows:
        print("No prediction files found - run scripts/run_experiments.py first.")
        return 1

    det = pd.DataFrame(rows)
    det.to_csv(os.path.join(M, "transition_recall_by_station.csv"), index=False)

    # ---- pooled over stations -------------------------------------------
    pooled = (det.groupby(["model", "transition"])[["n_positive", "n_detected"]].sum()
                 .reset_index())
    pooled["recall"] = pooled["n_detected"] / pooled["n_positive"]
    tbl = (pooled[pooled.transition.isin(["0->1", "1->1"])]
           .pivot(index="model", columns="transition",
                  values=["n_detected", "recall"]).reindex(OVERALL))
    flat = pd.DataFrame({
        "model": tbl.index,
        "detected_1to1": tbl[("n_detected", "1->1")].astype(int).values,
        "recall_1to1": tbl[("recall", "1->1")].round(3).values,
        "detected_0to1": tbl[("n_detected", "0->1")].astype(int).values,
        "recall_0to1": tbl[("recall", "0->1")].round(3).values,
    })
    n11 = int(pooled[pooled.transition == "1->1"]["n_positive"].iloc[0])
    n01 = int(pooled[pooled.transition == "0->1"]["n_positive"].iloc[0])
    flat["n_positive_1to1"] = n11
    flat["n_positive_0to1"] = n01

    fa = (pooled[pooled.transition == "false_alarm"].set_index("model")["n_detected"]
          .reindex(OVERALL))
    flat["false_alarms"] = fa.astype(int).values
    tp = flat["detected_1to1"] + flat["detected_0to1"]
    flat["precision"] = (tp / (tp + flat["false_alarms"])).round(3)
    flat.to_csv(os.path.join(T, "table_transition_recall.csv"), index=False)

    print(f"Overall-task positives pooled over {det.station.nunique()} stations: "
          f"{n11 + n01:,}  (1->1 {n11:,} = {n11/(n11+n01)*100:.1f}%, 0->1 {n01:,})")
    print(flat.to_string(index=False))
    print()

    # ---- prevalence vs advantage over Persistence ------------------------
    met = pd.read_csv(os.path.join(M, "station_metrics.csv"))
    ov = met[met.task == "overall"]
    f1 = ov.pivot_table(index="station", columns="model", values="f1")
    prev = (ov[ov.model == "Persistence"].set_index("station")["positive_prevalence_test"] * 100)

    # Explicit column name: this is the prevalence of the COMMON EVALUATION SUBSET.
    gap = pd.DataFrame({"common_eval_fog_prevalence_percent": prev})
    for model in ["XGBoost", "1D-CNN", "LSTM"]:
        gap[f"{model}_minus_Persistence_f1"] = f1[model] - f1["Persistence"]
    gap = gap.sort_values("common_eval_fog_prevalence_percent", ascending=False).round(4)
    gap.to_csv(os.path.join(M, "prevalence_gap.csv"))

    corr = []
    for model in ["XGBoost", "1D-CNN", "LSTM"]:
        y = gap[f"{model}_minus_Persistence_f1"].to_numpy()
        x = gap["common_eval_fog_prevalence_percent"].to_numpy()
        rho, p_rho = spearmanr(x, y)
        r, p_r = pearsonr(np.log(x), y)
        corr.append({"model": model, "n_stations": len(x),
                     "spearman_rho": round(float(rho), 3),
                     "spearman_p_raw": float(p_rho),
                     "pearson_r_vs_log_prevalence": round(float(r), 3),
                     "pearson_p_raw": float(p_r)})
    cdf = pd.DataFrame(corr)

    # Three correlations are computed per statistic, one per trained model, so the
    # p-values are Holm-corrected within each family for consistency with the
    # model-comparison tests in scripts/significance_tests.py.
    def holm(pvals):
        order = np.argsort(pvals)
        adj = np.empty(len(pvals))
        running = 0.0
        for i, idx in enumerate(order):
            running = max(running, (len(pvals) - i) * pvals[idx])
            adj[idx] = min(running, 1.0)
        return adj

    for stat in ("spearman", "pearson"):
        cdf[f"{stat}_p_holm"] = np.round(holm(cdf[f"{stat}_p_raw"].to_numpy()), 4)
        cdf[f"{stat}_p_raw"] = cdf[f"{stat}_p_raw"].round(4)
    cdf["significant_005_holm"] = cdf["spearman_p_holm"] < 0.05
    cdf = cdf[["model", "n_stations", "spearman_rho", "spearman_p_raw", "spearman_p_holm",
               "pearson_r_vs_log_prevalence", "pearson_p_raw", "pearson_p_holm",
               "significant_005_holm"]]
    cdf.to_csv(os.path.join(T, "table_prevalence_gap_correlation.csv"), index=False)

    print("Advantage over Persistence vs. common-evaluation-subset fog prevalence:")
    print(cdf.to_string(index=False))
    print(f"\nWrote transition_recall_by_station.csv, table_transition_recall.csv, "
          f"prevalence_gap.csv, table_prevalence_gap_correlation.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
