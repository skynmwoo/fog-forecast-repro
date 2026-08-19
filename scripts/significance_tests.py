# -*- coding: utf-8 -*-
"""Paired significance tests across the twelve stations.

    python scripts/significance_tests.py

Each station contributes one paired observation (its test-set F1 under two
models), so the twelve stations are the units of analysis.  The Wilcoxon
signed-rank test is used because twelve paired differences are too few to
justify a normality assumption, and station-level F1 values are strongly
heterogeneous.  p-values are corrected within each task with the Holm
step-down procedure.

Effect size is the matched-pairs rank-biserial correlation
    r = (W+ - W-) / (W+ + W-),
which is +1 when the first model wins at every station and -1 when it loses at
every station.

Output: results/tables/table_significance_tests.csv
"""
from __future__ import annotations

import itertools
import os

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M = os.path.join(ROOT, "results", "metrics")
T = os.path.join(ROOT, "results", "tables")

OVERALL = ["Persistence", "XGBoost", "LSTM", "1D-CNN"]
ONSET = ["XGBoost-Onset", "LSTM-Onset", "1D-CNN-Onset"]


def rank_biserial(diff: np.ndarray) -> float:
    nz = diff[diff != 0]
    if len(nz) == 0:
        return 0.0
    r = rankdata(np.abs(nz))
    w_pos = r[nz > 0].sum()
    w_neg = r[nz < 0].sum()
    return float((w_pos - w_neg) / (w_pos + w_neg))


def holm(pvals: list[float]) -> list[float]:
    n = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(n)
    running = 0.0
    for i, idx in enumerate(order):
        val = (n - i) * pvals[idx]
        running = max(running, val)
        adj[idx] = min(running, 1.0)
    return adj.tolist()


def run(df: pd.DataFrame, task: str, models: list[str], metric: str) -> pd.DataFrame:
    piv = df[df["task"] == task].pivot_table(index="station", columns="model", values=metric)
    rows = []
    for a, b in itertools.combinations(models, 2):
        pair = piv[[a, b]].dropna()
        diff = (pair[a] - pair[b]).to_numpy()
        stat, p = wilcoxon(pair[a], pair[b], zero_method="wilcox",
                           alternative="two-sided", method="exact")
        rows.append({
            "task": task, "metric": metric, "model_a": a, "model_b": b,
            "n_stations": int(len(pair)),
            "mean_a": round(float(pair[a].mean()), 4),
            "mean_b": round(float(pair[b].mean()), 4),
            "median_diff": round(float(np.median(diff)), 4),
            "a_wins": int((diff > 0).sum()),
            "b_wins": int((diff < 0).sum()),
            "wilcoxon_W": float(stat),
            "p_raw": float(p),
            "rank_biserial_r": round(rank_biserial(diff), 3),
        })
    out = pd.DataFrame(rows)
    out["p_holm"] = np.round(holm(out["p_raw"].tolist()), 4)
    out["p_raw"] = out["p_raw"].round(4)
    out["significant_005"] = out["p_holm"] < 0.05
    return out


def main() -> int:
    df = pd.read_csv(os.path.join(M, "station_metrics.csv"))
    parts = [
        run(df, "overall", OVERALL, "f1"),
        run(df, "onset", ONSET, "f1"),
        run(df, "onset", ONSET, "pr_auc"),
    ]
    out = pd.concat(parts, ignore_index=True)
    os.makedirs(T, exist_ok=True)
    out.to_csv(os.path.join(T, "table_significance_tests.csv"), index=False)
    print(out.to_string(index=False))
    print(f"\nWrote {T}/table_significance_tests.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
