# -*- coding: utf-8 -*-
"""PHASE 10: summarise the multi-seed stability diagnostic.

    python scripts/run_experiments.py --seeds 42 1337 2024 --stability [--stations ...]
    python scripts/report_stability.py

This is a DIAGNOSTIC, not the reporting protocol.  The paper reports the
seed-42 run; this script quantifies how much of the reported difference could
be attributed to initialisation alone.
"""
from __future__ import annotations

import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M = os.path.join(ROOT, "results", "metrics")
T = os.path.join(ROOT, "results", "tables")


def main() -> int:
    p = os.path.join(M, "stability_metrics.csv")
    if not os.path.exists(p):
        print("stability_metrics.csv not found — run run_experiments.py --stability first")
        return 1
    d = pd.read_csv(p)
    d = d[d["seed"].notna()]

    g = (d.groupby(["task", "model", "station"])["f1"]
           .agg(["mean", "std", "min", "max", "count"])
           .reset_index())
    g["range"] = g["max"] - g["min"]
    g = g.round(4)
    g.to_csv(os.path.join(T, "table_seed_stability_by_station.csv"), index=False)

    s = (g.groupby(["task", "model"])[["std", "range"]].mean()
           .join(g.groupby(["task", "model"])["mean"].mean().rename("f1_mean"))
           .round(4).reset_index())
    s = s[["task", "model", "f1_mean", "std", "range"]]
    s.columns = ["task", "model", "mean_f1_across_seeds",
                 "mean_within_station_sd", "mean_within_station_range"]
    s.to_csv(os.path.join(T, "table_seed_stability_summary.csv"), index=False)

    print(s.to_string(index=False))
    print("\nPer station:")
    print(g[["task", "model", "station", "mean", "std", "min", "max"]].to_string(index=False))
    print(f"\nWrote {T}/table_seed_stability_summary.csv and _by_station.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
