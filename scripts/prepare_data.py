# -*- coding: utf-8 -*-
"""PHASE 7 step 1: raw ASOS CSV -> deterministic processed parquet per station.

    python scripts/prepare_data.py [--config configs/default.yaml]

Outputs
-------
data/processed/<Station>.parquet          cleaned + feature-engineered table
results/metrics/fog_prevalence.csv        per-station fog prevalence (all splits)
results/metrics/transition_counts.csv     0->0 / 0->1 / 1->0 / 1->1 counts
results/metrics/split_summary.csv         row counts per split
results/logs/sanity_real_batch.npy        a real scaled batch for the model gate
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.features import assert_no_future_leakage, build_features, finalize
from src.data.preprocess import clean, load_station
from src.data.sequences import fit_scaler, make_sequences

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def temporal_split(df: pd.DataFrame, train_end: str, val_end: str):
    t = df["일시"]
    train = df[t < pd.Timestamp(train_end)].copy()
    val = df[(t >= pd.Timestamp(train_end)) & (t < pd.Timestamp(val_end))].copy()
    test = df[t >= pd.Timestamp(val_end)].copy()
    # hard guarantee: the splits are disjoint and strictly ordered in time
    assert train["일시"].max() < val["일시"].min(), "train/val overlap"
    assert val["일시"].max() < test["일시"].min(), "val/test overlap"
    return train, val, test


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "default.yaml"))
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    raw_dir = os.path.join(ROOT, cfg["paths"]["raw_data_dir"])
    proc_dir = os.path.join(ROOT, cfg["paths"]["processed_dir"])
    res = os.path.join(ROOT, cfg["paths"]["results_dir"])
    os.makedirs(proc_dir, exist_ok=True)
    for sub in ("metrics", "logs"):
        os.makedirs(os.path.join(res, sub), exist_ok=True)

    feat_xgb = cfg["features"]["xgb"]
    feat_dl = cfg["features"]["dl"]
    assert_no_future_leakage(feat_xgb)
    assert_no_future_leakage(feat_dl)
    all_feats = list(dict.fromkeys(feat_xgb + feat_dl))

    prevalence, transitions, splits = [], [], []
    real_batch = None

    for station, meta in cfg["data"]["stations"].items():
        print(f"[{station}]")
        raw = load_station(raw_dir, meta["prefix"], cfg["data"]["years"])
        cleaned = clean(raw, cfg["data"]["valid_ranges"])
        feats = build_features(cleaned, cfg["data"]["fog_threshold_vis10m"])
        df = finalize(feats, all_feats)

        train, val, test = temporal_split(df, cfg["data"]["train_end"], cfg["data"]["val_end"])
        df["split"] = np.where(
            df["일시"] < pd.Timestamp(cfg["data"]["train_end"]), "train",
            np.where(df["일시"] < pd.Timestamp(cfg["data"]["val_end"]), "val", "test"),
        )
        df.to_parquet(os.path.join(proc_dir, f"{station}.parquet"), index=False)

        for name, part in (("train", train), ("val", val), ("test", test), ("all", df)):
            splits.append({
                "station": station, "split": name, "n_rows": len(part),
                "start": str(part["일시"].min()), "end": str(part["일시"].max()),
                "n_fog_tplus1": int(part["fog"].sum()),
                "fog_prevalence": float(part["fog"].mean()),
                "n_onset_candidates": int((part["fog_now"] == 0).sum()),
                "n_onset": int(((part["fog_now"] == 0) & (part["fog"] == 1)).sum()),
                "onset_prevalence": float(
                    ((part["fog_now"] == 0) & (part["fog"] == 1)).sum()
                    / max((part["fog_now"] == 0).sum(), 1)
                ),
            })

        prevalence.append({
            "station": station,
            "n_total": len(df),
            "n_fog": int(df["fog_now"].sum()),
            "fog_ratio": float(df["fog_now"].mean()),
            "fog_percent": round(float(df["fog_now"].mean()) * 100, 3),
            "n_test": len(test),
            "n_fog_test_tplus1": int(test["fog"].sum()),
            "fog_prevalence_test": float(test["fog"].mean()),
        })

        for part_name, part in (("test", test), ("all", df)):
            cur, nxt = part["fog_now"].to_numpy(), part["fog"].to_numpy()
            transitions.append({
                "station": station, "scope": part_name,
                "n_00": int(((cur == 0) & (nxt == 0)).sum()),
                "n_01": int(((cur == 0) & (nxt == 1)).sum()),
                "n_10": int(((cur == 1) & (nxt == 0)).sum()),
                "n_11": int(((cur == 1) & (nxt == 1)).sum()),
                "n_total": int(len(part)),
            })

        if real_batch is None:
            scaler = fit_scaler(train, feat_dl)
            Xs, _, _ = make_sequences(
                test, feat_dl, cfg["models"]["deep"]["seq_len"], "fog", scaler
            )
            if len(Xs):
                real_batch = Xs[:64]

        print(f"    rows={len(df):,}  train={len(train):,} val={len(val):,} test={len(test):,}"
              f"  fog(t+1) test={int(test['fog'].sum())}")

    m = os.path.join(res, "metrics")
    pd.DataFrame(prevalence).to_csv(os.path.join(m, "fog_prevalence.csv"), index=False)
    pd.DataFrame(transitions).to_csv(os.path.join(m, "transition_counts.csv"), index=False)
    pd.DataFrame(splits).to_csv(os.path.join(m, "split_summary.csv"), index=False)
    if real_batch is not None:
        np.save(os.path.join(res, "logs", "sanity_real_batch.npy"), real_batch)

    print("\nWrote fog_prevalence.csv, transition_counts.csv, split_summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
