# -*- coding: utf-8 -*-
"""Derived-feature construction.

TEMPORAL CONTRACT
-----------------
Target      : fog(t+1)  ==  visibility(t+1) <= 100  (units of 10 m, i.e. <= 1 km)
Predictors  : every feature below is a function of observations at time <= t.
              No feature reads visibility(t+1) or any other t+1 quantity.

The only column that touches t+1 is ``target_vis_tplus1`` / ``fog``, which are
labels and are never placed in a feature list.
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

# Columns that must never appear in a feature list.
FORBIDDEN_IN_FEATURES = {"target_vis_tplus1", "fog", "일시"}


def build_features(df: pd.DataFrame, fog_threshold: int) -> pd.DataFrame:
    out = df.copy()

    # --- physical ------------------------------------------------------
    out["dew_point_depression"] = out["기온(°C)"] - out["이슬점온도(°C)"]
    out["air_ground_diff"] = out["기온(°C)"] - out["지면온도(°C)"]
    out["ground_air_diff"] = out["지면온도(°C)"] - out["기온(°C)"]
    out["ground_air_diff_1h"] = out["ground_air_diff"].diff()          # (t) - (t-1)
    out["rel_humid"] = out["습도(%)"] / 100.0

    denom = out["기온(°C)"].where(out["기온(°C)"].abs() > 1e-6, np.nan)
    out["dpd_ratio"] = out["dew_point_depression"] / denom

    # --- calendar ------------------------------------------------------
    out["hour"] = out["일시"].dt.hour
    out["weekday"] = out["일시"].dt.weekday
    out["is_weekend"] = (out["weekday"] >= 5).astype(int)
    month = out["일시"].dt.month
    out["month"] = month
    out["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * month / 12.0)

    # --- lags (strictly backward) --------------------------------------
    for lag in (1, 2, 3):
        out[f"temp_lag_{lag}"] = out["기온(°C)"].shift(lag)
        out[f"humidity_lag_{lag}"] = out["습도(%)"].shift(lag)
        out[f"vis_lag_{lag}"] = out["시정(10m)"].shift(lag)

    # --- rolling statistics over {t-1, t} ------------------------------
    # Backward-looking window; contains no information from t+1.
    out["temp_roll_mean_2h"] = out["기온(°C)"].rolling(window=2, min_periods=2).mean()
    out["humid_roll_std_2h"] = out["습도(%)"].rolling(window=2, min_periods=2).std()

    # --- labels --------------------------------------------------------
    out["target_vis_tplus1"] = out["시정(10m)"].shift(-1)
    out["fog"] = (out["target_vis_tplus1"] <= fog_threshold).astype(float)
    out.loc[out["target_vis_tplus1"].isna(), "fog"] = np.nan

    # current fog state at t  (Persistence baseline / onset candidate filter)
    out["fog_now"] = (out["시정(10m)"] <= fog_threshold).astype(float)
    out.loc[out["시정(10m)"].isna(), "fog_now"] = np.nan

    # onset label: defined only where fog_now == 0
    out["onset"] = np.where(
        (out["fog_now"] == 0) & out["fog"].notna(), out["fog"], np.nan
    )

    return out.replace([np.inf, -np.inf], np.nan)


def finalize(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """Drop rows with any missing required value; keep chronological order."""
    required = list(dict.fromkeys(feature_cols)) + ["fog", "fog_now", "target_vis_tplus1"]
    out = df.dropna(subset=required).reset_index(drop=True)
    return out


def assert_no_future_leakage(feature_cols: List[str]) -> None:
    """Structural guard: no label / target-time column may be used as a predictor."""
    bad = sorted(set(feature_cols) & FORBIDDEN_IN_FEATURES)
    if bad:
        raise AssertionError(f"Target-time columns present in feature list: {bad}")
    suspicious = [c for c in feature_cols if "tplus" in c or c.endswith("_lead")]
    if suspicious:
        raise AssertionError(f"Forward-looking feature names detected: {suspicious}")
