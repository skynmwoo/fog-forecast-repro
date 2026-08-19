# -*- coding: utf-8 -*-
"""Sequence construction for the 1D-CNN and LSTM models.

WINDOW DEFINITION (corrected relative to the legacy code)
---------------------------------------------------------
For a row at time t the input window covers

        t-(L-1), ..., t-1, t          (L = seq_len, INCLUSIVE of t)

and the label is fog(t+1).  The legacy implementation used ``X[i-L:i]``,
i.e. t-L ... t-1, which withheld the current observation from the deep models
while XGBoost received it — the deep models were effectively forecasting two
hours ahead and were compared against one-hour-ahead models.  See
REPRODUCIBILITY_AUDIT.md, finding L1.

Additional guarantees
---------------------
* windows are built **after** the temporal split, so no window ever spans a
  train/validation/test boundary;
* windows are rejected unless all L timestamps are exactly one hour apart,
  so a window can never silently bridge a data gap.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def make_sequences(
    df: pd.DataFrame,
    features: List[str],
    seq_len: int,
    label_col: str,
    scaler: Optional[StandardScaler] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (X, y, row_index) with X shaped (N, seq_len, n_features)."""
    values = df[features].to_numpy(dtype=np.float32)
    if scaler is not None:
        values = scaler.transform(values).astype(np.float32)

    labels = df[label_col].to_numpy(dtype=np.float32)
    times = df["일시"].to_numpy(dtype="datetime64[h]")

    n = len(df)
    xs, ys, idx = [], [], []
    step = np.timedelta64(1, "h")
    for i in range(seq_len - 1, n):
        lo = i - seq_len + 1
        if not np.all(np.diff(times[lo : i + 1]) == step):
            continue  # window crosses a gap in the hourly record
        if np.isnan(labels[i]):
            continue
        xs.append(values[lo : i + 1])
        ys.append(labels[i])
        idx.append(i)

    if not xs:
        return (
            np.empty((0, seq_len, len(features)), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )
    return (
        np.asarray(xs, dtype=np.float32),
        np.asarray(ys, dtype=np.float32),
        np.asarray(idx, dtype=np.int64),
    )


def fit_scaler(train_df: pd.DataFrame, features: List[str]) -> StandardScaler:
    """Fit StandardScaler on TRAINING rows only (never val/test)."""
    scaler = StandardScaler()
    scaler.fit(train_df[features].to_numpy(dtype=np.float64))
    return scaler
