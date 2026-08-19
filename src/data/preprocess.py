# -*- coding: utf-8 -*-
"""Deterministic loading + cleaning of KMA ASOS station data.

Contract enforced here:
    * rows are sorted strictly by observation time
    * the prediction target is fog at t+1
    * every predictor is observable at or before time t
"""
from __future__ import annotations

import os
from typing import Dict, List

import numpy as np
import pandas as pd

BASE_COLS: List[str] = [
    "일시",
    "기온(°C)",
    "이슬점온도(°C)",
    "습도(%)",
    "풍속(m/s)",
    "현지기압(hPa)",
    "지면온도(°C)",
    "시정(10m)",
]

_ENCODINGS = ["euc-kr", "cp949", "utf-8-sig", "utf-8"]


def safe_read_csv(path: str) -> pd.DataFrame:
    last_error: Exception | None = None
    for enc in _ENCODINGS:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    raise RuntimeError(f"Failed to read {path}: {last_error}")


def load_station(raw_dir: str, prefix: str, years: List[int]) -> pd.DataFrame:
    frames = []
    missing = []
    for year in years:
        fpath = os.path.join(raw_dir, f"{prefix}{year}.csv")
        if not os.path.exists(fpath):
            missing.append(fpath)
            continue
        frames.append(safe_read_csv(fpath))
    if not frames:
        raise FileNotFoundError(f"No CSV found for prefix '{prefix}' in {raw_dir}")
    if missing:
        print(f"    [warn] missing files: {[os.path.basename(m) for m in missing]}")
    return pd.concat(frames, ignore_index=True)


def clean(df: pd.DataFrame, valid_ranges: Dict[str, List[float]]) -> pd.DataFrame:
    """Type coercion, chronological sort, physical-range filtering, hourly regridding.

    Out-of-range observations are set to NaN rather than having their rows deleted,
    and the series is then reindexed onto a complete hourly grid.  This guarantees
    that ``shift(k)`` really means "k hours earlier" — deleting rows first would
    silently turn a lag-1 feature into a lag-of-unknown-length feature.
    Rows that remain incomplete are dropped later, after feature construction.
    """
    missing = [c for c in BASE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}; got {list(df.columns)}")

    out = df[BASE_COLS].copy()
    out["일시"] = pd.to_datetime(out["일시"], errors="coerce")
    out = out.dropna(subset=["일시"])
    out = out.sort_values("일시").drop_duplicates(subset="일시", keep="first")

    for col in BASE_COLS[1:]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan)

    # Out-of-range -> NaN (do NOT drop the row: that would corrupt the time grid)
    for col, (lo, hi) in valid_ranges.items():
        out.loc[~out[col].between(lo, hi), col] = np.nan

    # Reindex onto a complete hourly grid spanning the observed period
    full_index = pd.date_range(out["일시"].min(), out["일시"].max(), freq="h")
    out = out.set_index("일시").reindex(full_index)
    out.index.name = "일시"
    return out.reset_index()
