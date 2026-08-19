# -*- coding: utf-8 -*-
"""PHASE 7: run every model on every station for both prediction tasks.

    python scripts/run_experiments.py                    # all stations, both tasks
    python scripts/run_experiments.py --stations Seoul Inje
    python scripts/run_experiments.py --task onset
    python scripts/run_experiments.py --seeds 42 1337 2024 --stability

Tasks
-----
overall : predict fog(t+1) for every time step t
onset   : predict fog(t+1) restricted to candidates with fog(t) == 0

Models
------
overall : Persistence, XGBoost, 1D-CNN, LSTM
onset   : No-Onset Baseline, XGBoost-Onset, 1D-CNN-Onset, LSTM-Onset

Every number that reaches the manuscript is written to results/metrics/.
Nothing is reported from stdout alone.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import yaml
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.features import assert_no_future_leakage
from src.data.sequences import fit_scaler, make_sequences
from src.evaluate.metrics import compute_metrics, select_threshold
from src.models.cnn1d import CNN1D, assert_is_cnn
from src.models.lstm import LSTMClassifier, assert_is_lstm
from src.train.deep import predict_proba, set_seed, train_model

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TASKS = {
    "overall": {"label": "fog", "candidate": None},
    "onset": {"label": "onset", "candidate": "fog_now == 0"},
}


def load_station_df(cfg, station):
    path = os.path.join(ROOT, cfg["paths"]["processed_dir"], f"{station}.parquet")
    return pd.read_parquet(path)


def split_parts(df):
    return (df[df["split"] == "train"].reset_index(drop=True),
            df[df["split"] == "val"].reset_index(drop=True),
            df[df["split"] == "test"].reset_index(drop=True))


def tabular_subset(part, task):
    """Rows evaluated by the tabular models for a given task (training/validation)."""
    if task == "onset":
        return part[part["fog_now"] == 0].reset_index(drop=True)
    return part


def common_test_index(cfg, test_df, task):
    """Row positions in ``test_df`` that EVERY model can score.

    The deep models can only score a row when the preceding ``seq_len`` hours
    form an unbroken hourly window, so a handful of rows after a data gap are
    unavailable to them.  Restricting the tabular models to the same rows makes
    the four-model comparison exactly like-for-like; without it Persistence and
    XGBoost would be scored on ~2% more cases than 1D-CNN and LSTM.
    """
    label = "fog" if task == "overall" else "onset"
    _, _, idx = make_sequences(test_df, cfg["features"]["dl"],
                               cfg["models"]["deep"]["seq_len"], label, scaler=None)
    return idx


# ---------------------------------------------------------------- baselines
def run_baseline(station, task, test_df, eval_idx):
    te = test_df.iloc[eval_idx].reset_index(drop=True)
    y_true = te["fog"].to_numpy(int)
    if task == "overall":
        y_pred = te["fog_now"].to_numpy(int)           # carry current state forward
        name = "Persistence"
    else:
        y_pred = np.zeros_like(y_true)                 # persistence never predicts onset
        name = "No-Onset Baseline"
    return compute_metrics(station, task, name, y_true, y_pred,
                           y_prob=y_pred.astype(float), threshold=None)


# ---------------------------------------------------------------- XGBoost
def run_xgboost(cfg, station, task, train_df, val_df, test_df, eval_idx, seed, save_dir):
    feats = cfg["features"]["xgb"]
    tr, va = (tabular_subset(p, task) for p in (train_df, val_df))
    te = test_df.iloc[eval_idx].reset_index(drop=True)   # common evaluation set

    Xtr, ytr = tr[feats], tr["fog"].to_numpy(int)
    pos, neg = int(ytr.sum()), int((ytr == 0).sum())
    spw = neg / pos if pos else 1.0

    params = dict(cfg["models"]["xgboost"])
    params.update(scale_pos_weight=spw, random_state=seed, n_jobs=-1, verbosity=0)
    model = XGBClassifier(**params)
    model.fit(Xtr, ytr)

    p_val = model.predict_proba(va[feats])[:, 1]
    thr, val_f1 = select_threshold(va["fog"].to_numpy(int), p_val,
                                   cfg["evaluation"]["threshold_min"],
                                   cfg["evaluation"]["threshold_max"],
                                   cfg["evaluation"]["threshold_steps"])
    p_test = model.predict_proba(te[feats])[:, 1]
    name = "XGBoost" if task == "overall" else "XGBoost-Onset"
    res = compute_metrics(station, task, name, te["fog"].to_numpy(int),
                          (p_test >= thr).astype(int), p_test, thr, val_f1, seed)
    res["scale_pos_weight"] = spw

    # default-threshold (0.5) variant, reported separately in the manuscript
    res_def = compute_metrics(station, task, name + " (thr=0.5)", te["fog"].to_numpy(int),
                              (p_test >= 0.5).astype(int), p_test, 0.5, None, seed)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        model.save_model(os.path.join(save_dir, f"xgb_{task}_{station}.json"))
        pred = te[["일시", "fog", "fog_now", "시정(10m)", "target_vis_tplus1"]].copy()
        pred["prob"] = p_test
        pred["pred"] = (p_test >= thr).astype(int)
        pred.to_csv(os.path.join(save_dir, f"predictions_{task}_{station}.csv"), index=False)
    return res, res_def, model, te, p_test, thr


# ---------------------------------------------------------------- deep models
def run_deep(cfg, kind, station, task, train_df, val_df, test_df, seed):
    feats = cfg["features"]["dl"]
    d = cfg["models"]["deep"]
    label = "fog" if task == "overall" else "onset"

    scaler = fit_scaler(train_df, feats)        # TRAIN ONLY
    Xtr, ytr, _ = make_sequences(train_df, feats, d["seq_len"], label, scaler)
    Xva, yva, _ = make_sequences(val_df, feats, d["seq_len"], label, scaler)
    Xte, yte, ite = make_sequences(test_df, feats, d["seq_len"], label, scaler)
    if min(len(Xtr), len(Xva), len(Xte)) == 0 or ytr.sum() == 0 or yva.sum() == 0:
        print(f"      [skip] {kind}/{task}/{station}: insufficient positive sequences")
        return None, None

    set_seed(seed, cfg["reproducibility"]["deterministic"])
    if kind == "cnn":
        model = CNN1D(len(feats), d["seq_len"], tuple(d["cnn"]["channels"]),
                      d["cnn"]["kernel_size"], d["dropout"], tuple(d["cnn"]["fc"]))
        assert_is_cnn(model)
        name = "1D-CNN" if task == "overall" else "1D-CNN-Onset"
    else:
        model = LSTMClassifier(len(feats), d["lstm"]["hidden_size"], d["lstm"]["num_layers"],
                               d["dropout"], tuple(d["lstm"]["fc"]))
        assert_is_lstm(model)
        name = "LSTM" if task == "overall" else "LSTM-Onset"

    model, info = train_model(model, Xtr, ytr, Xva, yva, cfg, seed,
                              log_prefix=f"{name}/{station}")

    p_val = predict_proba(model, Xva, d["batch_size"])
    thr, val_f1 = select_threshold(yva.astype(int), p_val,
                                   cfg["evaluation"]["threshold_min"],
                                   cfg["evaluation"]["threshold_max"],
                                   cfg["evaluation"]["threshold_steps"])
    p_test = predict_proba(model, Xte, d["batch_size"])
    res = compute_metrics(station, task, name, yte.astype(int),
                          (p_test >= thr).astype(int), p_test, thr, val_f1, seed)
    res["best_epoch"] = info["best_epoch"]
    res["n_train_sequences"] = int(len(Xtr))
    return res, model


# ---------------------------------------------------------------- driver
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "default.yaml"))
    ap.add_argument("--stations", nargs="*", default=None)
    ap.add_argument("--task", choices=["overall", "onset", "both"], default="both")
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--stability", action="store_true",
                    help="write to stability_metrics.csv instead of the main result files")
    ap.add_argument("--out-suffix", default="")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    assert_no_future_leakage(cfg["features"]["xgb"])
    assert_no_future_leakage(cfg["features"]["dl"])

    stations = args.stations or list(cfg["data"]["stations"].keys())
    tasks = ["overall", "onset"] if args.task == "both" else [args.task]
    seeds = args.seeds or [cfg["reproducibility"]["seed"]]

    res_dir = os.path.join(ROOT, cfg["paths"]["results_dir"], "metrics")
    ckpt_dir = os.path.join(ROOT, cfg["paths"]["checkpoints_dir"])
    os.makedirs(res_dir, exist_ok=True)

    rows, rows_default = [], []
    t0 = time.time()

    for station in stations:
        df = load_station_df(cfg, station)
        train_df, val_df, test_df = split_parts(df)
        for task in tasks:
            print(f"  {station} / {task}")
            eval_idx = common_test_index(cfg, test_df, task)
            rows.append(run_baseline(station, task, test_df, eval_idx))
            for seed in seeds:
                save = None if args.stability else os.path.join(ckpt_dir, task)
                xr, xd, *_ = run_xgboost(cfg, station, task, train_df, val_df, test_df,
                                         eval_idx, seed, save)
                rows.append(xr)
                rows_default.append(xd)
                for kind in ("cnn", "lstm"):
                    r, _ = run_deep(cfg, kind, station, task, train_df, val_df, test_df, seed)
                    if r:
                        rows.append(r)
        print(f"    elapsed {time.time()-t0:.0f}s")

    out = pd.DataFrame(rows)
    suffix = args.out_suffix
    if args.stability:
        path = os.path.join(res_dir, f"stability_metrics{suffix}.csv")
    else:
        path = os.path.join(res_dir, f"station_metrics{suffix}.csv")
    out.to_csv(path, index=False)
    print(f"\nWrote {path}  ({len(out)} rows)")

    if rows_default and not args.stability:
        dpath = os.path.join(res_dir, f"station_metrics_default_threshold{suffix}.csv")
        pd.DataFrame(rows_default).to_csv(dpath, index=False)
        print(f"Wrote {dpath}")

    env = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "seeds": seeds,
        "stations": stations,
        "tasks": tasks,
        "config": os.path.relpath(args.config, ROOT),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    with open(os.path.join(ROOT, cfg["paths"]["results_dir"], "logs",
                           f"run_env{suffix}.json"), "w") as fh:
        json.dump(env, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
