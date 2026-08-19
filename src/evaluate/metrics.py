# -*- coding: utf-8 -*-
"""Metric computation and validation-set threshold selection.

Threshold rule (identical for every probability-output model and every task):
    1. sweep a fixed grid on the VALIDATION set,
    2. keep the threshold that maximises validation F1 of the positive class,
    3. FREEZE it and apply it unchanged to the test set.
The test set is never consulted during threshold selection.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def select_threshold(
    y_val: np.ndarray, p_val: np.ndarray, lo: float, hi: float, steps: int
) -> Tuple[float, float]:
    grid = np.linspace(lo, hi, steps)
    scores = [f1_score(y_val, (p_val >= t).astype(int), zero_division=0) for t in grid]
    best = int(np.argmax(scores))
    return float(grid[best]), float(scores[best])


def compute_metrics(
    station: str,
    task: str,
    model: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    threshold: Optional[float] = None,
    val_f1: Optional[float] = None,
    seed: Optional[int] = None,
) -> Dict:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    roc = pr = None
    if y_prob is not None and len(np.unique(y_true)) == 2:
        roc = float(roc_auc_score(y_true, y_prob))
        pr = float(average_precision_score(y_true, y_prob))

    return {
        "station": station,
        "task": task,
        "model": model,
        "seed": seed,
        "threshold": threshold,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": roc,
        "pr_auc": pr,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "n_test": int(len(y_true)),
        "n_positive_test": int(y_true.sum()),
        "positive_prevalence_test": float(y_true.mean()) if len(y_true) else float("nan"),
        "val_f1": val_f1,
    }
