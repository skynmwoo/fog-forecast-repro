# -*- coding: utf-8 -*-
"""Training loop shared by the 1D-CNN and the LSTM."""
from __future__ import annotations

import copy
import os
import random
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def _loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    gen = torch.Generator()
    gen.manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=gen if shuffle else None)


@torch.no_grad()
def predict_proba(model: nn.Module, X: np.ndarray, batch_size: int) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(X), batch_size):
        chunk = torch.from_numpy(X[i : i + batch_size])
        out.append(torch.sigmoid(model(chunk)).cpu().numpy())
    return np.concatenate(out) if out else np.empty(0, dtype=np.float32)


def train_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: Dict,
    seed: int,
    log_prefix: str = "",
) -> Tuple[nn.Module, Dict]:
    """Train with BCEWithLogitsLoss(pos_weight) + Adam, early stopping on validation F1.

    The epoch with the highest validation F1 is restored as the final model.
    Validation F1 during training uses the fixed 0.5 logit midpoint; the
    operating threshold is selected afterwards on the same validation set.
    """
    set_seed(seed, cfg["reproducibility"]["deterministic"])
    d = cfg["models"]["deep"]

    pos = float(y_train.sum())
    neg = float(len(y_train) - pos)
    pos_weight = torch.tensor([neg / pos if pos > 0 else 1.0], dtype=torch.float32)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=d["learning_rate"])
    train_loader = _loader(X_train, y_train, d["batch_size"], True, seed)

    best_f1 = -1.0
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    bad_epochs = 0
    history = []

    for epoch in range(1, d["max_epochs"] + 1):
        model.train()
        total = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(xb)
        train_loss = total / max(len(X_train), 1)

        p_val = predict_proba(model, X_val, d["batch_size"])
        val_f1 = f1_score(y_val.astype(int), (p_val >= 0.5).astype(int), zero_division=0)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_f1": float(val_f1)})

        if val_f1 > best_f1:
            best_f1, best_epoch, bad_epochs = float(val_f1), epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad_epochs += 1
            if bad_epochs >= d["early_stopping_patience"]:
                break

    model.load_state_dict(best_state)
    if log_prefix:
        print(f"      {log_prefix} best_epoch={best_epoch} val_f1@0.5={best_f1:.4f} "
              f"epochs_run={len(history)}")
    return model, {"best_epoch": best_epoch, "best_val_f1_at_0.5": best_f1, "history": history}
