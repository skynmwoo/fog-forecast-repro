# -*- coding: utf-8 -*-
"""Explicit 1D convolutional network.

Architecture (as reported in the manuscript Methods):
    Conv1d(F -> 32, k=3, padding='same') + ReLU
    Conv1d(32 -> 64, k=3, padding='same') + ReLU
    Global average pooling over the time axis
    Dropout(0.3)
    Linear(64 -> 32) + ReLU
    Linear(32 -> 1)          -> raw logit (BCEWithLogitsLoss)

Input tensor is (batch, seq_len, n_features); it is transposed to
(batch, n_features, seq_len) so that the convolution kernel slides along the
TIME axis, which is what makes this a temporal 1D-CNN rather than a
per-timestep feature mixer.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class CNN1D(nn.Module):
    def __init__(
        self,
        n_features: int,
        seq_len: int,
        channels=(32, 64),
        kernel_size: int = 3,
        dropout: float = 0.3,
        fc=(64, 32),
    ) -> None:
        super().__init__()
        c1, c2 = channels
        pad = kernel_size // 2  # 'same' padding for odd kernel sizes
        self.n_features = n_features
        self.seq_len = seq_len

        self.conv1 = nn.Conv1d(n_features, c1, kernel_size=kernel_size, padding=pad)
        self.conv2 = nn.Conv1d(c1, c2, kernel_size=kernel_size, padding=pad)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)  # global average pooling over time
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(fc[0], fc[1])
        self.out = nn.Linear(fc[1], 1)

        if c2 != fc[0]:
            raise ValueError(f"conv2 out_channels ({c2}) must match fc input ({fc[0]})")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F) -> (B, F, T) so convolution runs along time
        if x.dim() != 3:
            raise ValueError(f"expected (batch, seq_len, n_features), got {tuple(x.shape)}")
        x = x.transpose(1, 2)
        x = self.relu(self.conv1(x))          # (B, 32, T)
        x = self.relu(self.conv2(x))          # (B, 64, T)
        x = self.pool(x).squeeze(-1)          # (B, 64)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))            # (B, 32)
        return self.out(x).squeeze(-1)        # (B,)


def assert_is_cnn(model: nn.Module) -> None:
    """Structural guard: refuse to run an experiment on a model that is not a CNN."""
    convs = [m for m in model.modules() if isinstance(m, nn.Conv1d)]
    assert len(convs) >= 2, f"1D-CNN must contain >=2 nn.Conv1d layers, found {len(convs)}"
    assert not any(isinstance(m, nn.LSTM) for m in model.modules()), "CNN must not contain nn.LSTM"
    assert not any(isinstance(m, nn.Flatten) for m in model.modules()), (
        "CNN must not flatten the time axis into a dense layer (that would be an MLP)"
    )
