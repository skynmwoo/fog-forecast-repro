# -*- coding: utf-8 -*-
"""Explicit recurrent LSTM classifier.

Architecture (as reported in the manuscript Methods):
    LSTM(input=F, hidden=64, num_layers=1, batch_first=True)
    take the final hidden state h_n[-1]      -> (batch, 64)
    Dropout(0.3)
    Linear(64 -> 32) + ReLU
    Linear(32 -> 1)                          -> raw logit (BCEWithLogitsLoss)

The classifier consumes the LAST HIDDEN STATE of the recurrence, not a
flattened copy of the input window, so the temporal ordering of the sequence
genuinely affects the prediction.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.3,
        fc=(64, 32),
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(fc[0], fc[1])
        self.relu = nn.ReLU()
        self.out = nn.Linear(fc[1], 1)

        if hidden_size != fc[0]:
            raise ValueError(f"hidden_size ({hidden_size}) must match fc input ({fc[0]})")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"expected (batch, seq_len, n_features), got {tuple(x.shape)}")
        seq_out, (h_n, _c_n) = self.lstm(x)   # seq_out: (B, T, H); h_n: (L, B, H)
        last_hidden = h_n[-1]                 # (B, H)
        z = self.dropout(last_hidden)
        z = self.relu(self.fc1(z))            # (B, 32)
        return self.out(z).squeeze(-1)        # (B,)


def assert_is_lstm(model: nn.Module) -> None:
    """Structural guard: refuse to run an experiment on a model that is not recurrent."""
    lstms = [m for m in model.modules() if isinstance(m, nn.LSTM)]
    assert len(lstms) >= 1, "LSTM model must contain at least one nn.LSTM module"
    assert not any(isinstance(m, nn.Conv1d) for m in model.modules()), "LSTM must not contain nn.Conv1d"
    assert not any(isinstance(m, nn.Flatten) for m in model.modules()), (
        "LSTM must not flatten the time axis into a dense layer (that would be an MLP)"
    )
