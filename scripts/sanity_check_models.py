# -*- coding: utf-8 -*-
"""PHASE 6 gate: prove the deep models are really a CNN and a recurrent LSTM.

Run this BEFORE any experiment.  It fails loudly if either model degenerates
into an MLP, if the tensor shapes are wrong, or if the convolution / recurrent
parameters receive no gradient.

    python scripts/sanity_check_models.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.nn as nn
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.cnn1d import CNN1D, assert_is_cnn
from src.models.lstm import LSTMClassifier, assert_is_lstm
from src.train.deep import set_seed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _hook_shapes(model: nn.Module, names):
    rec = {}
    handles = []
    for name in names:
        mod = dict(model.named_modules())[name]

        def make(n):
            def fn(_m, inp, out):
                i = inp[0]
                o = out[0] if isinstance(out, tuple) else out
                rec[n] = (tuple(i.shape), tuple(o.shape))
            return fn

        handles.append(mod.register_forward_hook(make(name)))
    return rec, handles


def check_cnn(cfg, n_features, seq_len, batch):
    print("=" * 72)
    print("1D-CNN STRUCTURAL SANITY CHECK")
    print("=" * 72)
    d = cfg["models"]["deep"]
    model = CNN1D(
        n_features=n_features,
        seq_len=seq_len,
        channels=tuple(d["cnn"]["channels"]),
        kernel_size=d["cnn"]["kernel_size"],
        dropout=d["dropout"],
        fc=tuple(d["cnn"]["fc"]),
    )
    assert_is_cnn(model)
    print(model)

    rec, handles = _hook_shapes(model, ["conv1", "conv2", "pool", "fc1", "out"])
    x = torch.from_numpy(batch.astype(np.float32))
    print(f"\nInput tensor shape        : {tuple(x.shape)}  (batch, seq_len, n_features)")
    logits = model(x)
    for h in handles:
        h.remove()
    print(f"Conv1  in/out             : {rec['conv1'][0]} -> {rec['conv1'][1]}")
    print(f"Conv2  in/out             : {rec['conv2'][0]} -> {rec['conv2'][1]}")
    print(f"Pool   in/out             : {rec['pool'][0]} -> {rec['pool'][1]}")
    print(f"FC1    in/out             : {rec['fc1'][0]} -> {rec['fc1'][1]}")
    print(f"Output in/out             : {rec['out'][0]} -> {rec['out'][1]}")
    print(f"Final logits shape        : {tuple(logits.shape)}")
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters      : {n_par:,}")

    # convolution must slide over TIME, not over features
    assert rec["conv1"][0][1] == n_features, "conv1 input channels must be n_features"
    assert rec["conv1"][0][2] == seq_len, "conv1 must receive the time axis as its length"
    assert rec["conv2"][1][2] == seq_len, "'same' padding must preserve the time axis"
    assert rec["pool"][1][1:] == (d["cnn"]["channels"][1], 1), "global pooling must collapse time"

    # gradients must actually reach the convolution kernels
    loss = nn.BCEWithLogitsLoss()(logits, torch.zeros(len(x)))
    loss.backward()
    g1 = model.conv1.weight.grad
    g2 = model.conv2.weight.grad
    assert g1 is not None and torch.isfinite(g1).all() and g1.abs().sum() > 0, "conv1 got no gradient"
    assert g2 is not None and torch.isfinite(g2).all() and g2.abs().sum() > 0, "conv2 got no gradient"
    print(f"conv1.weight grad |sum|   : {float(g1.abs().sum()):.6f}  (non-zero OK)")
    print(f"conv2.weight grad |sum|   : {float(g2.abs().sum()):.6f}  (non-zero OK)")

    # permuting the time axis must change the output of a *temporal* model
    model.eval()
    with torch.no_grad():
        a = model(x)
        b = model(torch.flip(x, dims=[1]))
    assert not torch.allclose(a, b, atol=1e-6), "output is invariant to time order -> not temporal"
    print(f"Time-order sensitivity    : max|f(x)-f(reverse x)| = {float((a-b).abs().max()):.6f}  (>0 OK)")
    print("RESULT: PASS — real nn.Conv1d over the time axis, not an MLP.\n")
    return True


def check_lstm(cfg, n_features, seq_len, batch):
    print("=" * 72)
    print("LSTM STRUCTURAL SANITY CHECK")
    print("=" * 72)
    d = cfg["models"]["deep"]
    model = LSTMClassifier(
        n_features=n_features,
        hidden_size=d["lstm"]["hidden_size"],
        num_layers=d["lstm"]["num_layers"],
        dropout=d["dropout"],
        fc=tuple(d["lstm"]["fc"]),
    )
    assert_is_lstm(model)
    print(model)

    rec, handles = _hook_shapes(model, ["lstm", "fc1", "out"])
    x = torch.from_numpy(batch.astype(np.float32))
    print(f"\nInput tensor shape        : {tuple(x.shape)}  (batch, seq_len, n_features)")
    logits = model(x)
    for h in handles:
        h.remove()

    with torch.no_grad():
        seq_out, (h_n, c_n) = model.lstm(x)
    print(f"LSTM input               : {rec['lstm'][0]}")
    print(f"LSTM sequence output     : {tuple(seq_out.shape)}  (batch, seq_len, hidden)")
    print(f"LSTM h_n                 : {tuple(h_n.shape)}  (num_layers, batch, hidden)")
    print(f"LSTM c_n                 : {tuple(c_n.shape)}")
    print(f"FC1    in/out            : {rec['fc1'][0]} -> {rec['fc1'][1]}")
    print(f"Output in/out            : {rec['out'][0]} -> {rec['out'][1]}")
    print(f"Final logits shape       : {tuple(logits.shape)}")
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters     : {n_par:,}")

    assert rec["lstm"][0][1] == seq_len and rec["lstm"][0][2] == n_features, \
        "LSTM must receive (batch, seq_len, n_features)"
    assert tuple(h_n.shape) == (d["lstm"]["num_layers"], len(x), d["lstm"]["hidden_size"])

    loss = nn.BCEWithLogitsLoss()(logits, torch.zeros(len(x)))
    loss.backward()
    for pname in ["weight_ih_l0", "weight_hh_l0"]:
        g = getattr(model.lstm, pname).grad
        assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0, f"{pname} got no gradient"
        print(f"lstm.{pname} grad |sum| : {float(g.abs().sum()):.6f}  (non-zero OK)")

    model.eval()
    with torch.no_grad():
        a = model(x)
        b = model(torch.flip(x, dims=[1]))
    assert not torch.allclose(a, b, atol=1e-6), "output is invariant to time order -> not recurrent"
    print(f"Time-order sensitivity   : max|f(x)-f(reverse x)| = {float((a-b).abs().max()):.6f}  (>0 OK)")
    print("RESULT: PASS — real nn.LSTM consuming the last hidden state, not an MLP.\n")
    return True


def main() -> int:
    cfg = yaml.safe_load(open(os.path.join(ROOT, "configs", "default.yaml"), encoding="utf-8"))
    set_seed(cfg["reproducibility"]["seed"], cfg["reproducibility"]["deterministic"])

    seq_len = cfg["models"]["deep"]["seq_len"]
    n_features = len(cfg["features"]["dl"])

    print(f"\nconfig: seq_len={seq_len}, n_dl_features={n_features}, torch={torch.__version__}\n")

    print("### (a) synthetic batch ###")
    synth = np.random.randn(8, seq_len, n_features).astype(np.float32)
    check_cnn(cfg, n_features, seq_len, synth)
    check_lstm(cfg, n_features, seq_len, synth)

    real_path = os.path.join(ROOT, "results", "logs", "sanity_real_batch.npy")
    if os.path.exists(real_path):
        print("### (b) real ASOS batch ###")
        real = np.load(real_path)
        assert real.shape[1:] == (seq_len, n_features), \
            f"real batch shape {real.shape} does not match config ({seq_len}, {n_features})"
        check_cnn(cfg, n_features, seq_len, real[:8])
        check_lstm(cfg, n_features, seq_len, real[:8])
    else:
        print(f"### (b) real ASOS batch SKIPPED — run scripts/prepare_data.py first "
              f"(expected {real_path}) ###")
        return 2

    print("ALL SANITY CHECKS PASSED — experiments may proceed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
