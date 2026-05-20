"""Server-side aggregation, momentum, evaluation and BatchNorm calibration."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn


def score_weighted_aggregate(
    global_model: nn.Module,
    deltas: Sequence[torch.Tensor],
    scores_sel: Sequence[float],
    momentum_buf: torch.Tensor,
    beta_s: float,
) -> torch.Tensor:
    """Score-weighted aggregation feeding a server-side momentum buffer.

    ``delta_scored = sum_k (S_k / sum_j S_j) * delta_k``
    ``M_t = beta_s * M_{t-1} + delta_scored``
    ``w_{t+1} = w_t + M_t``

    All work happens on the same device as ``deltas[0]``; ``global_model``
    is updated in place and the new momentum buffer is returned.
    """
    s = np.asarray(scores_sel, dtype=np.float32)
    s_sum = s.sum()
    if s_sum < 1e-8:
        s = np.ones(len(s), dtype=np.float32) / len(s)
    else:
        s = s / s_sum

    delta_scored = torch.zeros_like(deltas[0])
    for w, d in zip(s, deltas):
        delta_scored.add_(d, alpha=float(w))

    new_buf = beta_s * momentum_buf + delta_scored

    i = 0
    for p in global_model.parameters():
        n = p.numel()
        p.data.add_(new_buf[i:i + n].view(p.shape))
        i += n
    return new_buf


@torch.no_grad()
def calibrate_bn(
    model: nn.Module,
    sel_loaders: Iterable,
    n_batches: int,
    device: torch.device,
) -> None:
    """Refresh BatchNorm running statistics on a few selected-client batches."""
    model.train(True)
    count = 0
    for ld in sel_loaders:
        for x, _ in ld:
            if count >= n_batches:
                break
            model(x.to(device))
            count += 1
    model.train(False)


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device) -> float:
    """Top-1 test accuracy."""
    model.train(False)
    correct = total = 0
    for x, y in loader:
        pred = model(x.to(device)).argmax(1)
        correct += (pred == y.to(device)).sum().item()
        total += len(y)
    return correct / total
