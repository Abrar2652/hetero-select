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
    model.train(False)
    correct = total = 0
    for x, y in loader:
        pred = model(x.to(device)).argmax(1)
        correct += (pred == y.to(device)).sum().item()
        total += len(y)
    return correct / total
