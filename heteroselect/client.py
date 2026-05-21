from __future__ import annotations

import copy
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def fedprox_train(
    global_model: nn.Module,
    loader,
    H: int,
    lr_k: float,
    mu: float,
    device: torch.device,
    grad_clip: float,
) -> Tuple[torch.Tensor, nn.Module]:
    local = copy.deepcopy(global_model).train()
    g_params = [p.data.clone() for p in global_model.parameters()]
    opt = torch.optim.SGD(local.parameters(), lr=lr_k, momentum=0.9)
    step = 0
    while step < H:
        for x, y in loader:
            if step >= H:
                break
            x, y = x.to(device), y.to(device)
            local.zero_grad()
            ce = F.cross_entropy(local(x), y)
            prox = sum(
                ((p - gp) ** 2).sum()
                for p, gp in zip(local.parameters(), g_params)
            )
            (ce + (mu / 2.0) * prox).backward()
            nn.utils.clip_grad_norm_(local.parameters(), grad_clip)
            opt.step()
            step += 1
    delta = torch.cat([
        (p.data - gp).flatten()
        for p, gp in zip(local.parameters(), g_params)
    ])
    return delta, local
