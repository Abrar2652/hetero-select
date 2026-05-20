"""Client-side local training (FedProx with score-adaptive learning rate)."""

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
    """Run ``H`` FedProx steps starting from ``global_model``.

    Parameters
    ----------
    global_model : nn.Module
        Server-side model at the start of this round. ``global_model`` is
        deep-copied before any local optimization step so the server state
        is never modified in place.
    loader : DataLoader
        Local training data for this client.
    H : int
        Number of local steps.
    lr_k : float
        Per-client adaptive learning rate
        ``lr_k = lr_base(t) * (1 + S_k)``, capped at ``fl['lr_scale_cap']``.
    mu : float
        FedProx proximal coefficient.

    Returns
    -------
    delta : torch.Tensor
        Flattened model delta ``w_local - w_global``.
    local_model : nn.Module
        Locally updated model (used downstream for the Hutchinson HVP).
    """
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
