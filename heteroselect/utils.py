"""Cross-module utilities: device selection and global seeding."""

from __future__ import annotations

import random

import numpy as np
import torch


def get_device() -> torch.device:
    """Return the CUDA device if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int) -> np.random.RandomState:
    """Seed Python, NumPy, and PyTorch RNGs and return a NumPy RandomState.

    The returned ``np.random.RandomState`` is used by the trainer for
    selection sampling and bandwidth draws so that the per-round
    randomness is reproducible independently of any global state.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return np.random.RandomState(seed)
