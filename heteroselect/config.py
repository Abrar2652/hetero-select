"""Default hyperparameters and YAML/JSON config loaders.

All defaults exactly match the FedCG simulation protocol (100 clients,
M=10, H=50, T=100, 1-5 Mb/s uplink) and the HeteRo-Select v4 settings
reported in the paper.
"""

from __future__ import annotations

import json
import os
from copy import deepcopy
from typing import Any, Dict


DEFAULT_FL_CONFIG: Dict[str, Any] = dict(
    # FedCG protocol (fixed)
    num_clients       = 100,
    clients_per_round = 10,
    num_rounds        = 100,
    local_steps       = 50,
    mu                = 0.1,
    bw_min_mbps       = 1.0,
    bw_max_mbps       = 5.0,
    comp_min_s        = 0.10,
    comp_max_s        = 0.50,
    target_acc        = {"cifar10": 0.70, "cifar100": 0.54},

    # Compression schedule
    theta_total       = 0.20,
    theta_floor       = 0.08,
    warmup_rounds     = 1,
    alpha_cos         = 0.4,

    # Local training
    local_lr          = 0.05,
    lr_scale_cap      = 0.15,
    grad_clip         = 2.0,
    batch_size        = 32,
    eval_batches      = 8,

    # Scoring weights
    lambda_D          = 0.3,
    lambda_F          = 0.2,
    lambda_St         = 0.2,
    gamma_St          = 0.5,

    # Softmax temperature
    tau_0             = 1.0,

    # Server momentum
    beta_s            = 0.5,

    # Markov-Newton sparsification
    newton_Q          = 3,
    newton_lambda     = 0.2,

    # BatchNorm calibration
    bn_calib_batches  = 20,
)


def _merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    out = deepcopy(base)
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | None = None) -> Dict[str, Any]:
    """Return ``DEFAULT_FL_CONFIG`` merged with overrides from ``path``.

    Supports ``.json`` and ``.yaml`` / ``.yml`` (the latter requires PyYAML).
    ``path=None`` returns an unmodified copy of ``DEFAULT_FL_CONFIG``.
    """
    if path is None:
        return deepcopy(DEFAULT_FL_CONFIG)
    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        with open(path, "r") as f:
            overlay = json.load(f)
    elif ext in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError(
                "PyYAML is required to read YAML configs. Install with 'pip install pyyaml'."
            ) from exc
        with open(path, "r") as f:
            overlay = yaml.safe_load(f) or {}
    else:
        raise ValueError(f"Unknown config extension '{ext}'. Use .json or .yaml.")
    return _merge(DEFAULT_FL_CONFIG, overlay)


def default_experiments() -> list[dict]:
    """The standard experiment grid used to produce the paper's tables.

    Each dict represents one run; ``variant`` defaults to ``'adaptive'``
    when omitted.  Supported variants are ``'adaptive'`` (the full
    HeteRo-Select algorithm), ``'uniform'`` (uniform compression
    ablation) and ``'stress'`` (inverted informativeness--bandwidth
    coupling).
    """
    return [
        dict(dataset="cifar10",  psi=0.4, seed=42),
        dict(dataset="cifar10",  psi=0.4, seed=43),
        dict(dataset="cifar10",  psi=0.4, seed=44),
        dict(dataset="cifar100", psi=40,  seed=42),
        dict(dataset="cifar100", psi=40,  seed=43),
        dict(dataset="cifar100", psi=40,  seed=44),
    ]


def ablation_experiments() -> list[dict]:
    """Every ablation run reported in the paper."""
    return [
        dict(dataset="cifar10", psi=0.4, seed=42, variant="uniform"),
        dict(dataset="cifar10", psi=0.4, seed=42, mu=0.0),
        dict(dataset="cifar10", psi=0.4, seed=42, mu=0.01),
        dict(dataset="cifar10", psi=0.4, seed=42, mu=0.5),
        dict(dataset="cifar10", psi=0.2, seed=42),
        dict(dataset="cifar10", psi=0.6, seed=42),
        dict(dataset="cifar10", psi=0.4, seed=42, variant="stress"),
    ]
