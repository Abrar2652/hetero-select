from __future__ import annotations

import numpy as np


def sim_round_time(
    theta_k: np.ndarray,
    bw_bps_sel: np.ndarray,
    model_bits: int,
    H: int,
    comp_times_sel: np.ndarray,
) -> float:
    t_com = theta_k * model_bits / bw_bps_sel
    t_cmp = H * comp_times_sel
    return float((t_com + t_cmp).max())


def sim_traffic_mb(theta_k: np.ndarray, model_mb: float) -> float:
    return float((theta_k * model_mb).sum())
