from __future__ import annotations

import os
import sys

import torch
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from heteroselect.config import DEFAULT_FL_CONFIG
from heteroselect.data import load_data
from heteroselect.trainer import run_experiment
from heteroselect.utils import seed_everything


def test_two_round_cifar10_smoke() -> None:
    fl = dict(DEFAULT_FL_CONFIG)
    fl["num_rounds"] = 2
    fl["num_clients"] = 8
    fl["clients_per_round"] = 4
    fl["local_steps"] = 2
    fl["eval_batches"] = 1
    fl["batch_size"] = 16

    cfg = dict(dataset="cifar10", psi=0.4, seed=0)
    rng = seed_everything(0)

    train_ds, test_ds = load_data("cifar10")
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)

    result = run_experiment(
        cfg, fl, train_ds, test_loader, rng,
        device=torch.device("cpu"), log_every=99, verbose=False,
    )

    assert "summary" in result
    assert len(result["rounds"]) == 2
    assert 0.0 <= result["summary"]["peak_acc"] <= 1.0


if __name__ == "__main__":
    test_two_round_cifar10_smoke()
    print("OK")
