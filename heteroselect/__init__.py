"""HeteRo-Select: informativeness-aware client selection and gradient compression.

A reference implementation of the framework described in

    Masud, Jahin, & Hasan, "HeteRo-Select: Informativeness-Aware Client Selection
    and Gradient Compression for Communication-Efficient Federated Learning."

The package exposes the building blocks of one federated round (selection,
compression, local training, aggregation) as well as the end-to-end
``run_experiment`` driver used to reproduce the paper's results.
"""

from __future__ import annotations

from .config import DEFAULT_FL_CONFIG, load_config
from .trainer import run_experiment

__all__ = ["DEFAULT_FL_CONFIG", "load_config", "run_experiment"]

__version__ = "1.0.0"
