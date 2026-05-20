#!/usr/bin/env python3
"""Regenerate every supplementary figure from JSON logs.

This is a thin wrapper around the original ``make_extra_figures.py`` so
that the figure-building logic still lives in one place but can be
called from inside the public repo as

    python scripts/make_figures.py --exp results/experiment \
                                   --abl results/ablation  \
                                   --out figs

All figures are written as PNG into ``--out``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


# Plot style
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 160,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": ":",
})


CIFAR10_SEEDS = (42, 43, 44)
CIFAR100_SEEDS = (42, 43, 44)


def _load(path: Path) -> dict:
    with path.open("r") as f:
        return json.load(f)


def _series(run: dict, key: str) -> np.ndarray:
    return np.asarray([r[key] for r in run["rounds"]], dtype=float)


def fig_score_components(exp: Path, out: Path) -> None:
    run = _load(exp / "cifar10_psi0p4_mu0p1_seed42_adaptive.json")
    rounds = _series(run, "round")
    V = _series(run, "score_V_mean")
    D = _series(run, "score_D_mean")
    F = _series(run, "score_F_mean")
    St = _series(run, "score_St_mean")

    fig, ax = plt.subplots(figsize=(7.0, 3.3))
    ax.plot(rounds, V,  label=r"$V'_k$ loss",       lw=1.7)
    ax.plot(rounds, D,  label=r"$D_k$ diversity",   lw=1.7)
    ax.plot(rounds, F,  label=r"$F'_k$ fairness",   lw=1.7)
    ax.plot(rounds, St, label=r"$St'_k$ staleness", lw=1.7)
    ax.set_xlabel("Round")
    ax.set_ylabel("Mean score component")
    ax.set_ylim(-0.1, 1.05)
    ax.legend(loc="upper center", ncol=4, frameon=False,
              bbox_to_anchor=(0.5, 1.18), fontsize=9.5)
    ax.set_title("Composite-score components over a CIFAR-10 run "
                 r"($\psi=0.4$, seed 42)")
    fig.savefig(out / "fig5_score_components.png")
    plt.close(fig)


def fig_seed_bands(exp: Path, out: Path) -> None:
    def stack(name_fmt: str, seeds: Iterable[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        accs = []
        rounds = None
        for s in seeds:
            run = _load(exp / name_fmt.format(s=s))
            if rounds is None:
                rounds = _series(run, "round")
            accs.append(_series(run, "test_acc"))
        arr = np.stack(accs, axis=0)
        return rounds, arr.mean(axis=0), arr.std(axis=0)

    r10, m10, s10 = stack(
        "cifar10_psi0p4_mu0p1_seed{s}_adaptive.json", CIFAR10_SEEDS
    )
    r100, m100, s100 = stack(
        "cifar100_psi40_mu0p1_seed{s}_adaptive.json", CIFAR100_SEEDS
    )

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.3))
    for ax, r, mu, sd, target, title in [
        (axes[0], r10,  m10,  s10,  0.70, "CIFAR-10 / AlexNet"),
        (axes[1], r100, m100, s100, 0.54, "CIFAR-100 / ResNet9"),
    ]:
        ax.plot(r, mu, color="tab:blue", lw=2.0, label="Mean over seeds")
        ax.fill_between(r, mu - sd, mu + sd, color="tab:blue", alpha=0.18,
                        label=r"$\pm 1\sigma$ band")
        ax.axhline(target, color="tab:red", lw=1.0, ls="--", label="Target")
        ax.set_xlabel("Round")
        ax.set_ylabel("Test accuracy")
        ax.set_title(title)
        ax.legend(loc="lower right", fontsize=9, frameon=False)
    fig.savefig(out / "fig7_seed_bands.png")
    plt.close(fig)


def fig_stress_curve(abl: Path, exp: Path, out: Path) -> None:
    normal = _load(exp / "cifar10_psi0p4_mu0p1_seed42_adaptive.json")
    stress = _load(abl / "cifar10_psi0p4_mu0p1_seed42_stress.json")

    rn, an = _series(normal, "round"), _series(normal, "test_acc")
    rs, as_ = _series(stress, "round"), _series(stress, "test_acc")
    tn, ts = _series(normal, "cum_traffic_mb"), _series(stress, "cum_traffic_mb")

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.3))
    axes[0].plot(rn, an, label="Normal bandwidth", lw=1.9)
    axes[0].plot(rs, as_, label="Inverted coupling", lw=1.9, ls="--")
    axes[0].axhline(0.70, color="tab:red", lw=1.0, ls=":")
    axes[0].set_xlabel("Round")
    axes[0].set_ylabel("Test accuracy")
    axes[0].set_title("Accuracy vs. round")
    axes[0].legend(loc="lower right", fontsize=9, frameon=False)

    axes[1].plot(tn, an, label="Normal bandwidth", lw=1.9)
    axes[1].plot(ts, as_, label="Inverted coupling", lw=1.9, ls="--")
    axes[1].axhline(0.70, color="tab:red", lw=1.0, ls=":")
    axes[1].set_xlabel("Cumulative uplink traffic (MB)")
    axes[1].set_ylabel("Test accuracy")
    axes[1].set_title("Accuracy vs. traffic")
    axes[1].legend(loc="lower right", fontsize=9, frameon=False)

    fig.savefig(out / "fig6_stress_curve.png")
    plt.close(fig)


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--exp", type=Path, default=Path("results/experiment"),
                   help="Directory with the main experiment JSON logs.")
    p.add_argument("--abl", type=Path, default=Path("results/ablation"),
                   help="Directory with the ablation JSON logs.")
    p.add_argument("--out", type=Path, default=Path("figs"),
                   help="Output directory for the PNGs.")
    return p.parse_args()


def main() -> None:
    args = _parse()
    args.out.mkdir(parents=True, exist_ok=True)

    fig_score_components(args.exp, args.out)
    fig_seed_bands(args.exp, args.out)
    fig_stress_curve(args.abl, args.exp, args.out)

    print(f"✓  Figures written to {args.out.resolve()}")


if __name__ == "__main__":
    main()
