#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


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


DATASET_TITLE = {
    "mnist":        "MNIST / LogisticRegression",
    "cifar10":      "CIFAR-10 / AlexNet",
    "cifar100":     "CIFAR-100 / ResNet9",
    "tinyimagenet": "TinyImageNet / ResNet-18",
}

PANEL_ORDER = ["mnist", "cifar10", "cifar100", "tinyimagenet"]

FNAME_RE = re.compile(
    r"^(?P<ds>[a-z0-9]+)_psi(?P<psi>[0-9p]+)_mu(?P<mu>[0-9p]+)_seed(?P<seed>\d+)_(?P<variant>adaptive|uniform|stress)\.json$"
)


def _load(path: Path) -> dict:
    with path.open("r") as f:
        return json.load(f)


def _series(run: dict, key: str) -> np.ndarray:
    return np.asarray([r[key] for r in run["rounds"]], dtype=float)


def _discover(exp: Path) -> Dict[str, List[Path]]:
    """Group adaptive-variant logs by dataset."""
    by_ds: Dict[str, List[Path]] = defaultdict(list)
    for p in sorted(exp.glob("*.json")):
        m = FNAME_RE.match(p.name)
        if not m:
            continue
        if m.group("variant") != "adaptive":
            continue
        by_ds[m.group("ds")].append(p)
    return by_ds


def _stack(runs: List[Path], key: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rounds = None
    cols = []
    for p in runs:
        run = _load(p)
        if rounds is None:
            rounds = _series(run, "round")
        cols.append(_series(run, key))
    arr = np.stack(cols, axis=0)
    return rounds, arr.mean(axis=0), arr.std(axis=0)


def _target_for(ds: str, runs: List[Path]) -> float | None:
    blob = _load(runs[0])
    return blob["fl_config"].get("target_acc", {}).get(ds)


def _annotate_target(ax, run_summaries: List[dict], target: float | None, x_key: str) -> None:
    """Mark target line + first crossing for the (first) run on this panel."""
    if target is None:
        return
    ax.axhline(target, color="tab:red", lw=1.0, ls="--",
               label=f"Target = {target:g}")
    if not run_summaries:
        return
    s = run_summaries[0]
    if not s.get("target_hit"):
        return
    if x_key == "round":
        xv = s.get("rounds_to_target")
    elif x_key == "cum_traffic_mb":
        xv = s.get("traffic_to_target_mb")
    elif x_key == "cum_time_s":
        xv = s.get("time_to_target_s")
    else:
        xv = None
    if xv is None:
        return
    ax.axvline(xv, color="tab:green", lw=1.0, ls=":",
               label=f"Hit @ {xv:g}")


def _panel(ax, runs: List[Path], ds: str, x_key: str, x_label: str) -> bool:
    if not runs:
        return False
    rounds_x, _, _ = _stack(runs, x_key)
    _, mu_acc, sd_acc = _stack(runs, "test_acc")
    ax.plot(rounds_x, mu_acc, color="tab:blue", lw=2.0,
            label=f"Mean over {len(runs)} seed(s)")
    if len(runs) > 1:
        ax.fill_between(rounds_x, mu_acc - sd_acc, mu_acc + sd_acc,
                        color="tab:blue", alpha=0.18, label=r"$\pm 1\sigma$")
    summaries = [_load(p)["summary"] for p in runs]
    _annotate_target(ax, summaries, _target_for(ds, runs), x_key)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Test accuracy")
    ax.set_title(DATASET_TITLE.get(ds, ds))
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    return True


def _grid(by_ds: Dict[str, List[Path]], x_key: str, x_label: str,
          out_path: Path, suptitle: str) -> None:
    datasets = [d for d in PANEL_ORDER if d in by_ds]
    if not datasets:
        return
    n = len(datasets)
    cols = 2 if n > 1 else 1
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols,
                             figsize=(5.2 * cols, 3.4 * rows),
                             squeeze=False)
    flat = axes.ravel()
    for i, ds in enumerate(datasets):
        _panel(flat[i], by_ds[ds], ds, x_key, x_label)
    for j in range(len(datasets), len(flat)):
        flat[j].set_visible(False)
    fig.suptitle(suptitle, y=1.005, fontsize=12.5)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def fig_score_components(by_ds: Dict[str, List[Path]], out: Path) -> None:
    pick = next((by_ds[d][0] for d in PANEL_ORDER if d in by_ds), None)
    if pick is None:
        return
    run = _load(pick)
    rounds = _series(run, "round")
    V = _series(run, "score_V_mean")
    D = _series(run, "score_D_mean")
    F = _series(run, "score_F_mean")
    St = _series(run, "score_St_mean")
    ds = run["config"]["dataset"]

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
    ax.set_title(f"Composite-score components — {DATASET_TITLE.get(ds, ds)} "
                 f"(seed {run['config']['seed']})")
    fig.savefig(out / "fig_score_components.png")
    plt.close(fig)


def fig_stress_curve(by_ds: Dict[str, List[Path]], abl: Path, out: Path) -> None:
    if "cifar10" not in by_ds:
        return
    stress_path = abl / "cifar10_psi0p4_mu0p1_seed42_stress.json"
    if not stress_path.is_file():
        return
    normal = _load(by_ds["cifar10"][0])
    stress = _load(stress_path)

    rn, an = _series(normal, "round"), _series(normal, "test_acc")
    rs, as_ = _series(stress, "round"), _series(stress, "test_acc")
    tn, ts = _series(normal, "cum_traffic_mb"), _series(stress, "cum_traffic_mb")
    target = _target_for("cifar10", by_ds["cifar10"]) or 0.70

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.3))
    axes[0].plot(rn, an, label="Normal bandwidth", lw=1.9)
    axes[0].plot(rs, as_, label="Inverted coupling", lw=1.9, ls="--")
    axes[0].axhline(target, color="tab:red", lw=1.0, ls=":")
    axes[0].set_xlabel("Round"); axes[0].set_ylabel("Test accuracy")
    axes[0].set_title("Accuracy vs. round")
    axes[0].legend(loc="lower right", fontsize=9, frameon=False)
    axes[1].plot(tn, an, label="Normal bandwidth", lw=1.9)
    axes[1].plot(ts, as_, label="Inverted coupling", lw=1.9, ls="--")
    axes[1].axhline(target, color="tab:red", lw=1.0, ls=":")
    axes[1].set_xlabel("Cumulative uplink traffic (MB)")
    axes[1].set_ylabel("Test accuracy"); axes[1].set_title("Accuracy vs. traffic")
    axes[1].legend(loc="lower right", fontsize=9, frameon=False)
    fig.savefig(out / "fig_stress_curve.png")
    plt.close(fig)


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Regenerate figures from JSON logs.")
    p.add_argument("--exp", type=Path, default=Path("results"),
                   help="Directory with the main experiment JSON logs.")
    p.add_argument("--abl", type=Path, default=Path("results"),
                   help="Directory with the ablation JSON logs.")
    p.add_argument("--out", type=Path, default=Path("figs"),
                   help="Output directory for the PNGs.")
    return p.parse_args()


def main() -> None:
    args = _parse()
    args.out.mkdir(parents=True, exist_ok=True)
    by_ds = _discover(args.exp)
    if not by_ds:
        print(f"No adaptive-variant logs found under {args.exp}.")
        return

    print("Found:")
    for d in PANEL_ORDER:
        if d in by_ds:
            print(f"  {d:14s} {len(by_ds[d])} seed(s)")

    _grid(by_ds, "round", "Communication round",
          args.out / "fig_acc_vs_rounds.png",
          "HeteRo-Select: accuracy vs. communication round")
    _grid(by_ds, "cum_traffic_mb", "Cumulative uplink traffic (MB)",
          args.out / "fig_acc_vs_traffic.png",
          "HeteRo-Select: accuracy vs. cumulative traffic (InfoCom-2023 Fig.1d setup)")
    _grid(by_ds, "cum_time_s", "Cumulative simulated time (s)",
          args.out / "fig_acc_vs_time.png",
          "HeteRo-Select: accuracy vs. simulated wall-clock")
    fig_score_components(by_ds, args.out)
    fig_stress_curve(by_ds, args.abl, args.out)

    print(f"\nFigures written to {args.out.resolve()}")
    for f in sorted(args.out.glob("*.png")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
