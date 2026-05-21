#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np


def _load_summaries(results_dir: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for entry in sorted(os.listdir(results_dir)):
        if not entry.endswith(".json") or entry == "summary.json":
            continue
        with open(os.path.join(results_dir, entry), "r") as f:
            blob = json.load(f)
        cfg = blob["config"]
        row: Dict[str, Any] = dict(
            dataset=cfg["dataset"], psi=cfg["psi"],
            mu=cfg.get("mu", blob["fl_config"].get("mu", 0.1)),
            seed=cfg["seed"], variant=cfg.get("variant", "adaptive"),
        )
        row.update(blob["summary"])
        rows.append(row)
    return rows


def _ms(vals: List[float]) -> str:
    v = [x for x in vals if x is not None]
    if not v:
        return "    N/A          "
    if len(v) == 1:
        return f"{v[0]:8.4f}        "
    return f"{np.mean(v):8.4f}±{np.std(v):.4f}"


def main() -> None:
    p = argparse.ArgumentParser(description="Print results table from JSON logs.")
    p.add_argument("results_dir", help="Directory containing per-run JSON logs.")
    args = p.parse_args()

    rows = _load_summaries(args.results_dir)
    if not rows:
        print(f"No JSON logs found in {args.results_dir!r}.")
        return

    groups: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        key = (r["dataset"], r["psi"], r["mu"], r["variant"])
        groups[key].append(r)

    print("\n" + "═" * 110)
    print("HETERO-SELECT  —  RESULTS")
    print("═" * 110)
    print(
        f"{'Dataset':10} {'ψ':5} {'μ':5} {'Variant':10} "
        f"{'Peak%':16} {'Final%':16} {'Drop%':10} "
        f"{'Traffic(MB)':16} {'Time(s)':14}"
    )
    print("─" * 110)
    for (ds, psi, mu, var), runs in sorted(groups.items()):
        print(
            f"{ds:10} {str(psi):<5} {mu:<5} {var:10} "
            f"{_ms([r['peak_acc'] for r in runs]):16} "
            f"{_ms([r['final_acc'] for r in runs]):16} "
            f"{_ms([r['stability_drop'] for r in runs]):10} "
            f"{_ms([r['traffic_to_target_mb'] for r in runs]):16} "
            f"{_ms([r['time_to_target_s']     for r in runs]):14}"
        )
    print("─" * 110)
    print("FedCG simulation targets (Fig. 3, 100 clients):")
    print("  CIFAR-10:  Acc=70%  Time=5170s   Traffic≈2480 MB")
    print("  CIFAR-100: Acc=54%  Time=10069s  Traffic=8402 MB")


if __name__ == "__main__":
    main()
