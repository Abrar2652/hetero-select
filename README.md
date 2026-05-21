# HeteRo-Select

Official PyTorch implementation of **HeteRo-Select: Informativeness-Aware Client Selection and Gradient Compression for Communication-Efficient Federated Learning** (Masud, Jahin, & Hasan).

HeteRo-Select drives client selection, compression ratio, local learning rate, and server aggregation from one normalized informativeness score; bandwidth is a hard ceiling only.

## Requirements

- Python 3.10+
- PyTorch 2.x (CUDA recommended)

```bash
pip install -e .
# or: pip install -r requirements.txt
```

## Quick start

```bash
# smoke test (5 rounds)
python scripts/train.py --dataset cifar10 --psi 0.4 --seed 42 --rounds 5

# primary CIFAR-10 run (100 rounds)
python scripts/train.py --dataset cifar10 --psi 0.4 --seed 42
```

Logs are written to `results/<dataset>_psi<psi>_mu<mu>_seed<seed>_<variant>.json`.

## Datasets

| Dataset | Train / test | Classes | Source |
|---------|--------------|---------|--------|
| CIFAR-10 | 50k / 10k | 10 | [Krizhevsky](https://www.cs.toronto.edu/~kriz/cifar.html) via `torchvision` |
| CIFAR-100 | 50k / 10k | 100 | same |

Preprocessing: random crop (pad 4), horizontal flip, dataset normalization (see `heteroselect/data.py`). No extra cleaning or exclusions.

## Computing environment

- Linux, single NVIDIA GPU, CUDA-enabled PyTorch 2.1+
- Python 3.10–3.11, `torchvision` 0.16+
- Typical wall time: ~10–15 min/seed (CIFAR-10, 100 rounds), ~25–35 min/seed (CIFAR-100)

## Hyperparameters

Primary run uses fixed values in `configs/default.yaml` (no validation search): theta_avg=0.20, mu=0.1, psi=0.4 (CIFAR-10) or FedCG 40-class-missing split (CIFAR-100), M=10, H=50, T=100, seeds 42–44 for main table. Ablations vary one knob at a time (`--variant uniform|stress`, `--mu`, `--psi`).

## Reproduce paper experiments

| Paper result | Command | Output log | Primary metric |
|--------------|---------|------------|----------------|
| Table I, CIFAR-10 (3 seeds) | `python scripts/train.py --dataset cifar10 --psi 0.4 --seed 42` (repeat seeds 43, 44) | `results/cifar10_psi0p4_mu0p1_seed*_adaptive.json` | peak test acc., rounds/time/traffic to 70% |
| Table I, CIFAR-100 | `python scripts/train.py --dataset cifar100 --psi 40 --seed 42` (repeat 43, 44) | `results/cifar100_psi40_mu0p1_seed*_adaptive.json` | peak acc., efficiency to 45% |
| Abl. uniform compression | `python scripts/train.py --dataset cifar10 --variant uniform --seed 42` | `*_uniform.json` | peak acc. vs adaptive |
| Abl. mu | `python scripts/train.py --dataset cifar10 --mu 0.01 --seed 42` (repeat for 0, 0.1, 0.5) | `*_mu*.json` | peak acc. |
| Abl. psi | `python scripts/train.py --dataset cifar10 --psi 0.2 --seed 42` (repeat for 0.4, 0.6) | `*_psi*.json` | peak acc. |
| Stress (inverted coupling) | `python scripts/train.py --dataset cifar10 --variant stress --seed 42` | `*_stress.json` | peak acc., time/traffic |

Batch all runs:

```bash
python scripts/train.py --grid main      # 3-seed CIFAR-10 / CIFAR-100
python scripts/train.py --grid ablation  # uniform, mu, psi, stress
python scripts/train.py --grid all --zip
```

Regenerate tables and figures from JSON logs:

```bash
python scripts/print_table.py results/
python scripts/make_figures.py --exp results/ --abl results/ --out figs/
```

## Project layout

```
heteroselect/   # core package (config, data, scoring, compression, trainer, ...)
scripts/        # train.py, print_table.py, make_figures.py
configs/        # default.yaml
tests/          # smoke test
```

Key defaults live in `heteroselect.config.DEFAULT_FL_CONFIG` (or `configs/default.yaml`).

## Citation

```bibtex
@article{masud2026heteroselect,
  title   = {HeteRo-Select: Informativeness-Aware Client Selection and Gradient
             Compression for Communication-Efficient Federated Learning},
  author  = {Masud, M. A. and Jahin, Md Abrar and Hasan, M.},
  journal = {Manuscript under review},
  year    = {2026}
}
```

## License

MIT — see [LICENSE](LICENSE).
