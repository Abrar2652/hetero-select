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

## Reproduce paper experiments

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
