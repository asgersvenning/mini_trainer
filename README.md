# Mini trainer

[![Python version](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://github.com/asgersvenning/mini_trainer/actions/workflows/ci.yml/badge.svg)](https://github.com/asgersvenning/mini_trainer/actions)
[![codecov](https://codecov.io/github/asgersvenning/mini_trainer/graph/badge.svg?token=3BCL6NH5GC)](https://codecov.io/github/asgersvenning/mini_trainer)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

---

This is an attempt to create a minimal extendable framework for development and research on classification models.

All code in `mini_trainer` should follow the following core principles:

* There should be **NO** additional dependencies beyond core `Python`, `PyTorch` (`torch`, `torchvision`, etc.), `matplotlib` and `tqdm`.
* The required portion of any API should be as minimal as possible (i.e. to train a model we only require `python train.py -i <TRAINING_DATA>`)
* All hyperparameters and system configuration should have smart defaults that are as general as possible
* All functionality should be extendable to custom model architectures, loss functions, training regimes, data formats etc.

# Installation

We recommend using `uv` for package and environment management.

> [!NOTE]
> See [Install uv](https://docs.astral.sh/uv/getting-started/installation/) for instructions.

## PyPi

```bash
# install via pip (uv)
[uv] pip install mini_trainer
# or add to your existing project
uv add mini_trainer
```

## Local Installation

```bash
git clone ssh://git@github.com:asgersvenning/mini_trainer.git
cd mini_trainer
uv sync
```

## Weights & Biases Integration

`mini_trainer` supports logging your training runs, including metrics, confusion matrices, and the probabilistic dendrogram, directly to [Weights & Biases](https://wandb.ai). 

Because we strive for a minimal core, `wandb` is an optional dependency. To use it:

1. **Install with wandb support**:
   ```bash
   uv sync --extra wandb
   # or if using pip directly: pip install "mini_trainer[wandb]"
   ```
2. **Login to your wandb account**:
   ```bash
   uv run wandb login
   ```
3. **Train with the `--wandb` flag**:
   Simply append the `--wandb` flag to your training command.
   ```bash
   uv run mt_train -i path/to/dataset --wandb
   ```

## Acknowledgements
This repository repository draws inspiration from https://github.com/pytorch/vision/tree/main/references/classification.

## Contribution
Feel free to contribute, but here are a few tips:

* Follow the installation guide to setup a proper dev environment.
* Setup linting via `uv ruff` with `uv add --dev ruff` and then do `uv run ruff check mini_trainer`.
* Please avoid adding new dependencies 🙂
