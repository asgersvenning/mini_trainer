# Mini trainer

[![Python version](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://github.com/asgersvenning/mini_trainer/actions/workflows/ci.yml/badge.svg)](https://github.com/asgersvenning/mini_trainer/actions)
[![codecov](https://codecov.io/github/asgersvenning/mini_trainer/graph/badge.svg?token=3BCL6NH5GC)](https://codecov.io/github/asgersvenning/mini_trainer)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

---

This is an attempt to create a minimal extendable framework for development and research on classification models.

All code in `mini_trainer` should follow the following core principles:

* There should be **NO** additional dependencies beyond core `Python`, `PyTorch` (`torch`, `torchvision`, etc.), `matplotlib` and `tqdm`.
* The required portion of any API should be as minimal as possible (i.e. to train a model we only require `mt_train -i <TRAINING_DATA>`)
* All hyperparameters and system configuration should have smart defaults that are as general as possible
* All functionality should be extendable to custom model architectures, loss functions, training regimes, data formats etc.

# Installation

We recommend using `uv` for package and environment management.

> [!NOTE]
> See [Install uv](https://docs.astral.sh/uv/getting-started/installation/) for instructions.

## PyPi

```bash
# install via pip (uv)
uv pip install mini_trainer --torch-backend=auto
# or add to your existing project
uv add mini_trainer
```

## Local Installation

```bash
git clone ssh://git@github.com:asgersvenning/mini_trainer.git
cd mini_trainer
uv sync --extra recommended --extra [cpu/cu126/cu130/cu132]
source .venv/bin/activate
```

> [!TIP]
> We highly recommend installing `torch` and `torchvision` with native CUDA support via either `uv sync ... --extra [cpu/cu126/cu130/cu132]` or `uv pip install ... --torch-backend=auto`, **and** crucially running scripts or tools associated with your `uv` virtual environment by **activating the venv:**
> ```bash
> source .venv/bin/activate
> ```
> Using `uv run ...` is likely to automatically install CUDA-incompatible wheels. If you really want to use `uv run`, we suggest using the `--no-sync` flag every time.
> Note that if you are *"lucky"* you might have the default CUDA version on your system, meaning that `uv run` might in fact use the correct wheels. This is, however, not guaranteed.

## Weights & Biases Integration

`mini_trainer` supports logging your training runs, including metrics, confusion matrices, and the probabilistic dendrogram, directly to [Weights & Biases](https://wandb.ai). 

To use this feature you must install `mini_trainer` with the `recommended` extras. See [Installation](#installation) for more information.

1. **Login to your wandb account**:
   ```bash
   wandb login
   ```
2. **Train with the `--wandb` flag**:
   Simply append the `--wandb` flag to your training command.
   ```bash
   mt_train -i path/to/dataset --wandb
   ```

## Acknowledgements
This repository draws inspiration from https://github.com/pytorch/vision/tree/main/references/classification.

## Contribution
Feel free to contribute, but here are a few tips:

* Follow the installation guide to setup a proper dev environment.
* Setup linting via `ruff`; verify with: `ruff check mini_trainer`.
* Please avoid adding new dependencies 🙂
