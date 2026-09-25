# Mini trainer

[![Python version](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://github.com/asgersvenning/mini_trainer/actions/workflows/ci.yml/badge.svg)](https://github.com/asgersvenning/mini_trainer/actions)
[![codecov](https://codecov.io/github/asgersvenning/mini_trainer/graph/badge.svg?token=3BCL6NH5GC)](https://codecov.io/github/asgersvenning/mini_trainer)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

---

This is an attempt to create a minimal extendable framework for development and research on classification models.

For the MAMBO model release candidate, see the [local deployment guide](deployment/README.md)
for PyTorch/ONNX inference, regional presets, custom class lists and embeddings.

All code in `mini_trainer` should follow the following core principles:

* Keep core dependencies minimal (see `pyproject.toml` for the current set); third-party integrations should remain optional.
* The required portion of any API should be as minimal as possible (i.e. to train a model we only require `mt_train -i <TRAINING_DATA>`)
* All hyperparameters and system configuration should have smart defaults that are as general as possible
* All functionality should be extendable to custom model architectures, loss functions, training regimes, data formats etc.

## Installation

Use [uv](https://docs.astral.sh/uv/getting-started/installation/) for environment
and package management. Choose a published package or a source checkout.

### PyPI

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install "mini_trainer[recommended]" --torch-backend=auto
```

| Package choice | Includes |
| --- | --- |
| `mini_trainer` | Core training and inference |
| `mini_trainer[recommended]` | Core plus logging, visualization and optional utilities |
| `mini_trainer[all]` | Recommended extras plus notebooks, model backends and ONNX export |

Substitute the desired package in the install command. Standard `pip install` also
works; select its PyTorch CPU/CUDA installation separately for your environment.

### Local installation

Choose one backend: `cpu`, `cu126`, `cu130` or `cu132`. The example selects CUDA 13.0;
change `TORCH_BACKEND` to match your intended environment before synchronizing.

```bash
git clone https://github.com/asgersvenning/mini_trainer.git
cd mini_trainer
TORCH_BACKEND=cu130
uv sync --extra recommended --extra "$TORCH_BACKEND"
source .venv/bin/activate
```

Replace `recommended` with `all` for the additional backends/export tools above.
Activate the environment, use its executables directly, or use `uv run --no-sync`.
An implicit sync can replace the deliberately selected PyTorch backend. Select the
backend explicitly whenever installing or synchronizing dependencies.

## Data loading on shared machines

Defaults use process CPU availability, affinity, visible cgroup quotas and Slurm
allocation limits. Shared resources may still need an explicit per-process budget.
Set `--num_workers N` for loading (`0` runs in the main process), and
`--cache-workers N` for training-cache preparation. CUDA-cached datasets use zero
DataLoader workers. See [automatic budgets](dev/README.md#automatic-cpu-budgets)
for caps and fallback behavior; cache readers and DataLoader workers are separate.

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
* Run `bash dev/check.sh static` for lint, formatting, and architecture checks.
* Run `bash dev/check.sh test` for the test suite; see the [development guide](dev/README.md) for focused checks and environment details.
* Run `bash dev/check-wheel.sh` to validate a minimal wheel installation in a disposable CPU environment.
* Please avoid adding new dependencies 🙂

Repository agents should start with [AGENTS.md](AGENTS.md). Planned improvements and
their acceptance criteria are tracked in the [roadmap](docs/roadmap.md).
Remaining quantization work has a focused [target-qualification roadmap](docs/quantization-roadmap.md).

## ONNX export

Export trained models with `mt_export --weights weights.pt --output exported-model`
after installing the `export` extra and the relevant model backend. The generic
exporter preserves evaluation outputs and verifies ONNX Runtime parity. See the
[export guide](docs/onnx.md) for the Python API, preprocessing contract and coverage.

## Continuous benchmarks

Follow the [benchmark results and coverage](docs/benchmarks.md) and
[continuous run history](https://github.com/asgersvenning/mini_trainer/actions/workflows/benchmarks.yml).
The suite progresses from an exact synthetic oracle to MNIST and hierarchical Blair,
with separate CPU and GPU profiles, visible summaries, and retained reproduction artifacts.

For configured GPU runners, the opt-in [TensorRT deployment workflow](dev/benchmarks/reporting.md#opt-in-target-gpu-workflow)
rebuilds engines on the target and reports paired quality, latency and memory.

## Temporarily unsupported feature

EMA (`--ema` / `ema=True`) is currently nonfunctional: classifier caches populated
by evaluation can break later EMA updates. Leave it disabled. Enabling it emits a
runtime warning; its API and checkpoint compatibility are retained, and repair is
deferred. See [known limitations](docs/roadmap.md).

## INT8 quantization

An opt-in [PTQ and QAT Python API](docs/quantization.md) targets native x86 INT8
inference. This is an initial backend increment; CPU float32 QAT, integer inference
and ordinary AMP are distinct capabilities.

Opt-in [CUDA INT8 training](docs/quantized-training.md) supports Linear weights,
integer forward/backward products, checkpoint restoration and CUDA inference.
See the [validation audit](docs/quantized-training-validation.md) for measured
memory, speed and loading benefits, supported configurations and limitations.
[x86 PTQ/QAT inference](docs/quantization.md) is a separate backend.
