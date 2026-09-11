# Mini trainer

[![Python version](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://github.com/asgersvenning/mini_trainer/actions/workflows/ci.yml/badge.svg)](https://github.com/asgersvenning/mini_trainer/actions)
[![codecov](https://codecov.io/github/asgersvenning/mini_trainer/graph/badge.svg?token=3BCL6NH5GC)](https://codecov.io/github/asgersvenning/mini_trainer)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

---

This is an attempt to create a minimal extendable framework for development and research on classification models.

All code in `mini_trainer` should follow the following core principles:

* Keep core dependencies minimal (see `pyproject.toml` for the current set); third-party integrations should remain optional.
* The required portion of any API should be as minimal as possible (i.e. to train a model we only require `mt_train -i <TRAINING_DATA>`)
* All hyperparameters and system configuration should have smart defaults that are as general as possible
* All functionality should be extendable to custom model architectures, loss functions, training regimes, data formats etc.

## Explore model prototypes

With the optional `explorer` dependencies installed, run `mt_explore weights.pt`
to open the interactive prototype explorer, or `mt_explore` to choose weights in
the browser. See the [prototype explorer guide](docs/prototype-explorer.md) for
installation, portable exports, supported checkpoints, and interpretation.

# Installation

We recommend using `uv` for package and environment management.

> See [Install uv](https://docs.astral.sh/uv/getting-started/installation/) for instructions.

## PyPi

```bash
# Recommended installation (includes logging, visualization, and optional utilities)
uv pip install "mini_trainer[recommended]" --torch-backend=auto
# or standard pip
pip install "mini_trainer[recommended]"

# Installation with all features (timm, transformers, BioCLIP, etc.)
uv pip install "mini_trainer[all]" --torch-backend=auto
# or standard pip
pip install "mini_trainer[all]"

# Minimal installation (core training & inference loop only)
uv pip install mini_trainer --torch-backend=auto
# or standard pip
pip install mini_trainer
```

## Local Installation

```bash
git clone ssh://git@github.com:asgersvenning/mini_trainer.git
cd mini_trainer

# Sync with recommended extras:
uv sync --extra recommended --extra [cpu/cu126/cu130/cu132]

# Or sync with all features (timm, transformers, BioCLIP):
uv sync --extra all --extra [cpu/cu126/cu130/cu132]

source .venv/bin/activate
```

> [!TIP]
> We highly recommend installing `torch` and `torchvision` with native CUDA support via either `uv sync ... --extra [cpu/cu126/cu130/cu132]` or `uv pip install ... --torch-backend=auto`, **and** crucially running scripts or tools associated with your `uv` virtual environment by **activating the venv:**
> ```bash
> source .venv/bin/activate
> ```
> Using `uv run ...` is likely to automatically install CUDA-incompatible wheels. If you really want to use `uv run`, we suggest using the `--no-sync` flag every time.
> Note that if you are *"lucky"* you might have the default CUDA version on your system, meaning that `uv run` might in fact use the correct wheels. This is, however, not guaranteed.

## Data loading on shared machines

Automatic DataLoader worker selection uses the CPUs available to the process when
the OS exposes that information, including CPU affinity. It reserves four CPUs,
rounds down to an even worker count, and caps workers at 16 for training and 32 for
prediction. For example, an 8-CPU affinity limit selects four workers, even on a
larger shared machine. Four or fewer available CPUs selects zero workers.

Set `--num_workers 2` to choose a count explicitly, or `--num_workers 0` to load in
the main process. CUDA-cached datasets always use zero DataLoader workers.
RAM-cache preloading uses a separate thread pool that also respects process CPU
availability, reserves two CPUs, and uses between 1 and 128 threads.

Affinity does not describe all container CPU quotas or competition from other jobs.
If an allocation shares an unrestricted CPU set, choose a conservative explicit
worker count per training process. `--num_workers` does not control RAM-cache
preloading; use uncached loading when you need that explicit bound.

## Weights & Biases Integration

`mini_trainer` supports logging your training runs, including metrics, confusion matrices, and the probabilistic dendrogram, directly to [Weights & Biases](https://wandb.ai). 

Class-matrix diagnostics use [log-domain probabilities](docs/prototype-diagnostics.md)
to retain small tails during evaluation logging.

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
The quantization branch has a focused [bottleneck and handoff roadmap](docs/quantization-roadmap.md).

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
