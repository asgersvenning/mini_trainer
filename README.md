[![codecov](https://codecov.io/github/asgersvenning/mini_trainer/graph/badge.svg?token=3BCL6NH5GC)](https://codecov.io/github/asgersvenning/mini_trainer)
<!-- # Mini trainer
This is an attempt to create a minimal extendable framework for development and research on classification models.

All code in `mini_trainer` should follow the following core principles:

* There should be **NO** additional dependencies beyond core `Python`, `PyTorch` (`torch`, `torchvision`, etc.), `matplotlib` and `tqdm`.
* The required portion of any API should be as minimal as possible (i.e. to train a model we only require `python train.py -i <TRAINING_DATA>`)
* All hyperparameters and system configuration should have smart defaults that are as general as possible
* All functionality should be extendable to custom model architectures, loss functions, training regimes, data formats etc. -->

# Installation
As we currently aren't distributing the `mini_trainer` module on PyPi or conda-forge, the installation unfortunately requires two steps:
```sh
[conda/mamba/micromamba] install -f conda.yaml
[conda/mamba/micromamba] activate mini_trainer
pip install -e .
```

# Usage
After installation predictions can be made simply with:
```sh
python predict.py -i [DIRECTORY/FILE]
```
The most up-to-date model will be automatically downloaded and used.

## Model Details
The current model is a global hierarchical Lepidoptera model based on **EfficientNetV2 (Medium)**, originally trained on approximately **12,500 species** and **6,000,000 images**.

To provide a more local model for Northern Europe, the model output layer has been constrained to the ~2,000 species relevant to the region. To be included in this specific version, a species must appear more than 25 times in the training data in at least one of the following countries:

* Denmark (DK)
* United Kingdom (UK)
* Germany (DE)
* Poland (PL)
* Sweden (SE)
* Norway (NO)
* Netherlands (NL)
* Finland (FI)
* Estonia (EE)
* Latvia (LV)
* Lithuania (LT)

The original full list of species (by GBIF ID) can be found in [full.txt](full.txt), while the reduced list of local species for this model can be found in [reduced.txt](reduced.txt).

<!-- ## Acknowledgements
This repository repository draws inspiration from https://github.com/pytorch/vision/tree/main/references/classification.

## Contribution
Feel free to contribute, but here are a few tips:

* Follow the installation guide to setup a proper dev environment.
* Setup linting via `uv ruff` with `uv add --dev ruff` and then do `uv run ruff check mini_trainer`.
* Please avoid adding new dependencies 🙂 -->
