# Mini trainer
This is an attempt to create a minimal extendable framework for development and research on classification models.

All code in `mini_trainer` should follow the following core principles:

* There should be **NO** additional dependencies beyond core `Python`, `PyTorch` (`torch`, `torchvision`, etc.), `matplotlib` and `tqdm`.
* The required portion of any API should be as minimal as possible (i.e. to train a model we only require `python train.py -i <TRAINING_DATA>`)
* All hyperparameters and system configuration should have smart defaults that are as general as possible
* All functionality should be extendable to custom model architectures, loss functions, training regimes, data formats etc.

# Installation
As we currently aren't distributing the `mini_trainer` module on PyPi or conda-forge, the installation unfortunately requires two steps:
```sh
[conda/mamba/micromamba] install -f conda.yaml
pip install -e .
```

## Acknowledgements
This repository repository draws inspiration from https://github.com/pytorch/vision/tree/main/references/classification.