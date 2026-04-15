# Mini trainer MAMBO release

## Standalone installation

```bash
wget https://github.com/asgersvenning/mini_trainer/archive/refs/tags/MAMBO_v0.zip
unzip MAMBO_v0.zip
cd mini_trainer-MAMBO_v0
# wget -qO- https://astral.sh/uv/install.sh | sh # If you haven't already installed `uv`
uv sync
uv pip install -e .
```

## Dependency installation

```bash
# Assuming you already have a virtual environment
uv pip install https://github.com/asgersvenning/mini_trainer/archive/refs/tags/MAMBO_v0.zip
```

## CLI Usage
See the help string.

```bash
uv run mambo_predict --help
```

## Python API
It's that simple!

```py
from mini_trainer.deploy import Predictor

model = Predictor() # No arguments yields the default canonical model for this release
files = [
    "image_A.jpg",
    "image_B.jpg"
]

print(model(files))
```


## Model Details
The current model is a global hierarchical Lepidoptera model fine-tuned from [**BioCLIP-2**](https://imageomics.github.io/bioclip-2/), originally trained on approximately **12,500 species** and **6,000,000 images**.

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

The original full list of species (by GBIF ID) can be found in [data/full.txt](./data/full.txt), while the reduced list of local species for this model can be found in [data/reduced.txt](./data/reduced.txt).