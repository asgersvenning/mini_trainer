# Mini trainer MAMBO release

## Standalone installation

```bash
wget https://github.com/asgersvenning/mini_trainer/archive/refs/tags/MAMBO_v0.zip
unzip MAMBO_v0.zip
cd mini_trainer-MAMBO_v0
# wget -qO- https://astral.sh/uv/install.sh | sh # If you haven't already installed `uv`
uv sync
```

## Dependency installation

```bash
# Assuming you already have a virtual environment
uv add --url https://github.com/asgersvenning/mini_trainer/archive/refs/tags/MAMBO_v0.zip
```

> [!TIP]
> For development, including running in notebooks, install via:
>
> ```bash
> uv sync --dev
> ```

## CLI Usage

See the help string.

```bash
uv run mambo_predict --help
```

> [!TIP]
> The most important CLI argument is `--model`/`-M`, e.g `-M full` (global), `-M europe` (Europe), `-M north_europe` (northern Europe). 

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

To provide a more local model for Europe, the model output layer has been constrained to the ~3,000 species relevant to the region. To be included in this specific version, a species must appear more than 25 times in the training data in Europe.
