# MAMBO V3 notices

## Model weights

The MAMBO V3 trained weights, in PyTorch and ONNX form, are prepared for release
under **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International**
(`CC-BY-NC-SA-4.0`). The full terms are in `MODEL_LICENSE.txt` and at
<https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.en>.

Attribute **MAMBO V3 / mini_trainer project**, link to
<https://github.com/asgersvenning/mini_trainer/releases/tag/MAMBO_v3> and the
license, and identify modifications when sharing. The license permits
non-commercial use and sharing; distributed adaptations must retain the same
license elements under the license's ShareAlike conditions. It does not require
private adaptations to be published. Commercial use requires separate permission
from the relevant rights holders. The license grants only rights the licensor
has authority to grant; it does not replace third-party rights or notices.

## Initialization and attribution

The retained UCloud preparation code creates `initial_seed42.pt` before training:
it initializes the torchvision EfficientNetV2-S backbone with `DEFAULT` pretrained
weights and a new normalized hierarchical classification head using seed 42.
Torchvision documents this default as `EfficientNet_V2_S_Weights.IMAGENET1K_V1`,
trained on ImageNet-1K. Production then loads the saved starting checkpoint with
`pretrained=false` to avoid loading the upstream weights again.

This lineage is reconstructed from preparation source, not verified against the
original initialization file: its bytes/hash and the run-specific preparation
manifest were not retained in the release archive. `MODEL_PROVENANCE.toml` records
the source revision and this limitation. The deployed trained checkpoint is
independently identified by its SHA-256; inference does not require the initial file.

Acknowledgements: the TorchVision maintainers and contributors, ImageNet, and
Mingxing Tan and Quoc V. Le for EfficientNetV2. Upstream references:

- [TorchVision EfficientNetV2-S weights](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_v2_s.html)
- [EfficientNetV2 paper](https://arxiv.org/abs/2104.00298)
- [TorchVision model terms](https://github.com/pytorch/vision#pre-trained-model-license)

TorchVision notes that pretrained models may have terms derived from their training
data. Its software license alone is not a blanket license for pretrained weights
or photographs. The MAMBO license does not relicense those upstream materials.

## Repository code and runtime dependencies

The deployment adapter and mini_trainer code remain **MIT-licensed** (`CODE_LICENSE`).
The model-weight license does not relicense independently written application code.
PyTorch/torchvision, ONNX Runtime, NumPy and Pillow are installed separately by the
application's package manager, not vendored in the model bundle. Their distributions
carry their own licenses and notices. CUDA/cuDNN components have separate terms.
Keep dependency notices when redistributing an environment or container.

## Dataset, taxonomy and photographs

Global-lepi metadata and GBIF taxon identifiers informed training and regional
presets. Training/evaluation photographs and the Flemming dataset are not included
in this deployment release. Their original source permissions remain separate;
this release grants no permission to redistribute those datasets or images.
Preset files contain taxon IDs and filter definitions, not photographs. Retained
private evaluation inputs must remain outside publication assets.
