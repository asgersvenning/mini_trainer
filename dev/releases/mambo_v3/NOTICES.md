# MAMBO V3 notices

## Repository code

The deployment adapter and mini_trainer code are distributed under the MIT license
in `CODE_LICENSE`. This notice does not grant rights to independently licensed
runtime libraries, model weights, datasets or photographs.

## Model weights — owner decision outstanding

A license for the trained MAMBO V3 weights has not yet been designated in the
retained release metadata. Public download availability alone is not a license.
The model owner must designate the license and confirm any required attribution
from the initialization checkpoint before public release. The current training
configuration loads an earlier local checkpoint; its original initialization
lineage is not established by that configuration alone.

## Runtime dependencies

PyTorch/torchvision, ONNX Runtime, NumPy, Pillow and optional timm are installed
separately by the application's package manager, not vendored in the model bundle.
Their distributions carry their own license files and notices. CUDA/cuDNN components
are optional third-party dependencies with their own terms. Keep dependency notices
when redistributing an environment or container. The release manifest records the
qualified versions, not a requirement to use one universal environment lock.

## Dataset, taxonomy and photographs

Global-lepi metadata and GBIF taxon identifiers informed training and regional
presets. Training/evaluation photographs and the Flemming dataset are not included
in this deployment release. Their original source permissions remain separate;
this release grants no permission to redistribute those datasets or images.
Preset files contain taxon IDs and filter definitions, not photographs. Retained
private evaluation inputs must remain outside publication assets.
