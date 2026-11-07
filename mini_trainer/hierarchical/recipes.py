"""Training recipes assembled from opt-in mini_trainer components."""

import torch
import torchvision.transforms.v2 as tt

from mini_trainer.data import RandomAffineResize, split_domain_augmentation

from .integration import HierarchicalBuilder


class LepinetBuilder(HierarchicalBuilder):
    """The lepinet recipe (fastai, EfficientNetV2-S baseline) built from opt-in components.

    - Data: square crops read at ``READ_SIZE`` (random in training, centred otherwise), square-root
      class oversampling with replacement, and rows outside the class specification dropped (for a
      class-count floor such as ``spec_model_dataloader_kwargs: {min_count: 50}``).
    - Augmentation: per-image dihedral flip, rotation and zoom sampled straight to the model size;
      lepinet's camera-trap transforms are opt-in (``domain_aug``).
    - Head: cosine head with trainable per-class scale and bias, run in float32 without TF32, and the
      spherical-repulsion step lepinet used.

    Explicit keyword arguments override these defaults. Optimiser, schedule, loss and AMP settings are
    ordinary builder keywords; ``dev/lepinet/`` holds complete presets and the parity protocol.
    """

    READ_SIZE = 460

    @staticmethod
    def build_model(*args, float32_matmul_precision: str = "highest", **kwargs):
        # lepinet's float32 head ran without TF32; mini_trainer's device setup enables it.
        torch.set_float32_matmul_precision(float32_matmul_precision)
        defaults = {
            "resize_method": "crop",
            "read_size": LepinetBuilder.READ_SIZE,
            "trainable_scale": True,
            "trainable_bias": True,
            "float32_head": True,
            "legacy_spherical_init": True,
        }
        return HierarchicalBuilder.build_model(*args, **{**defaults, **kwargs})

    @staticmethod
    def build_dataloader(*args, **kwargs):
        defaults = {
            "resize_method": "crop",
            "read_size": LepinetBuilder.READ_SIZE,
            "class_sampling_power": 0.5,
            "drop_unknown_classes": True,
        }
        return HierarchicalBuilder.build_dataloader(*args, **{**defaults, **kwargs})

    @staticmethod
    def build_augmentation(
        dtype: torch.dtype,
        size: int = 256,
        dihedral: bool = True,
        flip_p: float = 0.5,
        max_rotate: float = 15.0,
        min_zoom: float = 1.0,
        max_zoom: float = 1.1,
        affine_p: float = 0.75,
        domain_aug: str | None = None,
    ):
        """Return the per-image affine augmentation; ``size`` must equal the model input size.

        ``domain_aug`` adds opt-in transforms by name, ``+``-joined (e.g. ``"geometric+trap"``). lepinet's
        camera-trap presets (``"trap"``: motion blur, low light, JPEG-like quantization; ``"blur"`` and
        ``"lowlight"`` alone) are drawn once per batch after the affine and before normalization, as lepinet
        did. ``"geometric"`` (per-image rotation up to 45 degrees on an expanded canvas, then 0-25 % edge
        padding, as ``mambo_deploy``'s TTA views) acts on the read images before the affine. ``None`` leaves
        the recipe unchanged. (TrivialAugmentWide is a loader option: ``dataloader_builder_kwargs``
        ``sample_augmentation: trivialwide``.)
        """
        before, after = split_domain_augmentation(domain_aug)
        return tt.Compose(
            [
                *before,
                RandomAffineResize(
                    size=size,
                    dihedral=dihedral,
                    flip_p=flip_p,
                    max_rotate=max_rotate,
                    min_zoom=min_zoom,
                    max_zoom=max_zoom,
                    affine_p=affine_p,
                ),
                *after,
            ]
        )
