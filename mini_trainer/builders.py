import json
import os
import warnings
from collections import Counter
from collections.abc import Callable
from tempfile import NamedTemporaryFile
from typing import Any

import numpy as np
import torch
import torchvision.transforms.v2 as tt
from torch import nn
from torch.amp import GradScaler
from torch.optim.swa_utils import get_ema_multi_avg_fn

from mini_trainer.data import (
    SaltAndPepper,
    create_metadata,
    get_dataset_dataloader,
    get_inference_dataloader,
    get_metadata,
    parse_class_spec,
)
from mini_trainer.logging import MultiLogger
from mini_trainer.modeling import Classifier, EMATeacher, backbone, ema_lambda_per_update, last_layer_weights
from mini_trainer.training import EMLACrossEntropy, class_weight_distribution_regularization
from mini_trainer.utils import (
    broadcast_from_master,
    cosine_schedule_with_warmup,
)


class BaseBuilder:
    """The base builder used in the `mini_trainer` training pipeline.

    Subclass this builder and override the relevant functions to alter the `mini_trainer` training pipeline.

    Methods:
        **`spec_model_dataloader`** : Parses or builds the model specification from the class index or input directory.
        **`build_model`**: Builds the model.
        **`build_dataloader`**: Builds the training and validation dataloaders.
        **`build_augmentation`**: Builds the augmentation method(s).
        **`build_optimizer`**: Builds the model optimizer method (e.g. SGD/ADAM).
        **`build_criterion`**: Builds the optimization criterion (i.e. loss function).
        **`build_lr_scheduler`**: Builds the learning rate scheduler (shape only, magnitude defined by optimizer).
        **`build_logger`**: Builds the training diagnostics logger(s).
    """

    def __init__(self):  # noqa: D107
        pass

    @staticmethod
    def build_class_spec(path: str | None = None, dir: str | None = None, species : bool=False, *args, **kwargs):
        """TODO.

        Returns:
            (extra_model_kwargs, extra_dataloader_kwargs):
                Extra keyword arguments for the model and dataloader building functions.
        """
        if args or kwargs:
            raise ValueError(
                f"`BaseBuilder.spec_model_dataloader` expects only `path` and `dir`, "
                f"but got {len(args)} additional positional arguments ({args}) "
                f"and {len(kwargs)} additional keyword arguments ({list(kwargs.keys())})."
            )
        return parse_class_spec(path=path, dir=dir, species=species)

    @staticmethod
    def build_model(
        fine_tune: bool = False, cls: type[Classifier] = Classifier, fine_tune_dtype: torch.dtype = torch.bfloat16, **kwargs: Any
    ) -> tuple[nn.Module, Callable[[torch.Tensor], torch.Tensor]]:
        """TODO.

        The mandatory keyword arguments depend on the class of `cls`,
          but are likely to be ["model_type", "weights", "device", "dtype" and "num_classes"]
          or a superset containing these.

        Returns:
            (model, model_preprocess) (`tuple[torch.nn.Module, Callable[[torch.Tensor], torch.Tensor]]`):
                The loaded model and an appropriate preprocessing function (e.g. RGB[0,1] normalizer).
        """
        if fine_tune:
            kwargs["preprocess_dtype"] = fine_tune_dtype
        model, model_preprocess = cls.build(**kwargs)
        if fine_tune:
            _backbone = backbone(model)
            _backbone.requires_grad_(False)
            _backbone.to(dtype=fine_tune_dtype)
            for param in _backbone.parameters():
                # param.to(dtype=fine_tune_dtype)
                param.requires_grad_(False)

        return model, model_preprocess

    @staticmethod
    def build_dataloader(
        input_dir: str,
        output_dir: str | None,
        cls2idx: dict[str, int],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        resize_size: int,
        data_index: str | None = None,
        splits: tuple[str, ...] = ("train", "val"),
        num_workers: int | None = None,
        subsample: int | None = None,
        cache: int | str | None = None,
        train_proportion: float = 0.9,
        labels: Any | None = None,
        num_classes: int | None = None,
        hook: Callable[[torch.Tensor], torch.Tensor] | None = None,
        **kwargs,
    ):
        """TODO.

        Returns:
            (train_label_cls, train_loader, validation_loader): The training and validation dataloaders.
        """

        def _resolve_metadata():
            if data_index is None:
                metadata = create_metadata(directory=input_dir, cls2idx=cls2idx, labels=labels, train_proportion=train_proportion)
                index_path = (
                    os.path.join(output_dir, "data_index.json")
                    if output_dir is not None
                    else NamedTemporaryFile(suffix="data_index.json", delete=False).name
                )
                if not os.path.exists(index_path):
                    with open(index_path, "w") as f:
                        json.dump(metadata, f)
            else:
                metadata = get_metadata(data_index, cls2idx=cls2idx)
            return metadata

        metadata = broadcast_from_master(_resolve_metadata)
        metasplits = metadata["split"].copy()
        metadata = [
            {k: [vi for vi, s in zip(v, metasplits) if s.startswith(split.strip().lower())] for k, v in metadata.items()}
            for split in splits
        ]
        datasets, loaders = get_dataset_dataloader(
            *metadata,
            resize_size=resize_size,
            modes=splits,
            batch_size=batch_size,
            num_workers=num_workers,
            subsample=subsample,
            device=device,
            dtype=dtype,
            cache=cache,
            hook=hook,
            **kwargs,
        )

        return metadata[0]["class"], *loaders

    @staticmethod
    def build_inference_dataloader(
        images: list[str],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        num_workers: int | None = None,
        resize_size: int | None = None,
        subsample: int | None = None,
        hook: Callable[[torch.Tensor], torch.Tensor] | None = None,
        preprocess: Callable[[torch.Tensor], torch.Tensor] | None = None,
        data_index: str | None = None,
        splits: tuple[str, ...] = ("train", "val"),
        cache: int | str | None = None,
        train_proportion: float = 0.9,
        labels: Any | None = None,
        num_classes: int | None = None,
        **kwargs,
    ):
        return get_inference_dataloader(
            images=images,
            resize_size=resize_size,
            batch_size=batch_size,
            num_workers=num_workers,
            subsample=subsample,
            device=device,
            dtype=dtype,
            hook=hook,
            **kwargs,
        )[1]

    @staticmethod
    def build_augmentation(dtype: torch.dtype):
        """Returns a training augmentation pipeline for normalized tensors.

        Assumes the input tensor is already normalized (e.g., in the range [0, 1] or standardized).

        Returns:
            A composition of augmentations.
        """
        return tt.Compose(
            [
                # tt.AugMix(severity=3),
                SaltAndPepper(proportion=(0.001, 0.05), probability=0.75),
                tt.RandomHorizontalFlip(),
                tt.RandomVerticalFlip(),
                tt.RandomRotation(15),
                tt.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                # # tt.RandomResizedCrop(size=(224, 224), scale=(0.9, 1.0)),
                # tt.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                # # Convert back to tensor (in case some augmentations convert to PIL Image)
                # tt.ToTensor()
            ]
        )

    @staticmethod
    def parameter_groups(
        model: nn.Module,
        head_lr: float,
        backbone_lr: float,
        head_weight_decay: float,
        backbone_weight_decay: float,
    ) -> list[dict[str, Any]]:
        """Groups all model parameters into 'head' and 'backbone'.

        The 'head' is identified by the attribute name stored in `model._backbone_output_name`.
        All other parameters are considered 'backbone'.
        This method does not filter by requires_grad; it groups all parameters.
        """
        if not hasattr(model, "_backbone_output_name"):
            raise AttributeError("Model does not have `_backbone_output_name` attribute to identify the head.")
        head_attr_name = getattr(model, "_backbone_output_name")
        if not isinstance(head_attr_name, str):
            raise TypeError(f"`model._backbone_output_name` must be a string, got {type(head_attr_name)}.")
        if not hasattr(model, head_attr_name):
            raise AttributeError(f'Model does not have an attribute named "{head_attr_name}" as specified by `_backbone_output_name`.')
        head_module = getattr(model, head_attr_name)
        if not isinstance(head_module, nn.Module):
            raise TypeError(f'The attribute "{head_attr_name}" (value of `_backbone_output_name`) must be an nn.Module.')

        norm_cls = (
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.LayerNorm,
            nn.GroupNorm,
            nn.InstanceNorm1d,
            nn.InstanceNorm2d,
            nn.InstanceNorm3d,
            nn.SyncBatchNorm,
        )
        norm_param_ids = set(id(p) for m in model.modules() if isinstance(m, norm_cls) for p in m.parameters(recurse=False))
        head_module_param_ids = set(id(p) for p in head_module.parameters())

        param_groups = {
            "head": {"params": [], "lr": head_lr, "weight_decay": head_weight_decay},
            "backbone": {"params": [], "lr": backbone_lr, "weight_decay": backbone_weight_decay},
        }

        n_params = 0
        for name, p in model.named_parameters():  # Iterate through all parameters the model exposes
            if not p.requires_grad:
                continue
            grp_name = "head" if id(p) in head_module_param_ids else "backbone"
            # Heuristic to detect last-layer-weights:
            #   Parameters in classification module with "linear" or "layer" in name
            # If using the Muon optimizer it is important not to use it on the final layer(s)!
            if grp_name == "head" and ("linear" in name or "layer" in name):
                new_grp_name = f"{grp_name}_nomuon"
                if new_grp_name not in param_groups:
                    param_groups[new_grp_name] = {
                        "params": [],
                        "lr": param_groups[grp_name]["lr"],
                        "weight_decay": param_groups[grp_name]["weight_decay"],
                    }
                grp_name = new_grp_name
            if "parametrizations.weight" in name or id(p) in norm_param_ids or name.endswith(".bias"):
                new_grp_name = f"{grp_name}_nodecay"
                if new_grp_name not in param_groups:
                    param_groups[new_grp_name] = {"params": [], "lr": param_groups[grp_name]["lr"], "weight_decay": 0}
                grp_name = new_grp_name
            n_params += 1
            param_groups[grp_name]["params"].append(p)

        param_groups = [grp for k, grp in param_groups.items() if len(grp["params"]) > 0 and grp.update({"name": k}) is None]
        if len(param_groups) == 0 and n_params > 0:
            raise RuntimeError(f"No parameter groups detected for model with {n_params} trainable parameters!")

        return param_groups

    @classmethod
    def build_optimizer(
        cls,
        model: nn.Module,
        optimizer_cls: type[torch.optim.Optimizer],
        lr: float,
        weight_decay: float,
        backbone_lr: float | None = None,
        backbone_weight_decay: float | None = None,
        **optimizer_kwargs,  # Other optimizer_cls arguments (e.g., betas, eps for AdamW)
    ) -> torch.optim.Optimizer:
        """Builds an optimizer with separate parameter groups for head and backbone.

        All parameters of the model are assigned to groups.
        Requires `model` to have `_backbone_output_name` attribute.
        """
        head_lr = lr
        backbone_lr = backbone_lr or head_lr / 3
        head_weight_decay = weight_decay
        backbone_weight_decay = backbone_weight_decay or head_weight_decay

        param_groups = cls.parameter_groups(
            model,
            head_lr=head_lr,
            backbone_lr=backbone_lr,
            head_weight_decay=head_weight_decay,
            backbone_weight_decay=backbone_weight_decay,
        )

        if not param_groups and not list(model.parameters()):  # Model has no parameters
            final_params_for_optimizer = []  # Optimizer on empty list
        elif not param_groups and list(model.parameters()):  # Should be caught by group_parameters warning.
            raise ValueError("Model has parameters, but group_parameters returned no groups. This indicates an issue.")
        else:
            final_params_for_optimizer = param_groups

        # Default LR for the optimizer itself (used if a group has no LR or if not using groups)
        if "lr" not in optimizer_kwargs:
            optimizer_kwargs["lr"] = head_lr  # A sensible default

        return optimizer_cls(params=final_params_for_optimizer, **optimizer_kwargs)

    @staticmethod
    def build_scaler(device: torch.types.Device, init_scale: float = 2**14, growth_interval: int = 100, **kwargs):
        return GradScaler(device=device, init_scale=init_scale, growth_interval=growth_interval, **kwargs)

    @staticmethod
    def build_ema(
        enable: bool,
        model: torch.nn.Module,
        epochs: int,
        steps_per_epoch: int,
        device: torch.types.Device,
        update_rate: int = 1,
        epoch_halflife: float | int | None = None,
        decay_rate: float | None = None,
        distill_start: int | None = None,
        use_buffers: bool = True,
        **kwargs,
    ):
        """Produces a Exponential Moving Average (EMA) copy
        of the provided model, with a decay rate derived
        from the given half-life in fractional epochs.

        Args:
            enable: Flag to enable EMA.
            model: The model to track.
            epochs: Total number of epochs that ``model`` is trained for.
            steps_per_epoch: Number of steps per epoch (training phase only).
            device: Device of the model. TODO: This should be inferred from the model.
            update_rate: How often the EMA is updated with a snapshot of the ``model`` weights.
            epoch_halflife: Number of fraction epochs after which the EMA is only 50% of what it was:
                EMA(t + epoch_halflife) = EMA(t) * 0.5 + X
            decay_rate: If specified, manually overrides the decay rate
                calculated from the `update_rate` and half-life.
            distill_start: When EMA distillation/self-supervision starts in fractional epochs.
                Defaults to two epochs or a maximum of 5000 steps (batches).
            use_buffers: Passed to ``torch.optim.swa_utils.AveragedModel`` (default=True).
            **kwargs: Additional arguments passed to ``mini_trainer.builders.EMATeacher``
                or ``torch.optim.swa_utils.AveragedModel``.

        Returns:
            The EMA copy tracking the original model.
        """
        default_step_halflife = min(epochs * steps_per_epoch, 5000)
        if decay_rate is None:
            if epoch_halflife is None:
                step_halflife : float = (epochs * steps_per_epoch) ** (1 / 2) / 2
                epoch_halflife = step_halflife / steps_per_epoch
                default_ratio = step_halflife / default_step_halflife
                if not ((1 / 5) < default_ratio < 5):
                    warnings.warn(
                        f"EMA decay rate half-life is very low/high: {epoch_halflife:.2f} epochs, which "
                        f"is {default_ratio:.1f}X the default ({default_step_halflife / steps_per_epoch:.2f} epochs)."
                    )
            decay_rate = ema_lambda_per_update(epoch_halflife * steps_per_epoch, update_rate)
        if distill_start is None:
            distill_start = min(default_step_halflife, 2 * steps_per_epoch)
        else:
            distill_start = round(distill_start * steps_per_epoch)
        return EMATeacher(
            enable=enable,
            total_steps=epochs * steps_per_epoch,
            distill_start=distill_start,
            update_rate=update_rate,
            model=model,
            device=device,
            multi_avg_fn=get_ema_multi_avg_fn(decay_rate),
            use_buffers=use_buffers,
        )

    @staticmethod
    def build_criterion(
        *args,
        num_classes: int,
        weighted: bool = False,
        labels: np.ndarray | torch.Tensor | list | tuple | None = None,
        label_smoothing: float | None=None,
        device: torch.types.Device | None = None,
        dtype: torch.dtype | None = None,
        **kwargs,
    ):
        """TODO.

        Returns:
            The loss function for optimization (e.g. `torch.nn.CrossEntropyLoss` for classification).
        """
        if label_smoothing is None:
            label_smoothing = 1/num_classes
        if not weighted or labels is None:
            return nn.CrossEntropyLoss(*args, label_smoothing=label_smoothing, **kwargs)
        if not isinstance(labels, (list, tuple)):
            labels = labels.tolist()
        counts = Counter(labels)
        counts = [counts.get(i, 0) for i in range(num_classes)]
        return EMLACrossEntropy(class_frequencies=counts, label_smoothing=label_smoothing, device=device, *args, **kwargs)

    @staticmethod
    def build_regularizer(strength: float = 1e-3, *args, **kwargs):
        strength = float(strength)
        if strength == 0.0:
            return lambda x: 0

        def func(model):
            llw = last_layer_weights(model)
            return strength * class_weight_distribution_regularization(llw)

        return func

    @staticmethod
    def build_lr_scheduler(
        optimizer: torch.optim.Optimizer,
        epochs: int,
        warmup_epochs: float,
        steps_per_epoch: int,
        min_factor: float = 1e-3,
        start_factor: float = 1e-4,
        pretrained_backbone: bool = True,
    ) -> torch.optim.lr_scheduler.LRScheduler:
        """Only the *shape* of the LR curve is defined here; the *magnitude* should be set in the optimizer.
        I suggest using `torch.optim.lr_scheduler.LambdaLR`.

        Returns:
            The learning rate scheduler (shape only).
        """
        warmup_steps = round(warmup_epochs * steps_per_epoch)
        head_schedule = cosine_schedule_with_warmup(epochs * steps_per_epoch, warmup_steps, start_factor, min_factor)

        if pretrained_backbone:

            def backbone_schedule(step: int):
                if step < warmup_steps:
                    return 0.0  # Completely frozen during head warmup

                # Re-align step count so backbone starts cosine decay from index 0
                # after warmup finishes
                return cosine_schedule_with_warmup(
                    epochs * steps_per_epoch - warmup_steps,  # Backbone trains for fewer total steps
                    0,
                    1.0,
                    min_factor,
                )(step - warmup_steps)
        else:
            backbone_schedule = head_schedule

        lr_lambdas = []  # [head_schedule for _ in optimizer.param_groups]
        for grp in optimizer.param_groups:
            lr_lambdas.append(head_schedule if "head" in grp.get("name", "head") else backbone_schedule)

        return torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lr_lambdas)

    @staticmethod
    def build_logger(**kwargs):
        return MultiLogger(**kwargs)


class AutoEmbedder(nn.Module):  # noqa: D101 TODO
    def __init__(self, original_model: Classifier):  # noqa: D107
        super().__init__()
        self.original_model = original_model
        self.embeddings = None
        self._hook_handle = None

        # Attempt to find the final nn.Linear layer
        last_linear_layer = None
        for module in self.original_model.modules():
            if isinstance(module, nn.Linear):
                last_linear_layer = module

        if last_linear_layer is None:
            if isinstance(self.original_model, nn.Linear):  # Model itself is Linear
                last_linear_layer = self.original_model
            else:  # Check last child
                children = list(self.original_model.children())
                if children and isinstance(children[-1], nn.Linear):
                    last_linear_layer = children[-1]
                else:
                    raise ValueError("Could not automatically find a final nn.Linear layer to hook.")

        self._hook_handle = last_linear_layer.register_forward_hook(self._capture_embeddings_hook)

    def _capture_embeddings_hook(self, module, input_args, output):
        # input_args[0] is the input tensor to the hooked nn.Linear layer
        if input_args:
            self.embeddings = input_args[0].detach()  # .clone() if modification is a concern

    def forward(self, x):
        self.embeddings = None  # Reset
        predictions = self.original_model(x)
        return predictions, self.embeddings

    def remove_hook(self):
        if self._hook_handle:
            self._hook_handle.remove()
            self._hook_handle = None

    def __del__(self):
        self.remove_hook()
