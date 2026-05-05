from mini_trainer.hierarchical.integration import HierarchicalBuilder
from mini_trainer.hierarchical.model import (
    AutoregressiveClassifier,
    AutoregressiveClassifierV2,
    ConditionalClassifier,
    HierarchicalClassifier,
    IndependentClassifier,
)
from mini_trainer.train import cli as mt_train_args
from mini_trainer.train import main as mt_train

overrides = []

HEAD_OPTIONS = {
    "hierarchical": HierarchicalClassifier,
    "conditional": ConditionalClassifier,
    "independent": IndependentClassifier,
    "autoregressive": AutoregressiveClassifier,
    "v2.autoregressive": AutoregressiveClassifierV2,
}


def head_name_to_cls(name: str | type):  # noqa: D103
    if isinstance(name, type):
        return name
    name = name.strip().lower()
    try:
        return HEAD_OPTIONS[name]
    except KeyError as e:
        e.add_note(f"Available options are: {list(HEAD_OPTIONS.keys())}")
        raise


def cli(desc="Train a hierarchical classifier", **kwargs):  # noqa: D103
    kwargs = mt_train_args(
        description=desc,
        loss_weights={
            "type": float,
            "nargs": "+",
            "dest": "criterion_builder_kwargs.weights",
            "default": (1.0, 1.0, 1.0),
            "required": False,
            "help": "Weights for the hierarchical loss terms (species, genus, family). Three numbers should be supplied.",
        },
        head={
            "type": str,
            "dest": "model_builder_kwargs.cls",
            "default": "hierarchical",
            "required": False,
            "help": "Which type of classification head architecture to use. "
            'Options are "hierarchical" (default), "conditional" and "independent".',
        },
        no_gbif={
            "action": "store_false",
            "dest": "spec_model_dataloader_kwargs.species",
            "default": True,
            "required": False,
            "help": "Class names are not GBIF IDs.",
        },
        **kwargs,
    )
    for key in overrides:
        kwargs.pop(key, None)

    kwargs["model_builder_kwargs"]["cls"] = head_name_to_cls(kwargs["model_builder_kwargs"]["cls"])
    kwargs["spec_model_dataloader_kwargs"]["levels"] = len(kwargs["criterion_builder_kwargs"]["weights"])
    kwargs["builder"] = HierarchicalBuilder

    return kwargs


def run():
    mt_train(**cli())


if __name__ == "__main__":
    run()
