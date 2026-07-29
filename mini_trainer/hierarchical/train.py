from mini_trainer.modeling import Classifier
from mini_trainer.train import cli as mt_train_args
from mini_trainer.train import main as mt_train

from .integration import HierarchicalBuilder
from .model import head_name_to_cls

overrides = []


def cli(desc="Train a hierarchical classifier", **kwargs):  # noqa: D103
    kwargs = mt_train_args(
        description=desc,
        loss_weights={
            "type": float,
            "nargs": "+",
            "dest": "criterion_builder_kwargs.weights",
            "default": None,
            "required": False,
            "help": "Weights for the hierarchical loss terms (species, genus, family). Three numbers should be supplied.",
        },
        head={
            "type": str,
            "dest": "head",
            "default": "hierarchical",
            "required": False,
            "help": "Which type of classification head architecture to use. "
            'Options are "hierarchical" (default), "conditional" and "independent".',
        },
        no_gbif={
            "action": "store_false",
            "dest": "species",
            "default": True,
            "required": False,
            "help": "Class names are not GBIF IDs.",
        },
        **kwargs,
    )
    for key in overrides:
        kwargs.pop(key, None)

    head = head_name_to_cls(kwargs.pop("head", "hierarchical").strip().lower())
    weights = kwargs.pop("weights", None)
    kwargs["spec_model_dataloader_kwargs"]["species"] = kwargs.pop("species", True)
    if head is Classifier:
        kwargs["criterion_builder_kwargs"].pop("weights")
        return kwargs
    kwargs["model_builder_kwargs"]["cls"] = head
    if weights is not None:
        kwargs["spec_model_dataloader_kwargs"]["weights"] = weights
        kwargs["spec_model_dataloader_kwargs"]["levels"] = len(kwargs["criterion_builder_kwargs"]["weights"])
    kwargs["builder"] = HierarchicalBuilder

    return kwargs


def run():
    mt_train(**cli())


if __name__ == "__main__":
    run()
