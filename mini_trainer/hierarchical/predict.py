from mini_trainer.hierarchical.integration import HierarchicalBuilder, HierarchicalResultCollector
from mini_trainer.hierarchical.train import head_name_to_cls
from mini_trainer.predict import cli as mt_predict_cli
from mini_trainer.predict import main as mt_predict
from mini_trainer.utils.logging import RawResultCollector


def cli(description : str="Predict with a trained hierarchical model", **extra_kwargs): # noqa: D103
    kwargs = mt_predict_cli(
        description=description,
        head={
            "type" : str,
            "dest" : "model_builder_kwargs.cls",
            "default" : None,
            "required" : False,
            "help" : 'Which type of classification head architecture to use. '
        }, 
        scientific_names={
            None : "-S",
            "action" : "store_true",
            "dest" : "collector_cls_kwargs.scientific_names",
            "default" : False,
            "required" : False,
            "help" : "Convert GBIF IDs in output to scientific names via GBIF API."
        },
        **extra_kwargs
    )
    head : str | None = kwargs["model_builder_kwargs"].get("cls", None)
    if head is not None:
        kwargs["model_builder_kwargs"]["cls"] = head_name_to_cls(head)
    kwargs["builder"] = HierarchicalBuilder
    if kwargs["collector_cls"] is not RawResultCollector:
        kwargs["collector_cls"] = HierarchicalResultCollector
    return kwargs


def run():
    mt_predict(**cli())


if __name__ == "__main__":
    run()