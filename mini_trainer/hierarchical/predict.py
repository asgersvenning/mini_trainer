from mini_trainer.hierarchical.integration import HierarchicalBuilder, HierarchicalResultCollector
from mini_trainer.hierarchical.train import head_name_to_cls
from mini_trainer.predict import cli as mt_predict_cli
from mini_trainer.predict import main as mt_predict


def cli(description : str="Predict with a trained hierarchical model", **extra_kwargs): # noqa: D103
    kwargs = mt_predict_cli(
        description=description,
        loss_weights={
            "type" : float, 
            "nargs" : "+", 
            "dest" : "criterion_builder_kwargs.weights",
            "default" : (1., 1., 1.), 
            "required" : False, 
            "help" : 'Weights for the hierarchical loss terms (species, genus, family). '
            'Three numbers should be supplied.'
        },
        head={
            "type" : str,
            "dest" : "model_builder_kwargs.cls",
            "default" : "hierarchical",
            "required" : False,
            "help" : 'Which type of classification head architecture to use. '
            'Options are "hierarchical" (default), "conditional" and "independent".'
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
    kwargs["model_builder_kwargs"]["cls"] = head_name_to_cls(kwargs["model_builder_kwargs"]["cls"])
    kwargs["builder"] = HierarchicalBuilder
    kwargs["collector_cls"] = HierarchicalResultCollector
    return kwargs


def run():
    mt_predict(**cli())


if __name__ == "__main__":
    run()