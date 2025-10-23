from mini_trainer.hierarchical.integration import HierarchicalBuilder
from mini_trainer.train import cli as mt_train_args
from mini_trainer.train import main as mt_train
from mini_trainer.hierarchical.model import HierarchicalClassifier, ConditionalClassifier, IndependentClassifier

overrides = []

HEAD_OPTIONS = {
    "hierarchical" : HierarchicalClassifier,
    "conditional" : ConditionalClassifier,
    "independent" : IndependentClassifier
}

def head_name_to_cls(name : str | type):
    if isinstance(name, type):
        return name
    name = name.strip().lower()
    try:
        return HEAD_OPTIONS[name]
    except KeyError as e:
        e.add_note(f"Available options are: {list(HEAD_OPTIONS.keys())}")
        raise e
    

def cli(desc="Train a hierarchical classifier", **kwargs):
    kwargs = mt_train_args(
        description=desc,
        loss_weights={
            "type" : float, 
            "nargs" : "+", 
            "dest" : "criterion_builder_kwargs.weights",
            "default" : (1., 1., 1.), 
            "required" : False, 
            "help" : "Weights for the hierarchical loss terms (species, genus, family). Three numbers should be supplied"
        },
        head={
            "type" : str,
            "dest" : "model_builder_kwargs.cls",
            "default" : "hierarchical",
            "required" : False,
            "help" : "Which type of classification head architecture to use. Options are 'hierarchical' (default), 'conditional' and 'independent'."
        },
        **kwargs
    ) 
    for key in overrides:
        kwargs.pop(key, None)
    
    kwargs["model_builder_kwargs"]["cls"] = head_name_to_cls(kwargs["model_builder_kwargs"]["cls"])
    kwargs["builder"] = HierarchicalBuilder

    return kwargs

if __name__ == "__main__":

    # import torch
    # torch.autograd.set_detect_anomaly(True)
    mt_train(
        **cli()
    )