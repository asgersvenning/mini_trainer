from mini_trainer.hierarchical.integration import HierarchicalBuilder
from mini_trainer.train import cli as mt_train_args
from mini_trainer.train import main as mt_train

overrides = []

def cli(desc="Train a hierarchical classifier", **kwargs):
    kwargs = mt_train_args(
        description=desc,
        loss_weights={
            "type" : float, 
            "nargs" : "+", 
            "default" : (1., 1., 1.), 
            "required" : False, 
            "help" : "Weights for the hierarchical loss terms (species, genus, family). Three numbers should be supplied"
        },
        **kwargs
    ) 
    for key in overrides:
        kwargs.pop(key, None)
    
    kwargs["criterion_builder_kwargs"]["weights"] = kwargs.pop("loss_weights")

    return kwargs

if __name__ == "__main__":

    # import torch
    # torch.autograd.set_detect_anomaly(True)
    mt_train(
        **cli(),
        builder=HierarchicalBuilder
    )