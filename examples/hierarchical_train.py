from mini_trainer.train import cli as mt_train_args
from mini_trainer.train import main as mt_train

from hierarchical.base.integration import HierarchicalBuilder

overrides = [] # Unused ATM

if __name__ == "__main__":
    kwargs = mt_train_args(
        description="Train a hierarchical classifier",
        loss_weights={
            "type" : float, 
            "nargs" : "+", 
            "default" : (1., 1., 1.), 
            "required" : False, 
            "help" : "Weights for the hierarchical loss terms (species, genus, family). Three numbers should be supplied"
        }
    ) 
    for key in overrides:
        kwargs.pop(key, None)
    
    kwargs["criterion_builder_kwargs"]["weights"] = kwargs.pop("loss_weights")
    # import torch
    # torch.autograd.set_detect_anomaly(True)
    mt_train(
        **kwargs,
        builder=HierarchicalBuilder
    )