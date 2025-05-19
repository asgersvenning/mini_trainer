from mini_trainer.predict import cli as mt_pred_args
from mini_trainer.predict import main as mt_predict

from hierarchical.base.integration import HierarchicalBuilder, HierarchicalResultCollector

overrides = [] # Unused ATM

if __name__ == "__main__":
    kwargs = mt_pred_args(description="Predict with a hierarchical classifier") 
    for key in overrides:
        kwargs.pop(key, None)
    kwargs["result_collector_kwargs"]["levels"] = 3
    mt_predict(
        **kwargs,
        builder=HierarchicalBuilder,
        result_collector=HierarchicalResultCollector
    )