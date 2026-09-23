"""MAMBO deployment compatibility entry point; shared core code remains unchanged."""


def _runtime():
    try:
        from mambo_deploy import Predictor
    except ImportError as error:
        raise ImportError("Install the matching mambo_deploy deployment wheel and provide a local MAMBO bundle") from error
    return Predictor


class Predictor:
    """Legacy native/CUDA defaults with additive bundle and backend selection."""

    def __init__(self, device="cuda", model=None, weights=None, class_mask=None, **kwargs):
        import torch

        if model is not None and weights is not None:
            raise ValueError("model and weights are mutually exclusive")
        if isinstance(model, str) and model.endswith((".pt", ".pth")):
            weights, model = model, None
        self.device = torch.device(device)
        self._predictor = _runtime()(
            device=str(self.device), model=model, weights=weights, class_mask=class_mask, backend=kwargs.pop("backend", "torch"), **kwargs
        )

    def _convert(self, result):
        if self._predictor.backend != "torch":
            return result
        import torch

        from mini_trainer.hierarchical.model import HierarchicalPrediction

        class DeploymentPrediction(HierarchicalPrediction):
            def _process(self, raw_prediction):
                return (
                    torch.as_tensor(result.logits, device=self_device),
                    torch.as_tensor(result.indices, device=self_device),
                )

            def _extract_confidence(self, raw_prediction):
                return torch.as_tensor(result.confidence, device=self_device)

        # Preserve the native container while using the shared score and tie policy.
        self_device = self.device
        prediction = DeploymentPrediction(
            [torch.as_tensor(rank, device=self.device) for rank in result.raw_logits],
            topk=result.topk,
            cls2idx=result.cls2idx,
            **result.metadata,
        )
        prediction.global_indices = torch.as_tensor(result.global_indices, device=self.device)
        return prediction

    def predict(self, x, **kwargs):
        return self._convert(self._predictor.predict(x, **kwargs))

    def __call__(self, x, **kwargs):
        return self.predict(x, **kwargs)

    def predict_with_embeddings(self, x, **kwargs):
        result, embeddings = self._predictor.predict_with_embeddings(x, **kwargs)
        if self._predictor.backend == "torch":
            import torch

            embeddings = torch.as_tensor(embeddings, device=self.device)
        return self._convert(result), embeddings

    def _apply_class_mask(self, mask):
        self._predictor._apply_class_mask(mask)

    def available_presets(self):
        return self._predictor.available_presets()


def run():
    try:
        from mambo_deploy.cli import run as deploy_run
    except ImportError:
        from argparse import ArgumentParser

        parser = ArgumentParser(description="MAMBO local prediction. Install the matching mambo_deploy wheel for inference.")
        parser.parse_args()
        parser.error("Install the matching mambo_deploy wheel and provide --bundle or MAMBO_BUNDLE")

    deploy_run(default_backend="torch", default_device="cuda")
