import torch


class SupervisionContext:
    """Process-global target passed to classifier heads; cleared on context exit."""

    _target: torch.Tensor | None = None

    @classmethod
    def set(cls, target):
        cls._target = target

    @classmethod
    def get(cls):
        return cls._target

    @classmethod
    def clear(cls):
        cls._target = None

    def __init__(self, target):  # noqa: D107
        self.target = target

    def __enter__(self):
        SupervisionContext.set(self.target)

    def __exit__(self, exc_type, exc_val, exc_tb):
        SupervisionContext.clear()


class EmbeddingContext:
    """Process-global embedding handoff preserving gradients; nested contexts are rejected."""

    # Dynamo can carry dictionary mutations out of a compiled graph. Assigning
    # a Tensor to a class attribute instead forces a graph break at publication.
    _state: dict[str, torch.Tensor | None] = {"embeddings": None}
    _active: bool = False

    @classmethod
    def set(cls, embeddings):
        cls._state["embeddings"] = embeddings

    @classmethod
    def get(cls):
        return cls._state["embeddings"]

    @classmethod
    def clear(cls):
        cls._state["embeddings"] = None
        cls._active = False

    @classmethod
    def activate(cls):
        if cls._active:
            raise RuntimeError("EmbeddingContext is already active")
        cls._active = True

    @classmethod
    def active(cls):
        return cls._active

    def __enter__(self):
        self.activate()

    def __exit__(self, exc_type, exc_val, exc_tb):
        EmbeddingContext.clear()


class AuxiliaryLogitsContext:
    """Process-global list of auxiliary head outputs collected during one training forward; nested contexts are rejected.

    An opt-in auxiliary head (e.g. ``Classifier(detached_learned_head=True)``) adds its output while the context is
    active; the training step adds the criterion of each to the loss. Outside the context nothing is computed.
    """

    _state: dict[str, list] = {"logits": []}
    _active: bool = False

    @classmethod
    def add(cls, logits):
        cls._state["logits"] = [*cls._state["logits"], logits]

    @classmethod
    def get(cls) -> list:
        return cls._state["logits"]

    @classmethod
    def clear(cls):
        cls._state["logits"] = []
        cls._active = False

    @classmethod
    def active(cls):
        return cls._active

    def __enter__(self):
        if AuxiliaryLogitsContext._active:
            raise RuntimeError("AuxiliaryLogitsContext is already active")
        AuxiliaryLogitsContext._active = True

    def __exit__(self, exc_type, exc_val, exc_tb):
        AuxiliaryLogitsContext.clear()
