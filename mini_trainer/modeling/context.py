import torch


class SupervisionContext:
    """Used for passing a target to the classification module."""

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
    """Used for passing embeddings from the classification module to the criterion (or elsewhere)."""

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
