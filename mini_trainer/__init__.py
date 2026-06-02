# noqa: D104
import inspect
import logging as std_logging


def get_logger(name: str | None = None) -> std_logging.Logger:
    """Get a logger instance unified under the 'mini_trainer' hierarchy."""
    if name is None:
        frame = inspect.currentframe()
        if frame is not None and frame.f_back is not None:
            name = frame.f_back.f_globals.get("__name__", "mini_trainer")
        else:
            name = "mini_trainer"

    if name == "__main__":
        name = "mini_trainer"
    elif name != "mini_trainer" and not name.startswith("mini_trainer."):
        name = f"mini_trainer.{name}"

    return std_logging.getLogger(name)
