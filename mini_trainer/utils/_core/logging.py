import inspect
import logging
import sys


class DDPFilter(logging.Filter):
    """Filter to ensure INFO and higher logs only run on Rank 0, but DEBUG runs on all ranks."""

    def filter(self, record):
        from mini_trainer.utils._concurrent.distributed import get_rank
        if record.levelno < logging.INFO:
            return True
        return get_rank() == 0


class DDPFormatter(logging.Formatter):
    """Formatter to automatically prepend [Rank X] in DDP mode."""

    def format(self, record):
        from mini_trainer.utils._concurrent.distributed import get_rank, is_dist_avail_and_initialized
        if is_dist_avail_and_initialized():
            rank_str = f"[Rank {get_rank()}] "
        else:
            rank_str = ""
        msg = super().format(record)
        if rank_str and not msg.startswith("[Rank"):
            msg = f"{rank_str}{msg}"
        return msg


def setup_logging(verbose: bool = False):
    """Configure the 'mini_trainer' logger with a custom DDP filter and formatter."""
    logger = logging.getLogger("mini_trainer")
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.propagate = False

    handler = logging.StreamHandler(sys.__stdout__ or sys.stdout)
    handler.setFormatter(DDPFormatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    handler.addFilter(DDPFilter())
    logger.addHandler(handler)


def get_logger(name: str | None = None) -> logging.Logger:
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

    return logging.getLogger(name)
