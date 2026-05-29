import logging
import sys

from mini_trainer.utils._concurrent.distributed import get_rank, is_dist_avail_and_initialized


class DDPFilter(logging.Filter):
    """Filter to ensure INFO and higher logs only run on Rank 0, but DEBUG runs on all ranks."""

    def filter(self, record):

        if record.levelno < logging.INFO:
            return True
        return get_rank() == 0


class DDPFormatter(logging.Formatter):
    """Formatter to automatically prepend [Rank X] in DDP mode."""

    def format(self, record):
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
