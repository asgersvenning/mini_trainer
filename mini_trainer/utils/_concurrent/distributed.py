import builtins as __builtin__
import functools
import inspect
import logging
import os
import sys
from contextlib import contextmanager

import torch
from torch import distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record

from mini_trainer.utils._core import increment_name_dir


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
    handler.setFormatter(DDPFormatter("%(message)s"))
    handler.addFilter(DDPFilter())
    logger.addHandler(handler)


def setup_for_distributed(is_master):
    """This function disables printing when not in master process."""
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():  # noqa: D103
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():  # noqa: D103
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():  # noqa: D103
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():  # noqa: D103
    return get_rank() == 0


def save_on_master(*args, **kwargs):  # noqa: D103
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed(device=None):
    """Initialize distributed training process group."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    elif "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["WORLD_SIZE"])
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        local_rank = rank % num_gpus if num_gpus > 0 else 0
    else:
        return None

    use_cuda = torch.cuda.is_available()
    if device is not None and "cpu" in str(device).lower():
        use_cuda = False

    if use_cuda:
        num_gpus = torch.cuda.device_count()
        if num_gpus > 0:
            gpu_idx = local_rank % num_gpus
            torch.cuda.set_device(gpu_idx)
            if world_size > num_gpus:
                backend = "gloo"
            else:
                backend = "nccl"
        else:
            backend = "gloo"
    else:
        backend = "gloo"

    dist.init_process_group(
        backend=backend,
        init_method="env://",
        world_size=world_size,
        rank=rank,
    )
    dist.barrier()
    setup_for_distributed(rank == 0)
    return {"rank": rank, "world_size": world_size, "local_rank": local_rank}


def broadcast_from_master(fn, *args, **kwargs):
    """Run ``fn(*args, **kwargs)`` on rank 0 and broadcast the result to all ranks.

    If DDP is not active, simply calls ``fn`` directly.
    """
    if is_dist_avail_and_initialized():
        result = [None]
        if get_rank() == 0:
            result[0] = fn(*args, **kwargs)
        dist.broadcast_object_list(result, src=0)
        return result[0]
    return fn(*args, **kwargs)


def sync_run_name(name: str, output: str | None) -> str:
    """Synchronize the name of the run across all processes under DDP."""
    return broadcast_from_master(increment_name_dir, name, output)


def ddp_train_wrapper(main_fn):
    """Decorator/wrapper for main training entry point to enable DDP."""
    @functools.wraps(main_fn)
    def wrapped(*args, **kwargs):
        sig = inspect.signature(main_fn)
        bound = sig.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        device_str = bound.arguments.get("device", "cuda:0")

        ddp_info = init_distributed(device=device_str)
        bound.arguments["ddp_info"] = ddp_info

        tgt_fn = main_fn
        if ddp_info is not None:
            tgt_fn = record(main_fn)
            # Override device argument for this rank if targeting GPU
            if "cpu" not in str(device_str).lower():
                if torch.cuda.is_available():
                    num_gpus = torch.cuda.device_count()
                    gpu_idx = ddp_info["local_rank"] % num_gpus
                    bound.arguments["device"] = f"cuda:{gpu_idx}"
                else:
                    bound.arguments["device"] = "cpu"

            # Disable logging on non-zero ranks
            if ddp_info["rank"] > 0:
                from mini_trainer.logging.core import MetricLogger

                logger_kwargs = bound.arguments.setdefault("logger_builder_kwargs", {})
                logger_kwargs["logger_cls"] = [MetricLogger]
                logger_kwargs["verbose"] = False
        try:
            return tgt_fn(*bound.args, **bound.kwargs)
        except Exception:
            # Do not call destroy_process_group on failure to avoid deadlocks
            raise
        else:
            if ddp_info is not None:
                dist.destroy_process_group()

    return wrapped


def init_distributed_mode(args):  # noqa: D103
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    elif hasattr(args, "rank"):
        pass
    else:
        print("Not using distributed mode")
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print(f"| distributed init (rank {args.rank}): {args.dist_url}", flush=True)
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)
    dist.barrier()
    setup_for_distributed(args.rank == 0)


def reduce_across_processes(val):  # noqa: D103
    if not is_dist_avail_and_initialized():
        # nothing to sync, but we still convert to tensor for consistency with the distributed case.
        return torch.tensor(val)

    device = torch.device(f"cuda:{torch.cuda.current_device()}") if torch.cuda.is_available() else torch.device("cpu")
    t = torch.tensor(val, device=device)
    dist.barrier()
    dist.all_reduce(t)
    return t


@contextmanager
def main_process_first(verbose=True):
    """Context manager to execute the wrapped block on the main process first, then other processes."""
    logger = logging.getLogger("mini_trainer")
    if is_dist_avail_and_initialized():
        if is_main_process():
            logger.debug("Entered main_process_first (master). Executing block...")
            try:
                yield
            finally:
                logger.debug("Master block completed. Reaching barrier...")
                dist.barrier()
                logger.debug("Barrier released on master.")
        else:
            logger.debug("Worker waiting at barrier...")
            dist.barrier()
            logger.debug("Worker barrier released. Executing block...")
            yield
            logger.debug("Worker block completed.")
    else:
        yield
