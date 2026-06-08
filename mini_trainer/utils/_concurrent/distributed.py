import builtins as __builtin__
import functools
import inspect
import os
from collections.abc import Callable
from contextlib import contextmanager

import torch
from torch import distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record

from mini_trainer import get_logger


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
    elif "SLURM_PROCID" in os.environ and int(os.environ.get("SLURM_NTASKS", 1)) > 1:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        local_rank = int(os.environ.get("SLURM_LOCALID", 0))        
        # Safeguard DDP with `env://` initialization method
        if "MASTER_ADDR" not in os.environ or "MASTER_PORT" not in os.environ:
            raise RuntimeError(
                "Slurm DDP detected, but MASTER_ADDR or MASTER_PORT is not set. "
                "Please set these in your Slurm batch script."
            )
    else:
        return None

    use_cuda = torch.cuda.is_available()
    if device is not None and "cpu" in str(device).lower():
        use_cuda = False

    pg_device = None

    if use_cuda:
        num_gpus = torch.cuda.device_count()
        if num_gpus > 0:
            gpu_idx = local_rank % num_gpus
            torch.cuda.set_device(gpu_idx)
            backend = "nccl"
            pg_device = torch.device(f"cuda:{gpu_idx}")
        else:
            backend = "gloo"
    else:
        backend = "gloo"

    dist.init_process_group(
        backend=backend,
        init_method="env://",
        world_size=world_size,
        rank=rank,
        device_id=pg_device,
    )
    dist.barrier()
    setup_for_distributed(rank == 0)
    return {"rank": rank, "world_size": world_size, "local_rank": local_rank}


def broadcast_from_master[**P, Q](fn: Callable[P, Q], *args, **kwargs) -> Q:
    """Run ``fn(*args, **kwargs)`` on rank 0 and broadcast the result to all ranks.

    If DDP is not active, simply calls ``fn`` directly.
    """
    if is_dist_avail_and_initialized():
        result: list = [None]
        if get_rank() == 0:
            result[0] = fn(*args, **kwargs)
        dist.broadcast_object_list(result, src=0)
        return result[0]
    return fn(*args, **kwargs)


def ddp_train_wrapper(main_fn):
    """Decorator/wrapper for main training entry point to enable DDP."""

    @functools.wraps(main_fn)
    def wrapped(*args, **kwargs):
        sig = inspect.signature(main_fn)
        bound = sig.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        device_str = bound.arguments.get("device", "cuda")

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

        result = tgt_fn(*bound.args, **bound.kwargs)
        if ddp_info is not None:
            dist.destroy_process_group()
        return result

    return wrapped


def reduce_across_processes(val):  # noqa: D103
    if not is_dist_avail_and_initialized():
        # nothing to sync, but we still convert to tensor for consistency with the distributed case.
        return torch.tensor(val)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    t = torch.tensor(val, device=device)
    dist.barrier()
    dist.all_reduce(t)
    return t


@contextmanager
def main_process_first(verbose=True):
    """Context manager to execute the wrapped block on the main process first, then other processes."""
    logger = get_logger()
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
