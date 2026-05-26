import functools
import os

import torch
from torch import distributed as dist


def setup_for_distributed(is_master):
    """This function disables printing when not in master process."""
    import builtins as __builtin__

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


def ddp_train_wrapper(main_fn):
    """Decorator/wrapper for main training entry point to enable DDP."""
    @functools.wraps(main_fn)
    def wrapped(*args, **kwargs):
        device = kwargs.get("device", None)
        ddp_info = init_distributed(device=device)
        if ddp_info is not None:
            # Override device argument for this rank if targeting GPU
            device_str = kwargs.get("device", "cuda:0")
            if "cpu" not in str(device_str).lower():
                if torch.cuda.is_available():
                    num_gpus = torch.cuda.device_count()
                    gpu_idx = ddp_info["local_rank"] % num_gpus
                    kwargs["device"] = f"cuda:{gpu_idx}"
                else:
                    kwargs["device"] = "cpu"

            # Disable logging on non-zero ranks
            if ddp_info["rank"] > 0:
                from mini_trainer.logging.core import MetricLogger
                logger_kwargs = kwargs.setdefault("logger_builder_kwargs", {})
                logger_kwargs["logger_cls"] = [MetricLogger]
                logger_kwargs["verbose"] = False
        try:
            return main_fn(*args, **kwargs)
        finally:
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
