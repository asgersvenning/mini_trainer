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

    t = torch.tensor(val, device="cuda")
    dist.barrier()
    dist.all_reduce(t)
    return t
