"""Measure steps per second of a synthetic 192k-class, seven-rank head under real DDP (nccl).

    python dev/benchmarks/ddp_head_sync.py RANKS

The input is random 1536-d features with no backbone, so the time isolates the head's forward, the all-reduce of its
roughly 1.11 GB of gradients inside ``backward`` and the optimizer step. Prints one JSON line on rank 0.
"""

import json
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from mini_trainer.hierarchical.loss import MultiLevelWeightedCrossEntropyLoss
from mini_trainer.hierarchical.model import HierarchicalClassifier

SIZES = [192000, 40000, 1200, 80, 10, 5, 2]
DIM, BATCH, STEPS, WARMUP = 1536, 96, 80, 20


def run(rank: int, world: int):
    """Train the head for ``STEPS`` steps on this rank and report the median timings."""
    dist.init_process_group("nccl", init_method="tcp://127.0.0.1:29655", rank=rank, world_size=world)
    device = torch.device("cuda", rank)
    masks = [torch.from_numpy(np.arange(a) % b).long() for a, b in zip(SIZES, SIZES[1:])]
    head = HierarchicalClassifier(
        in_features=DIM,
        out_features=SIZES[0],
        sparse_masks=masks,
        hidden=True,
        droprate=0.1,
        normalized=True,
        trainable_scale=True,
        trainable_bias=True,
        float32_head=True,
        skip_spherical_init=True,
    )
    head = head.to(device).train()
    # The same wrapping as mini_trainer/trainer.py: the head has parameters outside the loss graph.
    model = DistributedDataParallel(
        head, device_ids=[rank], find_unused_parameters=True, gradient_as_bucket_view=True, broadcast_buffers=False
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    criterion = MultiLevelWeightedCrossEntropyLoss(num_classes=SIZES, device=device, dtype=torch.float32, label_smoothing=1 / SIZES[0])
    target = [torch.randint(0, SIZES[0], (BATCH,), device=device)]
    for mask in masks:
        target.append(mask.to(device)[target[-1]])
    target = torch.stack(target, 1)
    steps, backwards = [], []
    for _ in range(STEPS):
        features = torch.randn(BATCH, DIM, device=device)
        dist.barrier()
        start = time.perf_counter()
        loss = sum(criterion(model(features), target))
        forward_done = time.perf_counter()
        loss.backward()  # DDP all-reduces the gradients inside backward
        torch.cuda.synchronize()
        backward_done = time.perf_counter()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        steps.append(time.perf_counter() - start)
        backwards.append(backward_done - forward_done)
    if rank == 0:
        step = float(np.median(steps[WARMUP:]))
        result = {
            "ranks": world,
            "step_ms": round(1000 * step, 1),
            "backward_with_sync_ms": round(1000 * float(np.median(backwards[WARMUP:])), 1),
            "steps_per_second": round(1 / step, 2),
            "implied_images_per_second_batch96": round(world * BATCH / step),
            "head_gb": round(sum(p.numel() for p in head.parameters()) * 4 / 2**30, 2),
        }
        print(json.dumps(result), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    ranks = int(sys.argv[1])
    torch.multiprocessing.spawn(run, args=(ranks,), nprocs=ranks)
