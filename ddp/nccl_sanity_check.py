import os
import time

import torch
import torch.distributed as dist


def run_diagnostics() -> None:
    if not dist.is_initialized():
        print("Distributed process group not initialized.")
        return

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # 1. Collect Telemetry
    device_name = torch.cuda.get_device_name(local_rank)
    device_cap = torch.cuda.get_device_capability(local_rank)
    total_mem = torch.cuda.get_device_properties(local_rank).total_memory / (1024**3)
    nccl_version = torch.cuda.nccl.version()

    # Log environmental status per rank
    print(f"--- [Rank {rank}/{world_size}] Node Status ---")
    print(f"  Device Name: {device_name} (Compute {device_cap[0]}.{device_cap[1]})")
    print(f"  Available VRAM: {total_mem:.2f} GB")
    print(f"  PyTorch NCCL Version: {nccl_version}")
    print(f"  CUDA Runtime Version: {torch.version.cuda}")

    # Barrier synchronization check
    dist.barrier()
    if rank == 0:
        print("\n[Master] Global handshake successful. Starting network stress test...")

    # 2. Interconnect Stress Test (All-Reduce)
    # 64M floats = 256 MB tensor
    tensor_size = 64 * 1024 * 1024
    stress_tensor = torch.randn(tensor_size, device=local_rank)

    warmup_iters = 5
    test_iters = 50

    # Warmup
    for _ in range(warmup_iters):
        dist.all_reduce(stress_tensor, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize(local_rank)
    dist.barrier()

    # Timed runs
    start_time = time.perf_counter()
    for _ in range(test_iters):
        dist.all_reduce(stress_tensor, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize(local_rank)
    end_time = time.perf_counter()

    # 3. Calculate and Report Bandwidth
    total_duration = end_time - start_time
    avg_duration = total_duration / test_iters

    # For a 2-node All-Reduce, data moved per node is exactly the tensor size
    data_gb = (tensor_size * 4) / (1024**3)  # 4 bytes per float
    bus_bandwidth = data_gb / avg_duration

    if rank == 0:
        print("\n--- Interconnect Performance Results ---")
        print(f"  Sustained Test Iterations: {test_iters}")
        print(f"  Average Iteration Time:   {avg_duration * 1000:.2f} ms")
        print(f"  Effective Bus Bandwidth:  {bus_bandwidth:.2f} GB/s")
        print(f"  Equivalent Network Speed: {bus_bandwidth * 8:.2f} Gbps")
        print("----------------------------------------\n")


def main() -> None:
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

    run_diagnostics()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
