# Distributed training launchers

The trainer accepts torchrun or Slurm rank variables and selects NCCL for CUDA,
Gloo for CPU. These launchers are machine-specific examples. First select and
activate the environment using the [installation guide](../README.md#local-installation).

- [Slurm](slurm.sh): adapt the allocation (currently two nodes, eight GPUs each),
  reachable master address and training paths to your cluster, then submit with `sbatch`.
- [Dual DGX Spark](spark.sh): assumes one GPU per node, interface `enp1s0f1np1`,
  an SSH worker alias and Python at `<checkout>/.venv/bin/python`. It requires sudo,
  exports the checkout and `~/.cache` over NFS, and mounts them on the worker.
  Its cleanup uses a broad remote `pkill -f torchrun` and lazy unmounts: use only
  on dedicated nodes where those processes and mounts belong to this run.

From the checkout root, with the environment active and training images available:

```sh
bash ddp/spark.sh -w spkc -m mini_trainer.train -i /path/to/train -o /path/to/output
```

[nccl_sanity_check.py](nccl_sanity_check.py) is an optional torchrun diagnostic:
GPU/version telemetry and a 256 MiB all-reduce timing, separate from training speed.
