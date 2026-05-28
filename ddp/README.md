# Using mini_trainer with DDP

`mini_trainer` has support for DDP training (and potentially inference in the future), here we include some helpful scripts for launching training jobs with DDP in two scenarios:

1. Dual NVIDIA DGX Spark with a direct QSFP connection: [./spark.sh](./spark.sh)
2. SLURM: [./slurm.sh](./slurm.sh)

Our DDP support is currently limited to NCCL.

## Launching dual DGX Spark

```sh
uv -qq run bash ../../ddp/spark.sh -w spkc \
    -m mini_trainer.train -i train -o . \
    --name mnist_model \
    --model efficientnet_b0 \
    --batch_size 256 \
    --epochs 5 \
    --warmup_epochs 1 \
    --lr 0.01 \
    --class_weighted \
    --size 27 \
    --cache CUDA \
    --output . \
    --verbose
```