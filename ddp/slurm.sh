#!/bin/bash
#SBATCH --job-name=mini_trainer_ddp
#SBATCH --nodes=2              # Target number of nodes
#SBATCH --ntasks-per-node=8    # 8 H100 GPUs per DGX node
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16     # Adjust based on dataloader workers
#SBATCH --exclusive            # Request full nodes on a SuperPOD

# 1. Extract the IP of the master (rank 0) node
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo "Master Node IP: $head_node_ip"

# 2. Export variables required by init_method="env://"
export MASTER_ADDR=$head_node_ip
export MASTER_PORT=29500

# 3. SuperPOD NCCL Tuning (Often handled by HPC admins, but good to ensure)
export NCCL_DEBUG=INFO
# export NCCL_IB_HCA=mlx5      # Uncomment and adjust if IB interfaces aren't auto-detected

# 4. Launch natively with srun
# srun automatically spawns 16 processes (2 nodes * 8 tasks) and sets SLURM_PROCID
srun python -m mini_trainer.train \
    -i train \
    -o . \
    --name superpod_model \
    --model efficientnet_b0 \
    --batch_size 1024