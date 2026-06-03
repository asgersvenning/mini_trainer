#!/usr/bin/env bash

usage() {
    cat << EOF
Usage: uv run bash launch.sh -w <worker_alias> <your_script.py> [script arguments...]

A launcher for two-node PyTorch Distributed Data Parallel (DDP) training over RoCE/NFS.

Required Arguments:
  -w <alias>    The SSH alias for the worker node (must exist in ~/.ssh/config)

Example:
  uv run bash launch.sh -w spkc -m mini_trainer.train -i train -o . --batch_size 32
EOF
    exit 1
}

# Enforce minimum number of arguments
if [ "$#" -lt 3 ]; then
    usage
fi

# Extract the worker alias
if [ "$1" == "-w" ]; then
    WORKER_ALIAS="$2"
    shift 2  # Shift arguments so "$@" only contains the Python script and its args
else
    echo -e "[Error] The first argument must be the worker alias flag (-w).\n"
    usage
fi

# --- Topology Configuration ---
MASTER_PORT="29500"
IFNAME="enp1s0f1np1"   # The QSFP interface
CURRENT_DIR=$(pwd)
MASTER_CACHE="$HOME/.cache"
RDZV_ID="uv_torch_ddp"

# --- Dynamic IP Extraction ---
echo "[Launcher] Resolving network topology..."

# 1. Extract worker IP from your SSH config
WORKER_IP=$(ssh -G "$WORKER_ALIAS" 2>/dev/null | awk '/^hostname / { print $2 }')

if [[ -z "$WORKER_IP" ]]; then
    echo -e "\n[Error] Could not resolve IP for worker alias '$WORKER_ALIAS'."
    echo "Please ensure '$WORKER_ALIAS' is a valid Host entry in your ~/.ssh/config."
    exit 1
fi

WORKER_HOME=$(ssh -q "$WORKER_ALIAS" "echo \$HOME")
WORKER_CACHE="$WORKER_HOME/.cache"

# 2. Extract master IP directly from the physical network interface
MASTER_IP=$(ip -4 addr show $IFNAME | awk '/inet / {print $2}' | cut -d/ -f1)

if [[ -z "$MASTER_IP" ]]; then
    echo -e "\n[Error] Could not resolve Master IP on interface $IFNAME."
    exit 1
fi

# 3. Resolve the Python executable
PYTHON_EXEC=$(command -v python)
if [[ -z "$PYTHON_EXEC" ]]; then
    echo "[Error] Could not resolve Python executable. Are you running via 'uv run'?"
    exit 1
fi

# 4. Dynamically resolve the absolute root project directory by stripping the virtualenv suffix
PROJECT_PATH="${PYTHON_EXEC%/.venv/bin/python}"
if [[ -z "$PROJECT_PATH" ]]; then
    echo "[Error] Could not resolve project root directory. Executable: $PYTHON_EXEC"
    exit 1
fi

# 5. Prepare ephemeral teardown
cleanup() {
    # Capture exit code
    local exit_code=$?
    
    # 2. Disable the trap to prevent an infinite recursion loop during cleanup
    trap - INT TERM EXIT
    
    echo -e "\n[Launcher] Tearing down ephemeral environment..."
    # Clean up worker processes and unmount cleanly using the root PROJECT_PATH
    ssh -q $WORKER_ALIAS "pkill -f torchrun"
    ssh -q $WORKER_ALIAS "sudo umount -l $WORKER_CACHE"
    ssh -q $WORKER_ALIAS "sudo umount -l $PROJECT_PATH"
    
    sudo exportfs -u $WORKER_IP:$MASTER_CACHE
    sudo exportfs -u $WORKER_IP:$PROJECT_PATH
    
    echo "[Launcher] Teardown complete. Restoring node state. Exit code: $exit_code"
    
     exit $exit_code
}
trap cleanup INT TERM EXIT

# 6. Setup ephemeral NFS to mirror the uv .venv, dependencies and data
echo "[Launcher] Establishing ephemeral NFS mount over QSFP..."

MASTER_USER_UID=$(id -u)
MASTER_USER_GID=$(id -g)

# We export and mount the parent PROJECT_PATH so the virtual environment (.venv) is fully included
sudo exportfs -o rw,sync,no_subtree_check,all_squash,anonuid=$MASTER_USER_UID,anongid=$MASTER_USER_GID $WORKER_IP:$PROJECT_PATH
sudo exportfs -o rw,sync,no_subtree_check,all_squash,anonuid=$MASTER_USER_UID,anongid=$MASTER_USER_GID $WORKER_IP:$MASTER_CACHE

# Mount with optimized FS-Cache and metadata performance parameters 
# nocto and nconnect=16 drastically speed up Python's sequential module imports [2]
ssh -q $WORKER_ALIAS "sudo mkdir -p $PROJECT_PATH && sudo mount -t nfs -o rw,noatime,rsize=32768,wsize=32768,tcp,intr,fsc,nocto,nconnect=16 $MASTER_IP:$PROJECT_PATH $PROJECT_PATH"
echo "[Launcher]  -> Mounted Project: $MASTER_IP:$PROJECT_PATH  ==>  $WORKER_ALIAS:$PROJECT_PATH"

ssh -q $WORKER_ALIAS "sudo mkdir -p $WORKER_CACHE && sudo mount -t nfs -o rw,noatime,rsize=32768,wsize=32768,tcp,intr,fsc,nocto,nconnect=16 $MASTER_IP:$MASTER_CACHE $WORKER_CACHE"
echo "[Launcher]  -> Mounted Cache:   $MASTER_IP:$MASTER_CACHE  ==>  $WORKER_ALIAS:$WORKER_CACHE"

# 7. Pre-flight validation
echo "[Launcher] Verifying environment synchronization..."
ssh -q $WORKER_ALIAS "test -x $PYTHON_EXEC" || {
    echo -e "\n[Error] Worker node ($WORKER_ALIAS) cannot execute the shadowed Python binary."
    exit 1
}

echo "[Launcher] Testing PyTorch import on worker node..."
ssh -q $WORKER_ALIAS "cd $CURRENT_DIR && $PYTHON_EXEC -c 'import torch; print(f\" PyTorch {torch.__version__} loaded on worker.\")'" || {
    echo -e "\n[Error] Worker node ($WORKER_ALIAS) failed to load PyTorch."
    exit 1
}

# 8. Launch torchrun processes 
# PyTorch Network Binding 
PT_VARS="TP_SOCKET_IFNAME=$IFNAME GLOO_SOCKET_IFNAME=$IFNAME NCCL_SOCKET_IFNAME=$IFNAME NCCL_DEBUG=INFO"

echo "[Launcher] Spawning worker on $WORKER_ALIAS ($WORKER_IP)..."
ssh $WORKER_ALIAS "cd $CURRENT_DIR && \
    env $PT_VARS $PYTHON_EXEC -m torch.distributed.run \
    --nnodes=2 \
    --nproc_per_node=1 \
    --node_rank=1 \
    --local_addr=$WORKER_IP \
    --rdzv_id=$RDZV_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_IP:$MASTER_PORT \
    --rdzv_conf=is_host=0 \
    $@" &

echo "[Launcher] Spawning master locally ($MASTER_IP)..."
env $PT_VARS $PYTHON_EXEC -m torch.distributed.run \
    --nnodes=2 \
    --nproc_per_node=1 \
    --node_rank=0 \
    --local_addr=$MASTER_IP \
    --rdzv_id=$RDZV_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_IP:$MASTER_PORT \
    --rdzv_conf=is_host=1 \
    "$@"

MASTER_EXIT_CODE=$?

exit $MASTER_EXIT_CODE
