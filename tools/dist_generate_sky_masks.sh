#!/usr/bin/env bash

# 分布式生成天空掩码
# 用法:
#   bash tools/dist_generate_sky_masks.sh 4 --dataroot data/nuScenes --output data/SAM3 --model-name ckpts/sam3 --batch-size 4 --resume

GPUS=$1
if [ -z "$GPUS" ]; then
    echo "Usage: bash tools/dist_generate_sky_masks.sh <gpus> [generate_sky_masks.py args...]"
    exit 1
fi

NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29510}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

while true; do
    if python -c "import socket; s=socket.socket(socket.AF_INET, socket.SOCK_STREAM); s.bind(('', $PORT)); s.close()" >/dev/null 2>&1; then
        break
    else
        echo "Port $PORT is already in use, trying $((PORT + 1))"
        PORT=$((PORT + 1))
    fi
done

PYTHON_BIN=${PYTHON_BIN:-python}

NCCL_DEBUG=${NCCL_DEBUG:-WARN} \
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
$PYTHON_BIN -m torch.distributed.run \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/generate_sky_masks.py \
    ${@:2}
