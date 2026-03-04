#!/usr/bin/env bash
# 分布式测试启动脚本 — 基于 DeviceMesh 统一并行抽象
#
# 用法:
#   bash tools/dist_test.sh configs/flashocc_r50.py ckpts/epoch_24.pth 4
#   bash tools/dist_test.sh configs/flashocc_r50.py work_dirs/flashocc_r50 8

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29501}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

while true; do
    if python -c "import socket; s=socket.socket(socket.AF_INET, socket.SOCK_STREAM); s.bind(('', $PORT)); s.close()" >/dev/null 2>&1; then
        break
    else
        echo "Port $PORT is already in use, trying $((PORT + 1))"
        PORT=$((PORT + 1))
    fi
done

NCCL_DEBUG=${NCCL_DEBUG:-WARN} \
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
torchrun \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/test.py \
    $CONFIG \
    $CHECKPOINT \
    --launcher pytorch \
    ${@:4}
