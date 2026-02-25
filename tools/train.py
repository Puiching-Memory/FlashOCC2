#!/usr/bin/env python
"""FlashOCC 训练脚本.

:
    # 单 GPU
    python tools/train.py configs/flashocc_r50.py

    # 多 GPU (DDP)
    torchrun --nproc_per_node=4 tools/train.py configs/flashocc_r50.py
"""
import argparse
import os
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from flashocc.config import load_experiment
from flashocc.core import init_dist, get_dist_info, load_checkpoint
from flashocc.core.env import collect_env, setup_multi_processes
from flashocc.core.log import logger, setup_logger
from flashocc.engine import train_model, init_random_seed, set_random_seed


def parse_args():
    parser = argparse.ArgumentParser(description="FlashOCC 训练")
    parser.add_argument("config", help="Python 配置文件路径 (.py)")
    parser.add_argument("--work-dir", help="工作目录")
    parser.add_argument("--resume-from", help="从 checkpoint 恢复训练")
    parser.add_argument("--validate", action="store_true", help="训练时验证")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU ID (单卡)")
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--launcher", choices=["none", "pytorch", "slurm"],
                        default="none")
    parser.add_argument("--local_rank", "--local-rank", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()

    # 加载实验配置
    exp = load_experiment(args.config)

    # 覆盖运行时选项
    if args.work_dir:
        exp.work_dir = args.work_dir
    if exp.work_dir is None:
        config_name = os.path.splitext(os.path.basename(args.config))[0]
        exp.work_dir = os.path.join("work_dirs", config_name)
    os.makedirs(exp.work_dir, exist_ok=True)

    if args.resume_from:
        exp.resume_from = args.resume_from

    # 分布式
    distributed = args.launcher != "none"
    if distributed:
        init_dist(args.launcher)
    rank, world_size = get_dist_info()

    # 日志
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_file = os.path.join(exp.work_dir, f"{timestamp}.log")
    setup_logger(log_file=log_file, level="INFO", rank0_only=True)
    logger.info(f"环境信息:\n{collect_env()}")

    # 随机种子
    seed = init_random_seed(args.seed if args.seed is not None else exp.seed)
    set_random_seed(seed, deterministic=args.deterministic)
    exp.seed = seed

    # GPU
    gpu_ids = [args.gpu_id] if not distributed else list(range(world_size))

    # 构建模型
    model = exp.build_model()
    logger.info(f"模型: {model.__class__.__name__}")

    # 加载预训练
    if exp.load_from:
        logger.info(f"加载预训练: {exp.load_from}")
        load_checkpoint(model, exp.load_from, map_location="cpu",
                        strict=False, logger=logger)

    # 训练
    meta = dict(seed=seed, timestamp=timestamp)
    train_model(
        experiment=exp,
        model=model,
        distributed=distributed,
        validate=args.validate,
        timestamp=timestamp,
        meta=meta,
        gpu_ids=gpu_ids,
    )


if __name__ == "__main__":
    main()
