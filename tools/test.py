#!/usr/bin/env python
"""FlashOCC 测试脚本.

:
    python tools/test.py configs/flashocc_r50.py ckpts/epoch_24.pth
    torchrun --nproc_per_node=4 tools/test.py configs/flashocc_r50.py ckpts/epoch_24.pth
"""
import argparse
import os
import sys
import pickle

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from flashocc.config import load_experiment
from flashocc.core import init_dist, get_dist_info, load_checkpoint
from flashocc.core.log import logger, setup_logger
from flashocc.engine import single_gpu_test
from flashocc.engine.trainer import build_dataloader


def parse_args():
    parser = argparse.ArgumentParser(description="FlashOCC 测试")
    parser.add_argument("config", help="Python 配置文件路径 (.py)")
    parser.add_argument("checkpoint", help="checkpoint 文件路径")
    parser.add_argument("--out", help="保存结果到 pkl 文件")
    parser.add_argument("--eval", nargs="+", help="评估指标")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--show-dir")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--launcher", choices=["none", "pytorch", "slurm"],
                        default="none")
    parser.add_argument("--local_rank", "--local-rank", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()

    exp = load_experiment(args.config)

    distributed = args.launcher != "none"
    if distributed:
        init_dist(args.launcher)
    rank, _ = get_dist_info()

    setup_logger(level="INFO", rank0_only=True)

    # 构建测试数据集
    dataset = exp.build_test_dataset()
    logger.info(f"测试集: {dataset.__class__.__name__} ({len(dataset)} 样本)")

    data_loader = build_dataloader(
        dataset, samples_per_gpu=1,
        workers_per_gpu=exp.workers_per_gpu,
        dist_mode=distributed,
    )

    # 构建模型
    model = exp.build_model()
    load_checkpoint(model, args.checkpoint, map_location="cpu")
    model = model.cuda()

    # 测试
    logger.info("开始测试...")
    results = single_gpu_test(model, data_loader, show=args.show,
                              out_dir=args.show_dir)

    # 保存
    if args.out and rank == 0:
        logger.info(f"保存结果到 {args.out}")
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "wb") as f:
            pickle.dump(results, f)

    # 评估
    if args.eval and rank == 0:
        eval_kwargs = {"metric": args.eval}
        if hasattr(dataset, "evaluate"):
            metrics = dataset.evaluate(results, **eval_kwargs)
            for k, v in metrics.items():
                logger.info(f"{k}: {v}")

    logger.info("测试完成。")


if __name__ == "__main__":
    main()
