#!/usr/bin/env python
"""DDP 8-卡真实训练吞吐量单次测量.

由 bench_ddp_sweep.py 通过 torchrun 调用, 不要直接运行。

输出: rank-0 在 stdout 最后一行打印 JSON 结果, sweep 脚本解析。

用法 (由 sweep 脚本自动调用):
    torchrun --nproc_per_node=8 tools/bench_ddp.py configs/flashocc_r50.py \\
        --samples-per-gpu 8 --workers-per-gpu 8 --prefetch-factor 4 \\
        --warmup-iters 10 --measure-iters 30
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
import torch.distributed as dist

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
for p in (ROOT, SRC):
    if p not in sys.path:
        sys.path.insert(0, p)

from flashocc.config import load_experiment
from flashocc.core import init_dist, get_dist_info, load_checkpoint
from flashocc.core.log import logger, setup_logger
from flashocc.engine import init_random_seed, set_random_seed
from flashocc.engine.trainer import (
    build_dataloader, scatter_data,
    _setup_torch_runtime, _get_amp_dtype,
)
from flashocc.datasets.dali_decode import dali_decode_batch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("config")
    p.add_argument("--samples-per-gpu", type=int, required=True)
    p.add_argument("--workers-per-gpu", type=int, required=True)
    p.add_argument("--prefetch-factor",  type=int, required=True)
    p.add_argument("--warmup-iters",  type=int, default=10)
    p.add_argument("--measure-iters", type=int, default=30)
    p.add_argument("--local_rank", "--local-rank", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()

    # ---- DDP 初始化 ----
    init_dist("pytorch")
    rank, world_size = get_dist_info()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    # 只在 rank-0 打日志 (避免 stdout 污染 JSON 解析)
    if rank == 0:
        setup_logger(log_file=None, level="WARNING", rank0_only=True)

    # ---- 加载配置并覆盖三个参数 ----
    exp = load_experiment(args.config)
    exp = exp.model_copy(update=dict(
        samples_per_gpu=args.samples_per_gpu,
        workers_per_gpu=args.workers_per_gpu,
        dataloader_prefetch_factor=args.prefetch_factor,
        use_compile=False,          # 关闭 compile, 不污染计时
        max_epochs=1,
    ))

    seed = init_random_seed(exp.seed)
    set_random_seed(seed)

    # ---- 数据 ----
    dataset = exp.build_train_dataset()
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=exp.samples_per_gpu,
        workers_per_gpu=exp.workers_per_gpu,
        num_gpus=1,
        dist_mode=True,
        seed=seed,
        pin_memory=exp.dataloader_pin_memory,
        persistent_workers=exp.dataloader_persistent_workers,
        prefetch_factor=exp.dataloader_prefetch_factor,
        drop_last=True,
    )

    _setup_torch_runtime(exp)

    # ---- 模型 ----
    model = exp.build_model()
    # 子模块预训练
    for name, m in model.named_modules():
        if hasattr(m, 'init_weights') and callable(m.init_weights):
            if getattr(m, '_raw_init_cfg', None) is not None:
                m.init_weights()
    if exp.load_from:
        load_checkpoint(model, exp.load_from, map_location="cpu",
                        strict=False, logger=logger)
    model = model.cuda()
    if getattr(exp, 'use_channels_last', False):
        model = model.to(memory_format=torch.channels_last)
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        broadcast_buffers=False,
        find_unused_parameters=getattr(exp, 'find_unused_parameters', False),
    )
    model.train()

    # ---- AMP ----
    amp_dtype = _get_amp_dtype(exp)
    use_amp = amp_dtype is not None
    scaler = torch.amp.GradScaler('cuda', enabled=(use_amp and amp_dtype == torch.float16))

    # ---- 优化器 ----
    optimizer = exp.build_optimizer(model)

    # ---- 训练循环 ----
    total_iters = args.warmup_iters + args.measure_iters
    data_iter = iter(data_loader)

    # CUDA events for wall-clock timing on GPU
    events_start = []
    events_end   = []

    for i in range(total_iters):
        try:
            data_batch = next(data_iter)
        except StopIteration:
            data_iter = iter(data_loader)
            data_batch = next(data_iter)

        data_batch = scatter_data(data_batch,
                                  non_blocking=exp.dataloader_non_blocking)

        if 'jpeg_bytes' in data_batch:
            data_batch = dali_decode_batch(data_batch)

        if getattr(exp, 'use_channels_last', False) and 'img_inputs' in data_batch:
            img_inputs = data_batch['img_inputs']
            if isinstance(img_inputs, (list, tuple)) and len(img_inputs) > 0:
                imgs = img_inputs[0]
                if hasattr(imgs, 'ndim') and imgs.ndim == 5:
                    B, N, C, H, W = imgs.shape
                    imgs = (imgs.view(B * N, C, H, W)
                                .to(memory_format=torch.channels_last)
                                .view(B, N, C, H, W))
                    data_batch['img_inputs'] = [imgs] + list(img_inputs[1:])

        is_measuring = (i >= args.warmup_iters)
        if is_measuring:
            evt_s = torch.cuda.Event(enable_timing=True)
            evt_s.record()

        with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
            losses = model(return_loss=True, **data_batch)

        total_loss = torch.tensor(0.0, device="cuda")
        for name, val in losses.items():
            if hasattr(val, 'backward'):
                total_loss = total_loss + val
            elif hasattr(val, '__iter__'):
                s = sum(v for v in val if hasattr(v, 'backward'))
                if hasattr(s, 'item'):
                    total_loss = total_loss + s

        optimizer.zero_grad(set_to_none=exp.optimizer_set_to_none)
        scaler.scale(total_loss).backward()
        if exp.grad_max_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), exp.grad_max_norm)
        scaler.step(optimizer)
        scaler.update()

        if is_measuring:
            evt_e = torch.cuda.Event(enable_timing=True)
            evt_e.record()
            events_start.append(evt_s)
            events_end.append(evt_e)

    # ---- 汇总计时 ----
    torch.cuda.synchronize()
    iter_times_ms = [s.elapsed_time(e)
                     for s, e in zip(events_start, events_end)]
    median_ms  = sorted(iter_times_ms)[len(iter_times_ms) // 2]
    mean_ms    = sum(iter_times_ms) / len(iter_times_ms)

    # 全局 batch size = samples_per_gpu * world_size
    global_batch = args.samples_per_gpu * world_size
    throughput   = global_batch / (median_ms / 1000.0)   # samples/s

    # rank-0 打印结果 JSON (sweep 脚本解析最后一行)
    if rank == 0:
        result = dict(
            samples_per_gpu=args.samples_per_gpu,
            workers_per_gpu=args.workers_per_gpu,
            prefetch_factor=args.prefetch_factor,
            world_size=world_size,
            global_batch=global_batch,
            median_ms=round(median_ms, 2),
            mean_ms=round(mean_ms, 2),
            throughput=round(throughput, 2),
            measure_iters=args.measure_iters,
        )
        # 用特殊前缀标记, 方便 grep
        print(f"BENCH_RESULT: {json.dumps(result)}", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
