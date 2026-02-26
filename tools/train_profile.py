#!/usr/bin/env python
"""FlashOCC Profiling 训练脚本.

集成 PyTorch Profiler, 兼容 nsys / ncu 外部 profiling.
仅运行 1 epoch, 前几个 iteration warmup, 后续 iteration profiling.

用法:
    # 1) PyTorch Profiler (生成 Chrome trace + TensorBoard)
    python tools/train_profile.py configs/flashocc_r50.py --profiler torch

    # 2) nsys (NVIDIA Nsight Systems)
    nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas \
        --capture-range=cudaProfilerApi --capture-range-end=stop \
        -o work_dirs/profile/nsys_report \
        python tools/train_profile.py configs/flashocc_r50.py --profiler nsys

    # 3) ncu (NVIDIA Nsight Compute) — 只profile少量iteration
    ncu --set full --target-processes all \
        -o work_dirs/profile/ncu_report \
        python tools/train_profile.py configs/flashocc_r50.py --profiler ncu --profile-iters 3

    # 4) 纯计时模式 (不使用外部工具)
    python tools/train_profile.py configs/flashocc_r50.py --profiler timer
"""
import argparse
import os
import sys
import time
import json

import torch
import torch.cuda

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from flashocc.config import load_experiment
from flashocc.core import load_checkpoint
from flashocc.core.env import collect_env
from flashocc.core.log import logger, setup_logger
from flashocc.engine import init_random_seed, set_random_seed
from flashocc.engine.trainer import (
    build_dataloader, scatter_data, _setup_torch_runtime, _get_warmup_lr,
    _get_amp_dtype,
)


def parse_args():
    parser = argparse.ArgumentParser(description="FlashOCC Profiling 训练")
    parser.add_argument("config", help="Python 配置文件路径 (.py)")
    parser.add_argument("--work-dir", default="work_dirs/profile",
                        help="profiling 输出目录")
    parser.add_argument("--profiler", choices=["torch", "nsys", "ncu", "timer"],
                        default="torch", help="profiling 模式")
    parser.add_argument("--warmup-iters", type=int, default=5,
                        help="profiling 前 warmup 迭代数")
    parser.add_argument("--profile-iters", type=int, default=20,
                        help="profiling 迭代数")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU ID")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    return parser.parse_args()


# =====================================================================
#  NVTX 标记辅助
# =====================================================================

class NVTXRange:
    """上下文管理器: 自动推送/弹出 NVTX range."""
    def __init__(self, name: str):
        self.name = name
    def __enter__(self):
        torch.cuda.nvtx.range_push(self.name)
        return self
    def __exit__(self, *args):
        torch.cuda.nvtx.range_pop()


# =====================================================================
#  Timer Profiler — 手动 CUDA event 计时
# =====================================================================

class TimerProfiler:
    """基于 CUDA events 的简易 profiler, 按阶段计时."""

    def __init__(self):
        self.records = {}  # name -> list of ms

    def start(self, name):
        if name not in self.records:
            self.records[name] = {"starts": [], "ends": [], "times": []}
        evt = torch.cuda.Event(enable_timing=True)
        evt.record()
        self.records[name]["starts"].append(evt)

    def end(self, name):
        evt = torch.cuda.Event(enable_timing=True)
        evt.record()
        self.records[name]["ends"].append(evt)

    def synchronize_and_summarize(self):
        torch.cuda.synchronize()
        summary = {}
        for name, rec in self.records.items():
            times = []
            for s, e in zip(rec["starts"], rec["ends"]):
                times.append(s.elapsed_time(e))  # ms
            summary[name] = {
                "count": len(times),
                "total_ms": sum(times),
                "avg_ms": sum(times) / len(times) if times else 0,
                "min_ms": min(times) if times else 0,
                "max_ms": max(times) if times else 0,
            }
        return summary


# =====================================================================
#  Profiling 主循环
# =====================================================================

def run_profiled_training(args, exp):
    """运行 profile 训练循环."""
    work_dir = args.work_dir
    os.makedirs(work_dir, exist_ok=True)

    # ---- 数据 ----
    dataset = exp.build_train_dataset()
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=exp.samples_per_gpu,
        workers_per_gpu=exp.workers_per_gpu,
        num_gpus=1,
        dist_mode=False,
        seed=exp.seed,
        pin_memory=exp.dataloader_pin_memory,
        persistent_workers=exp.dataloader_persistent_workers,
        prefetch_factor=exp.dataloader_prefetch_factor,
        drop_last=True,
    )

    _setup_torch_runtime(exp)

    # ---- 模型 ----
    model = exp.build_model()
    if exp.load_from:
        logger.info(f"加载预训练: {exp.load_from}")
        load_checkpoint(model, exp.load_from, map_location="cpu",
                        strict=False, logger=logger)
    model = model.cuda()

    # ---- channels_last ----
    if getattr(exp, 'use_channels_last', False):
        model = model.to(memory_format=torch.channels_last)
        logger.info("启用 channels_last 内存格式")

    model.train()

    # ---- AMP ----
    amp_dtype = _get_amp_dtype(exp)
    use_amp = amp_dtype is not None
    scaler = torch.amp.GradScaler('cuda', enabled=(use_amp and amp_dtype == torch.float16))
    if use_amp:
        logger.info(f"启用 AMP, dtype={amp_dtype}")

    # ---- 优化器 ----
    optimizer = exp.build_optimizer(model)
    base_lr = optimizer.param_groups[0]["lr"]
    grad_max_norm = exp.grad_max_norm

    total_iters = args.warmup_iters + args.profile_iters
    logger.info(f"Profiler={args.profiler}, warmup={args.warmup_iters}, "
                f"profile={args.profile_iters}, total={total_iters}")

    # ---- 不同 profiler 策略 ----
    timer = TimerProfiler()
    profiler_ctx = None

    if args.profiler == "torch":
        profiler_ctx = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=1,
                warmup=args.warmup_iters - 1,
                active=args.profile_iters,
                repeat=1,
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                os.path.join(work_dir, "torch_tb_trace")
            ),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=True,
        )
        profiler_ctx.__enter__()

    elif args.profiler == "nsys":
        # 用 cudaProfilerApi 控制 nsys 捕获
        torch.cuda.cudart().cudaProfilerStop()  # 先停, warmup 后再开

    # ---- 训练循环 ----
    data_iter = iter(data_loader)
    global_iter = 0

    for i in range(total_iters):
        is_profiling = (i >= args.warmup_iters)

        # 获取数据
        try:
            data_batch = next(data_iter)
        except StopIteration:
            data_iter = iter(data_loader)
            data_batch = next(data_iter)

        # nsys: warmup 结束后开始 profiling
        if args.profiler == "nsys" and i == args.warmup_iters:
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStart()
            logger.info("nsys: cudaProfilerStart()")

        # ---------- 数据搬运 ----------
        if is_profiling:
            timer.start("data_transfer")
        with NVTXRange("data_transfer"):
            data_batch = scatter_data(
                data_batch, non_blocking=exp.dataloader_non_blocking)
            # channels_last for input images
            if getattr(exp, 'use_channels_last', False):
                if 'img_inputs' in data_batch:
                    img_inputs = data_batch['img_inputs']
                    if isinstance(img_inputs, (list, tuple)) and len(img_inputs) > 0:
                        imgs = img_inputs[0]
                        if hasattr(imgs, 'ndim') and imgs.ndim == 5:
                            B, N, C, H, W = imgs.shape
                            imgs = imgs.view(B*N, C, H, W).to(
                                memory_format=torch.channels_last).view(B, N, C, H, W)
                            data_batch['img_inputs'] = [imgs] + list(img_inputs[1:])
        if is_profiling:
            timer.end("data_transfer")

        # ---------- warmup lr ----------
        with NVTXRange("lr_warmup"):
            if exp.warmup_iters > 0 and global_iter < exp.warmup_iters:
                lr = _get_warmup_lr(base_lr, exp.warmup_ratio,
                                    exp.warmup_iters, global_iter)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr

        # ---------- Forward ----------
        if is_profiling:
            timer.start("forward")
        with NVTXRange("forward"):
            with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
                losses = model(return_loss=True, **data_batch)
        if is_profiling:
            timer.end("forward")

        # ---------- Loss 聚合 ----------
        with NVTXRange("loss_aggregate"):
            total_loss = torch.tensor(0.0, device="cuda")
            for name, val in losses.items():
                if hasattr(val, 'backward'):
                    total_loss = total_loss + val
                elif hasattr(val, '__iter__'):
                    s = sum(v for v in val if hasattr(v, 'backward'))
                    if hasattr(s, 'item'):
                        total_loss = total_loss + s

        # ---------- Backward ----------
        if is_profiling:
            timer.start("backward")
        with NVTXRange("backward"):
            optimizer.zero_grad(set_to_none=exp.optimizer_set_to_none)
            scaler.scale(total_loss).backward()
        if is_profiling:
            timer.end("backward")

        # ---------- Optimizer step ----------
        if is_profiling:
            timer.start("optimizer_step")
        with NVTXRange("optimizer_step"):
            if grad_max_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_max_norm)
            scaler.step(optimizer)
            scaler.update()
        if is_profiling:
            timer.end("optimizer_step")

        global_iter += 1

        # torch profiler step
        if args.profiler == "torch" and profiler_ctx is not None:
            profiler_ctx.step()

        # 日志
        loss_val = total_loss.item()
        lr_now = optimizer.param_groups[0]["lr"]
        phase = "PROFILE" if is_profiling else "WARMUP"
        logger.info(f"[{phase}] iter {i:3d}/{total_iters} | "
                    f"loss={loss_val:.4f} | lr={lr_now:.2e}")

    # ---- 停止 profiling ----
    if args.profiler == "nsys":
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
        logger.info("nsys: cudaProfilerStop()")

    if args.profiler == "torch" and profiler_ctx is not None:
        profiler_ctx.__exit__(None, None, None)

    # ---- 输出 Timer 结果 ----
    torch.cuda.synchronize()
    summary = timer.synchronize_and_summarize()

    logger.info("=" * 70)
    logger.info("  Timer Profiler Summary (profiled iterations only)")
    logger.info("=" * 70)
    logger.info(f"{'Phase':<20} {'Count':>6} {'Total(ms)':>12} "
                f"{'Avg(ms)':>10} {'Min(ms)':>10} {'Max(ms)':>10}")
    logger.info("-" * 70)
    for name in ["data_transfer", "forward", "backward", "optimizer_step"]:
        if name in summary:
            s = summary[name]
            logger.info(f"{name:<20} {s['count']:>6} {s['total_ms']:>12.2f} "
                        f"{s['avg_ms']:>10.2f} {s['min_ms']:>10.2f} "
                        f"{s['max_ms']:>10.2f}")
    logger.info("=" * 70)

    # 同时保存到 JSON
    summary_path = os.path.join(work_dir, "timer_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Timer 结果已保存: {summary_path}")

    # ---- Torch Profiler: 额外输出 key_averages ----
    if args.profiler == "torch" and profiler_ctx is not None:
        # 输出 key_averages 表到文件
        key_avg_path = os.path.join(work_dir, "torch_key_averages.txt")
        key_avg = profiler_ctx.key_averages()
        table = key_avg.table(
            sort_by="cuda_time_total", row_limit=50)
        with open(key_avg_path, "w") as f:
            f.write(table)
        logger.info(f"Torch Profiler key_averages 已保存: {key_avg_path}")
        logger.info(f"\nTop CUDA kernels:\n{table}")

        # 输出 Chrome trace (如果 TensorBoard handler 已保存则跳过)
        chrome_path = os.path.join(work_dir, "torch_trace.json")
        try:
            profiler_ctx.export_chrome_trace(chrome_path)
            logger.info(f"Chrome trace 已保存: {chrome_path}")
        except RuntimeError:
            logger.info("Chrome trace 已由 TensorBoard handler 保存")

        # 输出 stacks (可用 flamegraph 分析)
        stacks_path = os.path.join(work_dir, "torch_stacks.txt")
        try:
            profiler_ctx.export_stacks(stacks_path, "self_cuda_time_total")
            logger.info(f"Stacks 已保存: {stacks_path}")
        except Exception as e:
            logger.warning(f"导出 stacks 失败: {e}")

    # ---- GPU 显存统计 ----
    if torch.cuda.is_available():
        logger.info("\n" + "=" * 50)
        logger.info("  GPU Memory Summary")
        logger.info("=" * 50)
        logger.info(torch.cuda.memory_summary(abbreviated=True))

    return summary


def main():
    args = parse_args()

    # 加载配置
    exp = load_experiment(args.config)

    # 覆盖 work_dir
    if args.work_dir:
        exp = exp.model_copy(update={"work_dir": args.work_dir})

    os.makedirs(args.work_dir, exist_ok=True)

    # 日志
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_file = os.path.join(args.work_dir, f"profile_{timestamp}.log")
    setup_logger(log_file=log_file, level="INFO", rank0_only=True)
    logger.info(f"环境信息:\n{collect_env()}")

    # 种子
    seed = init_random_seed(args.seed)
    set_random_seed(seed)

    # GPU
    torch.cuda.set_device(args.gpu_id)

    # 运行
    summary = run_profiled_training(args, exp)

    logger.info("Profiling 完成!")


if __name__ == "__main__":
    main()
