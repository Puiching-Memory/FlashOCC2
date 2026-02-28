#!/usr/bin/env python
"""DataLoader samples_per_gpu / workers_per_gpu / prefetch_factor 组合性能 Benchmark.

测试不同 samples_per_gpu、workers_per_gpu 和 dataloader_prefetch_factor 组合下,
数据加载的吞吐量 (samples/s) 与每批耗时 (ms/batch)。
只跑 DataLoader 迭代, 不走 GPU 前向，聚焦于数据加载瓶颈。

用法:
    python tools/bench_dataloader.py
    python tools/bench_dataloader.py --num-batches 60 --warmup 5
    python tools/bench_dataloader.py --samples 2 4 8 --workers 8 16 32 --prefetch 2 4
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from itertools import product

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
for p in (ROOT, SRC):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch

from flashocc.config import load_experiment
from flashocc.engine.trainer import build_dataloader


# =====================================================================
#  搜索空间
# =====================================================================
SAMPLES_CANDIDATES     = [1, 2, 4, 8]
WORKERS_CANDIDATES     = [2, 4, 8, 16, 24, 32]
PREFETCH_CANDIDATES    = [2, 4, 8]


# =====================================================================
#  单次测量
# =====================================================================

def measure_one(dataset, samples_per_gpu, workers, prefetch,
                num_batches, warmup, pin_memory, persistent_workers,
                drop_last, seed):
    """构建 DataLoader, 迭代 num_batches 批, 返回每批中位耗时(ms) 和 吞吐(samples/s).

    吞吐量以 samples/s 计算, 可直接比较不同 batch size 下的效率。
    """
    loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=workers,
        num_gpus=1,
        dist_mode=False,
        seed=seed,
        shuffle=False,          # 不shuffle，排除随机性影响
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch,
        drop_last=drop_last,
    )

    times = []
    it = iter(loader)

    # warmup
    for _ in range(warmup):
        try:
            _ = next(it)
        except StopIteration:
            it = iter(loader)
            _ = next(it)

    # 正式计时
    for _ in range(num_batches):
        t0 = time.perf_counter()
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    del loader   # 释放 worker 进程

    import statistics
    median_ms   = statistics.median(times) * 1000
    mean_ms     = sum(times) / len(times) * 1000
    # 吞吐: samples/s = batch_size / median_time(s)
    batch_size  = samples_per_gpu          # 单GPU, num_gpus=1
    throughput  = batch_size / statistics.median(times)

    return dict(
        samples=samples_per_gpu,
        workers=workers,
        prefetch=prefetch,
        median_ms=round(median_ms, 1),
        mean_ms=round(mean_ms, 1),
        throughput=round(throughput, 2),
    )


# =====================================================================
#  主程序
# =====================================================================

def parse_args():
    p = argparse.ArgumentParser(description="DataLoader samples/workers/prefetch benchmark")
    p.add_argument("--config", default="configs/flashocc_r50.py")
    p.add_argument("--num-batches", type=int, default=50,
                   help="每个配置计时的批次数 (默认50)")
    p.add_argument("--warmup", type=int, default=5,
                   help="预热批次数 (默认5)")
    p.add_argument("--samples", type=int, nargs="+", default=SAMPLES_CANDIDATES,
                   help="要测试的 samples_per_gpu 列表")
    p.add_argument("--workers", type=int, nargs="+", default=WORKERS_CANDIDATES,
                   help="要测试的 workers_per_gpu 列表")
    p.add_argument("--prefetch", type=int, nargs="+", default=PREFETCH_CANDIDATES,
                   help="要测试的 prefetch_factor 列表")
    return p.parse_args()


def print_table(results: list[dict], best: dict):
    """美观打印结果表格."""
    header = f"{'samples':>8} {'workers':>8} {'prefetch':>9} {'median_ms':>11} {'mean_ms':>9} {'samples/s':>11}"
    sep = "-" * len(header)
    print(f"\n{'DataLoader Benchmark 结果':^{len(header)}}")
    print(sep)
    print(header)
    print(sep)
    for r in results:
        marker = " <-- BEST" if r is best else ""
        print(f"{r['samples']:>8} {r['workers']:>8} {r['prefetch']:>9} {r['median_ms']:>11.1f} "
              f"{r['mean_ms']:>9.1f} {r['throughput']:>11.2f}{marker}")
    print(sep)


def main():
    args = parse_args()

    print(f"加载实验配置: {args.config}")
    exp = load_experiment(args.config)

    print("构建验证数据集 (val, 无数据增广, 更快构建)...")
    dataset = exp.build_val_dataset()
    print(f"  数据集大小: {len(dataset)} 样本")
    print(f"  pin_memory: {exp.dataloader_pin_memory}")
    print(f"  persistent_workers: {exp.dataloader_persistent_workers}")
    print(f"  当前配置 samples_per_gpu={exp.samples_per_gpu}, "
          f"workers_per_gpu={exp.workers_per_gpu}, "
          f"prefetch_factor={exp.dataloader_prefetch_factor}")
    print()

    combos = list(product(args.samples, args.workers, args.prefetch))
    total  = len(combos)
    print(f"共测试 {total} 个组合 "
          f"(samples={args.samples}, workers={args.workers}, prefetch={args.prefetch})")
    print(f"每组: warmup={args.warmup} 批 + 计时={args.num_batches} 批\n")

    results = []
    for idx, (spg, w, pf) in enumerate(combos, 1):
        print(f"[{idx:2d}/{total}] samples={spg}, workers={w:2d}, prefetch={pf} ...", end=" ", flush=True)
        try:
            r = measure_one(
                dataset,
                samples_per_gpu=spg,
                workers=w,
                prefetch=pf,
                num_batches=args.num_batches,
                warmup=args.warmup,
                pin_memory=exp.dataloader_pin_memory,
                persistent_workers=exp.dataloader_persistent_workers,
                drop_last=exp.dataloader_drop_last,
                seed=exp.seed,
            )
            results.append(r)
            print(f"median={r['median_ms']:.1f}ms  {r['throughput']:.2f} samples/s")
        except Exception as e:
            print(f"FAILED: {e}")

    if not results:
        print("没有成功的测试结果，退出。")
        return

    # 按吞吐量排序，取最优
    results.sort(key=lambda x: x['throughput'], reverse=True)
    best = results[0]

    print_table(results, best)

    print(f"\n最优配置:")
    print(f"  samples_per_gpu            = {best['samples']}")
    print(f"  workers_per_gpu            = {best['workers']}")
    print(f"  dataloader_prefetch_factor = {best['prefetch']}")
    print(f"  吞吐量: {best['throughput']:.2f} samples/s  "
          f"(median {best['median_ms']:.1f} ms/batch)")
    print()
    print("如需更新配置，将 configs/flashocc_r50.py 中对应字段改为上述值即可。")


if __name__ == "__main__":
    main()
