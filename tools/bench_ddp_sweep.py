#!/usr/bin/env python
"""DDP 8-卡训练吞吐量 Sweep — 自动遍历所有参数组合.

对每个 (samples_per_gpu, workers_per_gpu, prefetch_factor) 组合启动
真实 8-GPU DDP 训练, 测量 forward+backward 吞吐量 (samples/s)。

用法:
    python tools/bench_ddp_sweep.py
    python tools/bench_ddp_sweep.py --config configs/flashocc_r50.py \\
        --samples 4 8 --workers 4 8 16 --prefetch 2 4 \\
        --warmup-iters 8 --measure-iters 20
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from itertools import product

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
for p in (ROOT, SRC):
    if p not in sys.path:
        sys.path.insert(0, p)

# =====================================================================
#  搜索空间默认值
# =====================================================================
DEFAULT_SAMPLES  = [4, 8]
DEFAULT_WORKERS  = [4, 8, 16]
DEFAULT_PREFETCH = [2, 4]

# =====================================================================
#  命令行参数
# =====================================================================

def parse_args():
    p = argparse.ArgumentParser(description="DDP 8-GPU training throughput sweep")
    p.add_argument("--config", default="configs/flashocc_r50.py")
    p.add_argument("--ngpus", type=int, default=8)
    p.add_argument("--samples",  type=int, nargs="+", default=DEFAULT_SAMPLES,
                   help="samples_per_gpu 候选列表")
    p.add_argument("--workers",  type=int, nargs="+", default=DEFAULT_WORKERS,
                   help="workers_per_gpu 候选列表")
    p.add_argument("--prefetch", type=int, nargs="+", default=DEFAULT_PREFETCH,
                   help="prefetch_factor 候选列表")
    p.add_argument("--warmup-iters",  type=int, default=10,
                   help="每次运行的预热迭代数 (默认10)")
    p.add_argument("--measure-iters", type=int, default=25,
                   help="每次运行的计时迭代数 (默认25)")
    p.add_argument("--update-config", action="store_true", default=True,
                   help="完成后自动将最优配置写回 config 文件")
    p.add_argument("--no-update-config", dest="update_config", action="store_false")
    return p.parse_args()


# =====================================================================
#  单次运行
# =====================================================================

def run_one(config, ngpus, spg, wpg, pf, warmup, measure) -> dict | None:
    """启动 torchrun 子进程, 解析 BENCH_RESULT JSON, 返回结果 dict 或 None."""
    cmd = [
        "torchrun",
        f"--nproc_per_node={ngpus}",
        "--master_port=29600",
        "tools/bench_ddp.py",
        config,
        f"--samples-per-gpu={spg}",
        f"--workers-per-gpu={wpg}",
        f"--prefetch-factor={pf}",
        f"--warmup-iters={warmup}",
        f"--measure-iters={measure}",
    ]

    env = os.environ.copy()
    # 抑制 INFO 日志, 避免污染 stdout 解析
    env["PYTHONUNBUFFERED"] = "1"

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            cwd=ROOT,
        )
    except FileNotFoundError:
        print("  ERROR: torchrun not found. Please activate the venv.")
        return None

    # 解析 BENCH_RESULT 行
    result = None
    for line in proc.stdout.splitlines():
        if line.startswith("BENCH_RESULT:"):
            try:
                result = json.loads(line[len("BENCH_RESULT:"):].strip())
            except json.JSONDecodeError as e:
                print(f"  JSON parse error: {e}")

    if result is None:
        # 打印 stderr 辅助排查
        print(f"  FAILED (exit={proc.returncode})")
        if proc.stderr:
            tail = "\n".join(proc.stderr.strip().splitlines()[-20:])
            print(f"  stderr (last 20 lines):\n{tail}")
        if proc.stdout:
            tail = "\n".join(proc.stdout.strip().splitlines()[-10:])
            print(f"  stdout (last 10 lines):\n{tail}")

    return result


# =====================================================================
#  结果表格
# =====================================================================

def print_table(results: list[dict], best: dict):
    header = (f"{'samples':>8} {'workers':>8} {'prefetch':>9} "
              f"{'median_ms':>11} {'mean_ms':>9} {'global_bs':>10} {'samples/s':>11}")
    sep = "-" * len(header)
    print(f"\n{'DDP 8-GPU Benchmark 结果':^{len(header)}}")
    print(sep)
    print(header)
    print(sep)
    for r in results:
        marker = " <-- BEST" if r is best else ""
        print(f"{r['samples_per_gpu']:>8} {r['workers_per_gpu']:>8} "
              f"{r['prefetch_factor']:>9} {r['median_ms']:>11.1f} "
              f"{r['mean_ms']:>9.1f} {r['global_batch']:>10} "
              f"{r['throughput']:>11.2f}{marker}")
    print(sep)


# =====================================================================
#  更新配置文件
# =====================================================================

def update_config(config_path: str, best: dict):
    """将最优参数写回 Python 配置文件."""
    with open(config_path, "r") as f:
        src = f.read()

    import re

    def replace_field(text, field, new_val):
        pattern = rf"(\b{re.escape(field)}\s*=\s*)\d+"
        replacement = rf"\g<1>{new_val}"
        new_text, count = re.subn(pattern, replacement, text)
        if count == 0:
            print(f"  WARNING: field '{field}' not found in config, skipped.")
        return new_text

    src = replace_field(src, "samples_per_gpu",            best["samples_per_gpu"])
    src = replace_field(src, "workers_per_gpu",            best["workers_per_gpu"])
    src = replace_field(src, "dataloader_prefetch_factor", best["prefetch_factor"])

    with open(config_path, "w") as f:
        f.write(src)
    print(f"  配置已更新: {config_path}")


# =====================================================================
#  主程序
# =====================================================================

def main():
    args = parse_args()

    combos = list(product(args.samples, args.workers, args.prefetch))
    total  = len(combos)

    print(f"DDP {args.ngpus}-GPU 训练吞吐量 Benchmark")
    print(f"  config: {args.config}")
    print(f"  搜索空间: samples={args.samples}, workers={args.workers}, "
          f"prefetch={args.prefetch}")
    print(f"  每次运行: warmup={args.warmup_iters} iters + "
          f"measure={args.measure_iters} iters")
    print(f"  共 {total} 个组合\n")

    results = []
    for idx, (spg, wpg, pf) in enumerate(combos, 1):
        global_bs = spg * args.ngpus
        print(f"[{idx:2d}/{total}] samples={spg}(global_bs={global_bs}), "
              f"workers={wpg}, prefetch={pf} ...", end=" ", flush=True)
        t0 = time.time()
        r = run_one(args.config, args.ngpus, spg, wpg, pf,
                    args.warmup_iters, args.measure_iters)
        elapsed = time.time() - t0
        if r is not None:
            results.append(r)
            print(f"median={r['median_ms']:.1f}ms  "
                  f"{r['throughput']:.1f} samples/s  "
                  f"(took {elapsed:.0f}s)")
        else:
            print(f"FAILED  (took {elapsed:.0f}s)")

    if not results:
        print("\n没有成功的结果，退出。")
        return

    # 按吞吐量排序
    results.sort(key=lambda x: x['throughput'], reverse=True)
    best = results[0]

    print_table(results, best)

    print(f"\n最优配置 (DDP {best['world_size']}-GPU 真实训练):")
    print(f"  samples_per_gpu            = {best['samples_per_gpu']}"
          f"   (global batch = {best['global_batch']})")
    print(f"  workers_per_gpu            = {best['workers_per_gpu']}")
    print(f"  dataloader_prefetch_factor = {best['prefetch_factor']}")
    print(f"  吞吐量: {best['throughput']:.1f} samples/s  "
          f"(median {best['median_ms']:.1f} ms/iter)")

    if args.update_config:
        print(f"\n正在更新配置文件 {args.config} ...")
        update_config(args.config, best)
    else:
        print(f"\n(--no-update-config: 配置文件未修改)")


if __name__ == "__main__":
    main()
