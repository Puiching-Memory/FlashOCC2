#!/usr/bin/env python3
"""BEV Pool v2 / v3 (CUDA) / v3 (Triton) — 综合性能基准测试.

使用真实维度 (FlashOCC-R50 config):
  C=64, D=88, fH=16, fW=44, N=6, B=1, Dx=Dy=200, Dz=1

输出:
  • forward / backward 延迟 (ms)
  • 吞吐量 (intervals/sec)
  • torch.profiler chrome trace (可选)
"""

from __future__ import annotations

import argparse
import time
import torch
import torch.cuda

# ─── 项目导入 ───
from flashocc.core.ops import bev_pool_v2, bev_pool_v3, bev_pool_v3_triton
from flashocc.core.ops.voxel_pooling_prepare_v3 import (
    voxel_pooling_prepare_v3_pytorch,
)

try:
    from flashocc.core.ops.voxel_pooling_prepare_v3 import voxel_pooling_prepare_v3
    _HAS_FUSED_PREPARE = True
except ImportError:
    _HAS_FUSED_PREPARE = False


# ─── 数据生成 ───
def generate_test_data(
    B: int = 1, N: int = 6, D: int = 88, fH: int = 16, fW: int = 44,
    C: int = 64, Dx: int = 200, Dy: int = 200, Dz: int = 1,
    dtype=torch.float32, device: str = "cuda:0",
):
    """生成用于 BEV pooling 的测试数据.

    Returns:
        depth, feat, ranks_depth, ranks_feat, ranks_bev,
        bev_feat_shape, interval_starts, interval_lengths, feat_intervals
    """
    torch.manual_seed(42)

    # 生成随机坐标 (模拟 ego 坐标系)
    # x in [-40, 40], y in [-40, 40], z in [-1, 5.4]
    coor = torch.zeros(B, N, D, fH, fW, 3, device=device)
    coor[..., 0] = torch.rand(B, N, D, fH, fW, device=device) * 80 - 40    # x
    coor[..., 1] = torch.rand(B, N, D, fH, fW, device=device) * 80 - 40    # y
    coor[..., 2] = torch.rand(B, N, D, fH, fW, device=device) * 6.4 - 1    # z

    grid_lower = torch.tensor([-40.0, -40.0, -1.0], device=device)
    grid_intv  = torch.tensor([0.4, 0.4, 6.4], device=device)
    grid_size  = torch.tensor([200.0, 200.0, 1.0], device=device)

    result = voxel_pooling_prepare_v3_pytorch(coor, grid_lower, grid_intv, grid_size)
    ranks_bev, ranks_depth, ranks_feat, interval_starts, interval_lengths = result[:5]
    feat_intervals = result[5] if len(result) > 5 else None

    depth = torch.rand(B, N, D, fH, fW, device=device, dtype=dtype)
    depth = depth.softmax(dim=2)  # 模拟 softmax 后的深度分布
    feat  = torch.randn(B, N, fH, fW, C, device=device, dtype=dtype)

    bev_feat_shape = (B, Dz, Dy, Dx, C)

    return (depth, feat, ranks_depth, ranks_feat, ranks_bev,
            bev_feat_shape, interval_starts, interval_lengths, feat_intervals)


# ─── Benchmark 核心 ───
def bench_forward(pool_fn, data, warmup=50, repeat=200, label="", feat_intervals=None):
    """Benchmark forward pass only."""
    depth, feat, rd, rf, rb, shape, starts, lengths = data[:8]
    fi = feat_intervals
    kwargs = {}
    if fi is not None and pool_fn is bev_pool_v3:
        kwargs['feat_intervals'] = fi
    # warmup
    for _ in range(warmup):
        out = pool_fn(depth, feat, rd, rf, rb, shape, starts, lengths, **kwargs)
    torch.cuda.synchronize()

    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev   = torch.cuda.Event(enable_timing=True)
    start_ev.record()
    for _ in range(repeat):
        out = pool_fn(depth, feat, rd, rf, rb, shape, starts, lengths, **kwargs)
    end_ev.record()
    torch.cuda.synchronize()

    ms = start_ev.elapsed_time(end_ev) / repeat
    n_intervals = starts.shape[0]
    print(f"  [{label}] forward : {ms:.4f} ms  "
          f"({n_intervals/ms/1e3:.1f} M-intervals/s, "
          f"n_points={rd.shape[0]}, n_intervals={n_intervals})")
    return ms


def bench_backward(pool_fn, data, warmup=30, repeat=100, label="", feat_intervals=None):
    """Benchmark forward + backward."""
    depth, feat, rd, rf, rb, shape, starts, lengths = data[:8]
    fi = feat_intervals
    depth = depth.clone().requires_grad_(True)
    feat  = feat.clone().requires_grad_(True)
    kwargs = {}
    if fi is not None and pool_fn is bev_pool_v3:
        kwargs['feat_intervals'] = fi

    # warmup
    for _ in range(warmup):
        out = pool_fn(depth, feat, rd, rf, rb, shape, starts, lengths, **kwargs)
        loss = out.sum()
        loss.backward()
        depth.grad = None
        feat.grad = None
    torch.cuda.synchronize()

    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev   = torch.cuda.Event(enable_timing=True)
    start_ev.record()
    for _ in range(repeat):
        out = pool_fn(depth, feat, rd, rf, rb, shape, starts, lengths, **kwargs)
        loss = out.sum()
        loss.backward()
        depth.grad = None
        feat.grad = None
    end_ev.record()
    torch.cuda.synchronize()

    ms = start_ev.elapsed_time(end_ev) / repeat
    print(f"  [{label}] fwd+bwd: {ms:.4f} ms")
    return ms


def bench_prepare(coor_shape, device, warmup=20, repeat=100):
    """Benchmark voxel_pooling_prepare."""
    B, N, D, fH, fW = coor_shape
    coor = torch.zeros(B, N, D, fH, fW, 3, device=device)
    coor[..., 0] = torch.rand(B, N, D, fH, fW, device=device) * 80 - 40
    coor[..., 1] = torch.rand(B, N, D, fH, fW, device=device) * 80 - 40
    coor[..., 2] = torch.rand(B, N, D, fH, fW, device=device) * 6.4 - 1

    grid_lower = torch.tensor([-40.0, -40.0, -1.0], device=device)
    grid_intv  = torch.tensor([0.4, 0.4, 6.4], device=device)
    grid_size  = torch.tensor([200.0, 200.0, 1.0], device=device)

    # pytorch
    for _ in range(warmup):
        voxel_pooling_prepare_v3_pytorch(coor, grid_lower, grid_intv, grid_size)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeat):
        voxel_pooling_prepare_v3_pytorch(coor, grid_lower, grid_intv, grid_size)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    pytorch_ms = (t1 - t0) / repeat * 1000
    print(f"  [PyTorch]  prepare: {pytorch_ms:.4f} ms")

    if _HAS_FUSED_PREPARE:
        for _ in range(warmup):
            voxel_pooling_prepare_v3(coor, grid_lower, grid_intv, grid_size)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(repeat):
            voxel_pooling_prepare_v3(coor, grid_lower, grid_intv, grid_size)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        fused_ms = (t1 - t0) / repeat * 1000
        print(f"  [Fused v3] prepare: {fused_ms:.4f} ms")


def correctness_check(data):
    """检查 v2, v3-CUDA, v3-CUDA(precomp), v3-Triton 输出一致性."""
    depth, feat, rd, rf, rb, shape, starts, lengths = data[:8]
    fi = data[8] if len(data) > 8 else None
    d_f32 = depth.float()
    f_f32 = feat.float()

    out_v2 = bev_pool_v2(d_f32, f_f32, rd, rf, rb, shape, starts, lengths)
    out_v3 = bev_pool_v3(d_f32, f_f32, rd, rf, rb, shape, starts, lengths)
    out_v3p = bev_pool_v3(d_f32, f_f32, rd, rf, rb, shape, starts, lengths,
                          feat_intervals=fi) if fi else out_v3
    out_tri = bev_pool_v3_triton(d_f32, f_f32, rd, rf, rb, shape, starts, lengths)

    diff_v3  = (out_v2 - out_v3).abs().max().item()
    diff_v3p = (out_v2 - out_v3p).abs().max().item()
    diff_tri = (out_v2 - out_tri).abs().max().item()
    print(f"  v2 vs v3-cuda: {diff_v3:.2e}")
    print(f"  v2 vs v3-cuda(precomp): {diff_v3p:.2e}")
    print(f"  v2 vs v3-triton: {diff_tri:.2e}")

    passed = diff_v3 < 1e-4 and diff_v3p < 1e-4 and diff_tri < 1e-4
    if not passed:
        print("  *** CORRECTNESS FAILED ***")
    else:
        print("  ✓ correctness OK")

    # Also check backward correctness
    d_v2 = d_f32.clone().requires_grad_(True)
    f_v2 = f_f32.clone().requires_grad_(True)
    d_v3 = d_f32.clone().requires_grad_(True)
    f_v3 = f_f32.clone().requires_grad_(True)
    d_v3p = d_f32.clone().requires_grad_(True)
    f_v3p = f_f32.clone().requires_grad_(True)

    bev_pool_v2(d_v2, f_v2, rd, rf, rb, shape, starts, lengths).sum().backward()
    bev_pool_v3(d_v3, f_v3, rd, rf, rb, shape, starts, lengths).sum().backward()
    bev_pool_v3(d_v3p, f_v3p, rd, rf, rb, shape, starts, lengths,
                feat_intervals=fi).sum().backward() if fi else None

    if fi:
        dd = (d_v2.grad - d_v3p.grad).abs().max().item()
        df = (f_v2.grad - f_v3p.grad).abs().max().item()
        print(f"  backward (precomp): depth_grad diff={dd:.2e}, feat_grad diff={df:.2e}")
        passed = passed and dd < 1e-3 and df < 1e-3

    dd = (d_v2.grad - d_v3.grad).abs().max().item()
    df = (f_v2.grad - f_v3.grad).abs().max().item()
    print(f"  backward (no precomp): depth_grad diff={dd:.2e}, feat_grad diff={df:.2e}")
    passed = passed and dd < 1e-3 and df < 1e-3

    if passed:
        print("  ✓ all correctness checks passed")
    else:
        print("  *** BACKWARD CORRECTNESS FAILED ***")
    return passed


def profile_with_torch_profiler(pool_fn, data, label="", feat_intervals=None):
    """使用 torch.profiler 生成 chrome trace."""
    depth, feat, rd, rf, rb, shape, starts, lengths = data[:8]
    fi = feat_intervals
    depth = depth.clone().requires_grad_(True)
    feat  = feat.clone().requires_grad_(True)
    kwargs = {}
    if fi is not None and pool_fn is bev_pool_v3:
        kwargs['feat_intervals'] = fi

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        with_stack=True,
    ) as prof:
        for _ in range(10):
            out = pool_fn(depth, feat, rd, rf, rb, shape, starts, lengths, **kwargs)
            loss = out.sum()
            loss.backward()
            depth.grad = None
            feat.grad = None

    trace_file = f"work_dirs/trace_bev_pool_{label}.json"
    prof.export_chrome_trace(trace_file)
    print(f"  torch.profiler trace → {trace_file}")

    # 打印 CUDA kernel 时间汇总
    print(prof.key_averages().table(
        sort_by="cuda_time_total", row_limit=20))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="float32",
                        choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--profile", action="store_true",
                        help="Generate torch.profiler traces")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeat", type=int, default=200)
    parser.add_argument("--B", type=int, default=1)
    args = parser.parse_args()

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]
    device = args.device

    print(f"═══ BEV Pool Benchmark ═══")
    print(f"  GPU: {torch.cuda.get_device_name(device)}")
    print(f"  dtype: {args.dtype}, B={args.B}")
    print(f"  Config: C=64, D=88, fH=16, fW=44, N=6, Dx=Dy=200, Dz=1")
    print()

    data = generate_test_data(B=args.B, dtype=dtype, device=device)
    fi = data[8] if len(data) > 8 else None

    print("─── Correctness ───")
    correctness_check(data)
    print()

    print("─── Prepare (voxel_pooling_prepare) ───")
    bench_prepare((args.B, 6, 88, 16, 44), device)
    print()

    print(f"─── Forward ({args.dtype}) ───")
    # v2 only supports float32
    if dtype == torch.float32:
        bench_forward(bev_pool_v2, data, args.warmup, args.repeat, "v2")
    bench_forward(bev_pool_v3, data, args.warmup, args.repeat, "v3-cuda")
    bench_forward(bev_pool_v3, data, args.warmup, args.repeat, "v3-cuda(precomp)", feat_intervals=fi)
    bench_forward(bev_pool_v3_triton, data, args.warmup, args.repeat, "v3-triton")
    print()

    print(f"─── Forward+Backward ({args.dtype}) ───")
    if dtype == torch.float32:
        bench_backward(bev_pool_v2, data, 30, 100, "v2")
    bench_backward(bev_pool_v3, data, 30, 100, "v3-cuda")
    bench_backward(bev_pool_v3, data, 30, 100, "v3-cuda(precomp)", feat_intervals=fi)
    bench_backward(bev_pool_v3_triton, data, 30, 100, "v3-triton")
    print()

    # half precision comparison
    if dtype == torch.float32:
        print("─── Forward (float16 comparison) ───")
        data_f16 = generate_test_data(B=args.B, dtype=torch.float16, device=device)
        fi_f16 = data_f16[8] if len(data_f16) > 8 else None
        bench_forward(bev_pool_v3, data_f16, args.warmup, args.repeat, "v3-cuda-fp16")
        bench_forward(bev_pool_v3, data_f16, args.warmup, args.repeat, "v3-cuda-fp16(precomp)", feat_intervals=fi_f16)
        bench_forward(bev_pool_v3_triton, data_f16, args.warmup, args.repeat, "v3-triton-fp16")
        print()

        print("─── Forward+Backward (float16) ───")
        bench_backward(bev_pool_v3, data_f16, 30, 100, "v3-cuda-fp16")
        bench_backward(bev_pool_v3, data_f16, 30, 100, "v3-cuda-fp16(precomp)", feat_intervals=fi_f16)
        bench_backward(bev_pool_v3_triton, data_f16, 30, 100, "v3-triton-fp16")
        print()

        print("─── Forward (bfloat16 comparison) ───")
        data_bf16 = generate_test_data(B=args.B, dtype=torch.bfloat16, device=device)
        fi_bf16 = data_bf16[8] if len(data_bf16) > 8 else None
        bench_forward(bev_pool_v3, data_bf16, args.warmup, args.repeat, "v3-cuda-bf16")
        bench_forward(bev_pool_v3, data_bf16, args.warmup, args.repeat, "v3-cuda-bf16(precomp)", feat_intervals=fi_bf16)
        bench_forward(bev_pool_v3_triton, data_bf16, args.warmup, args.repeat, "v3-triton-bf16")
        print()

    if args.profile:
        print("─── torch.profiler Traces ───")
        data_f32 = generate_test_data(B=args.B, dtype=torch.float32, device=device)
        fi_f32 = data_f32[8] if len(data_f32) > 8 else None
        if dtype == torch.float32:
            profile_with_torch_profiler(bev_pool_v2, data_f32, "v2_f32")
        profile_with_torch_profiler(bev_pool_v3, data_f32, "v3_cuda_f32")
        profile_with_torch_profiler(bev_pool_v3, data_f32, "v3_cuda_precomp_f32",
                                    feat_intervals=fi_f32)
        profile_with_torch_profiler(bev_pool_v3_triton, data_f32, "v3_triton_f32")


if __name__ == "__main__":
    main()
