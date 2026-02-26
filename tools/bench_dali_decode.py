#!/usr/bin/env python
"""NVIDIA DALI GPU JPEG 解码 vs CPU 解码 性能对比测试.

测试场景: NuScenes 6 路环视相机图像 (1600×900 JPEG) 的解码 + Resize + Normalize
===========================================================================
- **CPU 基线**: PIL.Image.open → resize → crop → numpy → normalize → torch.Tensor
  (FlashOCC2 当前管道，仅使用 CPU)
- **DALI GPU**: nvidia.dali 管道，JPEG 硬解 (nvJPEG) → GPU Resize → GPU Crop → GPU Normalize
  (全部在 GPU 上完成, CPU 零拷贝)

使用方法:
    python tools/bench_dali_decode.py [--num-samples 200] [--batch-size 6] [--warmup 20]
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

# =====================================================================
#  项目路径设置
# =====================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from flashocc.constants import IMAGENET_MEAN, IMAGENET_STD


# =====================================================================
#  工具函数
# =====================================================================

def collect_image_paths(pkl_path: str, num_samples: int) -> list[list[str]]:
    """从 nuScenes infos pkl 中提取前 N 个样本的 6 路相机图像路径.

    Returns:
        list of [cam_front_left, cam_front, ..., cam_back_right]  (6 paths per sample)
    """
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    infos = data["infos"][:num_samples]

    cam_order = [
        "CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
        "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT",
    ]
    samples = []
    for info in infos:
        paths = [info["cams"][cam]["data_path"] for cam in cam_order]
        # 验证文件存在
        for p in paths:
            assert os.path.isfile(p), f"Image not found: {p}"
        samples.append(paths)
    return samples


# =====================================================================
#  方案 A: CPU 基线 (PIL — 当前 FlashOCC2 管道的核心路径)
# =====================================================================

def decode_cpu_pil(image_paths: list[str],
                   target_h: int = 256,
                   target_w: int = 704) -> torch.Tensor:
    """PIL CPU 解码 + resize + crop + normalize → (N, 3, H, W) float32 Tensor.

    精确复刻 PrepareImageInputs.get_inputs() 中的图像处理逻辑:
      1. PIL.Image.open(path)         — CPU JPEG 软解码
      2. img.resize(resize_dims)      — Lanczos resize
      3. img.crop(crop)               — center crop to (target_h, target_w)
      4. numpy → float32 → normalize  — ImageNet 归一化
      5. torch.Tensor permute          — HWC → CHW
    """
    from PIL import Image

    src_h, src_w = 900, 1600
    resize_scale = float(target_w) / float(src_w)
    resize_dims = (int(src_w * resize_scale), int(src_h * resize_scale))
    new_w, new_h = resize_dims
    # center crop
    crop_h = int(new_h) - target_h
    crop_w = int(max(0, new_w - target_w) / 2)
    crop_box = (crop_w, crop_h, crop_w + target_w, crop_h + target_h)

    mean = np.float64(IMAGENET_MEAN.reshape(1, -1))
    stdinv = 1.0 / np.float64(IMAGENET_STD.reshape(1, -1))

    imgs = []
    for path in image_paths:
        img = Image.open(path)
        img = img.resize(resize_dims)
        img = img.crop(crop_box)
        arr = np.array(img).astype(np.float32)
        # BGR → RGB (原始代码 _imnormalize to_rgb=True, 但 PIL 已是 RGB，故此处
        # 直接对 RGB 归一化与原管道保持一致)
        arr = (arr - mean) * stdinv
        t = torch.from_numpy(np.ascontiguousarray(arr)).float().permute(2, 0, 1).contiguous()
        imgs.append(t)
    return torch.stack(imgs)  # (N, 3, H, W)


# =====================================================================
#  方案 B: NVIDIA DALI GPU 解码管道
# =====================================================================

def build_dali_pipeline(file_list_flat: list[str],
                        batch_size: int = 6,
                        target_h: int = 256,
                        target_w: int = 704,
                        device_id: int = 0,
                        num_threads: int = 4,
                        prefetch_queue_depth: int = 2):
    """构建 DALI 管道: nvJPEG GPU 硬解码 → GPU Resize → GPU Crop → GPU Normalize.

    管道流程:
        FileRead (CPU) → ImageDecoder (mixed/GPU) → Resize (GPU) → Crop (GPU)
                       → CropMirrorNormalize (GPU) → 输出 float32 NCHW Tensor
    """
    import nvidia.dali.fn as fn
    import nvidia.dali.types as types
    from nvidia.dali.pipeline import Pipeline

    src_h, src_w = 900, 1600
    resize_scale = float(target_w) / float(src_w)
    resize_h = int(src_h * resize_scale)
    resize_w = int(src_w * resize_scale)
    # center crop 偏移 (与 CPU baseline 一致)
    crop_y = resize_h - target_h
    crop_x = max(0, resize_w - target_w) // 2

    # ImageNet normalize (RGB, 0-255 range)
    # DALI CropMirrorNormalize expects mean/std in [0,255] range
    mean_rgb = [float(IMAGENET_MEAN[0]), float(IMAGENET_MEAN[1]), float(IMAGENET_MEAN[2])]
    std_rgb = [float(IMAGENET_STD[0]), float(IMAGENET_STD[1]), float(IMAGENET_STD[2])]

    pipe = Pipeline(batch_size=batch_size,
                    num_threads=num_threads,
                    device_id=device_id,
                    prefetch_queue_depth=prefetch_queue_depth)

    with pipe:
        jpegs, _ = fn.readers.file(files=file_list_flat,
                                   random_shuffle=False,
                                   name="Reader")
        # mixed backend: CPU 读取 → GPU nvJPEG 硬件解码
        images = fn.decoders.image(jpegs,
                                   device="mixed",        # GPU 解码
                                   output_type=types.RGB,
                                   hw_decoder_load=0.7)   # 优先使用硬件解码器

        # GPU Resize (双线性插值)
        images = fn.resize(images,
                           resize_x=resize_w,
                           resize_y=resize_h,
                           interp_type=types.INTERP_LINEAR)

        # GPU Crop + Normalize + HWC→CHW (合并到一个算子, 减少 kernel launch)
        images = fn.crop_mirror_normalize(
            images,
            crop=(target_h, target_w),
            crop_pos_x=crop_x / max(resize_w - target_w, 1),
            crop_pos_y=crop_y / max(resize_h - target_h, 1),
            mean=mean_rgb,
            std=std_rgb,
            output_layout="CHW",
            dtype=types.FLOAT,
        )

        pipe.set_outputs(images)

    pipe.build()
    return pipe


def decode_gpu_dali(pipe, batch_size: int = 6) -> torch.Tensor:
    """从已构建的 DALI 管道取一个 batch, 返回 (N, 3, H, W) GPU Tensor."""
    output = pipe.run()
    # DALI 输出为 TensorListGPU; 使用 as_tensor() 逐个转换为 DLPack → PyTorch Tensor
    dali_tensor = output[0]  # TensorListGPU
    tensors = []
    for i in range(batch_size):
        t = torch.empty(dali_tensor[i].shape(),
                        dtype=torch.float32,
                        device=f"cuda:{dali_tensor[i].device_id()}")
        # DALI → PyTorch 零拷贝
        from nvidia.dali.backend import TensorGPU
        import nvidia.dali.plugin.pytorch as dali_pytorch
        feed_ndarray = dali_pytorch.feed_ndarray
        feed_ndarray(dali_tensor[i], t)
        tensors.append(t)
    return torch.stack(tensors)


# =====================================================================
#  方案 C: torchvision + nvJPEG (纯 PyTorch GPU 解码, 作为额外对比)
# =====================================================================

def decode_gpu_torchvision(image_paths: list[str],
                           target_h: int = 256,
                           target_w: int = 704,
                           device: str = "cuda:0") -> torch.Tensor:
    """torchvision.io.decode_jpeg GPU 解码 (需要 torchvision >= 0.14).

    流程: 读取 JPEG bytes → GPU 解码 → GPU resize → GPU crop → normalize
    """
    import torchvision.transforms.functional as TF
    from torchvision.io import decode_jpeg, ImageReadMode

    src_h, src_w = 900, 1600
    resize_scale = float(target_w) / float(src_w)
    resize_dims = (int(src_h * resize_scale), int(src_w * resize_scale))
    new_h, new_w = resize_dims
    crop_h = new_h - target_h
    crop_w = max(0, new_w - target_w) // 2

    mean = torch.tensor([IMAGENET_MEAN[0], IMAGENET_MEAN[1], IMAGENET_MEAN[2]],
                        device=device).view(3, 1, 1) / 255.0
    std = torch.tensor([IMAGENET_STD[0], IMAGENET_STD[1], IMAGENET_STD[2]],
                       device=device).view(3, 1, 1) / 255.0

    imgs = []
    for path in image_paths:
        # 读取原始 JPEG bytes
        jpeg_bytes = torch.from_numpy(
            np.fromfile(path, dtype=np.uint8))
        # GPU 解码 (nvJPEG)
        img = decode_jpeg(jpeg_bytes, device=device, mode=ImageReadMode.RGB)  # (3, H, W) uint8
        # GPU resize
        img = TF.resize(img, resize_dims)  # (3, new_h, new_w)
        # GPU crop
        img = img[:, crop_h:crop_h + target_h, crop_w:crop_w + target_w]
        # normalize
        img = img.float() / 255.0
        img = (img - mean) / std
        imgs.append(img)
    return torch.stack(imgs)


# =====================================================================
#  Benchmark 运行器
# =====================================================================

class BenchmarkResult:
    def __init__(self, name: str, times: list[float], num_images: int):
        self.name = name
        self.times = times
        self.num_images = num_images

    @property
    def total(self) -> float:
        return sum(self.times)

    @property
    def mean(self) -> float:
        return np.mean(self.times)

    @property
    def std(self) -> float:
        return np.std(self.times)

    @property
    def median(self) -> float:
        return np.median(self.times)

    @property
    def images_per_sec(self) -> float:
        return self.num_images / self.total if self.total > 0 else 0

    @property
    def ms_per_image(self) -> float:
        return (self.mean * 1000) / 6  # 6 images per sample

    def __str__(self) -> str:
        return (
            f"  {self.name}:\n"
            f"    总耗时:        {self.total:.3f}s ({self.num_images} 张图像)\n"
            f"    每样本 (6图):  {self.mean * 1000:.2f} ± {self.std * 1000:.2f} ms\n"
            f"    每张图像:      {self.ms_per_image:.2f} ms\n"
            f"    中位数:        {self.median * 1000:.2f} ms/sample\n"
            f"    吞吐量:        {self.images_per_sec:.1f} images/s"
        )


def bench_cpu_pil(samples: list[list[str]], warmup: int) -> BenchmarkResult:
    """Benchmark CPU PIL 解码."""
    print(f"\n[CPU PIL] Warming up ({warmup} samples)...")
    for i in range(min(warmup, len(samples))):
        _ = decode_cpu_pil(samples[i])

    print(f"[CPU PIL] Benchmarking ({len(samples)} samples)...")
    times = []
    for paths in samples:
        t0 = time.perf_counter()
        result = decode_cpu_pil(paths)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return BenchmarkResult("CPU PIL (baseline)", times, len(samples) * 6)


def bench_dali_gpu(samples: list[list[str]], warmup: int) -> BenchmarkResult:
    """Benchmark DALI GPU 解码."""
    import nvidia.dali

    # DALI 需要一个平坦的文件列表, batch_size=6 (每样本 6 张)
    flat_files = [p for sample in samples for p in sample]

    pipe = build_dali_pipeline(flat_files, batch_size=6)

    print(f"\n[DALI GPU] Warming up ({warmup} iterations)...")
    for _ in range(warmup):
        try:
            _ = pipe.run()
        except StopIteration:
            pipe.reset()
            _ = pipe.run()

    # 重建管道以重置 reader
    pipe = build_dali_pipeline(flat_files, batch_size=6)

    print(f"[DALI GPU] Benchmarking ({len(samples)} samples)...")
    torch.cuda.synchronize()
    times = []
    for i in range(len(samples)):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            output = pipe.run()
        except StopIteration:
            break
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)
        # 取出数据确保内存同步
        dali_tensor = output[0]

    return BenchmarkResult("DALI GPU (nvJPEG)", times, len(times) * 6)


def bench_torchvision_gpu(samples: list[list[str]], warmup: int) -> BenchmarkResult:
    """Benchmark torchvision GPU 解码 (nvJPEG via torchvision.io.decode_jpeg)."""
    # 检查是否支持 GPU 解码
    try:
        from torchvision.io import decode_jpeg
        test_bytes = torch.from_numpy(np.fromfile(samples[0][0], dtype=np.uint8))
        _ = decode_jpeg(test_bytes, device="cuda:0")
        has_gpu_decode = True
    except Exception as e:
        print(f"  [WARN] torchvision GPU decode 不可用: {e}")
        has_gpu_decode = False

    if not has_gpu_decode:
        return BenchmarkResult("torchvision GPU (N/A)", [0.0], 0)

    print(f"\n[torchvision GPU] Warming up ({warmup} samples)...")
    for i in range(min(warmup, len(samples))):
        _ = decode_gpu_torchvision(samples[i])
    torch.cuda.synchronize()

    print(f"[torchvision GPU] Benchmarking ({len(samples)} samples)...")
    times = []
    for paths in samples:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = decode_gpu_torchvision(paths)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return BenchmarkResult("torchvision GPU (nvJPEG)", times, len(samples) * 6)


# =====================================================================
#  数值一致性检查
# =====================================================================

def check_consistency(samples: list[list[str]]) -> None:
    """对比 CPU 和 GPU 解码结果的数值差异."""
    print("\n" + "=" * 60)
    print("  数值一致性检查 (首个样本, 6 张图像)")
    print("=" * 60)

    paths = samples[0]
    cpu_out = decode_cpu_pil(paths)  # (6, 3, 256, 704)

    # DALI GPU
    try:
        import nvidia.dali
        import nvidia.dali.plugin.pytorch as dali_pytorch
        flat = paths
        pipe = build_dali_pipeline(flat, batch_size=6)
        output = pipe.run()
        dali_tensor = output[0]

        gpu_tensors = []
        for i in range(6):
            t = torch.empty(dali_tensor[i].shape(),
                            dtype=torch.float32, device=f"cuda:{dali_tensor[i].device_id()}")
            dali_pytorch.feed_ndarray(dali_tensor[i], t)
            gpu_tensors.append(t)
        dali_out = torch.stack(gpu_tensors).cpu()

        diff = (cpu_out - dali_out).abs()
        print(f"  CPU vs DALI GPU:")
        print(f"    max |diff|  = {diff.max().item():.6f}")
        print(f"    mean |diff| = {diff.mean().item():.6f}")
        print(f"    shape: CPU={cpu_out.shape}, DALI={dali_out.shape}")
    except Exception as e:
        print(f"  DALI 对比跳过: {e}")

    # torchvision GPU
    try:
        from torchvision.io import decode_jpeg
        tv_out = decode_gpu_torchvision(paths).cpu()
        diff = (cpu_out - tv_out).abs()
        print(f"  CPU vs torchvision GPU:")
        print(f"    max |diff|  = {diff.max().item():.6f}")
        print(f"    mean |diff| = {diff.mean().item():.6f}")
    except Exception as e:
        print(f"  torchvision 对比跳过: {e}")


# =====================================================================
#  Main
# =====================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="DALI GPU JPEG Decode Benchmark for NuScenes")
    parser.add_argument("--pkl", type=str,
                        default="data/flashocc2-nuscenes_infos_train.pkl",
                        help="nuScenes infos pickle 文件路径")
    parser.add_argument("--num-samples", type=int, default=200,
                        help="测试样本数 (每样本 6 张图像)")
    parser.add_argument("--warmup", type=int, default=20,
                        help="预热迭代次数")
    parser.add_argument("--target-h", type=int, default=256,
                        help="目标图像高度")
    parser.add_argument("--target-w", type=int, default=704,
                        help="目标图像宽度")
    parser.add_argument("--skip-consistency", action="store_true",
                        help="跳过数值一致性检查")
    parser.add_argument("--skip-torchvision", action="store_true",
                        help="跳过 torchvision GPU 测试")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("  NVIDIA DALI GPU JPEG Decode Benchmark")
    print("  NuScenes 6-Camera Surround View")
    print("=" * 60)

    # 环境信息
    print(f"\n  PyTorch:    {torch.__version__}")
    print(f"  CUDA:       {torch.version.cuda}")
    print(f"  GPU:        {torch.cuda.get_device_name(0)}")
    try:
        import nvidia.dali
        print(f"  DALI:       {nvidia.dali.__version__}")
    except ImportError:
        print("  DALI:       未安装")
    try:
        import torchvision
        print(f"  torchvision: {torchvision.__version__}")
    except ImportError:
        print("  torchvision: 未安装")

    print(f"\n  测试参数:")
    print(f"    样本数:       {args.num_samples} (共 {args.num_samples * 6} 张图像)")
    print(f"    原始尺寸:     1600 × 900 JPEG")
    print(f"    目标尺寸:     {args.target_w} × {args.target_h}")
    print(f"    预热:         {args.warmup} iterations")

    # 收集图像路径
    print(f"\n  加载数据信息: {args.pkl}")
    samples = collect_image_paths(args.pkl, args.num_samples)
    print(f"  收集到 {len(samples)} 个样本, 共 {len(samples) * 6} 张图像")

    # 数值一致性检查
    if not args.skip_consistency:
        check_consistency(samples)

    # ---- Benchmark ----
    results = []

    # 1) CPU PIL Baseline
    results.append(bench_cpu_pil(samples, args.warmup))

    # 2) DALI GPU
    try:
        results.append(bench_dali_gpu(samples, args.warmup))
    except Exception as e:
        print(f"  [ERROR] DALI benchmark failed: {e}")

    # 3) torchvision GPU
    if not args.skip_torchvision:
        try:
            results.append(bench_torchvision_gpu(samples, args.warmup))
        except Exception as e:
            print(f"  [ERROR] torchvision benchmark failed: {e}")

    # ---- 结果汇总 ----
    print("\n" + "=" * 60)
    print("  Benchmark 结果")
    print("=" * 60)
    for r in results:
        print(r)
        print()

    # 加速比
    if len(results) >= 2 and results[0].total > 0:
        baseline = results[0].mean
        print("-" * 60)
        print("  加速比 (相对于 CPU PIL baseline):")
        for r in results[1:]:
            if r.mean > 0:
                speedup = baseline / r.mean
                print(f"    {r.name}: {speedup:.2f}x")
        print("-" * 60)

    # 训练影响估算
    if len(results) >= 2 and results[0].mean > 0 and results[1].mean > 0:
        cpu_ms = results[0].mean * 1000
        gpu_ms = results[1].mean * 1000
        saved_ms = cpu_ms - gpu_ms
        print(f"\n  训练影响估算:")
        print(f"    每样本节省:     {saved_ms:.1f} ms")
        print(f"    batch_size=4:   {saved_ms * 4:.1f} ms/iter")
        print(f"    28130 训练样本: {saved_ms * 28130 / 1000:.1f}s/epoch")
        if saved_ms > 0:
            print(f"    24 epochs 共省: {saved_ms * 28130 * 24 / 1000 / 60:.1f} min")

    print("\n  ✓ Benchmark 完成")


if __name__ == "__main__":
    main()
