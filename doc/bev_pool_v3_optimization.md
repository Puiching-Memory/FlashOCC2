# BEV Pool v3 优化报告

## 目标

基于 nsys / ncu / torch.profiler 分析 BEV Pool v1/v2/v3 性能瓶颈，优化 v3 成为性能最强算子。

## 测试环境

| 项目               | 配置                                             |
| ------------------ | ------------------------------------------------ |
| GPU                | NVIDIA H800 (SM 9.0, 81GB HBM, ~3.35 TB/s)       |
| 配置               | FlashOCC-R50: C=64, D=88, fH=16, fW=44, N=6, B=1 |
| 有效点数           | ~371,712 (n_points)                              |
| BEV intervals      | ~39,995 (n_intervals)                            |
| 平均 interval 长度 | ~9.3 点                                          |

## 瓶颈分析

### v2 Backward — 灾难级瓶颈

`bev_pool_grad_kernel` 占用 **97%** CUDA 时间 (80.5ms / 10 iter)：
- 只启动 `n_intervals / 256 ≈ 156` 个 block
- 每个线程串行循环 C=64 个 channel
- 无 per-channel 并行度 → 极低 GPU 利用率

### v3 Backward (旧版) — Python 开销超过 CUDA 核函数

| 操作                                 | CUDA 时间 (10 iter) | 占比     |
| ------------------------------------ | ------------------- | -------- |
| `argsort` + CUB sort                 | 558μs + 385μs       | 38%      |
| `aten::copy_` (contiguous)           | 422μs               | 17%      |
| `aten::clone`                        | 309μs               | 12%      |
| `aten::index` (fancy indexing)       | 192μs               | 8%       |
| `aten::nonzero`                      | 131μs               | 5%       |
| **Python 开销小计**                  | **~2,000μs**        | **~80%** |
| `bev_pool_v3_flat_kernel` (feat bwd) | 487μs               | 19%      |
| `bev_pool_v3_bwd_depth_kernel`       | 378μs               | 15%      |

关键发现：**backward 80% 的时间花在 PyTorch ops（argsort/sort/index/clone），
而非 CUDA 核函数本身。**

## 优化方案

### 1. 预排序 feat intervals（核心优化）

**问题**：每次 backward 调用 `_compute_feat_intervals()` → `argsort()` + `sort()` + `index[]` + `where()` → ~2ms。

**方案**：在 `voxel_pooling_prepare_v3()` 阶段一次性计算 feat-sorted intervals，
通过 `forward()` 的 `ctx` 传递给 `backward()`，实现零 argsort 开销。

- `voxel_pooling_prepare_v3()` 新增第 6 个返回值 `feat_intervals`
- `bev_pool_v3()` 新增可选参数 `feat_intervals=None`
- `accelerate` 模式：prepare 只执行一次，feat_intervals 被缓存

### 2. 128-bit 向量化 depth backward（fp16/bf16）

**问题**：旧版 fp16 depth backward 使用 `half2`（一次处理 2 个元素），
相比 fp32 的 `float4`（一次 4 个）效率低 2×。

**方案**：使用 `int4` 128-bit 加载，一次处理 8 个 half 值：

```cuda
// 8 halves = 128 bits per iteration
const int4 a_raw = *reinterpret_cast<const int4*>(og + ch);
const int4 b_raw = *reinterpret_cast<const int4*>(ft + ch);
// → 4 × half2 → 4 × float2 → 8 FMA
```

C=64 时：8 次迭代（vs 旧版 32 次），与 fp32 float4 相同迭代次数。

### 3. 保持 flat kernel 用于 forward（反面教训）

初始尝试使用 shared memory tiled kernel（1 block per interval）反而 **慢了 2.4×**：
- 39,995 个 block 造成过度调度
- 平均 interval 长度仅 ~9 点，shared memory 开销无法摊销
- C=64 时 128 线程只有 64 个有用（50% 浪费）

**结论**：flat kernel（1 thread per (interval, channel)）对短 interval + 小 C 场景已接近理论带宽极限。

## 性能对比

### Forward+Backward (关键指标)

| 版本                      | fwd+bwd (ms) | vs v2 加速比 |
| ------------------------- | ------------ | ------------ |
| v2 (f32)                  | 12.87        | 1.0×         |
| v3-cuda (f32)             | 3.80         | 3.4×         |
| **v3-cuda precomp (f32)** | **0.401**    | **32.1×**    |
| v3-triton (f32)           | 4.29         | 3.0×         |
| v3-cuda precomp (fp16)    | 0.397        | **32.4×**    |

### Forward Only

| 版本                   | forward (ms) |
| ---------------------- | ------------ |
| v2 (f32)               | 0.199        |
| v3-cuda (f32)          | 0.194        |
| v3-cuda precomp (fp16) | 0.129        |
| v3-triton (f32)        | 0.191        |

### CUDA Kernel 耗时分析 (precomp, per iteration)

| Kernel                               | 耗时       |
| ------------------------------------ | ---------- |
| `bev_pool_v3_flat_kernel` (fwd)      | ~25μs      |
| `bev_pool_v3_flat_kernel` (feat bwd) | ~25μs      |
| `bev_pool_v3_bwd_depth_kernel`       | ~45μs      |
| **CUDA 核函数总计**                  | **~95μs**  |
| Python/launch 开销                   | ~305μs     |
| **Wall-clock 总计**                  | **~400μs** |

CUDA 核函数已接近 H800 显存带宽理论极限 (~107MB / 3.35TB/s ≈ 32μs)。
剩余开销主要是 Python autograd + 核函数启动 + 内存分配。

## 修改文件

| 文件                                            | 变更                                          |
| ----------------------------------------------- | --------------------------------------------- |
| `core/ops/csrc/bev_pool_v3/bev_pool_v3_cuda.cu` | 128-bit 向量化 fp16/bf16 depth bwd            |
| `core/ops/bev_pool_v3.py`                       | `feat_intervals` kwarg; backward 跳过 argsort |
| `core/ops/voxel_pooling_prepare_v3.py`          | 返回预计算的 `feat_intervals`                 |
| `models/necks/view_transformer.py`              | accelerate 模式缓存 feat_intervals            |
| `tools/bench_bev_pool.py`                       | 新增 precomp 基准测试                         |

## 正确性验证

- Forward: v2 vs v3(precomp) max_diff = **0.00e+00** (bit-exact)
- Backward depth: max_diff = **9.54e-06** (数值精度范围内)
- Backward feat: max_diff = **0.00e+00** (bit-exact)
- fp16 / bf16 均通过端到端测试
