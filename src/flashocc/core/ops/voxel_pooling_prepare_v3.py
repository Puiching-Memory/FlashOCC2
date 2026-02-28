"""voxel_pooling_prepare v3 — 融合 CUDA 核函数实现.

将 v2 的 ≈10 个独立 PyTorch 操作融合为:
  1. 一个 CUDA kernel: 坐标变换 + 边界检查 + rank 计算 + 流式压缩
  2. torch.sort (内部使用 CUB DeviceRadixSort)
  3. 一个 CUDA kernel: 区间检测 (或等价的 PyTorch fallback)

同时提供纯 PyTorch fallback (voxel_pooling_prepare_v3_pytorch).
"""

from __future__ import annotations

import torch

try:
    from flashocc.core.ops._ext import bev_pool_v3_ext
    _HAS_V3_CUDA = True
except ImportError:
    _HAS_V3_CUDA = False

__all__ = ["voxel_pooling_prepare_v3", "voxel_pooling_prepare_v3_pytorch"]


# =====================================================================
#                  FUSED CUDA IMPLEMENTATION
# =====================================================================

def _compute_intervals_cuda(sorted_keys: torch.Tensor):
    """Use CUDA kernel to detect interval starts, then diff for lengths."""
    n = sorted_keys.shape[0]
    flags = torch.empty(n, dtype=torch.int32, device=sorted_keys.device)
    bev_pool_v3_ext.compute_intervals_v3(sorted_keys.int().contiguous(), flags)
    torch.cuda.current_stream().synchronize()

    starts = torch.where(flags == 1)[0].int()
    if starts.numel() == 0:
        return None, None
    lengths = torch.zeros_like(starts)
    lengths[:-1] = starts[1:] - starts[:-1]
    lengths[-1] = n - starts[-1]
    return starts.contiguous(), lengths.contiguous()


def voxel_pooling_prepare_v3(coor, grid_lower_bound, grid_interval, grid_size):
    """Fused voxel pooling prepare using CUDA kernels.

    Args:
        coor: (B, N, D, H, W, 3) float — 世界坐标 (ego 坐标系)
        grid_lower_bound: (3,) float tensor — (min_x, min_y, min_z)
        grid_interval: (3,) float tensor — (dx, dy, dz)
        grid_size: (3,) float tensor — (Dx, Dy, Dz)

    Returns:
        ranks_bev:        (N_points,) int  — BEV 网格索引
        ranks_depth:      (N_points,) int  — depth 平坦索引
        ranks_feat:       (N_points,) int  — feat 平坦索引
        interval_starts:  (N_intervals,) int
        interval_lengths: (N_intervals,) int
    """
    if not _HAS_V3_CUDA:
        return voxel_pooling_prepare_v3_pytorch(coor, grid_lower_bound,
                                                 grid_interval, grid_size)

    B, N, D, H, W, _ = coor.shape
    num_points = B * N * D * H * W
    device = coor.device

    # Flatten coordinates to (num_points, 3)
    coor_flat = coor.reshape(num_points, 3).contiguous().float()

    # Pre-allocate output arrays (max size = num_points, trimmed after)
    ranks_bev   = torch.empty(num_points, dtype=torch.int32, device=device)
    ranks_depth = torch.empty(num_points, dtype=torch.int32, device=device)
    ranks_feat  = torch.empty(num_points, dtype=torch.int32, device=device)
    counter     = torch.zeros(1, dtype=torch.int32, device=device)

    lower = grid_lower_bound.float()
    intv  = grid_interval.float()
    gsz   = grid_size.float()

    Dx, Dy, Dz = int(gsz[0].item()), int(gsz[1].item()), int(gsz[2].item())

    n_valid = bev_pool_v3_ext.voxel_pooling_prepare_v3_fused(
        coor_flat,
        ranks_bev, ranks_depth, ranks_feat, counter,
        float(lower[0]), float(lower[1]), float(lower[2]),
        float(intv[0]),  float(intv[1]),  float(intv[2]),
        Dx, Dy, Dz,
        B, N, D, H, W)

    if n_valid == 0:
        return None, None, None, None, None

    # Trim to valid points
    ranks_bev   = ranks_bev[:n_valid].contiguous()
    ranks_depth = ranks_depth[:n_valid].contiguous()
    ranks_feat  = ranks_feat[:n_valid].contiguous()

    # Sort by ranks_bev
    order = ranks_bev.long().argsort()
    ranks_bev   = ranks_bev[order]
    ranks_depth = ranks_depth[order]
    ranks_feat  = ranks_feat[order]

    # Compute intervals (CUDA kernel for flag detection)
    starts, lengths = _compute_intervals_cuda(ranks_bev)
    if starts is None:
        return None, None, None, None, None

    return (ranks_bev.int().contiguous(),
            ranks_depth.int().contiguous(),
            ranks_feat.int().contiguous(),
            starts.int().contiguous(),
            lengths.int().contiguous())


# =====================================================================
#                  PURE PYTORCH FALLBACK
# =====================================================================

def voxel_pooling_prepare_v3_pytorch(coor, grid_lower_bound, grid_interval, grid_size):
    """Pure PyTorch implementation (identical to v2 logic, for reference/fallback).

    Args:
        coor: (B, N, D, H, W, 3) float
        grid_lower_bound: (3,) float tensor
        grid_interval: (3,) float tensor
        grid_size: (3,) float tensor

    Returns:
        (ranks_bev, ranks_depth, ranks_feat, interval_starts, interval_lengths)
    """
    B, N, D, H, W, _ = coor.shape
    num_points = B * N * D * H * W
    device = coor.device

    ranks_depth = torch.arange(0, num_points, dtype=torch.int, device=device)
    ranks_feat = torch.arange(0, num_points // D, dtype=torch.int, device=device)
    ranks_feat = ranks_feat.reshape(B, N, 1, H, W).expand(B, N, D, H, W).flatten()

    # Grid coordinate conversion
    coor = ((coor - grid_lower_bound.to(coor)) / grid_interval.to(coor))
    coor = coor.long().view(num_points, 3)
    batch_idx = torch.arange(0, B, device=coor.device).reshape(B, 1) \
        .expand(B, num_points // B).reshape(num_points, 1).to(coor)
    coor = torch.cat((coor, batch_idx), 1)

    # Filter out-of-bounds
    kept = ((coor[:, 0] >= 0) & (coor[:, 0] < grid_size[0]) &
            (coor[:, 1] >= 0) & (coor[:, 1] < grid_size[1]) &
            (coor[:, 2] >= 0) & (coor[:, 2] < grid_size[2]))
    if kept.sum() == 0:
        return None, None, None, None, None

    coor, ranks_depth, ranks_feat = coor[kept], ranks_depth[kept], ranks_feat[kept]

    # Compute BEV rank
    ranks_bev = (coor[:, 3] * (grid_size[2] * grid_size[1] * grid_size[0])
                 + coor[:, 2] * (grid_size[1] * grid_size[0])
                 + coor[:, 1] * grid_size[0] + coor[:, 0])
    order = ranks_bev.argsort()
    ranks_bev   = ranks_bev[order]
    ranks_depth = ranks_depth[order]
    ranks_feat  = ranks_feat[order]

    kept = torch.ones(ranks_bev.shape[0], device=device, dtype=torch.bool)
    kept[1:] = ranks_bev[1:] != ranks_bev[:-1]
    interval_starts = torch.where(kept)[0].int()
    if interval_starts.numel() == 0:
        return None, None, None, None, None
    interval_lengths = torch.zeros_like(interval_starts)
    interval_lengths[:-1] = interval_starts[1:] - interval_starts[:-1]
    interval_lengths[-1] = ranks_bev.shape[0] - interval_starts[-1]

    return (ranks_bev.int().contiguous(),
            ranks_depth.int().contiguous(),
            ranks_feat.int().contiguous(),
            interval_starts.int().contiguous(),
            interval_lengths.int().contiguous())
