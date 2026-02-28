"""BEV 池化算子 v3 — CUDA / Triton 双后端, 支持 FP16.

核心优化 vs v2:
  1. Block-per-interval 共享内存 tiling → 减少全局内存带宽
  2. 统一 interval 核函数复用于 forward + feat backward
  3. 独立 depth-backward 核函数 → 全点并行 (比 v2 多 100x+ 线程)
  4. FP16 输入 / FP32 累积 → 2× 带宽
  5. 融合 voxel_pooling_prepare → 1 kernel 代替 ~10 个 PyTorch ops
  6. 预排序 feat intervals → backward 零 argsort 开销 (节省 ~2ms)
"""

from __future__ import annotations

import torch

from flashocc.core.ops._ext import bev_pool_v3_ext

__all__ = ["bev_pool_v3", "TRTBEVPoolv3"]


def _compute_feat_intervals(ranks_feat, ranks_depth, ranks_bev):
    """Re-sort by ranks_feat and compute feat-grouping intervals for backward."""
    order = ranks_feat.argsort()
    rf = ranks_feat[order]
    rd = ranks_depth[order]
    rb = ranks_bev[order]

    kept = torch.ones(rf.shape[0], device=rf.device, dtype=torch.bool)
    kept[1:] = rf[1:] != rf[:-1]
    starts = torch.where(kept)[0].int()
    lengths = torch.zeros_like(starts)
    lengths[:-1] = starts[1:] - starts[:-1]
    lengths[-1] = rf.shape[0] - starts[-1]
    return rd.contiguous(), rf.contiguous(), rb.contiguous(), \
        starts.contiguous(), lengths.contiguous()


class _BevPoolV3Cuda(torch.autograd.Function):
    """Autograd wrapper for the v3 CUDA forward/backward kernels.

    When pre-computed feat intervals are provided via `feat_intervals`,
    backward skips the expensive argsort + interval computation (~2ms savings).
    """

    @staticmethod
    def forward(ctx, depth, feat, ranks_depth, ranks_feat, ranks_bev,
                bev_feat_shape, interval_starts, interval_lengths,
                feat_intervals=None):
        ranks_bev = ranks_bev.int()
        # Align dtypes: under AMP, depth (softmax) may be float32 while
        # feat (conv output) is bfloat16/float16. Cast depth to match feat.
        if depth.dtype != feat.dtype:
            depth = depth.to(feat.dtype)
        depth = depth.contiguous()
        feat = feat.contiguous()
        ranks_depth = ranks_depth.contiguous().int()
        ranks_feat = ranks_feat.contiguous().int()
        interval_starts = interval_starts.contiguous().int()
        interval_lengths = interval_lengths.contiguous().int()

        out = feat.new_zeros(bev_feat_shape)

        bev_pool_v3_ext.bev_pool_v3_forward(
            depth, feat, out,
            ranks_depth, ranks_feat, ranks_bev,
            interval_lengths, interval_starts)

        ctx.save_for_backward(ranks_bev, depth, feat, ranks_feat, ranks_depth)
        ctx.n_points = ranks_bev.shape[0]

        # Cache pre-computed feat intervals to avoid argsort in backward
        if feat_intervals is not None:
            ctx._feat_intervals = feat_intervals
        else:
            ctx._feat_intervals = None
        return out

    @staticmethod
    def backward(ctx, out_grad):
        ranks_bev, depth, feat, ranks_feat, ranks_depth = ctx.saved_tensors
        n_points = ctx.n_points

        # Align dtypes
        target_dtype = depth.dtype
        if out_grad.dtype != target_dtype:
            out_grad = out_grad.to(target_dtype)

        # --- Use pre-computed feat intervals if available ---
        if ctx._feat_intervals is not None:
            rd_fs, rf_fs, rb_fs, starts_fs, lengths_fs = ctx._feat_intervals
        else:
            rd_fs, rf_fs, rb_fs, starts_fs, lengths_fs = \
                _compute_feat_intervals(ranks_feat, ranks_depth, ranks_bev)

        depth_grad = torch.zeros_like(depth, dtype=torch.float32)
        feat_grad = feat.new_zeros(feat.shape)

        bev_pool_v3_ext.bev_pool_v3_backward(
            out_grad.contiguous(),
            depth_grad, feat_grad,
            depth, feat,
            # bev-sorted (for depth grad)
            ranks_depth, ranks_feat, ranks_bev,
            n_points,
            # feat-sorted (for feat grad)
            rd_fs, rf_fs, rb_fs,
            starts_fs, lengths_fs)

        # depth was originally float, cast grad back if needed
        if depth.dtype != torch.float32:
            depth_grad = depth_grad.to(depth.dtype)

        return depth_grad, feat_grad, None, None, None, None, None, None, None


def bev_pool_v3(depth, feat, ranks_depth, ranks_feat, ranks_bev,
                bev_feat_shape, interval_starts, interval_lengths,
                feat_intervals=None):
    """BEV 池化 v3.

    Args:
        depth: (B, N, D, fH, fW)                 — 支持 float32 / float16
        feat:  (B, N, fH, fW, C)                 — 与 depth 同 dtype
        ranks_depth:  (N_points,) int
        ranks_feat:   (N_points,) int
        ranks_bev:    (N_points,) int
        bev_feat_shape: tuple (B, Dz, Dy, Dx, C)
        interval_starts:  (N_intervals,) int
        interval_lengths: (N_intervals,) int
        feat_intervals:   tuple or None — 预排序的 feat intervals,
            格式: (rd_fs, rf_fs, rb_fs, starts_fs, lengths_fs)
            提供时 backward 跳过 argsort (节省 ~2ms).
    Returns:
        (B, C, Dz, Dy, Dx)
    """
    x = _BevPoolV3Cuda.apply(
        depth, feat, ranks_depth, ranks_feat, ranks_bev,
        bev_feat_shape, interval_starts, interval_lengths,
        feat_intervals)
    return x.permute(0, 4, 1, 2, 3).contiguous()


class TRTBEVPoolv3(torch.autograd.Function):
    """TensorRT 兼容的 BEV 池化 v3."""

    @staticmethod
    def symbolic(g, depth, feat, ranks_depth, ranks_feat, ranks_bev,
                 interval_starts, interval_lengths,
                 output_height=128, output_width=128, output_z=1):
        return g.op("mmdeploy::bev_pool_v3",
                     depth, feat, ranks_depth, ranks_feat, ranks_bev,
                     interval_starts, interval_lengths,
                     output_height_i=output_height,
                     output_width_i=output_width,
                     output_z_i=output_z)

    @staticmethod
    def forward(g, depth, feat, ranks_depth, ranks_feat, ranks_bev,
                interval_starts, interval_lengths,
                output_height=128, output_width=128, output_z=1):
        feat = feat.unsqueeze(0)
        depth = depth.unsqueeze(0)
        shape = (depth.shape[0], output_z, output_height, output_width, feat.shape[-1])
        bev_feat = bev_pool_v3(depth, feat, ranks_depth, ranks_feat, ranks_bev,
                               shape, interval_starts, interval_lengths)
        if output_z == 1:
            bev_feat = bev_feat.squeeze(2).permute(0, 2, 3, 1)
        return bev_feat
