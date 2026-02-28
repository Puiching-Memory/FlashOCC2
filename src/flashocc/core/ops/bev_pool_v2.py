"""BEV 池化算子 v2 (BEVPoolv2) — 需要编译 CUDA 扩展."""

from __future__ import annotations

import torch

from flashocc.core.ops._ext import bev_pool_v2_ext

__all__ = ["bev_pool_v2", "TRTBEVPoolv2"]


class _QuickCumsumCuda(torch.autograd.Function):
    @staticmethod
    def forward(ctx, depth, feat, ranks_depth, ranks_feat, ranks_bev,
                bev_feat_shape, interval_starts, interval_lengths):
        ranks_bev = ranks_bev.int()
        depth = depth.contiguous().float()
        feat = feat.contiguous().float()
        ranks_depth = ranks_depth.contiguous().int()
        ranks_feat = ranks_feat.contiguous().int()
        interval_lengths = interval_lengths.contiguous().int()
        interval_starts = interval_starts.contiguous().int()
        out = feat.new_zeros(bev_feat_shape)

        bev_pool_v2_ext.bev_pool_v2_forward(
            depth, feat, out, ranks_depth, ranks_feat, ranks_bev,
            interval_lengths, interval_starts)

        ctx.save_for_backward(ranks_bev, depth, feat, ranks_feat, ranks_depth)
        return out

    @staticmethod
    def backward(ctx, out_grad):
        ranks_bev, depth, feat, ranks_feat, ranks_depth = ctx.saved_tensors
        order = ranks_feat.argsort()
        ranks_feat, ranks_depth, ranks_bev = (
            ranks_feat[order], ranks_depth[order], ranks_bev[order])

        kept = torch.ones(ranks_bev.shape[0], device=ranks_bev.device, dtype=torch.bool)
        kept[1:] = ranks_feat[1:] != ranks_feat[:-1]
        interval_starts = torch.where(kept)[0].int()
        interval_lengths = torch.zeros_like(interval_starts)
        interval_lengths[:-1] = interval_starts[1:] - interval_starts[:-1]
        interval_lengths[-1] = ranks_bev.shape[0] - interval_starts[-1]

        depth_grad = depth.new_zeros(depth.shape)
        feat_grad = feat.new_zeros(feat.shape)
        bev_pool_v2_ext.bev_pool_v2_backward(
            out_grad.contiguous(), depth_grad, feat_grad, depth, feat,
            ranks_depth.contiguous(), ranks_feat.contiguous(),
            ranks_bev.contiguous(), interval_lengths.contiguous(),
            interval_starts.contiguous())
        return depth_grad, feat_grad, None, None, None, None, None, None


def bev_pool_v2(depth, feat, ranks_depth, ranks_feat, ranks_bev,
                bev_feat_shape, interval_starts, interval_lengths):
    """BEV 池化 v2.

    Args:
        depth: (B, N, D, fH, fW)
        feat:  (B, N, fH, fW, C)
    Returns:
        (B, C, Dz, Dy, Dx)
    """
    x = _QuickCumsumCuda.apply(
        depth, feat, ranks_depth, ranks_feat, ranks_bev,
        bev_feat_shape, interval_starts, interval_lengths)
    return x.permute(0, 4, 1, 2, 3).contiguous()


class TRTBEVPoolv2(torch.autograd.Function):
    """TensorRT 兼容的 BEV 池化 v2."""

    @staticmethod
    def symbolic(g, depth, feat, ranks_depth, ranks_feat, ranks_bev,
                 interval_starts, interval_lengths,
                 output_height=128, output_width=128, output_z=1):
        return g.op("mmdeploy::bev_pool_v2",
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
        bev_feat = bev_pool_v2(depth, feat, ranks_depth, ranks_feat, ranks_bev,
                               shape, interval_starts, interval_lengths)
        if output_z == 1:
            bev_feat = bev_feat.squeeze(2).permute(0, 2, 3, 1)
        return bev_feat
