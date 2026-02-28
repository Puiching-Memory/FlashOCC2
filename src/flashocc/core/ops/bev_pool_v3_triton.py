"""BEV Pool v3 — Triton JIT kernel 实现.

提供与 CUDA 实现等价的 Triton 核函数, 便于:
  • 快速原型迭代和 auto-tune
  • 无需 nvcc 编译即可运行
  • 作为基准与 CUDA 实现对比

使用方法::

    from flashocc.core.ops.bev_pool_v3_triton import bev_pool_v3_triton
    out = bev_pool_v3_triton(depth, feat, ranks_depth, ranks_feat, ranks_bev,
                              bev_feat_shape, interval_starts, interval_lengths)
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

__all__ = ["bev_pool_v3_triton"]


# =====================================================================
#                       FORWARD KERNEL
# =====================================================================

@triton.jit
def _bev_pool_v3_fwd_kernel(
    # pointers
    depth_ptr, feat_ptr, out_ptr,
    ranks_depth_ptr, ranks_feat_ptr, ranks_bev_ptr,
    interval_starts_ptr, interval_lengths_ptr,
    # scalars
    C: tl.constexpr,
    n_intervals,
    # block shape
    BLOCK_C: tl.constexpr,
):
    """One program per interval, dynamic loop over points.

    BLOCK_C: should be >= C (next power of 2).  Each thread handles 1 channel.
    Uses dynamic loop bounds (Triton 3.x) for early exit — no wasted iterations.
    """
    pid = tl.program_id(0)
    if pid >= n_intervals:
        return

    start  = tl.load(interval_starts_ptr + pid)
    length = tl.load(interval_lengths_ptr + pid)
    bev_idx = tl.load(ranks_bev_ptr + start)

    ch = tl.arange(0, BLOCK_C)
    ch_mask = ch < C
    acc = tl.zeros([BLOCK_C], dtype=tl.float32)

    for i in range(length):
        pt = start + i
        d_idx = tl.load(ranks_depth_ptr + pt)
        f_idx = tl.load(ranks_feat_ptr  + pt)
        d_val = tl.load(depth_ptr + d_idx).to(tl.float32)
        f_val = tl.load(feat_ptr + f_idx * C + ch, mask=ch_mask, other=0.0).to(tl.float32)
        acc += d_val * f_val

    tl.store(out_ptr + bev_idx * C + ch, acc.to(out_ptr.dtype.element_ty), mask=ch_mask)


# =====================================================================
#                       BACKWARD DEPTH KERNEL
# =====================================================================

@triton.jit
def _bev_pool_v3_bwd_depth_kernel(
    out_grad_ptr, feat_ptr,
    ranks_bev_ptr, ranks_feat_ptr, ranks_depth_ptr,
    depth_grad_ptr,
    C: tl.constexpr,
    n_points,
    BLOCK_C: tl.constexpr,
):
    """One program per point.  Dot-product over channels."""
    pid = tl.program_id(0)
    if pid >= n_points:
        return

    bev_i  = tl.load(ranks_bev_ptr   + pid)
    feat_i = tl.load(ranks_feat_ptr  + pid)
    dep_i  = tl.load(ranks_depth_ptr + pid)

    ch = tl.arange(0, BLOCK_C)
    ch_mask = ch < C

    og = tl.load(out_grad_ptr + bev_i  * C + ch, mask=ch_mask, other=0.0).to(tl.float32)
    ft = tl.load(feat_ptr     + feat_i * C + ch, mask=ch_mask, other=0.0).to(tl.float32)

    grad = tl.sum(og * ft, axis=0)
    tl.store(depth_grad_ptr + dep_i, grad)


# =====================================================================
#                       BACKWARD FEAT KERNEL
# =====================================================================

@triton.jit
def _bev_pool_v3_bwd_feat_kernel(
    depth_ptr, out_grad_ptr,
    ranks_depth_ptr, ranks_bev_ptr, ranks_feat_ptr,
    interval_starts_ptr, interval_lengths_ptr,
    feat_grad_ptr,
    C: tl.constexpr,
    n_intervals,
    BLOCK_C: tl.constexpr,
):
    """One program per feat-interval, dynamic loop over points."""
    pid = tl.program_id(0)
    if pid >= n_intervals:
        return

    start  = tl.load(interval_starts_ptr + pid)
    length = tl.load(interval_lengths_ptr + pid)
    feat_idx = tl.load(ranks_feat_ptr + start)

    ch = tl.arange(0, BLOCK_C)
    ch_mask = ch < C
    acc = tl.zeros([BLOCK_C], dtype=tl.float32)

    for i in range(length):
        pt = start + i
        d_idx = tl.load(ranks_depth_ptr + pt)
        b_idx = tl.load(ranks_bev_ptr   + pt)
        d_val = tl.load(depth_ptr + d_idx).to(tl.float32)
        og_val = tl.load(out_grad_ptr + b_idx * C + ch, mask=ch_mask, other=0.0).to(tl.float32)
        acc += d_val * og_val

    tl.store(feat_grad_ptr + feat_idx * C + ch, acc.to(feat_grad_ptr.dtype.element_ty), mask=ch_mask)


# =====================================================================
#                   PYTHON AUTOGRAD WRAPPER
# =====================================================================

def _compute_feat_intervals(ranks_feat, ranks_depth, ranks_bev):
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


def _next_power_of_2(n):
    """Return the smallest power-of-2 >= n, minimum 32."""
    p = 32
    while p < n:
        p *= 2
    return p


class _BevPoolV3Triton(torch.autograd.Function):

    @staticmethod
    def forward(ctx, depth, feat, ranks_depth, ranks_feat, ranks_bev,
                bev_feat_shape, interval_starts, interval_lengths):
        ranks_bev = ranks_bev.int()
        # Align dtypes: under AMP, depth (softmax) may be float32 while
        # feat (conv output) is bfloat16/float16.
        if depth.dtype != feat.dtype:
            depth = depth.to(feat.dtype)
        depth = depth.contiguous()
        feat = feat.contiguous()
        ranks_depth = ranks_depth.contiguous().int()
        ranks_feat = ranks_feat.contiguous().int()
        interval_starts = interval_starts.contiguous().int()
        interval_lengths = interval_lengths.contiguous().int()

        C = feat.shape[-1]
        n_intervals = interval_starts.shape[0]
        out = feat.new_zeros(bev_feat_shape)

        BLOCK_C = _next_power_of_2(C)

        if n_intervals > 0:
            _bev_pool_v3_fwd_kernel[(n_intervals,)](
                depth, feat, out,
                ranks_depth, ranks_feat, ranks_bev,
                interval_starts, interval_lengths,
                C=C, n_intervals=n_intervals,
                BLOCK_C=BLOCK_C,
                num_warps=max(1, BLOCK_C // 32),
            )

        ctx.save_for_backward(ranks_bev, depth, feat, ranks_feat, ranks_depth)
        ctx.n_points = ranks_bev.shape[0]
        ctx.C = C
        return out

    @staticmethod
    def backward(ctx, out_grad):
        ranks_bev, depth, feat, ranks_feat, ranks_depth = ctx.saved_tensors
        n_points = ctx.n_points
        C = ctx.C

        # Align dtypes
        target_dtype = depth.dtype
        if out_grad.dtype != target_dtype:
            out_grad = out_grad.to(target_dtype)
        out_grad = out_grad.contiguous()

        # --- depth grad ---
        depth_grad = torch.zeros(depth.shape, device=depth.device, dtype=torch.float32)
        BLOCK_C = _next_power_of_2(C)
        if n_points > 0:
            _bev_pool_v3_bwd_depth_kernel[(n_points,)](
                out_grad, feat,
                ranks_bev, ranks_feat, ranks_depth,
                depth_grad,
                C=C, n_points=n_points,
                BLOCK_C=BLOCK_C,
            )

        # --- feat grad ---
        rd_fs, rf_fs, rb_fs, starts_fs, lengths_fs = \
            _compute_feat_intervals(ranks_feat, ranks_depth, ranks_bev)
        feat_grad = feat.new_zeros(feat.shape)
        n_feat_intervals = starts_fs.shape[0]

        if n_feat_intervals > 0:
            BLOCK_C = _next_power_of_2(C)
            _bev_pool_v3_bwd_feat_kernel[(n_feat_intervals,)](
                depth, out_grad,
                rd_fs, rb_fs, rf_fs,
                starts_fs, lengths_fs,
                feat_grad,
                C=C, n_intervals=n_feat_intervals,
                BLOCK_C=BLOCK_C,
                num_warps=max(1, BLOCK_C // 32),
            )

        if depth.dtype != torch.float32:
            depth_grad = depth_grad.to(depth.dtype)

        return depth_grad, feat_grad, None, None, None, None, None, None


def bev_pool_v3_triton(depth, feat, ranks_depth, ranks_feat, ranks_bev,
                       bev_feat_shape, interval_starts, interval_lengths):
    """BEV 池化 v3 — Triton 实现.

    接口与 bev_pool_v3 (CUDA) 完全相同.

    Args:
        depth: (B, N, D, fH, fW)
        feat:  (B, N, fH, fW, C)
    Returns:
        (B, C, Dz, Dy, Dx)
    """
    x = _BevPoolV3Triton.apply(
        depth, feat, ranks_depth, ranks_feat, ranks_bev,
        bev_feat_shape, interval_starts, interval_lengths)
    return x.permute(0, 4, 1, 2, 3).contiguous()
