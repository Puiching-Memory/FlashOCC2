"""混合精度训练装饰器."""

from __future__ import annotations

import functools

import torch
import torch.nn as nn


def _maybe_to_fp32(x):
    """若 x 是浮点 Tensor 则转 fp32, 否则原样返回."""
    if hasattr(x, 'is_floating_point') and x.is_floating_point():
        return x.float()
    return x


def force_fp32(apply_to=None, out_fp16: bool = False):
    """装饰器: 将输入强制转为 fp32 再调用."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            new_args = [_maybe_to_fp32(a) for a in args]
            new_kwargs = {k: _maybe_to_fp32(v) for k, v in kwargs.items()}
            return func(*new_args, **new_kwargs)
        return wrapper

    # 支持 @force_fp32 和 @force_fp32() 两种写法
    if callable(apply_to):
        return apply_to
    return decorator


_BN_TYPES = (nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
_BN_TYPE_NAMES = frozenset(t.__name__ for t in _BN_TYPES)


def wrap_fp16_model(model: nn.Module) -> nn.Module:
    """将模型转为 fp16 (BN 保持 fp32)."""
    for m in model.modules():
        if type(m).__name__ in _BN_TYPE_NAMES:
            m.float()
    model.half()
    return model
