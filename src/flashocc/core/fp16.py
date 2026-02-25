"""混合精度训练装饰器."""

from __future__ import annotations

import functools

import torch
import torch.nn as nn


def force_fp32(apply_to=None, out_fp16: bool = False):
    """装饰器: 将输入强制转为 fp32 再调用."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            new_args = [
                a.float() if isinstance(a, torch.Tensor) and a.is_floating_point() else a
                for a in args
            ]
            new_kwargs = {
                k: v.float() if isinstance(v, torch.Tensor) and v.is_floating_point() else v
                for k, v in kwargs.items()
            }
            return func(*new_args, **new_kwargs)
        return wrapper

    # 支持 @force_fp32 和 @force_fp32() 两种写法
    if callable(apply_to):
        return apply_to
    return decorator


def wrap_fp16_model(model: nn.Module) -> nn.Module:
    """将模型转为 fp16 (BN 保持 fp32)."""
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)):
            m.float()
    model.half()
    return model
