"""混合精度工具 (兼容占位).

项目已全面使用 torch.amp.autocast, 此模块仅保留导入兼容性。
"""


def force_fp32(apply_to=None, out_fp16: bool = False):
    """No-op 装饰器, 保持向后兼容."""
    if callable(apply_to):
        return apply_to

    def decorator(func):
        return func
    return decorator



