"""Hook 共享工具函数."""


def is_parallel(model):
    """判断模型是否被 DataParallel/DistributedDataParallel 包裹."""
    return hasattr(model, 'module') and hasattr(model, 'device_ids')
