"""训练/推理核心工具: multi_apply, reduce_mean."""
import functools
import torch
import torch.distributed as dist


def multi_apply(func, *args, **kwargs):
    """将函数应用于参数列表的每个元素.

    Args:
        func: 要应用的函数.
        *args: 参数列表（按位置）.
        **kwargs: 固定关键字参数.

    Returns:
        tuple[list]: 每个返回值分别组成 list.
    """
    pfunc = functools.partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


def reduce_mean(tensor):
    """跨 GPU 求平均值."""
    if not (dist.is_available() and dist.is_initialized()):
        return tensor
    tensor = tensor.clone()
    dist.all_reduce(tensor.div_(dist.get_world_size()), op=dist.ReduceOp.SUM)
    return tensor


__all__ = ["multi_apply", "reduce_mean"]
