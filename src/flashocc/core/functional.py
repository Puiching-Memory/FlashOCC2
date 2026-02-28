"""训练/推理核心工具."""
import torch
import torch.distributed as dist


def reduce_mean(tensor):
    """跨 GPU 求平均值."""
    if not (dist.is_available() and dist.is_initialized()):
        return tensor
    tensor = tensor.clone()
    dist.all_reduce(tensor.div_(dist.get_world_size()), op=dist.ReduceOp.SUM)
    return tensor


__all__ = ["reduce_mean"]
