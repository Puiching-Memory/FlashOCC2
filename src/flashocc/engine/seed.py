"""随机种子工具."""
import os
import random
import numpy as np
import torch
import torch.distributed as dist


def init_random_seed(seed=None, device="cuda"):
    """初始化随机种子 (分布式同步).

    Args:
        seed: 基础种子。None 则自动生成。
        device: 设备字符串。

    Returns:
        int: 同步后的随机种子。
    """
    if seed is not None:
        return seed
    seed = np.random.randint(2 ** 31)
    if dist.is_available() and dist.is_initialized():
        seed_tensor = torch.tensor(seed, dtype=torch.int32, device=device)
        dist.broadcast(seed_tensor, src=0)
        seed = seed_tensor.item()
    return seed


def set_random_seed(seed, deterministic=False):
    """设置全局随机种子.

    Args:
        seed: 随机种子。
        deterministic: 是否启用确定性模式。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


__all__ = ["init_random_seed", "set_random_seed"]
