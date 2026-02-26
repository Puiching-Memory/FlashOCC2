"""分布式训练工具."""

from __future__ import annotations

import functools
import os

import torch
import torch.distributed as dist


def get_dist_info() -> tuple[int, int]:
    """返回 (rank, world_size)."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def init_dist(launcher: str = "pytorch", backend: str = "nccl", **kwargs):
    """初始化分布式训练."""
    if launcher == "pytorch":
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)

        if backend == "nccl" and torch.cuda.is_available():
            device_id = torch.device("cuda", local_rank)
            try:
                dist.init_process_group(backend=backend, device_id=device_id, **kwargs)
            except TypeError:
                dist.init_process_group(backend=backend, **kwargs)
        else:
            dist.init_process_group(backend=backend, **kwargs)

    elif launcher == "slurm":
        proc_id = int(os.environ.get("SLURM_PROCID", 0))
        ntasks = int(os.environ.get("SLURM_NTASKS", 1))
        node_list = os.environ.get("SLURM_NODELIST", "")
        num_gpus = torch.cuda.device_count()
        torch.cuda.set_device(proc_id % num_gpus)
        os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29500")
        if "MASTER_ADDR" not in os.environ:
            import subprocess
            addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
            os.environ["MASTER_ADDR"] = addr
        os.environ["WORLD_SIZE"] = str(ntasks)
        os.environ["RANK"] = str(proc_id)
        dist.init_process_group(backend=backend, **kwargs)

    else:
        raise ValueError(f"不支持的 launcher: {launcher}")


def master_only(func):
    """装饰器: 仅在 rank=0 上执行."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        rank, _ = get_dist_info()
        if rank == 0:
            return func(*args, **kwargs)
    return wrapper
