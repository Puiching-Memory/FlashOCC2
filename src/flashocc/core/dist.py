"""分布式训练工具 — 基于 DeviceMesh 统一抽象.

所有并行策略 (DDP / FSDP2 / 未来的张量并行) 统一通过 DeviceMesh 管理。
DeviceMesh 是 PyTorch 推荐的分布式拓扑抽象, 内部管理进程组创建与通信。

用法::

    mesh = init_dist("pytorch")          # torchrun 设置环境变量
    model = setup_parallel(model, mesh)   # 默认 DDP
    model = setup_parallel(model, mesh, mode="fsdp2")  # FSDP2
"""

from __future__ import annotations

import functools
import os
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh

# 全局 DeviceMesh 引用
_global_mesh: DeviceMesh | None = None


def get_mesh() -> DeviceMesh | None:
    """返回全局 DeviceMesh, 未初始化时返回 None."""
    return _global_mesh


def get_dist_info() -> tuple[int, int]:
    """返回 (rank, world_size)."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def init_dist(launcher: str = "pytorch", backend: str = "nccl", **kwargs) -> DeviceMesh:
    """初始化分布式训练 — 返回 DeviceMesh.

    使用 ``torch.distributed.device_mesh.init_device_mesh`` 作为统一入口,
    自动管理进程组创建, 为 DDP / FSDP2 / 张量并行提供统一底层。

    Parameters
    ----------
    launcher : str
        "pytorch" (torchrun) 或 "slurm".
    backend : str
        通信后端, 默认 "nccl".

    Returns
    -------
    DeviceMesh
        1D DeviceMesh, 维度名 ``"dp"`` (data parallel).
    """
    global _global_mesh
    from torch.distributed.device_mesh import init_device_mesh

    if launcher == "pytorch":
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)

        world_size = int(os.environ.get("WORLD_SIZE", 1))
        mesh = init_device_mesh(
            "cuda",
            mesh_shape=(world_size,),
            mesh_dim_names=("dp",),
        )

    elif launcher == "slurm":
        proc_id = int(os.environ.get("SLURM_PROCID", 0))
        ntasks = int(os.environ.get("SLURM_NTASKS", 1))
        node_list = os.environ.get("SLURM_NODELIST", "")
        num_gpus = torch.cuda.device_count()
        torch.cuda.set_device(proc_id % num_gpus)
        os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29500")
        if "MASTER_ADDR" not in os.environ:
            import subprocess
            addr = subprocess.getoutput(
                f"scontrol show hostname {node_list} | head -n1"
            )
            os.environ["MASTER_ADDR"] = addr
        os.environ["WORLD_SIZE"] = str(ntasks)
        os.environ["RANK"] = str(proc_id)
        mesh = init_device_mesh(
            "cuda",
            mesh_shape=(ntasks,),
            mesh_dim_names=("dp",),
        )

    else:
        raise ValueError(f"不支持的 launcher: {launcher}")

    _global_mesh = mesh
    return mesh


def setup_parallel(
    model: torch.nn.Module,
    mesh: DeviceMesh | None,
    mode: str = "ddp",
    **kwargs,
) -> torch.nn.Module:
    """根据并行策略和 DeviceMesh 包装模型.

    Parameters
    ----------
    model : nn.Module
        已放置在 CUDA 上的模型.
    mesh : DeviceMesh or None
        全局 DeviceMesh. ``None`` 时不做任何包装 (单卡).
    mode : str
        ``"ddp"``   — DistributedDataParallel, 全模型复制, 梯度 AllReduce.
        ``"fsdp2"`` — Fully Sharded Data Parallel v2, 参数/梯度分片.
    **kwargs
        DDP: ``broadcast_buffers``, ``find_unused_parameters`` 等.
        FSDP2: ``reshard_after_forward`` 等.

    Returns
    -------
    nn.Module
        并行包装后的模型.
    """
    if mesh is None:
        return model

    _, world_size = get_dist_info()
    if world_size <= 1:
        return model

    if mode == "ddp":
        return torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[torch.cuda.current_device()],
            process_group=mesh.get_group("dp"),
            **kwargs,
        )

    if mode == "fsdp2":
        try:
            from torch.distributed._composable.fsdp import fully_shard
        except ImportError:
            raise ImportError(
                "FSDP2 需要 PyTorch >= 2.4. "
                "请升级 PyTorch 或使用 parallel_mode='ddp'."
            ) from None
        # FSDP2 composable API — 就地修改模型
        fully_shard(model, mesh=mesh, **kwargs)
        return model

    raise ValueError(f"不支持的并行策略: {mode}, 可选: ddp, fsdp2")


def master_only(func):
    """装饰器: 仅在 rank=0 上执行."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        rank, _ = get_dist_info()
        if rank == 0:
            return func(*args, **kwargs)
    return wrapper
