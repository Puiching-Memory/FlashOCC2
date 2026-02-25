"""训练引擎 — 直接消费 Experiment 对象.

 dict 配置, 所有参数从 Experiment 字段读取。
"""
import os

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from flashocc.core import get_dist_info
from flashocc.core.log import logger
from flashocc.engine.parallel import DataContainer


# =====================================================================
#  DataLoader
# =====================================================================

def collate_fn(batch):
    """自定义 collate: 处理 DataContainer 和常规数据."""
    import numpy as np
    from collections.abc import Mapping, Sequence

    if not isinstance(batch, Sequence):
        raise TypeError(f"batch 应为 Sequence, 得到 {type(batch)}")

    if isinstance(batch[0], DataContainer):
        if batch[0].stack:
            stacked = torch.stack([s.data for s in batch], dim=0)
            return DataContainer(stacked, stack=batch[0].stack,
                                 padding_value=batch[0].padding_value,
                                 cpu_only=batch[0].cpu_only)
        else:
            return DataContainer([s.data for s in batch],
                                 stack=False, cpu_only=batch[0].cpu_only)
    elif isinstance(batch[0], Mapping):
        return {key: collate_fn([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], tuple) and hasattr(batch[0], '_fields'):
        return type(batch[0])(*(collate_fn(s) for s in zip(*batch)))
    elif isinstance(batch[0], Sequence) and not isinstance(batch[0], str):
        return [collate_fn(s) for s in zip(*batch)]
    elif isinstance(batch[0], torch.Tensor):
        return torch.stack(batch, 0)
    elif isinstance(batch[0], (int, float)):
        return torch.tensor(batch)
    elif isinstance(batch[0], np.ndarray):
        return torch.from_numpy(np.stack(batch))
    else:
        return batch


def build_dataloader(dataset, samples_per_gpu=2, workers_per_gpu=2,
                     num_gpus=1, dist_mode=False, seed=None, **kwargs):
    if dist_mode:
        sampler = DistributedSampler(dataset, shuffle=True, seed=seed or 0)
        batch_size = samples_per_gpu
        num_workers = workers_per_gpu
    else:
        sampler = None
        batch_size = num_gpus * samples_per_gpu
        num_workers = num_gpus * workers_per_gpu

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=(sampler is None),
        collate_fn=collate_fn,
    )


# =====================================================================
#  数据搬运
# =====================================================================

def scatter_data(data, device=None):
    if device is None:
        device = torch.cuda.current_device()

    if isinstance(data, DataContainer):
        if data.cpu_only:
            return data.data
        if data.stack:
            d = data.data
            return d.cuda(device) if isinstance(d, torch.Tensor) else d
        else:
            return [d.cuda(device) if isinstance(d, torch.Tensor) else d
                    for d in data.data]
    elif isinstance(data, dict):
        return {k: scatter_data(v, device) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return type(data)(scatter_data(v, device) for v in data)
    elif isinstance(data, torch.Tensor):
        return data.cuda(device)
    return data


# =====================================================================
#  Warmup 工具
# =====================================================================

def _get_warmup_lr(base_lr: float, warmup_ratio: float,
                   warmup_iters: int, cur_iter: int) -> float:
    """线性 warmup."""
    if cur_iter >= warmup_iters:
        return base_lr
    k = (1 - cur_iter / warmup_iters) * (1 - warmup_ratio)
    return base_lr * (1 - k)


# =====================================================================
#  训练主循环
# =====================================================================

def train_model(experiment, model, distributed=False, validate=False,
                timestamp=None, meta=None, gpu_ids=None):
    """训练入口 — 直接消费 Experiment 对象.

    Parameters
    ----------
    experiment : Experiment
        实验配置.
    model : nn.Module
        已构建的模型 (可能已加载预训练).
    distributed : bool
        是否 DDP.
    """
    exp = experiment

    # ---- 工作目录 ----
    work_dir = exp.work_dir or "work_dirs"
    os.makedirs(work_dir, exist_ok=True)

    # ---- 数据 ----
    dataset = exp.build_train_dataset()
    gpu_count = len(gpu_ids) if gpu_ids else 1
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=exp.samples_per_gpu,
        workers_per_gpu=exp.workers_per_gpu,
        num_gpus=gpu_count,
        dist_mode=distributed,
        seed=exp.seed,
    )

    # ---- 优化器 & 调度器 ----
    optimizer = exp.build_optimizer(model)
    lr_scheduler = exp.build_lr_scheduler(optimizer)

    # ---- GPU / DDP ----
    model = model.cuda()
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=exp.find_unused_parameters,
        )

    # ---- 训练参数 ----
    max_epochs = exp.max_epochs
    log_interval = exp.log_interval
    grad_max_norm = exp.grad_max_norm
    warmup_iters = exp.warmup_iters
    warmup_ratio = exp.warmup_ratio
    base_lr = optimizer.param_groups[0]["lr"]

    logger.info(f"开始训练, 共 {max_epochs} epochs, 工作目录: {work_dir}")

    global_iter = 0
    for epoch in range(max_epochs):
        model.train()
        if distributed and hasattr(data_loader.sampler, "set_epoch"):
            data_loader.sampler.set_epoch(epoch)

        for i, data_batch in enumerate(data_loader):
            # warmup
            if warmup_iters > 0 and global_iter < warmup_iters:
                lr = _get_warmup_lr(base_lr, warmup_ratio,
                                    warmup_iters, global_iter)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr

            data_batch = scatter_data(data_batch)
            losses = model(return_loss=True, **data_batch)

            total_loss = torch.tensor(0.0, device="cuda")
            loss_log = {}
            for name, val in losses.items():
                if isinstance(val, torch.Tensor):
                    total_loss = total_loss + val
                    loss_log[name] = val.item()
                elif isinstance(val, (list, tuple)):
                    s = sum(v for v in val if isinstance(v, torch.Tensor))
                    if isinstance(s, torch.Tensor):
                        total_loss = total_loss + s
                        loss_log[name] = s.item()

            optimizer.zero_grad()
            total_loss.backward()
            if grad_max_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), grad_max_norm)
            optimizer.step()
            global_iter += 1

            if i % log_interval == 0:
                lr = optimizer.param_groups[0]["lr"]
                loss_str = ", ".join(
                    f"{k}: {v:.4f}" for k, v in loss_log.items())
                logger.info(
                    f"Epoch [{epoch+1}/{max_epochs}][{i}/{len(data_loader)}] "
                    f"lr: {lr:.2e}, loss: {total_loss.item():.4f}, {loss_str}")

        # epoch 结束 → LR step
        if lr_scheduler is not None:
            lr_scheduler.step()

        # 保存 checkpoint
        ckpt_path = os.path.join(work_dir, f"epoch_{epoch+1}.pth")
        raw_model = model.module if hasattr(model, "module") else model
        state = {
            "meta": {"epoch": epoch + 1, **(meta or {})},
            "state_dict": raw_model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        torch.save(state, ckpt_path)
        logger.info(f"已保存 checkpoint: {ckpt_path}")

    logger.info("训练完成。")


__all__ = ["train_model", "build_dataloader", "collate_fn", "scatter_data"]
