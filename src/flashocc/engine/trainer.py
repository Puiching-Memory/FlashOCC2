"""训练引擎 — 直接消费 Experiment 对象.

 dict 配置, 所有参数从 Experiment 字段读取。
使用 plum-dispatch 替代 stdlib singledispatch 实现 collate 和 scatter。
"""
import os

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from plum import dispatch

from flashocc.core import get_dist_info
from flashocc.core.log import logger
from flashocc.engine.parallel import DataContainer


# =====================================================================
#  DataLoader — plum-dispatch collate
# =====================================================================

@dispatch
def _collate_elem(elem: DataContainer, batch: list) -> DataContainer:
    if batch[0].stack:
        stacked = torch.stack([s.data for s in batch], dim=0)
        return DataContainer(stacked, stack=batch[0].stack,
                             padding_value=batch[0].padding_value,
                             cpu_only=batch[0].cpu_only)
    else:
        return DataContainer([s.data for s in batch],
                             stack=False, cpu_only=batch[0].cpu_only)


@dispatch
def _collate_elem(elem: dict, batch: list):
    return {key: collate_fn([d[key] for d in batch]) for key in batch[0]}


@dispatch
def _collate_elem(elem: torch.Tensor, batch: list):
    return torch.stack(batch, 0)


@dispatch
def _collate_elem(elem: int, batch: list):
    return torch.tensor(batch)


@dispatch
def _collate_elem(elem: float, batch: list):
    return torch.tensor(batch)


@dispatch
def _collate_elem(elem: np.ndarray, batch: list):
    return torch.from_numpy(np.stack(batch))


@dispatch
def _collate_elem(elem: object, batch: list):
    """Fallback: 返回原始 batch."""
    return batch


def collate_fn(batch):
    """自定义 collate: 处理 DataContainer 和常规数据 — plum-dispatch 版."""
    if not hasattr(batch, '__getitem__'):
        raise TypeError(f"batch 应为 Sequence, 得到 {type(batch)}")

    if not isinstance(batch, list):
        batch = list(batch)

    elem = batch[0]

    # namedtuple: 有 _fields 属性的 tuple
    if hasattr(elem, '_fields'):
        return type(elem)(*(collate_fn(s) for s in zip(*batch)))

    # 普通 list/tuple (排除 str)
    if hasattr(elem, '__iter__') and not hasattr(elem, 'encode') and not hasattr(elem, 'data'):
        # 排除 DataContainer (有 data 属性) 和 str (有 encode 属性)
        # 也排除 dict (在 dispatch 中已处理)
        if not hasattr(elem, 'keys'):
            return [collate_fn(s) for s in zip(*batch)]

    return _collate_elem(elem, batch)


def build_dataloader(dataset, samples_per_gpu=2, workers_per_gpu=2,
                     num_gpus=1, dist_mode=False, seed=None, shuffle=True,
                     pin_memory=True, persistent_workers=True,
                     prefetch_factor=2, drop_last=False, **kwargs):
    if dist_mode:
        sampler = DistributedSampler(dataset, shuffle=shuffle, seed=seed or 0)
        batch_size = samples_per_gpu
        num_workers = workers_per_gpu
    else:
        sampler = None
        batch_size = num_gpus * samples_per_gpu
        num_workers = num_gpus * workers_per_gpu

    dataloader_kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=(sampler is None and shuffle),
        collate_fn=collate_fn,
        drop_last=drop_last,
    )
    if num_workers > 0:
        dataloader_kwargs["persistent_workers"] = persistent_workers
        if prefetch_factor is not None:
            dataloader_kwargs["prefetch_factor"] = prefetch_factor

    return DataLoader(**dataloader_kwargs)


# =====================================================================
#  数据搬运 — plum-dispatch
# =====================================================================

def _to_cuda_if_tensor(d, device, non_blocking=False):
    """将 tensor 搬到 GPU, 其他类型原样返回."""
    return d.cuda(device, non_blocking=non_blocking) if hasattr(d, 'cuda') else d


@dispatch
def scatter_data(data: DataContainer, device=None, non_blocking=False):
    if device is None:
        device = torch.cuda.current_device()
    if data.cpu_only:
        return data.data
    if data.stack:
        return _to_cuda_if_tensor(data.data, device, non_blocking=non_blocking)
    else:
        return [_to_cuda_if_tensor(d, device, non_blocking=non_blocking) for d in data.data]


@dispatch
def scatter_data(data: dict, device=None, non_blocking=False):
    return {k: scatter_data(v, device, non_blocking=non_blocking) for k, v in data.items()}


@dispatch
def scatter_data(data: list, device=None, non_blocking=False):
    return [scatter_data(v, device, non_blocking=non_blocking) for v in data]


@dispatch
def scatter_data(data: tuple, device=None, non_blocking=False):
    return tuple(scatter_data(v, device, non_blocking=non_blocking) for v in data)


@dispatch
def scatter_data(data: object, device=None, non_blocking=False):
    """递归将数据搬到 GPU — fallback: 原样返回."""
    if hasattr(data, 'cuda'):
        if device is None:
            device = torch.cuda.current_device()
        return data.cuda(device, non_blocking=non_blocking)
    return data


def _setup_torch_runtime(experiment):
    if experiment.allow_tf32:
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = True
    if experiment.cudnn_benchmark and hasattr(torch.backends, "cudnn"):
        if not torch.backends.cudnn.deterministic:
            torch.backends.cudnn.benchmark = True
    if experiment.float32_matmul_precision is not None and hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision(experiment.float32_matmul_precision)


def _get_amp_dtype(experiment):
    """Parse AMP dtype from experiment config."""
    if not experiment.use_amp:
        return None
    dtype_str = getattr(experiment, 'amp_dtype', 'bfloat16')
    if dtype_str == 'bfloat16':
        return torch.bfloat16
    elif dtype_str == 'float16':
        return torch.float16
    else:
        raise ValueError(f"Unsupported AMP dtype: {dtype_str}")


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
        pin_memory=exp.dataloader_pin_memory,
        persistent_workers=exp.dataloader_persistent_workers,
        prefetch_factor=exp.dataloader_prefetch_factor,
        drop_last=exp.dataloader_drop_last,
    )

    _setup_torch_runtime(exp)

    # ---- 优化器 & 调度器 ----
    optimizer = exp.build_optimizer(model)
    lr_scheduler = exp.build_lr_scheduler(optimizer)

    # ---- AMP (Automatic Mixed Precision) ----
    amp_dtype = _get_amp_dtype(exp)
    use_amp = amp_dtype is not None
    scaler = torch.amp.GradScaler('cuda', enabled=(use_amp and amp_dtype == torch.float16))
    if use_amp:
        logger.info(f"启用 AMP, dtype={amp_dtype}")

    # ---- GPU / DDP ----
    model = model.cuda()

    # ---- channels_last 内存格式 ----
    if getattr(exp, 'use_channels_last', False):
        model = model.to(memory_format=torch.channels_last)
        logger.info("启用 channels_last 内存格式")

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=exp.find_unused_parameters,
        )

    # ---- torch.compile ----
    if getattr(exp, 'use_compile', False):
        compile_backend = getattr(exp, 'compile_backend', 'inductor')
        compile_mode = getattr(exp, 'compile_mode', 'reduce-overhead')
        _model_to_compile = model.module if hasattr(model, 'module') else model
        _model_to_compile = torch.compile(
            _model_to_compile,
            backend=compile_backend,
            mode=compile_mode,
        )
        if hasattr(model, 'module'):
            model.module = _model_to_compile
        else:
            model = _model_to_compile
        logger.info(f"启用 torch.compile, backend={compile_backend}, mode={compile_mode}")

    # ---- 训练参数 ----
    max_epochs = exp.max_epochs
    grad_max_norm = exp.grad_max_norm
    warmup_iters = exp.warmup_iters
    warmup_ratio = exp.warmup_ratio
    base_lr = optimizer.param_groups[0]["lr"]
    rank, _ = get_dist_info()
    is_main_process = (rank == 0)

    logger.info(f"开始训练, 共 {max_epochs} epochs, 工作目录: {work_dir}")

    global_iter = 0
    for epoch in range(max_epochs):
        model.train()
        if distributed and hasattr(data_loader.sampler, "set_epoch"):
            data_loader.sampler.set_epoch(epoch)

        pbar = tqdm(
            data_loader,
            desc=f"Epoch [{epoch + 1}/{max_epochs}]",
            dynamic_ncols=True,
            disable=not is_main_process,
        )
        for i, data_batch in enumerate(pbar):
            # warmup
            if warmup_iters > 0 and global_iter < warmup_iters:
                lr = _get_warmup_lr(base_lr, warmup_ratio,
                                    warmup_iters, global_iter)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr

            data_batch = scatter_data(
                data_batch,
                non_blocking=exp.dataloader_non_blocking,
            )

            # ---- channels_last for input images ----
            if getattr(exp, 'use_channels_last', False):
                if 'img_inputs' in data_batch:
                    img_inputs = data_batch['img_inputs']
                    if isinstance(img_inputs, (list, tuple)) and len(img_inputs) > 0:
                        imgs = img_inputs[0]
                        if hasattr(imgs, 'ndim') and imgs.ndim == 5:
                            # (B, N, C, H, W) -> channels_last on (B*N, C, H, W)
                            B, N, C, H, W = imgs.shape
                            imgs = imgs.view(B*N, C, H, W).to(
                                memory_format=torch.channels_last).view(B, N, C, H, W)
                            data_batch['img_inputs'] = [imgs] + list(img_inputs[1:])

            # ---- Forward with AMP ----
            with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
                losses = model(return_loss=True, **data_batch)

            total_loss = torch.tensor(0.0, device="cuda")
            loss_log = {}
            for name, val in losses.items():
                if hasattr(val, 'backward'):
                    # torch.Tensor with grad
                    total_loss = total_loss + val
                    loss_log[name] = val.item()
                elif hasattr(val, '__iter__'):
                    # list/tuple of tensors
                    s = sum(v for v in val if hasattr(v, 'backward'))
                    if hasattr(s, 'item'):
                        total_loss = total_loss + s
                        loss_log[name] = s.item()

            optimizer.zero_grad(set_to_none=exp.optimizer_set_to_none)
            scaler.scale(total_loss).backward()
            if grad_max_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), grad_max_norm)
            scaler.step(optimizer)
            scaler.update()
            global_iter += 1

            lr = optimizer.param_groups[0]["lr"]
            pbar.set_postfix({
                "lr": f"{lr:.2e}",
                "loss": f"{total_loss.item():.4f}",
            })

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
