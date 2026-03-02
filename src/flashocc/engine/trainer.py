"""训练引擎 — 直接消费 Experiment 对象.

所有参数从 Experiment 字段读取。
"""
import csv
import os

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from flashocc.core import get_dist_info
from flashocc.core.dist import setup_parallel
from flashocc.core.log import logger
from flashocc.engine.parallel import DataContainer
from flashocc.datasets.dali_decode import dali_decode_batch


# =====================================================================
#  DataLoader — collate
# =====================================================================

def _collate_elem(elem, batch):
    """根据元素类型选择 collate 策略."""
    if isinstance(elem, DataContainer):
        if batch[0].stack:
            stacked = torch.stack([s.data for s in batch], dim=0)
            return DataContainer(stacked, stack=batch[0].stack,
                                 padding_value=batch[0].padding_value,
                                 cpu_only=batch[0].cpu_only)
        else:
            return DataContainer([s.data for s in batch],
                                 stack=False, cpu_only=batch[0].cpu_only)
    if isinstance(elem, dict):
        return {key: collate_fn([d[key] for d in batch]) for key in batch[0]}
    if isinstance(elem, torch.Tensor):
        return torch.stack(batch, 0)
    if isinstance(elem, (int, float)):
        return torch.tensor(batch)
    if isinstance(elem, np.ndarray):
        return torch.from_numpy(np.stack(batch))
    # fallback
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


def scatter_data(data, device=None, non_blocking=False):
    """递归将数据搬到 GPU, 处理 DataContainer / dict / list / tuple."""
    if device is None:
        device = torch.cuda.current_device()

    if isinstance(data, DataContainer):
        if data.cpu_only:
            return data.data
        if data.stack:
            return _to_cuda_if_tensor(data.data, device, non_blocking=non_blocking)
        else:
            return [_to_cuda_if_tensor(d, device, non_blocking=non_blocking) for d in data.data]

    if isinstance(data, dict):
        return {k: scatter_data(v, device, non_blocking=non_blocking) for k, v in data.items()}

    if isinstance(data, list):
        return [scatter_data(v, device, non_blocking=non_blocking) for v in data]

    if isinstance(data, tuple):
        return tuple(scatter_data(v, device, non_blocking=non_blocking) for v in data)

    # fallback: tensor or other
    if hasattr(data, 'cuda'):
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

def train_model(experiment, model, mesh=None, validate=False,
                timestamp=None, meta=None):
    """训练入口 — 直接消费 Experiment 对象.

    Parameters
    ----------
    experiment : Experiment
        实验配置.
    model : nn.Module
        已构建的模型 (可能已加载预训练).
    mesh : DeviceMesh or None
        全局 DeviceMesh, None 表示单卡训练.
    """
    exp = experiment
    distributed = mesh is not None

    # ---- 工作目录 ----
    work_dir = exp.work_dir or "work_dirs"
    os.makedirs(work_dir, exist_ok=True)

    # ---- 数据 ----
    dataset = exp.build_train_dataset()
    _, world_size = get_dist_info()
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=exp.samples_per_gpu,
        workers_per_gpu=exp.workers_per_gpu,
        num_gpus=world_size,
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

    # ---- 并行策略 (基于 DeviceMesh) ----
    parallel_mode = getattr(exp, 'parallel_mode', 'ddp')
    model = setup_parallel(
        model, mesh, mode=parallel_mode,
        broadcast_buffers=False,
        find_unused_parameters=exp.find_unused_parameters,
    )

    # ---- torch.compile ----
    if getattr(exp, 'use_compile', False):
        compile_backend = getattr(exp, 'compile_backend', 'inductor')
        compile_mode = getattr(exp, 'compile_mode', 'reduce-overhead')
        
        # 方案 A: 只编译纯 Tensor 计算的子模块，而不是整个模型
        # 这样可以避免 img_metas 等包含动态字符串的参数导致反复重新编译
        _model_to_compile = model.module if hasattr(model, 'module') else model
        
        if hasattr(_model_to_compile, 'img_backbone') and _model_to_compile.img_backbone is not None:
            _model_to_compile.img_backbone = torch.compile(
                _model_to_compile.img_backbone, backend=compile_backend, mode=compile_mode)
        if hasattr(_model_to_compile, 'img_neck') and _model_to_compile.img_neck is not None:
            _model_to_compile.img_neck = torch.compile(
                _model_to_compile.img_neck, backend=compile_backend, mode=compile_mode)
        if hasattr(_model_to_compile, 'img_bev_encoder_backbone') and _model_to_compile.img_bev_encoder_backbone is not None:
            _model_to_compile.img_bev_encoder_backbone = torch.compile(
                _model_to_compile.img_bev_encoder_backbone, backend=compile_backend, mode=compile_mode)
        if hasattr(_model_to_compile, 'img_bev_encoder_neck') and _model_to_compile.img_bev_encoder_neck is not None:
            _model_to_compile.img_bev_encoder_neck = torch.compile(
                _model_to_compile.img_bev_encoder_neck, backend=compile_backend, mode=compile_mode)
        if hasattr(_model_to_compile, 'pts_bbox_head') and _model_to_compile.pts_bbox_head is not None:
            _model_to_compile.pts_bbox_head = torch.compile(
                _model_to_compile.pts_bbox_head, backend=compile_backend, mode=compile_mode)
                
        logger.info(f"启用 torch.compile (子模块编译), backend={compile_backend}, mode={compile_mode}")

    # ---- 训练参数 ----
    max_epochs = exp.max_epochs
    grad_max_norm = exp.grad_max_norm
    warmup_iters = exp.warmup_iters
    warmup_ratio = exp.warmup_ratio
    base_lr = optimizer.param_groups[0]["lr"]
    rank, _ = get_dist_info()
    is_main_process = (rank == 0)

    # ---- EMA (Exponential Moving Average) ----
    ema_model = None
    if getattr(exp, 'use_ema', False):
        from flashocc.engine.hooks.ema import ModelEMA
        ema_decay = getattr(exp, 'ema_decay', 0.9990)
        ema_init_updates = getattr(exp, 'ema_init_updates', 0)
        ema_resume = getattr(exp, 'ema_resume', None)
        ema_model = ModelEMA(model, decay=ema_decay, updates=ema_init_updates)
        if ema_resume is not None:
            from flashocc.core import load_state_dict
            cpt = torch.load(ema_resume, map_location='cpu')
            load_state_dict(ema_model.ema, cpt['state_dict'])
            ema_model.updates = cpt.get('updates', 0)
            logger.info(f"恢复 EMA checkpoint: {ema_resume}")
        logger.info(f"启用 EMA, decay={ema_decay}")

    # ---- trackio 实验跟踪 ----
    trackio_run = None
    if is_main_process and getattr(exp, 'trackio_project', None):
        try:
            import trackio
            trackio_run = trackio.init(
                project=exp.trackio_project,
                name=getattr(exp, 'trackio_name', None) or timestamp,
                group=getattr(exp, 'trackio_group', None),
                config={
                    "max_epochs": max_epochs,
                    "samples_per_gpu": exp.samples_per_gpu,
                    "lr": base_lr,
                    "grad_max_norm": grad_max_norm,
                    "use_amp": exp.use_amp,
                    "amp_dtype": getattr(exp, 'amp_dtype', None),
                    "use_ema": getattr(exp, 'use_ema', False),
                    "use_compile": getattr(exp, 'use_compile', False),
                    "model": exp.model.cls.__name__,
                },
            )
            logger.info(f"启用 trackio 实验跟踪, project={exp.trackio_project}")
        except Exception as e:
            logger.warning(f"trackio 初始化失败: {e}, 继续训练但不记录")
            trackio_run = None

    logger.info(f"开始训练, 共 {max_epochs} epochs, 工作目录: {work_dir}")

    # ---- loss 日志 CSV ----
    loss_csv_path = os.path.join(work_dir, "train_loss.csv")
    loss_csv_header_written = False

    global_iter = 0
    for epoch in range(max_epochs):
        model.train()
        if distributed and hasattr(data_loader.sampler, "set_epoch"):
            data_loader.sampler.set_epoch(epoch)

        # 用于累计 epoch 级 loss 统计
        epoch_loss_sum = 0.0
        epoch_loss_detail: dict[str, float] = {}
        epoch_steps = 0

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

            # ---- DALI GPU 图像解码 ----
            if 'jpeg_bytes' in data_batch:
                data_batch = dali_decode_batch(
                    data_batch,
                    color_order=exp.image_color_order,
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
            # 每次前向前标记新 step，避免 CUDA Graphs 张量被后续 run 覆盖
            # 参见: https://pytorch.org/docs/stable/compiler_cudagraph_trees.html
            torch.compiler.cudagraph_mark_step_begin()
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

            # ---- EMA 更新 ----
            if ema_model is not None:
                ema_model.update(None, model)

            # 累计 epoch 级 loss
            epoch_loss_sum += total_loss.item()
            for k, v in loss_log.items():
                epoch_loss_detail[k] = epoch_loss_detail.get(k, 0.0) + v
            epoch_steps += 1

            lr = optimizer.param_groups[0]["lr"]
            pbar.set_postfix({
                "lr": f"{lr:.2e}",
                "loss": f"{total_loss.item():.4f}",
            })

            # ---- trackio: 记录每步 loss ----
            if trackio_run is not None:
                import trackio
                trackio.log({"loss": total_loss.item(), "lr": lr, **loss_log},
                            step=global_iter)

        # epoch 结束 → LR step
        if lr_scheduler is not None:
            lr_scheduler.step()

        # ---- 记录 epoch 平均 loss 到日志和 CSV ----
        if epoch_steps > 0 and is_main_process:
            avg_loss = epoch_loss_sum / epoch_steps
            avg_detail = {k: v / epoch_steps for k, v in epoch_loss_detail.items()}
            detail_str = ", ".join(f"{k}={v:.4f}" for k, v in avg_detail.items())
            logger.info(
                f"Epoch [{epoch + 1}/{max_epochs}] 平均 loss: {avg_loss:.4f} ({detail_str}), "
                f"lr: {optimizer.param_groups[0]['lr']:.2e}"
            )

            # 写 CSV
            row = {"epoch": epoch + 1, "avg_loss": avg_loss, "lr": optimizer.param_groups[0]["lr"]}
            row.update(avg_detail)
            if not loss_csv_header_written:
                with open(loss_csv_path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                    writer.writeheader()
                    writer.writerow(row)
                loss_csv_header_written = True
                logger.info(f"Loss CSV 创建: {loss_csv_path}")
            else:
                with open(loss_csv_path, "a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                    writer.writerow(row)

            # ---- trackio: 记录 epoch 级指标 ----
            if trackio_run is not None:
                import trackio
                epoch_metrics = {"epoch": epoch + 1, "avg_loss": avg_loss}
                epoch_metrics.update({f"avg_{k}": v for k, v in avg_detail.items()})
                trackio.log(epoch_metrics, step=global_iter)

        # 保存 checkpoint（仅主进程写盘，避免多 rank 竞争同一文件）
        if is_main_process:
            ckpt_path = os.path.join(work_dir, f"epoch_{epoch+1}.pth")
            raw_model = model.module if hasattr(model, "module") else model
            state = {
                "meta": {"epoch": epoch + 1, **(meta or {})},
                "state_dict": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            torch.save(state, ckpt_path)
            logger.info(f"已保存 checkpoint: {ckpt_path}")

            # ---- EMA checkpoint ----
            if ema_model is not None:
                ema_ckpt_path = os.path.join(work_dir, f"epoch_{epoch+1}_ema.pth")
                ema_state = {
                    "epoch": epoch + 1,
                    "state_dict": ema_model.ema.state_dict(),
                    "updates": ema_model.updates,
                }
                torch.save(ema_state, ema_ckpt_path)
                logger.info(f"已保存 EMA checkpoint: {ema_ckpt_path}")

            # 清理旧 checkpoint (max_keep_ckpts > 0 时保留最新 N 个, -1 表示不删除)
            max_keep = getattr(exp, 'max_keep_ckpts', -1)
            if max_keep > 0:
                import glob as _glob
                ckpt_files = sorted(
                    _glob.glob(os.path.join(work_dir, "epoch_*.pth")),
                    key=os.path.getmtime,
                )
                # 排除 EMA checkpoint (epoch_*_ema.pth)
                ckpt_files = [f for f in ckpt_files if not f.endswith("_ema.pth")]
                while len(ckpt_files) > max_keep:
                    old = ckpt_files.pop(0)
                    os.remove(old)
                    logger.info(f"已删除旧 checkpoint: {old}")

        # epoch 级同步，防止 rank 间进度漂移
        if distributed and dist.is_available() and dist.is_initialized():
            dist.barrier()

    # ---- trackio 结束 ----
    if trackio_run is not None:
        import trackio
        trackio.finish()
        logger.info("trackio 实验跟踪已结束")

    logger.info("训练完成。")


__all__ = ["train_model", "build_dataloader", "collate_fn", "scatter_data"]
