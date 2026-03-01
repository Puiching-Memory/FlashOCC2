#!/usr/bin/env python
"""FlashOCC 训练脚本.

:
    # 单 GPU
    python tools/train.py configs/flashocc_r50.py

    # 多 GPU (DDP)
    torchrun --nproc_per_node=4 tools/train.py configs/flashocc_r50.py
"""
import argparse
import csv
import os
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from flashocc.config import load_experiment
from flashocc.core import init_dist, get_dist_info, load_checkpoint
from flashocc.core.env import collect_env, setup_multi_processes
from flashocc.core.log import logger, setup_logger
from flashocc.engine import train_model, init_random_seed, set_random_seed

try:
    from thop import profile as thop_profile, clever_format
    HAS_THOP = True
except ImportError:
    HAS_THOP = False


def parse_args():
    parser = argparse.ArgumentParser(description="FlashOCC 训练")
    parser.add_argument("config", help="Python 配置文件路径 (.py)")
    parser.add_argument("--work-dir", help="工作目录")
    parser.add_argument("--resume-from", help="从 checkpoint 恢复训练")
    parser.add_argument("--validate", action="store_true", help="训练时验证")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU ID (单卡)")
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--launcher", choices=["none", "pytorch", "slurm"],
                        default="none")
    parser.add_argument("--parallel-mode", choices=["ddp", "fsdp2"],
                        default=None, help="并行策略 (覆盖配置文件)")
    parser.add_argument("--local_rank", "--local-rank", type=int, default=0)
    return parser.parse_args()


def _apply_freeze_modules(model, freeze_modules: list[str]) -> None:
    if not freeze_modules:
        return

    module_map = dict(model.named_modules())
    frozen_param_ids = set()
    frozen_names = []

    for module_name in freeze_modules:
        module = module_map.get(module_name)
        if module is None:
            logger.warning(f"冻结跳过: 未找到模块 '{module_name}'")
            continue

        module.eval()
        newly_frozen = 0
        for param in module.parameters():
            param_id = id(param)
            if param_id in frozen_param_ids:
                continue
            if param.requires_grad:
                newly_frozen += param.numel()
                param.requires_grad = False
            frozen_param_ids.add(param_id)

        frozen_names.append(module_name)
        logger.info(f"已冻结模块: {module_name} (新增冻结参数 {newly_frozen})")

    if frozen_names:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        frozen = total - trainable
        pct = trainable / total * 100 if total > 0 else 0.0
        logger.info(
            f"冻结完成: {len(frozen_names)} 个模块, "
            f"frozen {frozen:,} / total {total:,}, "
            f"trainable {trainable:,} ({pct:.2f}%)"
        )


def _log_model_profile(model, exp):
    """使用 ultralytics-thop 计算并打印各子模块的参数量与 FLOPs."""
    if not HAS_THOP:
        logger.warning("thop 未安装, 跳过 FLOPs 计算 (pip install ultralytics-thop)")
        return

    device = next(model.parameters()).device
    raw = model.module if hasattr(model, 'module') else model

    # ----- 各子模块独立 profile -----
    submodules = [
        ("img_backbone",            (1 * 6, 3, 256, 704)),
        ("img_neck",                None),  # 需要 backbone 输出, 见下方
        ("img_bev_encoder_backbone", None),
        ("img_bev_encoder_neck",     None),
        ("occ_head",                 None),
    ]

    # 先通过 backbone + neck 获得真实中间 shape
    dummy_shapes: dict[str, tuple] = {}
    try:
        # img_backbone
        bb = getattr(raw, 'img_backbone', None)
        if bb is not None:
            inp = torch.randn(6, 3, 256, 704, device=device)
            bb_out = bb(inp)
            if isinstance(bb_out, (list, tuple)):
                # neck 输入
                dummy_shapes['img_neck'] = [o.shape for o in bb_out]
                last = bb_out[-1] if isinstance(bb_out[-1], torch.Tensor) else bb_out[0]
            else:
                last = bb_out

        # img_neck
        nk = getattr(raw, 'img_neck', None)
        if nk is not None and 'img_neck' in dummy_shapes:
            neck_out = nk(bb_out)
            if isinstance(neck_out, (list, tuple)) and len(neck_out) > 0:
                neck_feat = neck_out[0]
            else:
                neck_feat = neck_out
            # view transformer 输出 -> bev_encoder_backbone 输入
            # 使用 grid_config 估计 bev shape
            numC_Trans = getattr(raw, 'img_view_transformer', None)
            if numC_Trans is not None:
                out_ch = getattr(numC_Trans, 'out_channels', 64)
            else:
                out_ch = 64
            dummy_shapes['img_bev_encoder_backbone'] = (1, out_ch, 200, 200)

        # bev_encoder_backbone
        bev_bb = getattr(raw, 'img_bev_encoder_backbone', None)
        if bev_bb is not None and 'img_bev_encoder_backbone' in dummy_shapes:
            bev_inp = torch.randn(*dummy_shapes['img_bev_encoder_backbone'], device=device)
            bev_out = bev_bb(bev_inp)
            if isinstance(bev_out, (list, tuple)):
                dummy_shapes['img_bev_encoder_neck'] = [o.shape for o in bev_out]
            else:
                dummy_shapes['img_bev_encoder_neck'] = bev_out.shape

        # bev_encoder_neck
        bev_nk = getattr(raw, 'img_bev_encoder_neck', None)
        if bev_nk is not None and 'img_bev_encoder_neck' in dummy_shapes:
            if isinstance(dummy_shapes['img_bev_encoder_neck'], list):
                bev_nk_inp = [torch.randn(*s, device=device) for s in dummy_shapes['img_bev_encoder_neck']]
            else:
                bev_nk_inp = torch.randn(*dummy_shapes['img_bev_encoder_neck'], device=device)
            bev_nk_out = bev_nk(bev_nk_inp)
            if isinstance(bev_nk_out, (list, tuple)) and len(bev_nk_out) > 0:
                bev_nk_out = bev_nk_out[0]
            dummy_shapes['occ_head'] = bev_nk_out.shape
    except Exception as e:
        logger.warning(f"中间 shape 推导失败: {e}, 将仅打印参数量")

    logger.info("═" * 60)
    logger.info("模型 Profile (ultralytics-thop)")
    logger.info("═" * 60)

    total_params_all = 0
    total_flops_all = 0

    for name, default_shape in submodules:
        mod = getattr(raw, name, None)
        if mod is None:
            continue
        n_params = sum(p.numel() for p in mod.parameters())
        total_params_all += n_params

        # 尝试 FLOPs profile
        flops = 0
        try:
            if name == 'img_backbone':
                inp = torch.randn(6, 3, 256, 704, device=device)
                flops, _ = thop_profile(mod, inputs=(inp,), verbose=False)
            elif name == 'img_neck' and 'img_neck' in dummy_shapes:
                inp = [torch.randn(*s, device=device) for s in dummy_shapes['img_neck']]
                flops, _ = thop_profile(mod, inputs=(inp,), verbose=False)
            elif name in dummy_shapes:
                shape = dummy_shapes[name]
                if isinstance(shape, list):
                    inp = [torch.randn(*s, device=device) for s in shape]
                    flops, _ = thop_profile(mod, inputs=(inp,), verbose=False)
                else:
                    inp = torch.randn(*shape, device=device)
                    flops, _ = thop_profile(mod, inputs=(inp,), verbose=False)
        except Exception as e:
            logger.debug(f"  {name} FLOPs 计算失败: {e}")
            flops = 0

        total_flops_all += flops
        p_str, f_str = clever_format([n_params, flops], "%.3f")
        trainable = sum(p.numel() for p in mod.parameters() if p.requires_grad)
        t_str = clever_format([trainable], "%.3f")[0]
        logger.info(f"  {name:30s}  Params: {p_str:>10s}  Trainable: {t_str:>10s}  FLOPs: {f_str:>10s}")

    # 统计 view_transformer (仅参数量, FLOPs 难以 profile)
    vt = getattr(raw, 'img_view_transformer', None)
    if vt is not None:
        n_params_vt = sum(p.numel() for p in vt.parameters())
        total_params_all += n_params_vt
        t_vt = sum(p.numel() for p in vt.parameters() if p.requires_grad)
        p_str = clever_format([n_params_vt], "%.3f")[0]
        t_str = clever_format([t_vt], "%.3f")[0]
        logger.info(f"  {'img_view_transformer':30s}  Params: {p_str:>10s}  Trainable: {t_str:>10s}  FLOPs: {'N/A':>10s}")

    logger.info("─" * 60)
    total_params_model = sum(p.numel() for p in raw.parameters())
    total_trainable = sum(p.numel() for p in raw.parameters() if p.requires_grad)
    p_str, f_str = clever_format([total_params_model, total_flops_all], "%.3f")
    t_str = clever_format([total_trainable], "%.3f")[0]
    pct = total_trainable / total_params_model * 100 if total_params_model > 0 else 0
    logger.info(f"  {'TOTAL':30s}  Params: {p_str:>10s}  Trainable: {t_str:>10s} ({pct:.2f}%)  FLOPs: {f_str:>10s}")
    logger.info("═" * 60)


def main():
    args = parse_args()

    # 加载实验配置
    exp = load_experiment(args.config)

    # 覆盖运行时选项
    if args.work_dir:
        exp.work_dir = args.work_dir
    if exp.work_dir is None:
        config_name = os.path.splitext(os.path.basename(args.config))[0]
        exp.work_dir = os.path.join("work_dirs", config_name)
    os.makedirs(exp.work_dir, exist_ok=True)

    if args.resume_from:
        exp.resume_from = args.resume_from

    # 分布式 — 基于 DeviceMesh 统一抽象
    mesh = None
    if args.launcher != "none":
        mesh = init_dist(args.launcher)
    rank, world_size = get_dist_info()

    # 日志
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_file = os.path.join(exp.work_dir, f"{timestamp}.log")
    setup_logger(log_file=log_file, level="INFO", rank0_only=True)
    logger.info(f"环境信息:\n{collect_env()}")

    # 随机种子
    seed = init_random_seed(args.seed if args.seed is not None else exp.seed)
    set_random_seed(seed, deterministic=args.deterministic)
    exp.seed = seed

    # GPU
    if args.parallel_mode:
        exp.parallel_mode = args.parallel_mode

    # 构建模型
    model = exp.build_model()
    logger.info(f"模型: {model.__class__.__name__}")

    # 子模块 init_cfg 预训练权重加载
    for name, m in model.named_modules():
        if hasattr(m, 'init_weights') and callable(m.init_weights):
            if getattr(m, '_raw_init_cfg', None) is not None:
                logger.info(f"加载子模块预训练: {name}")
                m.init_weights()

    # 加载预训练
    if exp.load_from:
        logger.info(f"加载预训练: {exp.load_from}")
        load_checkpoint(model, exp.load_from, map_location="cpu",
                        strict=False, logger=logger)

    # 按配置冻结任意模块
    _apply_freeze_modules(model, exp.freeze_modules)

    # 模型 profile: 参数量 & FLOPs
    model.cuda()
    _log_model_profile(model, exp)
    model.cpu()

    # 训练
    meta = dict(seed=seed, timestamp=timestamp)
    train_model(
        experiment=exp,
        model=model,
        mesh=mesh,
        validate=args.validate,
        timestamp=timestamp,
        meta=meta,
    )


if __name__ == "__main__":
    main()
