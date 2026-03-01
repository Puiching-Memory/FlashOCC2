#!/usr/bin/env python
"""FlashOCC 测试脚本.

用法:
    python tools/test.py configs/flashocc_r50.py ckpts/epoch_24.pth --eval occ
    torchrun --nproc_per_node=4 tools/test.py configs/flashocc_r50.py ckpts/epoch_24.pth --eval occ

性能优化参数 (AMP / channels_last / torch.compile / cudnn.benchmark / TF32)
自动从配置文件的 Experiment 对象中读取, 无需额外命令行参数。
"""
import argparse
import csv
import json
import os
import re
import sys

import numpy as np
import torch
import torch.distributed as dist

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from flashocc.config import load_experiment
from flashocc.core import init_dist, get_dist_info, load_checkpoint
from flashocc.core.log import logger, setup_logger
from flashocc.engine import single_gpu_test
from flashocc.engine.trainer import (
    build_dataloader, _setup_torch_runtime, _get_amp_dtype,
)


def parse_args():
    parser = argparse.ArgumentParser(description="FlashOCC 测试")
    parser.add_argument("config", help="Python 配置文件路径 (.py)")
    parser.add_argument("checkpoint", help="checkpoint 文件路径或目录路径")
    parser.add_argument(
        "--csv-out",
        help="评估结果 CSV 输出路径（默认: 单文件写到权重同目录，多文件写到输入目录）",
    )
    parser.add_argument("--eval", nargs="+", help="评估指标")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--show-dir")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--launcher", choices=["none", "pytorch", "slurm"],
                        default="none")
    parser.add_argument("--local_rank", "--local-rank", type=int, default=0)
    return parser.parse_args()


def collect_results_gpu(result_part, size):
    rank, world_size = get_dist_info()
    if world_size == 1:
        return result_part

    part_list = [None for _ in range(world_size)]
    dist.all_gather_object(part_list, result_part)

    if rank != 0:
        return None

    ordered_results = []
    for packed in zip(*part_list):
        ordered_results.extend(list(packed))
    return ordered_results[:size]


def _setup_model_for_test(model, exp):
    """根据 Experiment 配置, 为测试模型启用性能优化.

    注: torch.compile 在测试时不启用 — 推理只跑一遍, 编译开销 + CUDA graph
    分区问题得不偿失。AMP / channels_last 零开销, 保持启用。
    """
    model = model.cuda()

    # ---- channels_last 内存格式 ----
    if exp.use_channels_last:
        model = model.to(memory_format=torch.channels_last)
        logger.info("启用 channels_last 内存格式")

    return model


def _natural_sort_key(path):
    parts = re.split(r"(\d+)", os.path.basename(path))
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def _resolve_checkpoints(path):
    path = os.path.abspath(path)
    if os.path.isfile(path):
        return [path]

    if os.path.isdir(path):
        weight_exts = {".pth", ".pt", ".ckpt", ".bin"}
        candidates = []
        for name in os.listdir(path):
            full_path = os.path.join(path, name)
            if os.path.isfile(full_path) and os.path.splitext(name)[1].lower() in weight_exts:
                candidates.append(full_path)

        candidates.sort(key=_natural_sort_key)
        if not candidates:
            raise FileNotFoundError(f"目录 {path} 下未找到权重文件(.pth/.pt/.ckpt/.bin)")
        return candidates

    raise FileNotFoundError(f"输入路径不存在: {path}")


def _default_csv_out(input_path):
    input_path = os.path.abspath(input_path)
    if os.path.isfile(input_path):
        stem = os.path.splitext(os.path.basename(input_path))[0]
        return os.path.join(os.path.dirname(input_path), f"{stem}_eval.csv")
    return os.path.join(input_path, "eval_summary.csv")


def _to_csv_value(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass

    if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "tolist"):
        try:
            return json.dumps(value.detach().cpu().tolist(), ensure_ascii=False)
        except Exception:
            pass

    if hasattr(value, "tolist"):
        try:
            return json.dumps(value.tolist(), ensure_ascii=False)
        except Exception:
            pass

    if isinstance(value, (list, tuple)):
        normalized = [_to_csv_value(item) for item in value]
        return json.dumps(normalized, ensure_ascii=False)

    if isinstance(value, dict):
        normalized = {k: _to_csv_value(v) for k, v in value.items()}
        return json.dumps(normalized, ensure_ascii=False)

    return str(value)


def _save_metrics_csv(rows, output_path):
    """保存详细评估指标到 CSV — 将 per-class IoU 展开为独立列, 混淆矩阵单独保存."""
    # 构建扁平化行
    flat_rows = []
    confmats = []  # (checkpoint_name, confmat_array)

    for row in rows:
        flat = {}
        flat['checkpoint'] = row.get('checkpoint', '')
        flat['checkpoint_name'] = row.get('checkpoint_name', '')

        # 提取 epoch 编号 (从文件名如 epoch_24.pth)
        ckpt_name = flat.get('checkpoint_name', '')
        epoch_match = re.search(r'epoch[_\-]?(\d+)', ckpt_name)
        flat['epoch'] = int(epoch_match.group(1)) if epoch_match else ''

        for k, v in row.items():
            if k in ('checkpoint', 'checkpoint_name'):
                continue
            if k == 'per_class_iou' and isinstance(v, dict):
                for cls_name, iou_val in v.items():
                    flat[f'iou_{cls_name}'] = iou_val
            elif k == 'confusion_matrix':
                confmats.append((ckpt_name, v))
                continue  # 不写入主 CSV
            elif k == 'mIoU' and hasattr(v, 'tolist'):
                continue  # 已通过 per_class_iou 展开
            elif isinstance(v, np.ndarray):
                flat[k] = json.dumps(v.tolist(), ensure_ascii=False)
            elif isinstance(v, (dict, list, tuple)):
                flat[k] = json.dumps(v, ensure_ascii=False) if not isinstance(v, str) else v
            else:
                flat[k] = v
        flat_rows.append(flat)

    # 收集所有列名 (保持插入顺序)
    all_fields = []
    for row in flat_rows:
        for key in row.keys():
            if key not in all_fields:
                all_fields.append(key)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_fields)
        writer.writeheader()
        for row in flat_rows:
            writer.writerow({k: _to_csv_value(v) for k, v in row.items()})

    # 保存混淆矩阵为独立 .npz 和 CSV
    out_dir = os.path.dirname(output_path) or "."
    for ckpt_name, confmat in confmats:
        stem = os.path.splitext(ckpt_name)[0]
        # .npz 格式 (方便后续 numpy 加载)
        npz_path = os.path.join(out_dir, f"{stem}_confmat.npz")
        np.savez_compressed(npz_path, confusion_matrix=confmat)
        # CSV 格式 (方便 Excel 查看)
        csv_cm_path = os.path.join(out_dir, f"{stem}_confmat.csv")
        try:
            from flashocc.constants import OCC_CLASS_NAMES
            cls_names = list(OCC_CLASS_NAMES)
        except Exception:
            cls_names = [f"class_{i}" for i in range(confmat.shape[0])]
        with open(csv_cm_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["pred\\gt"] + cls_names[:confmat.shape[1]])
            for i, row_data in enumerate(confmat):
                label = cls_names[i] if i < len(cls_names) else f"class_{i}"
                writer.writerow([label] + [int(x) for x in row_data])
        logger.info(f"混淆矩阵已保存: {npz_path}, {csv_cm_path}")

    return flat_rows


def _plot_miou_curve(flat_rows, output_path):
    """使用 matplotlib 绘制 mIOU 曲线并保存."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib 未安装, 跳过 mIOU 曲线绘制")
        return

    # 收集有 epoch 和 mIoU_mean 的行
    epochs = []
    mious = []
    per_class_data: dict[str, list[float]] = {}

    for row in flat_rows:
        ep = row.get('epoch', '')
        miou = row.get('mIoU_mean')
        if ep == '' or miou is None:
            continue
        epochs.append(int(ep))
        mious.append(float(miou))

        # 收集 per-class iou
        for k, v in row.items():
            if k.startswith('iou_') and v is not None:
                cls = k[4:]
                per_class_data.setdefault(cls, []).append(float(v))

    if not epochs:
        logger.info("无有效 epoch mIOU 数据, 跳过曲线绘制")
        return

    # 按 epoch 排序
    order = sorted(range(len(epochs)), key=lambda i: epochs[i])
    epochs = [epochs[i] for i in order]
    mious = [mious[i] for i in order]
    for cls in per_class_data:
        per_class_data[cls] = [per_class_data[cls][i] for i in order]

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # ---- 左图: 总 mIOU 曲线 ----
    ax1 = axes[0]
    ax1.plot(epochs, mious, 'b-o', linewidth=2, markersize=5, label='mIoU')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('mIoU (%)', fontsize=12)
    ax1.set_title('mIoU vs Epoch', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)

    # 标注最大值
    if mious:
        max_idx = int(np.argmax(mious))
        ax1.annotate(f'{mious[max_idx]:.2f}%',
                     xy=(epochs[max_idx], mious[max_idx]),
                     xytext=(5, 10), textcoords='offset points',
                     fontsize=10, fontweight='bold', color='red',
                     arrowprops=dict(arrowstyle='->', color='red'))

    # ---- 右图: per-class IoU 曲线 ----
    ax2 = axes[1]
    cmap = plt.cm.get_cmap('tab20', len(per_class_data))
    for idx, (cls, vals) in enumerate(per_class_data.items()):
        ax2.plot(epochs, vals, '-o', color=cmap(idx), linewidth=1,
                 markersize=3, label=cls)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('IoU (%)', fontsize=12)
    ax2.set_title('Per-Class IoU vs Epoch', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=7, ncol=2, loc='best')

    plt.tight_layout()
    plot_path = output_path.replace('.csv', '_miou_curve.png')
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"mIOU 曲线图已保存: {plot_path}")


def main():
    args = parse_args()
    checkpoints = _resolve_checkpoints(args.checkpoint)

    exp = load_experiment(args.config)

    distributed = args.launcher != "none"
    if distributed:
        mesh = init_dist(args.launcher)  # DeviceMesh 统一管理进程组

    try:
        rank, _ = get_dist_info()

        setup_logger(level="INFO", rank0_only=True)

        # ---- 从配置读取运行时优化参数 ----
        _setup_torch_runtime(exp)
        amp_dtype = _get_amp_dtype(exp)
        logger.info(
            f"运行时配置: AMP={exp.use_amp} (dtype={exp.amp_dtype}), "
            f"channels_last={exp.use_channels_last}, "
            f"compile=disabled (test), "
            f"cudnn_benchmark={exp.cudnn_benchmark}, "
            f"allow_tf32={exp.allow_tf32}"
        )

        # 构建测试数据集
        dataset = exp.build_test_dataset()
        logger.info(f"测试集: {dataset.__class__.__name__} ({len(dataset)} 样本)")

        data_loader = build_dataloader(
            dataset, samples_per_gpu=1,
            workers_per_gpu=exp.workers_per_gpu,
            dist_mode=distributed,
            shuffle=False,
        )

        # 构建模型 (读取配置中的优化参数)
        model = exp.build_model()
        model = _setup_model_for_test(model, exp)

        if rank == 0:
            logger.info(f"待评估权重数: {len(checkpoints)}")

        csv_rows = []

        for index, checkpoint_path in enumerate(checkpoints, start=1):
            logger.info(f"[{index}/{len(checkpoints)}] 加载权重: {checkpoint_path}")
            load_checkpoint(model, checkpoint_path, map_location="cpu")

            logger.info(f"[{index}/{len(checkpoints)}] 开始测试...")
            results = single_gpu_test(
                model, data_loader,
                show=args.show, out_dir=args.show_dir,
                amp_dtype=amp_dtype,
                use_channels_last=exp.use_channels_last,
                non_blocking=exp.dataloader_non_blocking,
                img_color_order=exp.image_color_order,
            )
            if distributed:
                results = collect_results_gpu(results, len(dataset))

            if rank == 0:
                row = {
                    "checkpoint": checkpoint_path,
                    "checkpoint_name": os.path.basename(checkpoint_path),
                }
                if args.eval and hasattr(dataset, "evaluate"):
                    eval_kwargs = {"metric": args.eval}
                    metrics = dataset.evaluate(results, **eval_kwargs)
                    for k, v in metrics.items():
                        if k == 'per_class_iou':
                            # per_class_iou 是 dict, 保持原始结构
                            row[k] = v
                            for cls_name, iou_val in v.items():
                                logger.info(f"[{os.path.basename(checkpoint_path)}] iou_{cls_name}: {iou_val}")
                        elif k == 'confusion_matrix':
                            row[k] = v
                        elif k == 'mIoU' and hasattr(v, 'tolist'):
                            # 原始 per-class array, 已通过 per_class_iou 展开
                            row[k] = v
                        else:
                            logger.info(f"[{os.path.basename(checkpoint_path)}] {k}: {v}")
                            row[k] = v
                csv_rows.append(row)

        if rank == 0:
            csv_out = args.csv_out or _default_csv_out(args.checkpoint)
            flat_rows = _save_metrics_csv(csv_rows, csv_out)
            logger.info(f"评估结果已保存到 CSV: {csv_out}")

            # 绘制 mIOU 曲线
            _plot_miou_curve(flat_rows, csv_out)

        logger.info("测试完成。")
    finally:
        if distributed and dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
