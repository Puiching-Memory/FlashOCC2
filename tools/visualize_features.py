#!/usr/bin/env python
"""FlashOCC 特征可视化 + 3D 占用预测可视化工具.

功能:
    1. 可视化模型每个大层的输出特征 (img_backbone / img_neck /
       img_view_transformer / img_bev_encoder_backbone / img_bev_encoder_neck)
    2. 可视化最终 3D 占用预测结果, 按语义类别颜色区分

用法:
    python tools/visualize_features.py configs/flashocc_r50.py \\
        work_dirs/flashocc_r50/epoch_24.pth \\
        --sample-idx 0 \\
        --out-dir vis_output

    可选参数:
        --sample-idx   可视化的样本索引 (默认 0)
        --out-dir      输出目录 (默认 vis_output)
        --device       推理设备 (默认 cuda:0)
        --no-feat      跳过中间特征可视化
        --no-occ       跳过 3D 占用可视化
        --occ-thresh   占用可视化时忽略 free 类概率阈值 (默认 0.5)
        --voxel-step   3D 可视化体素下采样步长 (默认 1, 越大越快)
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import OrderedDict
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import matplotlib

matplotlib.use("Agg")  # 无头模式
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from flashocc.config import load_experiment
from flashocc.core import load_checkpoint
from flashocc.constants import OCC_CLASS_NAMES, POINT_CLOUD_RANGE, VOXEL_SIZE
from flashocc.engine.trainer import build_dataloader


# =====================================================================
#  NuScenes OCC 18 类颜色表 (RGB 0-1)
# =====================================================================

OCC_CLASS_COLORS: dict[int, tuple[float, float, float]] = {
    0:  (0.00, 0.00, 0.00),   # others          — 黑色
    1:  (1.00, 0.73, 0.47),   # barrier          — 橙黄
    2:  (1.00, 0.83, 0.00),   # bicycle          — 金黄
    3:  (0.00, 0.00, 0.90),   # bus              — 蓝色
    4:  (1.00, 0.00, 0.00),   # car              — 红色
    5:  (0.55, 0.27, 0.07),   # construction     — 棕色
    6:  (0.00, 0.00, 0.55),   # motorcycle       — 深蓝
    7:  (0.00, 0.75, 1.00),   # pedestrian       — 青色
    8:  (1.00, 0.40, 0.00),   # traffic_cone     — 橙色
    9:  (0.50, 0.00, 0.50),   # trailer          — 紫色
    10: (0.65, 0.16, 0.16),   # truck            — 暗红
    11: (0.60, 0.60, 0.60),   # driveable_surface— 灰色
    12: (0.75, 0.75, 0.75),   # other_flat        — 浅灰
    13: (0.85, 0.55, 0.85),   # sidewalk          — 粉紫
    14: (0.40, 0.70, 0.40),   # terrain           — 草绿
    15: (0.70, 0.70, 0.90),   # manmade           — 蓝灰
    16: (0.13, 0.55, 0.13),   # vegetation        — 绿色
    17: (1.00, 1.00, 1.00),   # free              — 白色 (不绘制)
}


# =====================================================================
#  特征图 Hook
# =====================================================================

class FeatureExtractor:
    """注册 forward hook, 自动捕获各层输出."""

    def __init__(self):
        self.features: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._hooks: list[Any] = []

    def register(self, model: nn.Module):
        """在 BEVDetOCC 的每个主要子模块上注册 hook."""
        target_layers = [
            ("img_backbone", "1. Image Backbone (ResNet50)"),
            ("img_neck", "2. Image Neck (FPN)"),
            ("img_view_transformer", "3. View Transformer (LSS)"),
            ("img_bev_encoder_backbone", "4. BEV Encoder Backbone"),
            ("img_bev_encoder_neck", "5. BEV Encoder Neck"),
            ("occ_head", "6. OCC Head"),
        ]
        for attr_name, display_name in target_layers:
            module = getattr(model, attr_name, None)
            if module is None:
                continue
            hook = module.register_forward_hook(
                self._make_hook(display_name)
            )
            self._hooks.append(hook)

    def _make_hook(self, name: str):
        def hook_fn(module, input, output):
            if isinstance(output, (list, tuple)):
                # 取第一个张量 (如 FPN 多尺度输出)
                for o in output:
                    if isinstance(o, torch.Tensor):
                        self.features[name] = o.detach().cpu()
                        return
            elif isinstance(output, torch.Tensor):
                self.features[name] = output.detach().cpu()
        return hook_fn

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# =====================================================================
#  特征可视化函数
# =====================================================================

def visualize_feature_map(feat: torch.Tensor, title: str, save_path: str):
    """可视化单个特征图, 支持 2D (B, C, H, W) 和高维.

    绘制方式:
        - 通道均值热力图
        - 通道最大值热力图
        - PCA 前 3 通道 → RGB
        - 随机 16 个通道的小图
    """
    # 取 batch 中第一个样本
    if feat.dim() == 5:
        # (B, N, C, H, W) → 取第一个视角
        feat = feat[0, 0]
    elif feat.dim() == 4:
        feat = feat[0]
    elif feat.dim() == 3:
        pass
    else:
        print(f"  [跳过] {title}: 形状 {feat.shape} 不適合可视化")
        return

    C, H, W = feat.shape[-3], feat.shape[-2], feat.shape[-1]
    feat_2d = feat.reshape(C, H, W).float()

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(f"{title}\nShape: {list(feat.shape)}", fontsize=14, fontweight="bold")

    # 1) 通道均值
    mean_map = feat_2d.mean(dim=0).numpy()
    im0 = axes[0, 0].imshow(mean_map, cmap="viridis", aspect="auto")
    axes[0, 0].set_title("Channel Mean")
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    # 2) 通道最大值
    max_map = feat_2d.max(dim=0).values.numpy()
    im1 = axes[0, 1].imshow(max_map, cmap="hot", aspect="auto")
    axes[0, 1].set_title("Channel Max")
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    # 3) 通道标准差
    std_map = feat_2d.std(dim=0).numpy()
    im2 = axes[0, 2].imshow(std_map, cmap="magma", aspect="auto")
    axes[0, 2].set_title("Channel Std")
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)

    # 4) PCA → RGB (前 3 主成分)
    try:
        from sklearn.decomposition import PCA
        feat_flat = feat_2d.reshape(C, -1).T.numpy()  # (H*W, C)
        n_components = min(3, C)
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(feat_flat)  # (H*W, 3)
        # 归一化到 [0, 1]
        for i in range(n_components):
            ch = pca_result[:, i]
            mn, mx = ch.min(), ch.max()
            if mx > mn:
                pca_result[:, i] = (ch - mn) / (mx - mn)
        if n_components < 3:
            pca_result = np.pad(pca_result, ((0, 0), (0, 3 - n_components)))
        pca_img = pca_result.reshape(H, W, 3)
        axes[1, 0].imshow(pca_img, aspect="auto")
        axes[1, 0].set_title(f"PCA RGB (var: {pca.explained_variance_ratio_.sum():.2%})")
    except Exception as e:
        axes[1, 0].text(0.5, 0.5, f"PCA failed:\n{e}", transform=axes[1, 0].transAxes,
                        ha="center", va="center")
        axes[1, 0].set_title("PCA RGB")

    # 5) 通道激活分布直方图
    vals = feat_2d.numpy().flatten()
    axes[1, 1].hist(vals, bins=100, color="steelblue", alpha=0.7, edgecolor="none")
    axes[1, 1].set_title(f"Activation Distribution\nmean={vals.mean():.3f}, std={vals.std():.3f}")
    axes[1, 1].set_xlabel("value")
    axes[1, 1].set_ylabel("count")

    # 6) 随机通道拼接 (4×4 grid)
    n_show = min(16, C)
    indices = np.linspace(0, C - 1, n_show, dtype=int)
    grid_rows, grid_cols = 4, 4
    tile_h, tile_w = H, W
    canvas = np.zeros((grid_rows * tile_h, grid_cols * tile_w))
    for idx_i, ch_idx in enumerate(indices):
        r, c = divmod(idx_i, grid_cols)
        if r >= grid_rows:
            break
        ch_data = feat_2d[ch_idx].numpy()
        ch_mn, ch_mx = ch_data.min(), ch_data.max()
        if ch_mx > ch_mn:
            ch_data = (ch_data - ch_mn) / (ch_mx - ch_mn)
        canvas[r * tile_h:(r + 1) * tile_h, c * tile_w:(c + 1) * tile_w] = ch_data
    axes[1, 2].imshow(canvas, cmap="viridis", aspect="auto")
    axes[1, 2].set_title(f"Sampled Channels ({n_show}/{C})")

    for ax in axes.flat:
        ax.tick_params(labelsize=7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → 已保存: {save_path}")


# =====================================================================
#  3D 占用可视化函数 — 等轴测体素渲染 + BEV 鸟瞰图
# =====================================================================

def _build_color_lut() -> np.ndarray:
    """构建 256 类颜色查找表, shape (256, 3), RGB 0-1."""
    lut = np.full((256, 3), 0.5)
    for cls_id, rgb in OCC_CLASS_COLORS.items():
        if cls_id < 256:
            lut[cls_id] = rgb
    return lut


_COLOR_LUT = _build_color_lut()


def _render_bev(ax, occ_pred: np.ndarray, title: str = "BEV"):
    """渲染干净的 BEV 鸟瞰图 — 每个 (x, y) 取最高非 free 类."""
    Dx, Dy, Dz = occ_pred.shape
    bev = np.full((Dx, Dy, 3), 1.0)  # 白色背景
    bev_filled = np.zeros((Dx, Dy), dtype=bool)

    for z_idx in range(Dz - 1, -1, -1):
        layer = occ_pred[:, :, z_idx]
        valid = (layer != 17) & (layer != 0) & ~bev_filled
        if not valid.any():
            continue
        for cls_id in np.unique(layer[valid]):
            mask = (layer == cls_id) & valid
            bev[mask] = OCC_CLASS_COLORS.get(int(cls_id), (0.5, 0.5, 0.5))
            bev_filled[mask] = True

    ax.imshow(bev, origin="lower", aspect="equal")
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=10)


def _render_isometric_voxels(
    ax,
    occ_pred: np.ndarray,
    voxel_step: int = 2,
    azim_deg: float = 45.0,
    elev_deg: float = 35.0,
    z_scale: float = 2.0,
    title: str = "3D Occupancy",
):
    """等轴测体素渲染 — 绘制实心方块, 仅渲染朝向相机的暴露面.

    使用 Painter's Algorithm (深度排序) 实现遮挡.
    面法向阴影模拟 3D 光照效果 (顶面最亮, 侧面渐暗).

    Args:
        ax: matplotlib Axes (普通 2D).
        occ_pred: (Dx, Dy, Dz) uint8 类别 ID.
        voxel_step: 下采样步长 (值越大渲染越快).
        azim_deg: 方位角 (绕 z 轴旋转, 度).
        elev_deg: 仰角 (度).
        z_scale: z 方向放大系数 (体素 z 维度通常远小于 xy, 放大便于查看).
        title: 子图标题.
    """
    from matplotlib.collections import PolyCollection

    occ = occ_pred[::voxel_step, ::voxel_step, ::voxel_step]
    Dx, Dy, Dz = occ.shape
    occupied = (occ != 17) & (occ != 0)

    if not occupied.any():
        ax.text(0.5, 0.5, "No occupied voxels", transform=ax.transAxes,
                ha="center", va="center", fontsize=14)
        ax.axis("off")
        return

    # 用 pad 检测暴露面 (邻居为空)
    padded = np.pad(occupied, 1, mode="constant", constant_values=False)

    # --- 等轴测投影参数 ---
    azim = np.radians(azim_deg)
    elev = np.radians(elev_deg)
    cos_a, sin_a = np.cos(azim), np.sin(azim)
    cos_e, sin_e = np.cos(elev), np.sin(elev)

    # 相机观察方向 (从相机指向场景) — 用于背面剔除和深度
    # 相机在 (+sin_a, +cos_a, +1) 方向 (归一化无关), 观察向原点
    # 面法线 dot 观察方向 < 0 表示面朝向相机 (可见)
    view_dir = np.array([sin_a * cos_e, cos_a * cos_e, -sin_e])

    # 只定义 6 个面方向, 实际只渲染朝向相机的 3 个面
    all_face_defs = [
        # (法线方向 dx,dy,dz, 顶点偏移 4×3, 阴影系数)
        ( 1, 0, 0, np.array([[1,0,0],[1,1,0],[1,1,1],[1,0,1]], dtype=np.float64), 0.80),
        (-1, 0, 0, np.array([[0,0,0],[0,0,1],[0,1,1],[0,1,0]], dtype=np.float64), 0.70),
        ( 0, 1, 0, np.array([[0,1,0],[0,1,1],[1,1,1],[1,1,0]], dtype=np.float64), 0.65),
        ( 0,-1, 0, np.array([[0,0,0],[1,0,0],[1,0,1],[0,0,1]], dtype=np.float64), 0.75),
        ( 0, 0, 1, np.array([[0,0,1],[1,0,1],[1,1,1],[0,1,1]], dtype=np.float64), 1.00),
        ( 0, 0,-1, np.array([[0,0,0],[0,1,0],[1,1,0],[1,0,0]], dtype=np.float64), 0.55),
    ]

    # 背面剔除: 仅保留面法线朝向相机的面 (dot < 0)
    visible_faces = []
    for dx, dy, dz, verts_tpl, shade in all_face_defs:
        normal = np.array([dx, dy, dz], dtype=float)
        if np.dot(normal, view_dir) < 0:
            visible_faces.append((dx, dy, dz, verts_tpl, shade))

    if not visible_faces:
        ax.text(0.5, 0.5, "No visible faces", transform=ax.transAxes,
                ha="center", va="center", fontsize=14)
        ax.axis("off")
        return

    # --- 提取暴露面, 投影到 2D ---
    all_polys = []
    all_fc = []
    all_depths = []

    for dx, dy, dz, verts_tpl, shade in visible_faces:
        neighbor = padded[1 + dx:Dx + 1 + dx,
                          1 + dy:Dy + 1 + dy,
                          1 + dz:Dz + 1 + dz]
        exposed = occupied & ~neighbor
        if not exposed.any():
            continue

        xi, yi, zi = np.where(exposed)
        n = len(xi)

        # 3D 顶点 (n, 4, 3)
        origins = np.stack([xi, yi, zi], axis=-1).astype(np.float64)
        verts_3d = origins[:, None, :] + verts_tpl[None, :, :]

        # z 放大
        verts_3d[..., 2] *= z_scale

        # 等轴测投影到 2D
        vx, vy, vz = verts_3d[..., 0], verts_3d[..., 1], verts_3d[..., 2]
        xr = vx * cos_a - vy * sin_a
        yr = vx * sin_a + vy * cos_a
        px = xr
        py = yr * sin_e + vz * cos_e
        verts_2d = np.stack([px, py], axis=-1)  # (n, 4, 2)

        # 深度 (越大越靠近相机, 后绘制)
        centers = origins.astype(np.float64)
        centers[:, 2] *= z_scale
        cx, cy, cz = centers[:, 0], centers[:, 1], centers[:, 2]
        depths = (cx * sin_a + cy * cos_a) * cos_e + cz * sin_e

        # 颜色: 查表 + 阴影
        cls_ids = occ[xi, yi, zi]
        fc = np.ones((n, 4))
        fc[:, :3] = np.clip(_COLOR_LUT[cls_ids] * shade, 0, 1)

        all_polys.append(verts_2d)
        all_fc.append(fc)
        all_depths.append(depths)

    if not all_polys:
        ax.text(0.5, 0.5, "No exposed faces", transform=ax.transAxes,
                ha="center", va="center", fontsize=14)
        ax.axis("off")
        return

    polys = np.concatenate(all_polys)       # (N, 4, 2)
    facecolors = np.concatenate(all_fc)     # (N, 4)
    depths = np.concatenate(all_depths)     # (N,)

    # Painter's Algorithm: 先绘远处 (depth 小), 后绘近处 (depth 大)
    order = np.argsort(depths)
    polys = polys[order]
    facecolors = facecolors[order]

    # 边缘颜色: 略暗半透明
    edgecolors = facecolors.copy()
    edgecolors[:, :3] = np.clip(edgecolors[:, :3] * 0.5, 0, 1)
    edgecolors[:, 3] = 0.3

    ax.set_facecolor("white")
    pc = PolyCollection(polys, closed=True)
    pc.set_facecolor(facecolors)
    pc.set_edgecolor(edgecolors)
    pc.set_linewidth(0.15)
    ax.add_collection(pc)
    ax.autoscale()
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=10)


def visualize_occ_3d(
    occ_pred: np.ndarray,
    save_path: str,
    voxel_step: int = 1,
    title: str = "3D Occupancy Prediction",
):
    """可视化 3D 占用预测 — 左: 等轴测体素渲染, 右: BEV 鸟瞰图.

    Args:
        occ_pred: (Dx, Dy, Dz) uint8, 值为类别 ID (0~17).
        save_path: 保存路径.
        voxel_step: 体素下采样步长 (加速渲染).
        title: 图标题.
    """
    fig = plt.figure(figsize=(28, 12))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.3, 1], wspace=0.05)

    # --- 左图: 3D 等轴测体素 ---
    ax_3d = fig.add_subplot(gs[0])
    _render_isometric_voxels(
        ax_3d, occ_pred,
        voxel_step=max(voxel_step, 1),
        azim_deg=45.0, elev_deg=35.0, z_scale=2.0,
        title="3D Isometric View",
    )

    # --- 右图: BEV 鸟瞰图 ---
    ax_bev = fig.add_subplot(gs[1])
    _render_bev(ax_bev, occ_pred, title="Bird's Eye View (BEV)")

    # --- 图例 ---
    present = occ_pred[(occ_pred != 17) & (occ_pred != 0)]
    if present.size > 0:
        legend_patches = []
        for cls_id in sorted(set(present.tolist())):
            color = OCC_CLASS_COLORS.get(cls_id, (0.5, 0.5, 0.5))
            name = OCC_CLASS_NAMES[cls_id] if cls_id < len(OCC_CLASS_NAMES) else f"class_{cls_id}"
            count = (occ_pred == cls_id).sum()
            legend_patches.append(
                Patch(facecolor=color, edgecolor="k", linewidth=0.5,
                      label=f"{name} ({count:,})")
            )
        fig.legend(
            handles=legend_patches, loc="lower center",
            ncol=min(9, len(legend_patches)), fontsize=8,
            frameon=True, fancybox=True, shadow=True,
            bbox_to_anchor=(0.5, -0.01),
        )

    fig.suptitle(title, fontsize=18, fontweight="bold", y=0.98)
    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  → 已保存: {save_path}")


# =====================================================================
#  输入图像可视化
# =====================================================================

def visualize_input_images(img_inputs: tuple, save_path: str):
    """可视化 6 个相机输入图像.

    img_inputs[0] 经 DALI 解码后为 (B, N, 3, H, W) float32 BGR normalized:
        pixel = (BGR - IMAGENET_MEAN) / IMAGENET_STD
    反归一化: pixel * std + mean → BGR [0,255] → RGB → [0,1]
    """
    imgs = img_inputs[0]  # (B, N, C, H, W)
    if isinstance(imgs, torch.Tensor):
        imgs = imgs[0].cpu().float()  # (N, C, H, W)
    N = imgs.shape[0]

    cam_names = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
                 "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    fig.suptitle("Input Camera Images (after augmentation)", fontsize=14, fontweight="bold")

    from flashocc.constants import IMAGENET_MEAN, IMAGENET_STD

    for i in range(min(N, 6)):
        img = imgs[i].permute(1, 2, 0).numpy()  # (H, W, 3) — BGR normalized
        # 反归一化: img_norm = (img_bgr - mean) / std  →  img_bgr = img_norm * std + mean
        img = img * IMAGENET_STD.reshape(1, 1, 3) + IMAGENET_MEAN.reshape(1, 1, 3)
        # BGR → RGB
        img = img[:, :, ::-1]
        img = np.clip(img / 255.0, 0, 1)

        r, c = divmod(i, 3)
        axes[r, c].imshow(img)
        name = cam_names[i] if i < len(cam_names) else f"cam_{i}"
        axes[r, c].set_title(name, fontsize=10)
        axes[r, c].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → 已保存: {save_path}")


# =====================================================================
#  OCC GT vs Pred 对比
# =====================================================================

def visualize_occ_gt_vs_pred(
    occ_gt: np.ndarray | None,
    occ_pred: np.ndarray,
    save_path: str,
):
    """GT 和预测对比 — 上方 3D 等轴测, 下方 BEV 鸟瞰图."""
    n_cols = 2 if occ_gt is not None else 1

    fig = plt.figure(figsize=(14 * n_cols, 22))
    gs = fig.add_gridspec(2, n_cols, hspace=0.08, wspace=0.05)

    occ_list = []
    titles = []
    if occ_gt is not None:
        occ_list.append(occ_gt)
        titles.append("Ground Truth")
    occ_list.append(occ_pred)
    titles.append("Prediction")

    for col, (occ, tag) in enumerate(zip(occ_list, titles)):
        # 上: 3D 等轴测
        ax_3d = fig.add_subplot(gs[0, col])
        _render_isometric_voxels(
            ax_3d, occ, voxel_step=2,
            azim_deg=45.0, elev_deg=35.0, z_scale=2.0,
            title=f"{tag} — 3D",
        )
        # 下: BEV
        ax_bev = fig.add_subplot(gs[1, col])
        _render_bev(ax_bev, occ, title=f"{tag} — BEV")

    # 图例
    all_classes: set[int] = set()
    for occ in occ_list:
        all_classes.update(occ[(occ != 17) & (occ != 0)].tolist())
    legend_patches = []
    for cls_id in sorted(all_classes):
        color = OCC_CLASS_COLORS.get(cls_id, (0.5, 0.5, 0.5))
        name = OCC_CLASS_NAMES[cls_id] if cls_id < len(OCC_CLASS_NAMES) else f"class_{cls_id}"
        legend_patches.append(Patch(facecolor=color, edgecolor="k", linewidth=0.5, label=name))

    if legend_patches:
        fig.legend(handles=legend_patches, loc="lower center",
                   ncol=min(9, len(legend_patches)), fontsize=9,
                   frameon=True, fancybox=True, shadow=True,
                   bbox_to_anchor=(0.5, -0.01))

    fig.suptitle("Occupancy: GT vs Prediction", fontsize=18, fontweight="bold", y=0.99)
    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  → 已保存: {save_path}")


# =====================================================================
#  类别统计
# =====================================================================

def print_occ_stats(occ_pred: np.ndarray, tag: str = "Prediction"):
    """打印占用预测的类别统计."""
    total = occ_pred.size
    print(f"\n{'=' * 50}")
    print(f"  {tag} — 类别统计 ({occ_pred.shape})")
    print(f"{'=' * 50}")
    print(f"  {'Class':<25s} {'Count':>10s} {'Ratio':>8s}")
    print(f"  {'-' * 43}")
    for cls_id in range(len(OCC_CLASS_NAMES)):
        count = (occ_pred == cls_id).sum()
        if count > 0:
            ratio = count / total * 100
            name = OCC_CLASS_NAMES[cls_id]
            print(f"  {name:<25s} {count:>10,d} {ratio:>7.2f}%")
    print(f"{'=' * 50}\n")


# =====================================================================
#  主流程
# =====================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="FlashOCC 特征 & 3D 占用可视化工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("config", help="Python 配置文件路径 (.py)")
    parser.add_argument("checkpoint", help="checkpoint 文件路径")
    parser.add_argument("--sample-idx", type=int, default=0,
                        help="可视化的样本索引 (默认 0)")
    parser.add_argument("--out-dir", default="vis_output",
                        help="输出目录 (默认 vis_output)")
    parser.add_argument("--device", default="cuda:0",
                        help="推理设备 (默认 cuda:0)")
    parser.add_argument("--no-feat", action="store_true",
                        help="跳过中间特征可视化")
    parser.add_argument("--no-occ", action="store_true",
                        help="跳过 3D 占用可视化")
    parser.add_argument("--voxel-step", type=int, default=1,
                        help="3D 可视化体素下采样步长 (默认 1)")
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)

    print("=" * 60)
    print("  FlashOCC 可视化工具")
    print("=" * 60)

    # ---------- 1. 加载实验配置 ----------
    print(f"\n[1/5] 加载配置: {args.config}")
    exp = load_experiment(args.config)

    # ---------- 2. 构建模型 ----------
    print(f"[2/5] 构建模型并加载权重: {args.checkpoint}")
    model = exp.build_model()
    load_checkpoint(model, args.checkpoint, map_location="cpu")
    model = model.to(device)
    model.eval()

    # ---------- 3. 注册特征 hook ----------
    extractor = FeatureExtractor()
    if not args.no_feat:
        print("[3/5] 注册特征提取 Hook ...")
        extractor.register(model)

    # ---------- 4. 获取数据样本 ----------
    print(f"[4/5] 加载验证集, 取第 {args.sample_idx} 个样本 ...")
    dataset = exp.build_test_dataset()
    print(f"       验证集大小: {len(dataset)}")

    if args.sample_idx >= len(dataset):
        print(f"  [错误] sample_idx={args.sample_idx} 超出范围 (max={len(dataset) - 1})")
        return

    sample = dataset[args.sample_idx]

    # DataLoader collate — 模拟 batch
    data_loader = build_dataloader(
        dataset, samples_per_gpu=1,
        workers_per_gpu=0, dist_mode=False, shuffle=False,
    )
    # 直接用迭代器取指定样本
    for i, batch in enumerate(data_loader):
        if i == args.sample_idx:
            break
    else:
        print(f"  [错误] 无法获取样本 {args.sample_idx}")
        return

    # ---------- 5. 前向推理 ----------
    print("[5/5] 执行前向推理 ...")

    # ---- DALI GPU 解码: jpeg_bytes → 实际图像 ----
    from flashocc.datasets.dali_decode import dali_decode_batch
    if 'jpeg_bytes' in batch:
        # DataContainer 解包 — DataLoader collate 后 DC 包装, dali_decode_batch 需要原始 list
        for key in ('jpeg_bytes', 'img_aug_params'):
            if key in batch and hasattr(batch[key], 'data'):
                batch[key] = batch[key].data
        batch = dali_decode_batch(batch)

    # 将数据移到 GPU
    img_inputs = batch["img_inputs"]
    if isinstance(img_inputs, (list, tuple)):
        img_inputs_gpu = [t.to(device) if isinstance(t, torch.Tensor) else t
                          for t in img_inputs]
    else:
        img_inputs_gpu = img_inputs.to(device)

    img_metas = batch.get("img_metas", [{}])
    if hasattr(img_metas, "data"):
        img_metas = img_metas.data[0]
    # forward_test 期望 img_metas 是 list[dict], 如果解包后是单个 dict 则包装
    if isinstance(img_metas, dict):
        img_metas = [img_metas]

    points = batch.get("points", None)
    if points is not None:
        if hasattr(points, "data"):
            points = points.data[0]
        if isinstance(points, (list, tuple)):
            points = [p.to(device) if isinstance(p, torch.Tensor) else p
                      for p in points]
        elif isinstance(points, torch.Tensor):
            points = points.to(device)

    # 调用模型的 forward_test
    results = model.forward_test(
        points=points,
        img_inputs=img_inputs_gpu,
        img_metas=img_metas,
    )

    # ============================================================
    #  可视化输入图像
    # ============================================================
    print("\n--- 可视化输入图像 ---")
    try:
        # 使用 DALI 解码后的真实图像 (img_inputs_gpu) 而非原始全零占位
        vis_inputs = img_inputs_gpu if isinstance(img_inputs_gpu, (list, tuple)) else (img_inputs_gpu,)
        visualize_input_images(
            vis_inputs,
            os.path.join(args.out_dir, "00_input_cameras.png"),
        )
    except Exception as e:
        print(f"  [跳过] 输入图像可视化失败: {e}")

    # ============================================================
    #  可视化中间层特征
    # ============================================================
    if not args.no_feat:
        print("\n--- 可视化中间层特征 ---")
        for idx, (name, feat) in enumerate(extractor.features.items()):
            safe_name = name.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")
            fname = f"{idx + 1:02d}_{safe_name}.png"
            save_path = os.path.join(args.out_dir, fname)
            print(f"  [{idx + 1}/{len(extractor.features)}] {name}: {list(feat.shape)}")
            try:
                visualize_feature_map(feat, name, save_path)
            except Exception as e:
                print(f"    [跳过] 可视化失败: {e}")
        extractor.remove()

    # ============================================================
    #  可视化 3D 占用预测
    # ============================================================
    if not args.no_occ:
        print("\n--- 可视化 3D 占用预测 ---")

        # 提取 OCC 预测
        occ_pred = None
        if isinstance(results, (list, tuple)) and len(results) > 0:
            r = results[0]
            if isinstance(r, np.ndarray):
                occ_pred = r
            elif isinstance(r, torch.Tensor):
                occ_pred = r.cpu().numpy()
            elif isinstance(r, dict):
                # 尝试从 dict 中取
                for key in ["occ", "occ_pred", "occ_preds", "pts_bbox"]:
                    if key in r:
                        val = r[key]
                        if isinstance(val, torch.Tensor):
                            occ_pred = val.cpu().numpy()
                        elif isinstance(val, np.ndarray):
                            occ_pred = val
                        break

        if occ_pred is None:
            print("  [错误] 无法从模型输出中提取占用预测")
            print(f"  模型输出类型: {type(results)}")
            if isinstance(results, (list, tuple)):
                for i, r in enumerate(results):
                    print(f"    [{i}] type={type(r)}, "
                          + (f"shape={r.shape}" if hasattr(r, "shape") else
                             f"keys={list(r.keys())}" if hasattr(r, "keys") else str(r)[:100]))
            return

        occ_pred = occ_pred.astype(np.uint8)
        print(f"  OCC 预测形状: {occ_pred.shape}")
        print_occ_stats(occ_pred, "Prediction")

        # 3D 可视化
        visualize_occ_3d(
            occ_pred,
            os.path.join(args.out_dir, "occ_3d_prediction.png"),
            voxel_step=args.voxel_step,
            title="3D Occupancy Prediction",
        )

        # GT vs Pred 对比
        occ_gt = None
        if "voxel_semantics" in batch:
            gt_data = batch["voxel_semantics"]
            if hasattr(gt_data, "data"):
                gt_data = gt_data.data[0]
            if isinstance(gt_data, torch.Tensor):
                occ_gt = gt_data[0].cpu().numpy().astype(np.uint8)
            elif isinstance(gt_data, list) and len(gt_data) > 0:
                if isinstance(gt_data[0], torch.Tensor):
                    occ_gt = gt_data[0].cpu().numpy().astype(np.uint8)

        if occ_gt is not None:
            print(f"  OCC GT 形状: {occ_gt.shape}")
            print_occ_stats(occ_gt, "Ground Truth")

            visualize_occ_3d(
                occ_gt,
                os.path.join(args.out_dir, "occ_3d_groundtruth.png"),
                voxel_step=args.voxel_step,
                title="3D Occupancy Ground Truth",
            )

        visualize_occ_gt_vs_pred(
            occ_gt, occ_pred,
            os.path.join(args.out_dir, "occ_bev_gt_vs_pred.png"),
        )

    # ============================================================
    #  完成
    # ============================================================
    print("\n" + "=" * 60)
    print(f"  可视化完成! 所有结果保存在: {args.out_dir}/")
    print("=" * 60)

    # 列出所有生成的文件
    for f in sorted(os.listdir(args.out_dir)):
        fpath = os.path.join(args.out_dir, f)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"    {f:<45s} {size_kb:>8.1f} KB")


if __name__ == "__main__":
    main()
