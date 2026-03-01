#!/usr/bin/env python
"""FlashOCC 特征可视化 + 3D 占用预测可视化工具 (重新设计版).

架构说明:
    输入图像:  (B, N=6, 3, H=256, W=704) — 6-摄像头环视
    Backbone:  (B*N, C, fH, fW) — 6相机拼入batch后提特征
    Neck/FPN:  (B*N, 256, fH, fW) — 每个相机独立输出
    ViewTrans: (B, 64, Dy=200, Dx=200) — 投影到BEV平面
    BEV Enc:   (B, 256, Dy=200, Dx=200) — BEV特征增强
    OCC Head:  (B, Dx=200, Dy=200, Dz=16, n_cls=18) — 语义占用预测
               Dx=200: X方向 (-40m→+40m, 0.4m/格, 车辆前后方向)
               Dy=200: Y方向 (-40m→+40m, 0.4m/格, 车辆左右方向)
               Dz=16:  Z方向 (-1m→+5.4m, 0.4m/格, 高度方向)
               n_cls=18: 语义类别概率 (softmax前的logits)

NuScenes 相机布局 (nuScenes ego坐标系: x=前, y=左, z=上):
    CAM_FRONT:       朝向  0°, HFoV ~70°
    CAM_FRONT_LEFT:  朝向 +55°, HFoV ~70°
    CAM_FRONT_RIGHT: 朝向 -55°, HFoV ~70°
    CAM_BACK_LEFT:   朝向+110°, HFoV ~70°
    CAM_BACK_RIGHT:  朝向-110°, HFoV ~70°
    CAM_BACK:        朝向±180°, HFoV~110°

自车: 雷诺Zoé (Renault Zoé) — 4.084m × 1.73m

用法:
    python tools/visualize_features.py configs/flashocc_convnext_tiny_dinov3.py \\
        work_dirs/flashocc_convnext_tiny_dinov3/epoch_12.pth \\
        --sample-idx 0 --out-dir vis_output

可选参数:
    --sample-idx   样本索引 (默认 0)
    --out-dir      输出目录 (默认 vis_output)
    --device       推理设备 (默认 cuda:0)
    --no-feat      跳过中间特征可视化
    --no-occ       跳过 3D 占用可视化
    --voxel-step   体素下采样步长 (默认 1, 越大越快)
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
import matplotlib.patches as mpatches
from matplotlib.patches import Patch, Rectangle
from matplotlib.collections import PolyCollection
from matplotlib.gridspec import GridSpec

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
#  全局常量
# =====================================================================

_PCR = POINT_CLOUD_RANGE    # [-40, -40, -1, 40, 40, 5.4]
_VS  = VOXEL_SIZE            # 0.4

# OCC格网维度
_DX = int((_PCR[3] - _PCR[0]) / _VS)  # 200
_DY = int((_PCR[4] - _PCR[1]) / _VS)  # 200
_DZ = int((_PCR[5] - _PCR[2]) / _VS)  # 16

# 雷诺 Zoé 尺寸 (m)
_ZOE_LENGTH = 4.084   # 车身总长 (x方向 ±2.042m)
_ZOE_WIDTH  = 1.730   # 车身总宽 (y方向 ±0.865m)
_ZOE_WB     = 2.588   # 轴距

# NuScenes 6相机布局
# 角度约定: 从+x轴(前方)逆时针为正 (+y=左=正角度)
_CAM_ORDER = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
              "CAM_BACK_LEFT",  "CAM_BACK",  "CAM_BACK_RIGHT"]

_CAM_INFO = {
    "CAM_FRONT_LEFT":  dict(heading=+55.0,  hfov=70.0,  range_m=50.0, color="#FF6B6B"),
    "CAM_FRONT":       dict(heading=  0.0,  hfov=70.0,  range_m=50.0, color="#FFD93D"),
    "CAM_FRONT_RIGHT": dict(heading=-55.0,  hfov=70.0,  range_m=50.0, color="#6BCB77"),
    "CAM_BACK_LEFT":   dict(heading=+110.0, hfov=70.0,  range_m=50.0, color="#4D96FF"),
    "CAM_BACK":        dict(heading= 180.0, hfov=110.0, range_m=50.0, color="#C77DFF"),
    "CAM_BACK_RIGHT":  dict(heading=-110.0, hfov=70.0,  range_m=50.0, color="#FF9F1C"),
}

# OCC 18类颜色表 (RGB 0-1)
OCC_CLASS_COLORS: dict[int, tuple] = {
    0:  (0.00, 0.00, 0.00),  # others
    1:  (1.00, 0.73, 0.47),  # barrier
    2:  (1.00, 0.83, 0.00),  # bicycle
    3:  (0.00, 0.00, 0.90),  # bus
    4:  (1.00, 0.00, 0.00),  # car
    5:  (0.55, 0.27, 0.07),  # construction_vehicle
    6:  (0.00, 0.00, 0.55),  # motorcycle
    7:  (0.00, 0.75, 1.00),  # pedestrian
    8:  (1.00, 0.40, 0.00),  # traffic_cone
    9:  (0.50, 0.00, 0.50),  # trailer
    10: (0.65, 0.16, 0.16),  # truck
    11: (0.60, 0.60, 0.60),  # driveable_surface
    12: (0.75, 0.75, 0.75),  # other_flat
    13: (0.85, 0.55, 0.85),  # sidewalk
    14: (0.40, 0.70, 0.40),  # terrain
    15: (0.70, 0.70, 0.90),  # manmade
    16: (0.13, 0.55, 0.13),  # vegetation
    17: (1.00, 1.00, 1.00),  # free
}


def _build_color_lut() -> np.ndarray:
    lut = np.full((256, 3), 0.5)
    for cls_id, rgb in OCC_CLASS_COLORS.items():
        if cls_id < 256:
            lut[cls_id] = rgb
    return lut

_COLOR_LUT = _build_color_lut()


# =====================================================================
#  BEV坐标系约定
#
#  occ_pred[i, j, k]  →  世界坐标:
#    x = _PCR[0] + i * _VS = -40 + i * 0.4   (前后, i↑=前方↑)
#    y = _PCR[1] + j * _VS = -40 + j * 0.4   (左右, j↑=左方↑)
#    z = _PCR[2] + k * _VS = -1  + k * 0.4   (高度)
#
#  imshow(occ_bev, origin="lower") 的 (Dx, Dy) 图:
#    水平轴 col = j  →  世界y方向 (j=0在左, j=Dy-1在右)
#    垂直轴 row = i  →  世界x方向 (origin=lower: i=0在底=后方, i=Dx-1在顶=前方)
#    自车: (row=100, col=100), BEV中心
#    前方=图像上方  |  左方=图像右方 (nuScenes BEV惯例)
# =====================================================================

def world_to_bev_px(x: float, y: float):
    """世界坐标(x,y) → BEV像素(col=j, row=i)."""
    return (y - _PCR[1]) / _VS, (x - _PCR[0]) / _VS   # col, row


def draw_ego_vehicle(ax, linewidth=2.0, color="#1E90FF", zorder=10):
    """绘制雷诺Zoé自车轮廓 + 前进方向箭头."""
    cx, cr = world_to_bev_px(0, 0)
    corners_w = [
        (-_ZOE_LENGTH/2, -_ZOE_WIDTH/2),
        (+_ZOE_LENGTH/2, -_ZOE_WIDTH/2),
        (+_ZOE_LENGTH/2, +_ZOE_WIDTH/2),
        (-_ZOE_LENGTH/2, +_ZOE_WIDTH/2),
        (-_ZOE_LENGTH/2, -_ZOE_WIDTH/2),
    ]
    cols = [(y - _PCR[1]) / _VS for (x, y) in corners_w]
    rows = [(x - _PCR[0]) / _VS for (x, y) in corners_w]
    ax.fill(cols, rows, color=color, alpha=0.30, zorder=zorder)
    ax.plot(cols, rows, color=color, linewidth=linewidth, zorder=zorder)
    # 前进方向箭头 (+x = 向上 in BEV)
    arrow_end = cr + (_ZOE_LENGTH / 2 + 2.5) / _VS
    ax.annotate("", xy=(cx, arrow_end), xytext=(cx, cr),
                arrowprops=dict(arrowstyle="->", color=color, lw=2.0),
                zorder=zorder + 1)
    # 轴距虚线
    wb_px = _ZOE_WB / _VS
    ax.plot([cx, cx], [cr - wb_px/2, cr + wb_px/2],
            color=color, linewidth=1.0, linestyle=":", alpha=0.7, zorder=zorder)
    # 车型标注
    ax.text(cx + 2, cr, "Renault Zoé\n4.08m×1.73m",
            fontsize=6, color=color, va="center",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color,
                      alpha=0.75, linewidth=0.5))


def draw_camera_fovs(ax, alpha_fill=0.07, alpha_line=0.65, zorder=5):
    """绘制6相机视场角扇形辅助线."""
    ego_col, ego_row = world_to_bev_px(0, 0)
    legend_patches = []

    for cam_name in _CAM_ORDER:
        info = _CAM_INFO[cam_name]
        heading_rad  = np.radians(info["heading"])
        half_fov_rad = np.radians(info["hfov"] / 2.0)
        color = info["color"]

        n_pts = 40
        angles = np.linspace(heading_rad - half_fov_rad,
                             heading_rad + half_fov_rad, n_pts)
        wx = np.cos(angles) * info["range_m"]
        wy = np.sin(angles) * info["range_m"]
        arc_cols = (wy - _PCR[1]) / _VS
        arc_rows = (wx - _PCR[0]) / _VS

        fan_cols = np.concatenate([[ego_col], arc_cols, [ego_col]])
        fan_rows = np.concatenate([[ego_row], arc_rows, [ego_row]])

        ax.fill(fan_cols, fan_rows, color=color, alpha=alpha_fill, zorder=zorder)
        ax.plot(fan_cols, fan_rows, color=color, linewidth=1.0,
                alpha=alpha_line, zorder=zorder)

        # 相机名标签 (6成处)
        f_col = arc_cols[n_pts//2] * 0.6 + ego_col * 0.4
        f_row = arc_rows[n_pts//2] * 0.6 + ego_row * 0.4
        short = cam_name.replace("CAM_", "").replace("_", "\n")
        ax.text(f_col, f_row, short, fontsize=5.5, ha="center", va="center",
                color=color, fontweight="bold", zorder=zorder + 1,
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec=color,
                          alpha=0.75, linewidth=0.5))
        legend_patches.append(
            Patch(facecolor=color, alpha=0.6, edgecolor=color,
                  label=f"{cam_name.replace('CAM_','').replace('_',' ')} "
                        f"(hdg={info['heading']:+.0f}°, HFoV={info['hfov']:.0f}°)"))

    return legend_patches


def add_bev_annotations(ax, title="", fontsize=13, tick_m=10.0):
    """BEV轴: 刻度、网格、方位文字."""
    ticks_m = np.arange(-40, 41, tick_m)
    ax.set_xticks([(t - _PCR[1]) / _VS for t in ticks_m])
    ax.set_xticklabels([f"{int(t)}" for t in ticks_m], fontsize=6)
    ax.set_yticks([(t - _PCR[0]) / _VS for t in ticks_m])
    ax.set_yticklabels([f"{int(t)}" for t in ticks_m], fontsize=6)
    ax.set_xlabel("Y / m  (right ← | → left )", fontsize=8)
    ax.set_ylabel("X / m  (rear  ↓ | ↑  front)", fontsize=8)
    for t in np.arange(-40, 41, tick_m):
        ax.axvline((t - _PCR[1]) / _VS, color="gray", lw=0.3, alpha=0.4)
        ax.axhline((t - _PCR[0]) / _VS, color="gray", lw=0.3, alpha=0.4)
    cx, cy = world_to_bev_px(0, 0)
    kw = dict(ha="center", fontsize=7, color="white", fontweight="bold",
              bbox=dict(boxstyle="round,pad=0.2", fc="#333", alpha=0.7))
    ax.text(cx, _DX - 2, "FRONT ▲", va="top",   **kw)
    ax.text(cx,        2, "REAR  ▼", va="bottom", **kw)
    if title:
        ax.set_title(title, fontsize=fontsize, fontweight="bold", pad=8)


# =====================================================================
#  特征图 Hook
#  layer_type: "multicam" | "bev" | "occ_head"
# =====================================================================

class FeatureExtractor:
    """注册 forward hook, 自动捕获各层输出并标记层类型."""

    LAYER_REGISTRY = [
        ("img_backbone",             "multicam",  "1. Image Backbone"),
        ("img_neck",                 "multicam",  "2. Image Neck (FPN)"),
        ("img_view_transformer",     "bev",       "3. View Transformer (LSS)"),
        ("img_bev_encoder_backbone", "bev",       "4. BEV Encoder Backbone"),
        ("img_bev_encoder_neck",     "bev",       "5. BEV Encoder Neck"),
        ("occ_head",                 "occ_head",  "6. OCC Head"),
    ]

    def __init__(self):
        self.features: OrderedDict[str, tuple] = OrderedDict()
        self._hooks: list[Any] = []

    def register(self, model: nn.Module):
        for attr_name, layer_type, display_name in self.LAYER_REGISTRY:
            module = getattr(model, attr_name, None)
            if module is None:
                continue
            hook = module.register_forward_hook(
                self._make_hook(display_name, layer_type))
            self._hooks.append(hook)

    def _make_hook(self, name: str, layer_type: str):
        def hook_fn(module, input_t, output):
            tensor = None
            if isinstance(output, torch.Tensor):
                tensor = output.detach().cpu()
            elif isinstance(output, (list, tuple)):
                for o in output:
                    if isinstance(o, torch.Tensor):
                        tensor = o.detach().cpu(); break
            if tensor is not None:
                self.features[name] = (layer_type, tensor)
        return hook_fn

    def remove(self):
        for h in self._hooks: h.remove()
        self._hooks.clear()


# =====================================================================
#  (1) 多相机层可视化  —  backbone / neck
#      Hook捕获: (B*N=6, C, fH, fW)
# =====================================================================

def visualize_multicam_features(feat: torch.Tensor, title: str, save_path: str,
                                 n_cams: int = 6):
    """6相机特征图可视化: 2×3网格(每相机通道均值热图) + 跨相机统计.

    backbone/neck的hook输出为 (B*N, C, fH, fW), N=6相机拼到batch维度.
    """
    if feat.dim() > 4:
        feat = feat.reshape(-1, *feat.shape[-3:])

    total_batch = feat.shape[0]
    if total_batch % n_cams == 0:
        feat = feat[:n_cams]   # 取第一个batch的6相机
    else:
        visualize_feature_map_bev(feat[0] if feat.dim() == 4 else feat,
                                  title, save_path)
        return

    C, fH, fW = feat.shape[1], feat.shape[2], feat.shape[3]
    feat_f = feat.float()
    cam_colors = [_CAM_INFO[c]["color"] for c in _CAM_ORDER]

    fig = plt.figure(figsize=(22, 18))
    fig.suptitle(
        f"{title}\n"
        f"Input: {n_cams}×(C={C}, fH={fH}, fW={fW})  "
        f"[6 cameras merged into batch dim]\n"
        f"Camera order: FL / F / FR / BL / B / BR",
        fontsize=13, fontweight="bold")

    gs_cams  = GridSpec(2, 3, figure=fig, top=0.89, bottom=0.45,
                        hspace=0.42, wspace=0.20)
    gs_stats = GridSpec(1, 3, figure=fig, top=0.40, bottom=0.24,
                        hspace=0.05, wspace=0.20)
    gs_pca   = GridSpec(1, 2, figure=fig, top=0.20, bottom=0.04,
                        hspace=0.05, wspace=0.20)

    vmin = float(feat_f.mean(dim=(1,2,3)).min())
    vmax = float(feat_f.mean(dim=1).max())

    # ---------- 上部 2×3: 每相机通道均值热力图 ----------
    for idx in range(n_cams):
        rg, cg = divmod(idx, 3)
        ax = fig.add_subplot(gs_cams[rg, cg])
        mean_map = feat_f[idx].mean(0).numpy()
        im = ax.imshow(mean_map, cmap="viridis", aspect="auto",
                       vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cam_name = _CAM_ORDER[idx]
        info = _CAM_INFO[cam_name]
        ax.set_title(
            f"{cam_name.replace('CAM_','')}\n"
            f"heading={info['heading']:+.0f}°  HFoV={info['hfov']:.0f}°",
            fontsize=9, fontweight="bold", color=cam_colors[idx])
        ax.tick_params(labelsize=6)
        ax.set_xlabel("fW", fontsize=7); ax.set_ylabel("fH", fontsize=7)
        # 最大激活轮廓线
        max_map = feat_f[idx].max(0).values.numpy()
        try:
            perc = np.percentile(max_map[max_map > max_map.mean()], [50, 90])
            ax.contour(max_map, levels=perc,
                       colors=["white", "red"], linewidths=0.6, alpha=0.6)
        except Exception:
            pass

    # ---------- 中部: 统计 ----------
    mean_per_cam = feat_f.mean(dim=(1,2,3)).numpy()
    std_per_cam  = feat_f.std(dim=(1,2,3)).numpy()

    ax_bar = fig.add_subplot(gs_stats[0, 0])
    ax_bar.bar(range(n_cams), mean_per_cam, color=cam_colors, alpha=0.85,
               yerr=std_per_cam, capsize=5, ecolor="gray")
    ax_bar.set_xticks(range(n_cams))
    ax_bar.set_xticklabels(
        [c.replace("CAM_","").replace("_","\n") for c in _CAM_ORDER],
        fontsize=8)
    ax_bar.set_title("Per-Camera Mean Activation ± Std", fontsize=9)
    ax_bar.set_ylabel("Channel Mean", fontsize=8)
    _ylo = float((mean_per_cam - std_per_cam).min())
    _yhi = float((mean_per_cam + std_per_cam).max())
    _ym  = max((_yhi - _ylo) * 0.12, 1e-6)
    ax_bar.set_ylim(_ylo - _ym, _yhi + _ym)

    ax_var = fig.add_subplot(gs_stats[0, 1])
    cross_std = feat_f.mean(1).std(0).numpy()
    im2 = ax_var.imshow(cross_std, cmap="hot", aspect="auto")
    plt.colorbar(im2, ax=ax_var, fraction=0.046, pad=0.04)
    ax_var.set_title("Cross-Camera Feature Variance\n(std of 6 camera mean-maps)",
                     fontsize=9)
    ax_var.tick_params(labelsize=7)
    ax_var.set_xlabel("fW", fontsize=7); ax_var.set_ylabel("fH", fontsize=7)

    ax_hist = fig.add_subplot(gs_stats[0, 2])
    for idx in range(n_cams):
        v = feat_f[idx].numpy().flatten()
        ax_hist.hist(v, bins=60, alpha=0.38, color=cam_colors[idx],
                     label=_CAM_ORDER[idx].replace("CAM_",""),
                     density=True, histtype="stepfilled")
    ax_hist.set_title("Activation Distribution (6 cameras overlay)", fontsize=9)
    ax_hist.set_xlabel("value"); ax_hist.legend(fontsize=6, ncol=2)

    # ---------- 下部: PCA RGB拼图 ----------
    try:
        from sklearn.decomposition import PCA
        flat = feat_f.permute(0,2,3,1).reshape(-1, C).numpy()
        pca = PCA(n_components=min(3, C))
        proj = pca.fit_transform(flat)
        for ch in range(proj.shape[1]):
            mn, mx = proj[:,ch].min(), proj[:,ch].max()
            if mx > mn: proj[:,ch] = (proj[:,ch]-mn)/(mx-mn)
        if proj.shape[1] < 3:
            proj = np.pad(proj, ((0,0),(0,3-proj.shape[1])))
        proj_imgs = proj.reshape(n_cams, fH, fW, 3)
        canvas = np.ones((2*fH, 3*fW, 3))
        for idx in range(n_cams):
            rr, cc = divmod(idx, 3)
            canvas[rr*fH:(rr+1)*fH, cc*fW:(cc+1)*fW] = proj_imgs[idx]
        ax_pca = fig.add_subplot(gs_pca[0, 0])
        ax_pca.imshow(canvas, aspect="auto")
        ax_pca.set_title(
            f"PCA RGB (6-camera tiled 2×3)  "
            f"explained var: {pca.explained_variance_ratio_.sum():.1%}",
            fontsize=9)
        ax_pca.axis("off")
        for idx in range(n_cams):
            rr, cc = divmod(idx, 3)
            ax_pca.text(cc*fW + fW*0.05, rr*fH + fH*0.15,
                        _CAM_ORDER[idx].replace("CAM_",""), fontsize=8,
                        color="white", fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.15",
                                  fc="#000", alpha=0.55))
    except Exception as e:
        print(f"    PCA失败: {e}")

    ax_info = fig.add_subplot(gs_pca[0, 1])
    ax_info.axis("off")
    lines = [
        f"Layer: Multi-camera feat (backbone/neck)",
        f"B*N={n_cams}: 6 cameras merged into batch",
        f"Feature: (C={C}, fH={fH}, fW={fW})",
        f"Spatial downsample: 256×704 → {fH}×{fW}",
        f"  (stride ×{256//fH}×{704//fW})",
        f"Global mean: {float(feat_f.mean()):.4f}",
        f"Global std:  {float(feat_f.std()):.4f}",
        f"Global max:  {float(feat_f.max()):.4f}",
        f"Global min:  {float(feat_f.min()):.4f}",
        "",
        "Next: View Transformer → BEV 200×200",
    ]
    ax_info.text(0.05, 0.97, "\n".join(lines),
                 transform=ax_info.transAxes, fontsize=9, va="top",
                 fontfamily="monospace",
                 bbox=dict(boxstyle="round", fc="lightyellow",
                           ec="gray", alpha=0.8))

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → 已保存: {save_path}")


# =====================================================================
#  (2) BEV层可视化  —  view_transformer / bev_encoder
#      Hook捕获: (B, C, Dy, Dx) 或多尺度列表取第一尺度
# =====================================================================

def visualize_feature_map_bev(feat: torch.Tensor, title: str, save_path: str):
    """BEV特征图可视化: 均值/最大/PCA/直方图/多通道拼图."""
    if feat.dim() == 5:
        feat = feat[0, 0]
    elif feat.dim() == 4:
        feat = feat[0]
    if feat.dim() != 3:
        print(f"  [跳过] {title}: shape {feat.shape}")
        return

    C, H, W = feat.shape
    feat_f = feat.float()
    extent = [_PCR[1], _PCR[4], _PCR[0], _PCR[3]]

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(
        f"{title}\n"
        f"BEV Feature Map — (C={C}, H={H}≡Dy, W={W}≡Dx)\n"
        f"X∈[-40,40]m (front-rear/vert),  Y∈[-40,40]m (left-right/horiz)",
        fontsize=12, fontweight="bold")

    def _im(ax, data, cmap, ttl):
        im = ax.imshow(data, cmap=cmap, aspect="equal", origin="lower",
                       extent=extent)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(ttl, fontsize=10)
        ax.set_xlabel("Y / m", fontsize=8); ax.set_ylabel("X / m", fontsize=8)
        ax.axhline(0, color="cyan", lw=0.5, alpha=0.6)
        ax.axvline(0, color="cyan", lw=0.5, alpha=0.6)
        ax.tick_params(labelsize=7)

    _im(axes[0,0], feat_f.mean(0).numpy(), "viridis", "Channel Mean")
    _im(axes[0,1], feat_f.max(0).values.numpy(), "hot", "Channel Max")
    _im(axes[0,2], feat_f.std(0).numpy(), "magma", "Channel Std")

    try:
        from sklearn.decomposition import PCA
        flat = feat_f.reshape(C,-1).T.numpy()
        pca = PCA(n_components=min(3,C))
        proj = pca.fit_transform(flat)
        for ch in range(proj.shape[1]):
            mn, mx = proj[:,ch].min(), proj[:,ch].max()
            if mx > mn: proj[:,ch] = (proj[:,ch]-mn)/(mx-mn)
        if proj.shape[1] < 3:
            proj = np.pad(proj, ((0,0),(0,3-proj.shape[1])))
        axes[1,0].imshow(proj.reshape(H,W,3), origin="lower",
                         extent=extent, aspect="equal")
        axes[1,0].set_title(
            f"PCA RGB ({pca.explained_variance_ratio_.sum():.1%} var)", fontsize=10)
        axes[1,0].set_xlabel("Y / m"); axes[1,0].set_ylabel("X / m")
        axes[1,0].axhline(0, color="cyan", lw=0.5, alpha=0.6)
        axes[1,0].axvline(0, color="cyan", lw=0.5, alpha=0.6)
    except Exception as e:
        axes[1,0].text(0.5, 0.5, str(e), transform=axes[1,0].transAxes, ha="center")

    vals = feat_f.numpy().flatten()
    axes[1,1].hist(vals, bins=100, color="steelblue", alpha=0.8)
    axes[1,1].set_title(
        f"Activation Distribution\n"
        f"mean={vals.mean():.3f}, std={vals.std():.3f}", fontsize=10)
    axes[1,1].set_xlabel("value"); axes[1,1].set_ylabel("count")

    n_show = min(16, C)
    idxs = np.linspace(0, C-1, n_show, dtype=int)
    canvas = np.zeros((4*H, 4*W))
    for ii, chi in enumerate(idxs):
        if ii >= 16: break
        rr, cc = divmod(ii, 4)
        ch = feat_f[chi].numpy()
        mn, mx = ch.min(), ch.max()
        if mx > mn: ch = (ch-mn)/(mx-mn)
        canvas[rr*H:(rr+1)*H, cc*W:(cc+1)*W] = ch
    axes[1,2].imshow(canvas, cmap="viridis", aspect="auto")
    axes[1,2].set_title(f"Sampled Channels ({n_show}/{C})", fontsize=10)
    axes[1,2].tick_params(labelsize=7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → 已保存: {save_path}")


# =====================================================================
#  (3) OCC Head 可视化
#      Hook捕获: (B, Dx=200, Dy=200, Dz=16, n_cls=18)
#
#      维度含义:
#        Dx=200: 世界X (-40→+40m, 0.4m, 前后/纵向)
#        Dy=200: 世界Y (-40→+40m, 0.4m, 左右/横向)
#        Dz=16:  世界Z (  -1→+5.4m, 0.4m, 高度)
#        n_cls=18: 每体素 18个语义类别的 logit 原始得分
# =====================================================================

def _bev_argmax_proj(pred: np.ndarray) -> np.ndarray:
    """(Dx,Dy,Dz) → BEV (Dx,Dy): 沿Z从高到低取首个非free非others类."""
    result = np.full(pred.shape[:2], 17, dtype=np.uint8)
    for z in range(pred.shape[2] - 1, -1, -1):
        layer = pred[:, :, z]
        valid = (layer != 17) & (layer != 0) & (result == 17)
        result[valid] = layer[valid]
    return result


def _side_argmax_proj(pred: np.ndarray) -> np.ndarray:
    """(Dx,Dy,Dz) → Side (Dx,Dz): 沿Y取首个非free类."""
    result = np.full((pred.shape[0], pred.shape[2]), 17, dtype=np.uint8)
    for y in range(pred.shape[1]):
        layer = pred[:, y, :]
        valid = (layer != 17) & (layer != 0) & (result == 17)
        result[valid] = layer[valid]
    return result


def _front_argmax_proj(pred: np.ndarray) -> np.ndarray:
    """(Dx,Dy,Dz) → Front (Dy,Dz): 沿X取首个非free类."""
    result = np.full((pred.shape[1], pred.shape[2]), 17, dtype=np.uint8)
    for x in range(pred.shape[0]):
        layer = pred[x, :, :]
        valid = (layer != 17) & (layer != 0) & (result == 17)
        result[valid] = layer[valid]
    return result


def _cls_to_rgb(cls_2d: np.ndarray) -> np.ndarray:
    return _COLOR_LUT[cls_2d.flatten()].reshape(*cls_2d.shape, 3)


def visualize_occ_head(feat: torch.Tensor, title: str, save_path: str):
    """OCC Head 原始输出可视化.

    输入: (B, Dx=200, Dy=200, Dz=16, n_cls=18)
    图布局:
      Row 0: BEV(XY)/Side(XZ)/Front(YZ) argmax投影 + 类别数量条形图
      Row 1: 三视图最大置信度图 + 全空间平均logit柱状图
      Row 2: Z层切片 (z=-1.0m ~ +1.8m 前8层)
    """
    if feat.dim() != 5:
        print(f"  [OCC Head] 期望5D (B,Dx,Dy,Dz,cls), 实际 {feat.shape}, 退化为BEV可视化")
        visualize_feature_map_bev(feat, title, save_path)
        return

    logits = feat[0].float()          # (Dx, Dy, Dz, n_cls)
    Dx, Dy, Dz, n_cls = logits.shape
    print(f"  OCC Head logits: Dx={Dx}, Dy={Dy}, Dz={Dz}, n_cls={n_cls}")
    print(f"    X(-40~+40m 前后) | Y(-40~+40m 左右) | Z(-1~+5.4m 高度) | {n_cls}类logit")

    x_range = (_PCR[0], _PCR[3])
    y_range = (_PCR[1], _PCR[4])
    z_range = (_PCR[2], _PCR[5])

    pred_cls = logits.argmax(dim=-1).numpy().astype(np.uint8)
    probs    = torch.softmax(logits, dim=-1)
    max_conf = probs.max(dim=-1).values.numpy()

    bev_cls   = _bev_argmax_proj(pred_cls)
    side_cls  = _side_argmax_proj(pred_cls)
    front_cls = _front_argmax_proj(pred_cls)
    bev_conf   = max_conf.max(axis=2)
    side_conf  = max_conf.max(axis=1)
    front_conf = max_conf.max(axis=0)

    colors_bar = [OCC_CLASS_COLORS.get(c, (0.5,0.5,0.5)) for c in range(n_cls)]
    cls_names  = [OCC_CLASS_NAMES[c] if c < len(OCC_CLASS_NAMES) else f"cls{c}"
                  for c in range(n_cls)]

    fig = plt.figure(figsize=(28, 24))
    fig.suptitle(
        f"{title}\n"
        f"OCC Head logits shape: (Dx={Dx}, Dy={Dy}, Dz={Dz}, n_cls={n_cls})\n"
        f"Dx: X front-rear[-40,+40m]  |  Dy: Y left-right[-40,+40m]  |  "
        f"Dz: Z height[-1,+5.4m]  |  n_cls: 18-class semantic logit (pre-softmax)",
        fontsize=13, fontweight="bold")

    gs0 = GridSpec(1, 4, figure=fig, top=0.87, bottom=0.61, wspace=0.28)
    gs1 = GridSpec(1, 4, figure=fig, top=0.57, bottom=0.31, wspace=0.28)
    gs2 = GridSpec(1, 8, figure=fig, top=0.27, bottom=0.04,
                   wspace=0.08, left=0.04, right=0.96)

    def _proj(ax, rgb, ttl, xl, yl, ext):
        ax.imshow(rgb, origin="lower", aspect="auto", extent=ext)
        ax.set_title(ttl, fontsize=10, fontweight="bold")
        ax.set_xlabel(xl, fontsize=8); ax.set_ylabel(yl, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.axhline(0, color="cyan", lw=0.5, alpha=0.6)
        ax.axvline(0, color="cyan", lw=0.5, alpha=0.6)

    _proj(fig.add_subplot(gs0[0,0]), _cls_to_rgb(bev_cls),
          "BEV / Top-Down (XY)\nFirst non-free class along Z",
          "Y / m  (left-right)", "X / m  (front-rear)",
          [y_range[0], y_range[1], x_range[0], x_range[1]])

    _proj(fig.add_subplot(gs0[0,1]), _cls_to_rgb(side_cls),
          "Side View (XZ)\nProjected along Y",
          "Z / m  (height)", "X / m  (front-rear)",
          [z_range[0], z_range[1], x_range[0], x_range[1]])

    _proj(fig.add_subplot(gs0[0,2]), _cls_to_rgb(front_cls),
          "Front View (YZ)\nProjected along X",
          "Z / m  (height)", "Y / m  (left-right)",
          [z_range[0], z_range[1], y_range[0], y_range[1]])

    ax_bar = fig.add_subplot(gs0[0,3])
    counts = np.array([(pred_cls == c).sum() for c in range(n_cls)])
    total  = pred_cls.size
    y_pos  = np.arange(n_cls)
    ax_bar.barh(y_pos, counts, color=colors_bar, edgecolor="gray", lw=0.3)
    ax_bar.set_yticks(y_pos); ax_bar.set_yticklabels(cls_names, fontsize=7)
    ax_bar.set_xlabel("Voxel Count", fontsize=8)
    ax_bar.set_title("Class Voxel Count\n(argmax prediction)", fontsize=10, fontweight="bold")
    for i, cnt in enumerate(counts):
        if cnt > 0:
            ax_bar.text(cnt + total*0.003, i, f"{cnt/total:.1%}",
                        va="center", fontsize=6, color="gray")

    def _conf(ax, conf2d, ttl, xl, yl, ext):
        im = ax.imshow(conf2d, cmap="plasma", origin="lower", aspect="auto",
                       extent=ext, vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(ttl, fontsize=10); ax.set_xlabel(xl, fontsize=8)
        ax.set_ylabel(yl, fontsize=8); ax.tick_params(labelsize=7)
        ax.axhline(0, color="cyan", lw=0.5, alpha=0.5)
        ax.axvline(0, color="cyan", lw=0.5, alpha=0.5)

    _conf(fig.add_subplot(gs1[0,0]), bev_conf,
          "BEV Max Confidence (max softmax along Z)",
          "Y / m", "X / m",
          [y_range[0], y_range[1], x_range[0], x_range[1]])
    _conf(fig.add_subplot(gs1[0,1]), side_conf,
          "Side Max Confidence (max softmax along Y)",
          "Z / m", "X / m",
          [z_range[0], z_range[1], x_range[0], x_range[1]])
    _conf(fig.add_subplot(gs1[0,2]), front_conf,
          "Front Max Confidence (max softmax along X)",
          "Z / m", "Y / m",
          [z_range[0], z_range[1], y_range[0], y_range[1]])

    ax_lgit = fig.add_subplot(gs1[0,3])
    mean_logits = logits.reshape(-1, n_cls).mean(0).numpy()
    ax_lgit.bar(range(n_cls), mean_logits, color=colors_bar, edgecolor="gray", lw=0.3)
    ax_lgit.set_xticks(range(n_cls))
    ax_lgit.set_xticklabels(cls_names, rotation=45, ha="right", fontsize=6)
    ax_lgit.set_ylabel("Mean Logit", fontsize=8)
    ax_lgit.set_title("Mean Logit over All Voxels\n(overall prediction bias)", fontsize=9)
    ax_lgit.axhline(0, color="k", lw=0.5)

    n_slices = min(Dz, 8)
    z_heights = [_PCR[2] + (k + 0.5) * _VS for k in range(Dz)]
    for k in range(n_slices):
        ax_s = fig.add_subplot(gs2[0, k])
        ax_s.imshow(_cls_to_rgb(pred_cls[:, :, k]),
                    origin="lower", aspect="equal")
        ax_s.set_title(f"Z={z_heights[k]:.1f}m", fontsize=7, pad=2)
        ax_s.axis("off")
        nf = int(((pred_cls[:,:,k] != 17) & (pred_cls[:,:,k] != 0)).sum())
        ax_s.text(0.5, -0.14, f"{nf}vox", transform=ax_s.transAxes,
                  ha="center", fontsize=6, color="gray")

    present = np.unique(pred_cls[pred_cls != 17])
    patches = [Patch(facecolor=OCC_CLASS_COLORS.get(int(c),(0.5,0.5,0.5)),
                     edgecolor="k", lw=0.4,
                     label=OCC_CLASS_NAMES[int(c)] if int(c) < len(OCC_CLASS_NAMES)
                           else f"cls{c}")
               for c in present]
    if patches:
        fig.legend(handles=patches, loc="lower center",
                   ncol=min(9, len(patches)), fontsize=7,
                   frameon=True, bbox_to_anchor=(0.5, 0.005))

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → 已保存: {save_path}")


# =====================================================================
#  (4) 3D OCC 4面板可视化  (完整重写)
#
#  布局:
#    [GT 3D 等轴测]    [Pred 3D 等轴测]
#    [GT BEV+辅助线]   [Pred BEV+辅助线]
#
#  BEV 辅助线内容:
#    · 6相机视场角扇形 (各自颜色)
#    · 雷诺Zoé自车轮廓 (4.084m×1.73m)
#    · 10m网格 + 前后方位标注
# =====================================================================

def _render_bev_panel(ax, occ: np.ndarray, title="BEV",
                      draw_helpers=True, fontsize=11):
    """渲染BEV面板."""
    Dx, Dy, Dz = occ.shape
    bev_rgb = np.ones((Dx, Dy, 3), dtype=np.float32)
    filled  = np.zeros((Dx, Dy), dtype=bool)
    for z in range(Dz - 1, -1, -1):
        layer = occ[:, :, z]
        valid = (layer != 17) & (layer != 0) & ~filled
        if not valid.any(): continue
        for cid in np.unique(layer[valid]):
            mask = (layer == cid) & valid
            bev_rgb[mask] = OCC_CLASS_COLORS.get(int(cid), (0.5,0.5,0.5))
            filled[mask] = True

    ax.imshow(bev_rgb, origin="lower", aspect="equal")
    if draw_helpers:
        draw_camera_fovs(ax, alpha_fill=0.09, alpha_line=0.70)
        draw_ego_vehicle(ax, linewidth=2.0, color="#1E90FF")
        add_bev_annotations(ax, title=title, fontsize=fontsize, tick_m=10.0)
    else:
        ax.axis("off")
        ax.set_title(title, fontsize=fontsize, fontweight="bold", pad=8)


def _render_isometric_panel(ax, occ: np.ndarray, voxel_step=2,
                             azim_deg=45.0, elev_deg=35.0, z_scale=2.5,
                             title="3D Isometric", fontsize=11):
    """等轴测体素渲染 (软件光照 + Painter's Algorithm)."""
    occ = occ[::voxel_step, ::voxel_step, ::voxel_step]
    Dx, Dy, Dz = occ.shape
    occupied = (occ != 17) & (occ != 0)
    if not occupied.any():
        ax.text(0.5, 0.5, "No occupied voxels", transform=ax.transAxes,
                ha="center", va="center", fontsize=14)
        ax.axis("off"); ax.set_title(title, fontsize=fontsize, fontweight="bold"); return

    padded = np.pad(occupied, 1, constant_values=False)
    azim = np.radians(azim_deg); elev = np.radians(elev_deg)
    cos_a, sin_a = np.cos(azim), np.sin(azim)
    cos_e, sin_e = np.cos(elev), np.sin(elev)
    view_dir = np.array([sin_a*cos_e, cos_a*cos_e, -sin_e])

    face_defs = [
        ( 1, 0, 0, np.array([[1,0,0],[1,1,0],[1,1,1],[1,0,1]], dtype=np.float64), 0.80),
        (-1, 0, 0, np.array([[0,0,0],[0,0,1],[0,1,1],[0,1,0]], dtype=np.float64), 0.70),
        ( 0, 1, 0, np.array([[0,1,0],[0,1,1],[1,1,1],[1,1,0]], dtype=np.float64), 0.65),
        ( 0,-1, 0, np.array([[0,0,0],[1,0,0],[1,0,1],[0,0,1]], dtype=np.float64), 0.75),
        ( 0, 0, 1, np.array([[0,0,1],[1,0,1],[1,1,1],[0,1,1]], dtype=np.float64), 1.00),
        ( 0, 0,-1, np.array([[0,0,0],[0,1,0],[1,1,0],[1,0,0]], dtype=np.float64), 0.55),
    ]
    visible = [f for f in face_defs
               if np.dot(np.array(f[:3], dtype=float), view_dir) < 0]

    all_p, all_c, all_d = [], [], []
    for dx, dy, dz, vtpl, shade in visible:
        nbr = padded[1+dx:Dx+1+dx, 1+dy:Dy+1+dy, 1+dz:Dz+1+dz]
        exp_mask = occupied & ~nbr
        if not exp_mask.any(): continue
        xi, yi, zi = np.where(exp_mask)
        n = len(xi)
        orig = np.stack([xi, yi, zi], axis=-1).astype(np.float64)
        v3 = orig[:,None,:] + vtpl[None,:,:]
        v3[...,2] *= z_scale
        vx, vy, vz = v3[...,0], v3[...,1], v3[...,2]
        xr = vx*cos_a - vy*sin_a; yr = vx*sin_a + vy*cos_a
        v2 = np.stack([xr, yr*sin_e + vz*cos_e], axis=-1)
        ctr = orig.copy(); ctr[:,2] *= z_scale
        dep = (ctr[:,0]*sin_a + ctr[:,1]*cos_a)*cos_e + ctr[:,2]*sin_e
        fc = np.ones((n,4)); fc[:,:3] = np.clip(_COLOR_LUT[occ[xi,yi,zi]]*shade, 0, 1)
        all_p.append(v2); all_c.append(fc); all_d.append(dep)

    if not all_p:
        ax.text(0.5, 0.5, "No exposed faces", transform=ax.transAxes,
                ha="center", va="center"); ax.axis("off"); return

    polys = np.concatenate(all_p); fc_ = np.concatenate(all_c)
    dep_  = np.concatenate(all_d)
    order = np.argsort(dep_)
    polys = polys[order]; fc_ = fc_[order]
    ec_ = fc_.copy(); ec_[:,:3] = np.clip(ec_[:,:3]*0.5, 0, 1); ec_[:,3] = 0.3

    ax.set_facecolor("#EEEEEE")
    pc = PolyCollection(polys, closed=True)
    pc.set_facecolor(fc_); pc.set_edgecolor(ec_); pc.set_linewidth(0.15)
    ax.add_collection(pc); ax.autoscale(); ax.set_aspect("equal"); ax.axis("off")
    ax.text(0.01, 0.03,
            f"↑X(front) ←Y(left) Z↑(height×{z_scale})",
            transform=ax.transAxes, fontsize=7, color="#333",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))
    ax.set_title(title, fontsize=fontsize, fontweight="bold", pad=8)


def visualize_occ_4panel(occ_gt, occ_pred: np.ndarray, save_path: str,
                          voxel_step: int = 1,
                          title: str = "3D Occupancy"):
    """4面板OCC可视化.

    布局 (has_gt=True):
        [GT 3D等轴测]    [Pred 3D等轴测]
        [GT BEV+辅助线]  [Pred BEV+辅助线]

    若 occ_gt=None: 2面板 (Pred 3D + Pred BEV).
    """
    has_gt = occ_gt is not None
    n_cols = 2 if has_gt else 1
    fig_w = 15 * n_cols + 1

    fig = plt.figure(figsize=(fig_w, 26))
    fig.patch.set_facecolor("white")
    fig.suptitle(title, fontsize=18, fontweight="bold", y=0.998)

    gs = GridSpec(2, n_cols, figure=fig,
                  hspace=0.10, wspace=0.06,
                  top=0.975, bottom=0.09,
                  height_ratios=[1.15, 1])

    occ_items = []
    if has_gt: occ_items.append(("Ground Truth", occ_gt, 0))
    occ_items.append(("Prediction", occ_pred, 1 if has_gt else 0))

    for tag, occ, ci in occ_items:
        ax_3d = fig.add_subplot(gs[0, ci])
        _render_isometric_panel(
            ax_3d, occ, voxel_step=max(voxel_step,1),
            azim_deg=45.0, elev_deg=35.0, z_scale=2.5,
            title=f"{tag} — 3D Isometric View", fontsize=12)

        ax_bev = fig.add_subplot(gs[1, ci])
        _render_bev_panel(ax_bev, occ,
                          title=f"{tag} — Bird's Eye View (BEV)",
                          draw_helpers=True, fontsize=12)

    # --- 图例 ---
    cam_patches = [
        Patch(facecolor=info["color"], edgecolor=info["color"], alpha=0.7,
              label=f"{name.replace('CAM_','').replace('_',' ')} "
                    f"[hdg={info['heading']:+.0f}° HFoV={info['hfov']:.0f}°]")
        for name, info in _CAM_INFO.items()
    ]
    cam_patches.append(Patch(facecolor="#1E90FF", alpha=0.35, edgecolor="#1E90FF",
                             label="Ego — Renault Zoé (4.084m×1.73m, wb=2.588m)"))

    all_cls: set = set(int(c) for c in occ_pred[(occ_pred != 17) & (occ_pred != 0)])
    if has_gt:
        all_cls |= set(int(c) for c in occ_gt[(occ_gt != 17) & (occ_gt != 0)])
    cls_patches = [
        Patch(facecolor=OCC_CLASS_COLORS.get(c, (0.5,0.5,0.5)),
              edgecolor="k", lw=0.4,
              label=f"{OCC_CLASS_NAMES[c] if c<len(OCC_CLASS_NAMES) else c} "
                    f"({(occ_pred==c).sum():,})")
        for c in sorted(all_cls)
    ]

    leg1 = fig.legend(handles=cam_patches, loc="lower center",
                      ncol=len(cam_patches), fontsize=8, frameon=True,
                      fancybox=True, bbox_to_anchor=(0.5, 0.068),
                      title="Camera FoV Lines & Ego Vehicle", title_fontsize=8)
    fig.legend(handles=cls_patches, loc="lower center",
               ncol=min(9, len(cls_patches)), fontsize=8,
               frameon=True, fancybox=True, bbox_to_anchor=(0.5, 0.005))
    fig.add_artist(leg1)

    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  → 已保存: {save_path}")


# =====================================================================
#  输入图像可视化
# =====================================================================

def visualize_input_images(img_inputs: tuple, save_path: str):
    """可视化6相机输入图像 + 右侧BEV相机布局示意."""
    imgs = img_inputs[0]
    if isinstance(imgs, torch.Tensor):
        imgs = imgs[0].cpu().float()
    N = imgs.shape[0]

    from flashocc.constants import IMAGENET_MEAN, IMAGENET_STD

    fig = plt.figure(figsize=(22, 9))
    fig.suptitle("Input Camera Images — 6-Camera Surround View (after augmentation)",
                 fontsize=14, fontweight="bold")

    cam_grid = [
        ("CAM_FRONT_LEFT",  0, 0),
        ("CAM_FRONT",       0, 1),
        ("CAM_FRONT_RIGHT", 0, 2),
        ("CAM_BACK_LEFT",   1, 0),
        ("CAM_BACK",        1, 1),
        ("CAM_BACK_RIGHT",  1, 2),
    ]
    gs = GridSpec(2, 3, figure=fig, left=0.01, right=0.83,
                  hspace=0.22, wspace=0.05)

    for i, (cam, row, col) in enumerate(cam_grid):
        if i >= N: continue
        img = imgs[i].permute(1,2,0).numpy()
        img = img * IMAGENET_STD.reshape(1,1,3) + IMAGENET_MEAN.reshape(1,1,3)
        img = img[:,:,::-1]
        img = np.clip(img / 255.0, 0, 1)
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(img)
        info = _CAM_INFO[cam]
        ax.set_title(
            f"{cam.replace('CAM_','')}\n"
            f"hdg={info['heading']:+.0f}°  HFoV={info['hfov']:.0f}°",
            fontsize=10, fontweight="bold", color=info["color"],
            bbox=dict(boxstyle="round", fc="white", ec=info["color"], alpha=0.7))
        ax.axis("off")
        h, w = img.shape[:2]
        ax.text(4, h-4, f"{w}×{h}px", fontsize=7, color="white", va="bottom",
                bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.5))

    # 右侧BEV布局示意
    ax_l = fig.add_axes([0.85, 0.06, 0.14, 0.88])
    ax_l.set_facecolor("#1a1a2e")
    ax_l.set_xlim(0, _DY); ax_l.set_ylim(0, _DX)
    ax_l.set_aspect("equal"); ax_l.axis("off")
    ax_l.set_title("Camera\nLayout\n(BEV)", fontsize=9, color="white",
                    fontweight="bold", pad=4)
    draw_camera_fovs(ax_l, alpha_fill=0.13, alpha_line=0.9)
    draw_ego_vehicle(ax_l, color="#E0E0FF", linewidth=2.5)
    cx, cy = world_to_bev_px(0, 0)
    ax_l.text(cx, _DX-3, "▲ FRONT", ha="center", va="top",
              color="white", fontsize=7, fontweight="bold")
    ax_l.text(cx, 3, "▼ REAR", ha="center", va="bottom",
              color="white", fontsize=7, fontweight="bold")

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → 已保存: {save_path}")


# =====================================================================
#  OCC 统计
# =====================================================================

def print_occ_stats(occ: np.ndarray, tag: str = "Prediction"):
    total = occ.size
    print(f"\n{'='*52}")
    print(f"  {tag} — shape {occ.shape}")
    print(f"{'='*52}")
    print(f"  {'Class':<26s} {'Count':>10s} {'Ratio':>8s}")
    print(f"  {'-'*44}")
    for cid in range(len(OCC_CLASS_NAMES)):
        cnt = int((occ == cid).sum())
        if cnt > 0:
            print(f"  {OCC_CLASS_NAMES[cid]:<26s} {cnt:>10,d} {cnt/total*100:>7.2f}%")
    print(f"{'='*52}\n")


# =====================================================================
#  主流程
# =====================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="FlashOCC 特征 & 3D 占用可视化工具 v2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("config",     help="Python 配置文件路径 (.py)")
    parser.add_argument("checkpoint", help="checkpoint 文件路径")
    parser.add_argument("--sample-idx", type=int, default=0)
    parser.add_argument("--out-dir",  default="vis_output")
    parser.add_argument("--device",   default="cuda:0")
    parser.add_argument("--no-feat",  action="store_true")
    parser.add_argument("--no-occ",   action="store_true")
    parser.add_argument("--voxel-step", type=int, default=1)
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)

    print("=" * 65)
    print("  FlashOCC 可视化工具  v2  (重新设计版)")
    print("=" * 65)

    print(f"\n[1/5] 加载配置: {args.config}")
    exp = load_experiment(args.config)

    print(f"[2/5] 构建模型并加载权重: {args.checkpoint}")
    model = exp.build_model()
    load_checkpoint(model, args.checkpoint, map_location="cpu")
    model = model.to(device); model.eval()

    extractor = FeatureExtractor()
    if not args.no_feat:
        print("[3/5] 注册特征提取 Hook ...")
        extractor.register(model)

    print(f"[4/5] 加载验证集, 取第 {args.sample_idx} 个样本 ...")
    dataset = exp.build_test_dataset()
    print(f"      验证集大小: {len(dataset)}")
    if args.sample_idx >= len(dataset):
        print(f"  [错误] sample_idx 超出范围 (max={len(dataset)-1})"); return

    data_loader = build_dataloader(
        dataset, samples_per_gpu=1, workers_per_gpu=0,
        dist_mode=False, shuffle=False)
    batch = None
    for i, b in enumerate(data_loader):
        if i == args.sample_idx: batch = b; break
    if batch is None:
        print(f"  [错误] 无法获取样本 {args.sample_idx}"); return

    print("[5/5] 执行前向推理 ...")
    from flashocc.datasets.dali_decode import dali_decode_batch
    if "jpeg_bytes" in batch:
        for k in ("jpeg_bytes", "img_aug_params"):
            if k in batch and hasattr(batch[k], "data"): batch[k] = batch[k].data
        batch = dali_decode_batch(batch, color_order=exp.image_color_order)

    img_inputs = batch["img_inputs"]
    if isinstance(img_inputs, (list, tuple)):
        img_inputs_gpu = [t.to(device) if isinstance(t, torch.Tensor) else t
                          for t in img_inputs]
    else:
        img_inputs_gpu = img_inputs.to(device)

    img_metas = batch.get("img_metas", [{}])
    if hasattr(img_metas, "data"): img_metas = img_metas.data[0]
    if isinstance(img_metas, dict): img_metas = [img_metas]

    points = batch.get("points", None)
    if points is not None:
        if hasattr(points, "data"): points = points.data[0]
        if isinstance(points, (list, tuple)):
            points = [p.to(device) if isinstance(p, torch.Tensor) else p for p in points]
        elif isinstance(points, torch.Tensor):
            points = points.to(device)

    results = model.forward_test(
        points=points, img_inputs=img_inputs_gpu, img_metas=img_metas)

    # ---- 可视化1: 输入图像 ----
    print("\n--- [可视化] 输入图像 ---")
    try:
        vis_in = img_inputs_gpu if isinstance(img_inputs_gpu, (list,tuple)) \
                 else (img_inputs_gpu,)
        visualize_input_images(vis_in,
                               os.path.join(args.out_dir, "00_input_cameras.png"))
    except Exception as e:
        import traceback; traceback.print_exc(); print(f"  [跳过] {e}")

    # ---- 可视化2: 中间层特征 ----
    if not args.no_feat:
        print("\n--- [可视化] 中间层特征 ---")
        for idx, (name, (layer_type, feat)) in enumerate(extractor.features.items()):
            print(f"  [{idx+1}/{len(extractor.features)}] {name}  "
                  f"type={layer_type}  shape={list(feat.shape)}")
            safe = (name.replace(" ","_").replace("/","_")
                       .replace("(","").replace(")","").replace(".","_"))
            sp = os.path.join(args.out_dir, f"{idx+1:02d}_{safe}.png")
            try:
                if layer_type == "multicam":
                    visualize_multicam_features(feat, name, sp, n_cams=6)
                elif layer_type == "occ_head":
                    visualize_occ_head(feat, name, sp)
                else:
                    visualize_feature_map_bev(feat, name, sp)
            except Exception as e:
                import traceback; traceback.print_exc(); print(f"    [跳过] {e}")
        extractor.remove()

    # ---- 可视化3: 3D OCC ----
    if not args.no_occ:
        print("\n--- [可视化] 3D 占用预测 ---")
        occ_pred = None
        if isinstance(results, torch.Tensor):
            t = results[0] if results.dim() == 4 else results
            occ_pred = t.cpu().numpy()
        elif isinstance(results, np.ndarray):
            occ_pred = results[0] if results.ndim == 4 else results
        elif isinstance(results, (list, tuple)) and results:
            r = results[0]
            if isinstance(r, np.ndarray): occ_pred = r
            elif isinstance(r, torch.Tensor): occ_pred = r.cpu().numpy()
            elif isinstance(r, dict):
                for k in ["occ","occ_pred","occ_preds","pts_bbox"]:
                    if k in r:
                        v = r[k]
                        occ_pred = v.cpu().numpy() if isinstance(v,torch.Tensor) else v
                        break

        if occ_pred is None:
            print("  [错误] 无法提取占用预测"); return

        occ_pred = occ_pred.astype(np.uint8)
        print(f"  OCC shape: {occ_pred.shape}")
        print_occ_stats(occ_pred, "Prediction")

        occ_gt = None
        if "voxel_semantics" in batch:
            gd = batch["voxel_semantics"]
            if hasattr(gd, "data"): gd = gd.data[0]
            if isinstance(gd, torch.Tensor):
                occ_gt = gd[0].cpu().numpy().astype(np.uint8)
            elif isinstance(gd, list) and gd:
                if isinstance(gd[0], torch.Tensor):
                    occ_gt = gd[0].cpu().numpy().astype(np.uint8)

        if occ_gt is not None:
            print(f"  GT shape: {occ_gt.shape}")
            print_occ_stats(occ_gt, "Ground Truth")

        # 4面板主图 (GT + Pred)
        visualize_occ_4panel(
            occ_gt, occ_pred,
            os.path.join(args.out_dir, "occ_4panel.png"),
            voxel_step=args.voxel_step,
            title="3D Occupancy — Ground Truth vs Prediction",
        )
        # 仅预测 (若有GT则额外生成单独预测图)
        if occ_gt is not None:
            visualize_occ_4panel(
                None, occ_pred,
                os.path.join(args.out_dir, "occ_prediction_only.png"),
                voxel_step=args.voxel_step,
                title="3D Occupancy Prediction",
            )

    print("\n" + "=" * 65)
    print(f"  可视化完成!  结果: {args.out_dir}/")
    print("=" * 65)
    for f in sorted(os.listdir(args.out_dir)):
        fp = os.path.join(args.out_dir, f)
        print(f"    {f:<52s} {os.path.getsize(fp)/1024:>8.1f} KB")


if __name__ == "__main__":
    main()
