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
from matplotlib.patches import Patch
from matplotlib.collections import PolyCollection
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

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

# 输入图像在 img_inputs 中的通道顺序 (由配置 experiment.image_color_order 控制)
# 可视化时统一转换为 RGB 再显示
_VIS_INPUT_COLOR_ORDER = "RGB"


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
        self.depth: torch.Tensor | None = None  # LSS depth: (B*N, D, fH, fW)
        self._hooks: list[Any] = []

    def register(self, model: nn.Module):
        for attr_name, layer_type, display_name in self.LAYER_REGISTRY:
            module = getattr(model, attr_name, None)
            if module is None:
                continue
            if attr_name == "img_view_transformer":
                hook = module.register_forward_hook(
                    self._make_vt_hook(display_name, layer_type))
            else:
                hook = module.register_forward_hook(
                    self._make_hook(display_name, layer_type))
            self._hooks.append(hook)

    def _make_vt_hook(self, name: str, layer_type: str):
        """view_transformer 专用 hook: 同时捕获 bev_feat 和 depth."""
        def hook_fn(module, input_t, output):
            if isinstance(output, (list, tuple)) and len(output) >= 2:
                bev_feat, depth = output[0], output[1]
                if isinstance(bev_feat, torch.Tensor):
                    self.features[name] = (layer_type, bev_feat.detach().cpu())
                if isinstance(depth, torch.Tensor):
                    self.depth = depth.detach().cpu()
            elif isinstance(output, torch.Tensor):
                self.features[name] = (layer_type, output.detach().cpu())
        return hook_fn

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
    hist_values = []
    for idx in range(n_cams):
        v = feat_f[idx].numpy().flatten()
        hist_values.append(v)
        ax_hist.hist(v, bins=60, alpha=0.38, color=cam_colors[idx],
                     label=_CAM_ORDER[idx].replace("CAM_",""),
                     density=True, histtype="stepfilled")
    ax_hist.set_title("Activation Distribution (6 cameras overlay)", fontsize=9)
    ax_hist.set_xlabel("value"); ax_hist.legend(fontsize=6, ncol=2)
    _auto_scale_hist_xy(ax_hist, hist_values)

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
#  (1b) LSS 深度预测可视化 — 2D 热力图 + 3D 透视热力图
#       depth: (B*N, D, fH, fW)  softmax 分布
#       depth_bins: 1.0, 1.5, ..., 44.5  (D=88)
# =====================================================================

def _make_depth_bins(depth_cfg: tuple = (1.0, 45.0, 0.5)) -> np.ndarray:
    """根据 grid_config.depth 生成深度离散化 bin 中心值.

    Returns:
        (D,) array of depth values in meters.
    """
    return np.arange(depth_cfg[0], depth_cfg[1], depth_cfg[2], dtype=np.float32)


def visualize_lss_depth_2d(
    depth: torch.Tensor,
    img_inputs: tuple,
    save_path: str,
    depth_cfg: tuple = (1.0, 45.0, 0.5),
    n_cams: int = 6,
    title: str = "LSS Depth Prediction — 2D Heatmap",
):
    """将 LSS 预测的深度分布投影回 2D 热力图叠加到输入图像上.

    对每个相机、每个像素位置，计算期望深度 E[d] = sum(p_i * d_i)，
    得到 (fH, fW) 的期望深度图, 上采样到原图大小后热力图叠加.

    Args:
        depth: (B*N, D, fH, fW) softmax depth distribution
        img_inputs: img_inputs tuple (包含原图/intrinsics等)
        save_path: 输出文件路径
        depth_cfg: (min_depth, max_depth, interval)
        n_cams: 相机数量
        title: 图表标题
    """
    from scipy.ndimage import zoom as ndimage_zoom

    depth_np = depth.float().numpy()   # (B*N, D, fH, fW)
    depth_np = depth_np[:n_cams]       # 取第一个 batch 的 6 相机
    D, fH, fW = depth_np.shape[1], depth_np.shape[2], depth_np.shape[3]

    bins = _make_depth_bins(depth_cfg)  # (D,)
    assert bins.shape[0] == D, f"depth bins {bins.shape[0]} != D={D}"

    # 期望深度: E[d] = sum(p_i * d_i) along depth axis
    expected_depth = (depth_np * bins[None, :, None, None]).sum(axis=1)  # (N, fH, fW)

    # 峰值深度 (argmax)
    peak_depth_idx = depth_np.argmax(axis=1)    # (N, fH, fW)
    peak_depth = bins[peak_depth_idx]            # (N, fH, fW)

    # 解码输入图像
    imgs = _decode_input_images_to_bgr01(img_inputs)

    cam_colors = [_CAM_INFO[c]["color"] for c in _CAM_ORDER]

    fig = plt.figure(figsize=(28, 22), facecolor="white")
    fig.suptitle(
        f"{title}\n"
        f"Depth bins: {depth_cfg[0]:.1f}m → {depth_cfg[1]:.1f}m, "
        f"step={depth_cfg[2]:.1f}m, D={D}, feature=({fH}×{fW})",
        fontsize=14, fontweight="bold")

    # 布局: 3行
    #   Row 0: 6相机 期望深度热力图叠加原图 (2×3)
    #   Row 1: 6相机 深度分布热力图 + argmax marker (2×3)
    #   Row 2: 统计面板
    gs_overlay = GridSpec(2, 3, figure=fig, top=0.91, bottom=0.52,
                          hspace=0.30, wspace=0.08)
    gs_dist    = GridSpec(2, 3, figure=fig, top=0.48, bottom=0.10,
                          hspace=0.30, wspace=0.08)
    gs_stats   = GridSpec(1, 3, figure=fig, top=0.08, bottom=0.01,
                          hspace=0.05, wspace=0.20)

    d_min, d_max = float(depth_cfg[0]), float(depth_cfg[1])

    # ---- Row 0: 期望深度叠加原图 ----
    for idx in range(n_cams):
        rg, cg = divmod(idx, 3)
        ax = fig.add_subplot(gs_overlay[rg, cg])
        if idx < imgs.shape[0]:
            ax.imshow(imgs[idx])
        H_img, W_img = imgs[idx].shape[:2] if idx < imgs.shape[0] else (256, 704)

        # 上采样期望深度到原图大小
        ed = expected_depth[idx]   # (fH, fW)
        scale_h, scale_w = H_img / fH, W_img / fW
        ed_up = ndimage_zoom(ed, (scale_h, scale_w), order=1)

        im = ax.imshow(ed_up, cmap="turbo", alpha=0.55,
                       vmin=d_min, vmax=d_max)
        plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="depth (m)")
        cam_name = _CAM_ORDER[idx]
        ax.set_title(f"{cam_name.replace('CAM_','')} — E[depth]",
                     fontsize=9, fontweight="bold", color=cam_colors[idx])
        ax.axis("off")

    # ---- Row 1: 深度分布热力图 (将 D 维压缩为可视化) ----
    for idx in range(n_cams):
        rg, cg = divmod(idx, 3)
        ax = fig.add_subplot(gs_dist[rg, cg])
        cam_name = _CAM_ORDER[idx]

        # 取中间行的深度分布作为代表
        mid_h = fH // 2
        dist_slice = depth_np[idx, :, mid_h, :]   # (D, fW)

        im = ax.imshow(dist_slice, aspect="auto", cmap="magma",
                       extent=[0, fW, d_max, d_min],
                       vmin=0, vmax=float(dist_slice.max()))
        plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="prob")
        ax.set_xlabel("fW pixel", fontsize=8)
        ax.set_ylabel("depth (m)", fontsize=8)
        ax.set_title(f"{cam_name.replace('CAM_','')} — depth dist (row={mid_h})",
                     fontsize=9, fontweight="bold", color=cam_colors[idx])

        # 叠加 argmax 线
        peak_line = peak_depth[idx, mid_h, :]  # (fW,)
        ax.plot(np.arange(fW) + 0.5, peak_line, 'c-', linewidth=0.8,
                alpha=0.8, label="argmax depth")
        ax.legend(fontsize=6, loc="upper right")

    # ---- Row 2: 全局统计 ----
    ax_s = fig.add_subplot(gs_stats[0, 0])
    mean_depths = [float(expected_depth[i].mean()) for i in range(n_cams)]
    ax_s.bar(range(n_cams), mean_depths, color=cam_colors, alpha=0.85)
    ax_s.set_xticks(range(n_cams))
    ax_s.set_xticklabels(
        [c.replace("CAM_","").replace("_","\n") for c in _CAM_ORDER], fontsize=7)
    ax_s.set_title("Mean Expected Depth per Camera", fontsize=9)
    ax_s.set_ylabel("depth (m)", fontsize=8)

    ax_e = fig.add_subplot(gs_stats[0, 1])
    # 深度分布的熵: H = -sum(p * log(p+eps))
    eps = 1e-8
    entropy = -(depth_np * np.log(depth_np + eps)).sum(axis=1).mean(axis=(1, 2))  # (N,)
    ax_e.bar(range(n_cams), entropy, color=cam_colors, alpha=0.85)
    ax_e.set_xticks(range(n_cams))
    ax_e.set_xticklabels(
        [c.replace("CAM_","").replace("_","\n") for c in _CAM_ORDER], fontsize=7)
    ax_e.set_title("Mean Depth Entropy per Camera", fontsize=9)
    ax_e.set_ylabel("entropy (nats)", fontsize=8)

    ax_i = fig.add_subplot(gs_stats[0, 2])
    ax_i.axis("off")
    info_lines = [
        f"Depth distribution: softmax over {D} bins",
        f"Bin range: {d_min:.1f}m → {d_max:.1f}m, step={depth_cfg[2]:.1f}m",
        f"Feature grid: {fH}×{fW} (stride 16 from 256×704)",
        f"Expected depth range: {float(expected_depth.min()):.1f}m — "
        f"{float(expected_depth.max()):.1f}m",
        f"Mean entropy: {float(entropy.mean()):.2f} nats "
        f"(max possible: {float(np.log(D)):.2f})",
    ]
    ax_i.text(0.05, 0.95, "\n".join(info_lines),
              transform=ax_i.transAxes, fontsize=9, va="top",
              fontfamily="monospace",
              bbox=dict(boxstyle="round", fc="lightyellow", ec="gray", alpha=0.8))

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → 已保存: {save_path}")


def visualize_lss_depth_3d_perspective(
    depth: torch.Tensor,
    img_inputs: tuple,
    save_path: str,
    depth_cfg: tuple = (1.0, 45.0, 0.5),
    n_cams: int = 6,
    input_size: tuple = (256, 704),
    title: str = "LSS Depth — 3D Ego Space",
):
    """将 LSS 期望深度投影到 3D ego 空间，按相机着色，绘制距离辅助圆环.

    对每个相机每个特征像素，计算期望深度 E[d] = sum(p_i * d_i)，
    然后反投影 (undo post_aug → K^{-1} → sensor2keyego) 到 ego 坐标系,
    在单张 3D 散点图上统一显示 6 个相机的深度预测点，不同相机用不同颜色.

    Args:
        depth: (B*N, D, fH, fW) softmax depth distribution
        img_inputs: img_inputs tuple (与 model(imgs, ...) 一致)
        save_path: 输出文件路径
        depth_cfg: (min_depth, max_depth, interval)
        n_cams: 相机数量
        input_size: 输入图像尺寸 (H_in, W_in)
        title: 标题
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from matplotlib.colors import to_rgba

    # ------ 数据准备 ------
    depth_np = depth.float().numpy()[:n_cams]  # (N, D, fH, fW)
    D, fH, fW = depth_np.shape[1], depth_np.shape[2], depth_np.shape[3]
    bins = _make_depth_bins(depth_cfg)  # (D,)

    # 期望深度 E[d] per pixel per camera
    expected_depth = (depth_np * bins[None, :, None, None]).sum(axis=1)  # (N, fH, fW)

    # ------ 相机参数 ------
    cam_params = _extract_cam_params_from_img_inputs(img_inputs)
    if cam_params is None:
        print("  [跳过] LSS depth 3D: 无法提取相机参数")
        return
    sensor2keyegos = cam_params["sensor2keyegos"]  # (N, 4, 4)
    intrins = cam_params["intrins"]                # (N, 3, 3)
    post_rots = cam_params["post_rots"]            # (N, 3, 3)
    post_trans = cam_params["post_trans"]           # (N, 3)

    H_in, W_in = input_size

    # 构建特征像素网格 (与模型 create_frustum 一致)
    u_grid = np.linspace(0, W_in - 1, fW, dtype=np.float32)
    v_grid = np.linspace(0, H_in - 1, fH, dtype=np.float32)
    uu, vv = np.meshgrid(u_grid, v_grid)  # (fH, fW) each
    ones_hw = np.ones((fH, fW), dtype=np.float32)
    aug_pts = np.stack([uu, vv, ones_hw], axis=-1).reshape(-1, 3)  # (fH*fW, 3)

    # ------ 反投影到 ego 空间 ------
    all_ego_pts: list[np.ndarray] = []   # per-camera ego points
    all_colors: list[str] = []

    for cam_idx in range(n_cams):
        cam_name = _CAM_ORDER[cam_idx]
        ed = expected_depth[cam_idx].flatten()  # (fH*fW,)

        K = intrins[cam_idx]           # (3, 3)
        post_rot = post_rots[cam_idx]  # (3, 3)
        post_tran = post_trans[cam_idx]  # (3,)
        s2ke = sensor2keyegos[cam_idx]   # (4, 4)

        # 1. 撤销 post augmentation: p_orig = inv(post_rot) @ (p_aug - post_trans)
        inv_post_rot = np.linalg.inv(post_rot)
        deaug = (aug_pts - post_tran[None, :]) @ inv_post_rot.T  # (M, 3)

        # 2. 乘以深度: [u'*d, v'*d, d]
        uvd = np.stack([deaug[:, 0] * ed, deaug[:, 1] * ed, ed], axis=-1)  # (M, 3)

        # 3. combine = R_sensor2keyego @ K^{-1},  p_ego = combine @ uvd + t
        K_inv = np.linalg.inv(K)
        R_c2e = s2ke[:3, :3]
        t_c2e = s2ke[:3, 3]
        combine = R_c2e @ K_inv  # (3, 3)
        p_ego = uvd @ combine.T + t_c2e[None, :]  # (M, 3)

        all_ego_pts.append(p_ego)
        all_colors.append(_CAM_INFO[cam_name]["color"])

    # ------ 绘图 ------
    bg_color = "#0d1117"
    fig = plt.figure(figsize=(16, 14), facecolor=bg_color)
    ax = fig.add_subplot(111, projection="3d", facecolor=bg_color)

    for cam_idx in range(n_cams):
        cam_name = _CAM_ORDER[cam_idx]
        pts = all_ego_pts[cam_idx]
        color = all_colors[cam_idx]
        ax.scatter(
            pts[:, 0], pts[:, 1], pts[:, 2],
            c=color, s=3.0, alpha=0.7,
            label=cam_name.replace("CAM_", ""),
            depthshade=False, rasterized=True,
        )

    # ------ 距离辅助圆环 (地面 z=0) ------
    theta = np.linspace(0, 2 * np.pi, 200)
    for r in [10, 20, 30, 40]:
        cx = r * np.cos(theta)
        cy = r * np.sin(theta)
        cz = np.zeros_like(theta)
        ax.plot(cx, cy, cz, color="white", alpha=0.35, linewidth=0.8)
        # 标签放在 +x 轴方向
        ax.text(r + 0.5, 0, 0, f"{r}m",
                color="white", fontsize=8, alpha=0.7,
                ha="left", va="center")

    # ego 标记
    ax.scatter([0], [0], [0], c="white", s=80, marker="*",
               zorder=10, edgecolors="none")

    # 参考线 (十字)
    lim = 48
    ax.plot([-lim, lim], [0, 0], [0, 0], color="white", alpha=0.15, lw=0.6)
    ax.plot([0, 0], [-lim, lim], [0, 0], color="white", alpha=0.15, lw=0.6)

    # ------ 轴 / 标签 / 视角 ------
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-4, 8)
    ax.set_xlabel("X / m  (front →)", color="white", fontsize=9, labelpad=8)
    ax.set_ylabel("Y / m  (← left)", color="white", fontsize=9, labelpad=8)
    ax.set_zlabel("Z / m  (up)", color="white", fontsize=9, labelpad=8)
    ax.view_init(elev=55, azim=-90)

    ax.tick_params(colors="white", labelsize=7)
    for spine in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        spine.fill = False
        spine.set_edgecolor((1, 1, 1, 0.08))
    ax.xaxis.line.set_color((1, 1, 1, 0.15))
    ax.yaxis.line.set_color((1, 1, 1, 0.15))
    ax.zaxis.line.set_color((1, 1, 1, 0.15))

    ax.set_title(title, fontsize=13, fontweight="bold", color="white", pad=18)
    leg = ax.legend(fontsize=9, loc="upper right", framealpha=0.3,
                    edgecolor="white", facecolor="#1a1a2e")
    for text in leg.get_texts():
        text.set_color("white")

    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
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
    _auto_scale_hist_xy(axes[1,1], vals)

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

    azim = np.radians(azim_deg); elev = np.radians(elev_deg)
    cos_a, sin_a = np.cos(azim), np.sin(azim)
    cos_e, sin_e = np.cos(elev), np.sin(elev)

    face_defs = [
        ( 1, 0, 0, np.array([[1,0,0],[1,1,0],[1,1,1],[1,0,1]], dtype=np.float64), 0.80),
        (-1, 0, 0, np.array([[0,0,0],[0,0,1],[0,1,1],[0,1,0]], dtype=np.float64), 0.70),
        ( 0, 1, 0, np.array([[0,1,0],[0,1,1],[1,1,1],[1,1,0]], dtype=np.float64), 0.65),
        ( 0,-1, 0, np.array([[0,0,0],[1,0,0],[1,0,1],[0,0,1]], dtype=np.float64), 0.75),
        ( 0, 0, 1, np.array([[0,0,1],[1,0,1],[1,1,1],[0,1,1]], dtype=np.float64), 1.00),
        ( 0, 0,-1, np.array([[0,0,0],[0,1,0],[1,1,0],[1,0,0]], dtype=np.float64), 0.55),
    ]

    all_p, all_c, all_d = [], [], []
    for dx, dy, dz, vtpl, shade in face_defs:
        xi, yi, zi = np.where(occupied)
        if len(xi) == 0: continue
        n = len(xi)
        orig = np.stack([xi, yi, zi], axis=-1).astype(np.float64)
        v3 = orig[:,None,:] + vtpl[None,:,:]
        v3[...,2] *= z_scale
        vx, vy, vz = v3[...,0], v3[...,1], v3[...,2]
        xr = vx*cos_a - vy*sin_a; yr = vx*sin_a + vy*cos_a
        v2 = np.stack([xr, yr*sin_e + vz*cos_e], axis=-1)
        ctr = orig.copy(); ctr[:,2] *= z_scale
        # depth = projection onto view_dir; 负sin_e因为相机在上方(+z), 高z更近
        dep = (ctr[:,0]*sin_a + ctr[:,1]*cos_a)*cos_e - ctr[:,2]*sin_e
        fc = np.ones((n,4)); fc[:,:3] = np.clip(_COLOR_LUT[occ[xi,yi,zi]]*shade, 0, 1)
        all_p.append(v2); all_c.append(fc); all_d.append(dep)

    if not all_p:
        ax.text(0.5, 0.5, "No exposed faces", transform=ax.transAxes,
                ha="center", va="center"); ax.axis("off"); return

    polys = np.concatenate(all_p); fc_ = np.concatenate(all_c)
    dep_  = np.concatenate(all_d)
    # Painter's algorithm: 从远到近绘制 (降序排列, 最远先画, 最近最后画在顶层)
    order = np.argsort(-dep_)
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


def _render_perspective_panel(ax, occ: np.ndarray, voxel_step=2,
                              heading_deg=0.0, elev_deg=8.0,
                              z_scale=1.5, fov_deg=90.0,
                              title="3D Perspective", fontsize=11):
    """从自车位置透视投影渲染体素 (Painter's Algorithm).

    Parameters:
        heading_deg: 相机朝向角 (从+x逆时针为正, °)
        elev_deg: 俯仰角 (正=向上看, °)
        z_scale: 高度方向拉伸系数
        fov_deg: 透视视场角 (°)
    """
    occ_sub = occ[::voxel_step, ::voxel_step, ::voxel_step]
    Dx, Dy, Dz = occ_sub.shape
    occupied = (occ_sub != 17) & (occ_sub != 0)
    if not occupied.any():
        ax.text(0.5, 0.5, "No occupied voxels", transform=ax.transAxes,
                ha="center", va="center", fontsize=14)
        ax.axis("off")
        ax.set_title(title, fontsize=fontsize, fontweight="bold")
        return

    heading = np.radians(heading_deg)
    elev = np.radians(elev_deg)
    cos_h, sin_h = np.cos(heading), np.sin(heading)
    cos_e, sin_e = np.cos(elev), np.sin(elev)

    # Camera at ego (grid center), at ~1.5m height
    # Z range: -1m to +5.4m (6.4m total), 1.5m => 39% through range
    ego_z_frac = (1.5 - _PCR[2]) / (_PCR[5] - _PCR[2])
    ego = np.array([Dx / 2.0, Dy / 2.0, ego_z_frac * Dz * z_scale])

    # Camera basis vectors (world → camera)
    cam_fwd = np.array([cos_h * cos_e, sin_h * cos_e, sin_e])
    cam_fwd /= np.linalg.norm(cam_fwd)
    cam_right = np.array([sin_h, -cos_h, 0.0])
    cam_right /= np.linalg.norm(cam_right)
    cam_up = np.cross(cam_right, cam_fwd)
    cam_up /= np.linalg.norm(cam_up)

    R = np.stack([cam_right, cam_up, cam_fwd])  # 3×3
    focal = 1.0 / np.tan(np.radians(fov_deg / 2.0))
    near_clip = 0.8
    proj_limit = 8.0

    face_defs = [
        ( 1, 0, 0, np.array([[1,0,0],[1,1,0],[1,1,1],[1,0,1]], dtype=np.float64), 0.80),
        (-1, 0, 0, np.array([[0,0,0],[0,0,1],[0,1,1],[0,1,0]], dtype=np.float64), 0.70),
        ( 0, 1, 0, np.array([[0,1,0],[0,1,1],[1,1,1],[1,1,0]], dtype=np.float64), 0.65),
        ( 0,-1, 0, np.array([[0,0,0],[1,0,0],[1,0,1],[0,0,1]], dtype=np.float64), 0.75),
        ( 0, 0, 1, np.array([[0,0,1],[1,0,1],[1,1,1],[0,1,1]], dtype=np.float64), 1.00),
        ( 0, 0,-1, np.array([[0,0,0],[0,1,0],[1,1,0],[1,0,0]], dtype=np.float64), 0.55),
    ]

    all_p, all_c, all_d = [], [], []
    for dx, dy, dz, vtpl, shade in face_defs:
        xi, yi, zi = np.where(occupied)
        if len(xi) == 0:
            continue
        n = len(xi)
        orig = np.stack([xi, yi, zi], axis=-1).astype(np.float64)

        # Vertex world positions (with z_scale)
        v3 = orig[:, None, :] + vtpl[None, :, :]  # (n, 4, 3)
        v3[..., 2] *= z_scale

        # Transform to camera space: v_cam = R @ (v3 - ego)
        v_rel = v3 - ego[None, None, :]
        v_cam = np.einsum('nvi,ji->nvj', v_rel, R)  # (n, 4, 3)

        # Depth along camera forward (index 2)
        # 对 z<=near 做 clamp, 不丢弃任何面
        depth_verts = v_cam[..., 2]
        any_visible = depth_verts.max(axis=1) > near_clip
        if not any_visible.any():
            continue

        v_cam = v_cam[any_visible]
        depth_verts = v_cam[..., 2]
        xi_f, yi_f, zi_f = xi[any_visible], yi[any_visible], zi[any_visible]
        n2 = v_cam.shape[0]

        # Perspective projection
        depth_safe = np.maximum(depth_verts, near_clip)
        x_proj = v_cam[..., 0] * focal / depth_safe
        y_proj = v_cam[..., 1] * focal / depth_safe
        v2 = np.stack([x_proj, y_proj], axis=-1)

        finite_mask = np.isfinite(v2).all(axis=(1, 2))
        bounded_mask = np.abs(v2).max(axis=(1, 2)) < proj_limit
        keep = finite_mask & bounded_mask
        if not keep.any():
            continue

        v2 = v2[keep]
        xi_f, yi_f, zi_f = xi_f[keep], yi_f[keep], zi_f[keep]
        n2 = v2.shape[0]

        # Center depth for painter's algorithm
        ctr = orig[any_visible][keep].copy() + 0.5
        ctr[:, 2] *= z_scale
        ctr_rel = ctr - ego[None, :]
        dep = ctr_rel @ R[2]  # dot with cam_fwd

        fc = np.ones((n2, 4))
        fc[:, :3] = np.clip(_COLOR_LUT[occ_sub[xi_f, yi_f, zi_f]] * shade, 0, 1)
        all_p.append(v2); all_c.append(fc); all_d.append(dep)

    if not all_p:
        ax.text(0.5, 0.5, "No visible faces", transform=ax.transAxes,
                ha="center", va="center")
        ax.axis("off")
        return

    polys = np.concatenate(all_p)
    fc_ = np.concatenate(all_c)
    dep_ = np.concatenate(all_d)
    order = np.argsort(-dep_)
    polys = polys[order]; fc_ = fc_[order]
    ec_ = fc_.copy(); ec_[:, :3] = np.clip(ec_[:, :3] * 0.5, 0, 1); ec_[:, 3] = 0.3

    ax.set_facecolor("#EEEEEE")
    pc = PolyCollection(polys, closed=True)
    pc.set_facecolor(fc_); pc.set_edgecolor(ec_); pc.set_linewidth(0.15)
    ax.add_collection(pc)
    ax.autoscale(); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(title, fontsize=fontsize, fontweight="bold", pad=8)


def visualize_occ_4panel(occ_gt, occ_pred: np.ndarray, save_path: str,
                          voxel_step: int = 1,
                          title: str = "3D Occupancy"):
    """4面板OCC可视化.

    布局 (has_gt=True):
        [GT 3D等轴测]    [Pred 3D等轴测]
        [GT BEV+辅助线]  [Pred BEV+辅助线]

    若 occ_gt=None: 左列显示 GT 缺失占位, 右列为 Pred (仍保持4格布局).
    """
    has_gt = occ_gt is not None
    n_cols = 2
    fig_w = 31

    fig = plt.figure(figsize=(fig_w, 26))
    fig.patch.set_facecolor("white")
    fig.suptitle(title, fontsize=18, fontweight="bold", y=0.998)

    gs = GridSpec(2, n_cols, figure=fig,
                  hspace=0.10, wspace=0.06,
                  top=0.975, bottom=0.09,
                  height_ratios=[1.15, 1])

    ax_gt_3d = fig.add_subplot(gs[0, 0])
    ax_gt_bev = fig.add_subplot(gs[1, 0])
    _render_occ_pair(
        ax_gt_3d, ax_gt_bev, occ_gt, voxel_step,
        iso_title="Ground Truth — 3D Isometric View",
        bev_title="Ground Truth — Bird's Eye View (BEV)",
        fontsize=12,
    )

    ax_pred_3d = fig.add_subplot(gs[0, 1])
    _render_isometric_panel(
        ax_pred_3d, occ_pred, voxel_step=max(voxel_step, 1),
        azim_deg=45.0, elev_deg=35.0, z_scale=2.5,
        title="Prediction — 3D Isometric View", fontsize=12)

    ax_pred_bev = fig.add_subplot(gs[1, 1])
    _render_bev_panel(ax_pred_bev, occ_pred,
                      title="Prediction — Bird's Eye View (BEV)",
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
    imgs = _decode_input_images_to_bgr01(img_inputs)
    N = imgs.shape[0]

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
        img = imgs[i]
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


def build_output_tag(sample_idx: int, checkpoint_path: str) -> str:
    """统一输出命名标签: s00000_<ckpt_stem>."""
    ckpt_stem = os.path.splitext(os.path.basename(checkpoint_path))[0]
    safe_ckpt = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in ckpt_stem)
    return f"s{sample_idx:05d}_{safe_ckpt}"


def build_output_path(out_dir: str, tag: str, name: str) -> str:
    """统一输出文件路径, 文件名格式: <tag>__<name>.png."""
    return os.path.join(out_dir, f"{tag}__{name}.png")


def _auto_scale_hist_xy(ax, values: np.ndarray | list[np.ndarray],
                        low_pct: float = 0.5, high_pct: float = 99.5):
    """对直方图做显式XY自动缩放，减少离群值对显示的影响."""
    if isinstance(values, list):
        arrays = [np.asarray(v).reshape(-1) for v in values if np.asarray(v).size > 0]
        if not arrays:
            return
        data = np.concatenate(arrays, axis=0)
    else:
        data = np.asarray(values).reshape(-1)

    data = data[np.isfinite(data)]
    if data.size == 0:
        return

    x_lo, x_hi = np.percentile(data, [low_pct, high_pct])
    if not np.isfinite(x_lo) or not np.isfinite(x_hi):
        return
    if x_hi <= x_lo:
        x_lo = float(data.min())
        x_hi = float(data.max())
        if x_hi <= x_lo:
            x_hi = x_lo + 1e-6

    ax.set_xlim(float(x_lo), float(x_hi))
    ax.relim(visible_only=True)
    ax.autoscale_view(scalex=False, scaley=True)
    ax.margins(x=0.02, y=0.10)


def _decode_input_images_to_bgr01(
    img_inputs: tuple,
    input_color_order: str | None = None,
) -> np.ndarray:
    """将 img_inputs 解码为 [N,H,W,3] 的 RGB 归一化图像 (0~1).

    参数:
        input_color_order: img_inputs 的通道顺序, 支持 "RGB"/"BGR"。
                          为空时使用全局 _VIS_INPUT_COLOR_ORDER。
    """
    imgs = img_inputs[0]
    if isinstance(imgs, torch.Tensor):
        imgs = imgs[0].cpu().float()

    from flashocc.constants import IMAGENET_MEAN, IMAGENET_STD

    mean = IMAGENET_MEAN.reshape(1, 1, 3)
    std = IMAGENET_STD.reshape(1, 1, 3)

    order = str(input_color_order or _VIS_INPUT_COLOR_ORDER).upper()
    if order not in {"RGB", "BGR"}:
        print(f"  [警告] 未识别的 image_color_order={order!r}, 回退为 RGB")
        order = "RGB"

    decoded = []
    for i in range(imgs.shape[0]):
        img = imgs[i].permute(1, 2, 0).numpy()
        img = img * std + mean
        # Matplotlib 以 RGB 显示；若输入为 BGR 则翻转到 RGB
        if order == "BGR":
            img = img[:, :, ::-1]
        decoded.append(np.clip(img / 255.0, 0, 1))
    return np.asarray(decoded)


def _render_occ_pair(
    ax_iso,
    ax_bev,
    occ: np.ndarray | None,
    voxel_step: int,
    iso_title: str,
    bev_title: str,
    fontsize: int,
):
    """渲染 OCC 的等轴测+BEV 两联图; occ 为 None 时绘制占位提示."""
    if occ is None:
        for ax, ttl in ((ax_iso, iso_title), (ax_bev, bev_title)):
            ax.axis("off")
            ax.text(0.5, 0.5, "Ground Truth Unavailable",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=16, fontweight="bold", color="gray")
            ax.set_title(ttl, fontsize=fontsize, fontweight="bold")
        return

    _render_isometric_panel(
        ax_iso, occ, voxel_step=max(voxel_step, 1),
        azim_deg=45.0, elev_deg=35.0, z_scale=2.5,
        title=iso_title, fontsize=fontsize)
    _render_bev_panel(
        ax_bev, occ,
        title=bev_title,
        draw_helpers=True, fontsize=fontsize)


def _draw_input_images_on_spec(fig, spec, img_inputs: tuple):
    """在给定 SubplotSpec 内绘制 6 相机输入与 BEV 布局示意."""
    imgs = _decode_input_images_to_bgr01(img_inputs)
    N = imgs.shape[0]

    sub = GridSpecFromSubplotSpec(2, 4, subplot_spec=spec,
                                  width_ratios=[1, 1, 1, 0.95],
                                  wspace=0.06, hspace=0.18)
    cam_grid = [
        ("CAM_FRONT_LEFT",  0, 0),
        ("CAM_FRONT",       0, 1),
        ("CAM_FRONT_RIGHT", 0, 2),
        ("CAM_BACK_LEFT",   1, 0),
        ("CAM_BACK",        1, 1),
        ("CAM_BACK_RIGHT",  1, 2),
    ]

    for i, (cam, row, col) in enumerate(cam_grid):
        ax = fig.add_subplot(sub[row, col])
        ax.axis("off")
        if i >= N:
            ax.set_facecolor("#f4f4f4")
            ax.text(0.5, 0.5, "N/A", transform=ax.transAxes,
                    ha="center", va="center", fontsize=10, color="gray")
            continue
        ax.imshow(imgs[i])
        info = _CAM_INFO[cam]
        ax.set_title(
            f"{cam.replace('CAM_','')}  (hdg={info['heading']:+.0f}°, HFoV={info['hfov']:.0f}°)",
            fontsize=8.5, fontweight="bold", color=info["color"],
            bbox=dict(boxstyle="round", fc="white", ec=info["color"], alpha=0.7))

    ax_bev = fig.add_subplot(sub[:, 3])
    ax_bev.set_facecolor("#1a1a2e")
    ax_bev.set_xlim(0, _DY)
    ax_bev.set_ylim(0, _DX)
    ax_bev.set_aspect("equal")
    ax_bev.axis("off")
    ax_bev.set_title("Camera Layout (BEV)", fontsize=10, color="white", fontweight="bold", pad=4)
    draw_camera_fovs(ax_bev, alpha_fill=0.13, alpha_line=0.9)
    draw_ego_vehicle(ax_bev, color="#E0E0FF", linewidth=2.2)


def visualize_scene_overview(img_inputs: tuple, occ_gt, occ_pred: np.ndarray,
                             save_path: str, voxel_step: int = 1,
                             title: str = "FlashOCC Scene Overview"):
    """单张总览图: 上部输入相机, 下部GT/Pred四宫格OCC."""
    fig = plt.figure(figsize=(30, 30), facecolor="white")
    gs = GridSpec(3, 2, figure=fig,
                  height_ratios=[1.05, 1.0, 1.0],
                  hspace=0.10, wspace=0.06,
                  top=0.97, bottom=0.04, left=0.02, right=0.98)

    _draw_input_images_on_spec(fig, gs[0, :], img_inputs)

    ax_gt_3d = fig.add_subplot(gs[1, 0])
    ax_gt_bev = fig.add_subplot(gs[2, 0])
    _render_occ_pair(
        ax_gt_3d, ax_gt_bev, occ_gt, voxel_step,
        iso_title="Ground Truth — 3D Isometric View",
        bev_title="Ground Truth — Bird's Eye View (BEV)",
        fontsize=12,
    )

    ax_pred_3d = fig.add_subplot(gs[1, 1])
    _render_isometric_panel(
        ax_pred_3d, occ_pred, voxel_step=max(voxel_step, 1),
        azim_deg=45.0, elev_deg=35.0, z_scale=2.5,
        title="Prediction — 3D Isometric View", fontsize=12)

    ax_pred_bev = fig.add_subplot(gs[2, 1])
    _render_bev_panel(ax_pred_bev, occ_pred,
                      title="Prediction — Bird's Eye View (BEV)",
                      draw_helpers=True, fontsize=12)

    fig.suptitle(title, fontsize=19, fontweight="bold", y=0.995)
    plt.savefig(save_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  → 已保存: {save_path}")


def _draw_input_cameras_compact(fig, spec, img_inputs: tuple):
    """在给定 SubplotSpec 内紧凑绘制 6 相机输入 (2×3), 不含BEV布局示意."""
    imgs = _decode_input_images_to_bgr01(img_inputs)
    N = imgs.shape[0]

    sub = GridSpecFromSubplotSpec(2, 3, subplot_spec=spec, wspace=0.04, hspace=0.12)
    cam_grid = [
        ("CAM_FRONT_LEFT",  0, 0),
        ("CAM_FRONT",       0, 1),
        ("CAM_FRONT_RIGHT", 0, 2),
        ("CAM_BACK_LEFT",   1, 0),
        ("CAM_BACK",        1, 1),
        ("CAM_BACK_RIGHT",  1, 2),
    ]

    for i, (cam, row, col) in enumerate(cam_grid):
        ax = fig.add_subplot(sub[row, col])
        ax.axis("off")
        if i >= N:
            ax.set_facecolor("#f4f4f4")
            ax.text(0.5, 0.5, "N/A", transform=ax.transAxes,
                    ha="center", va="center", fontsize=10, color="gray")
            continue
        ax.imshow(imgs[i])
        info = _CAM_INFO[cam]
        ax.set_title(
            f"{cam.replace('CAM_','')}",
            fontsize=9, fontweight="bold", color=info["color"],
            bbox=dict(boxstyle="round", fc="white", ec=info["color"], alpha=0.7))


def _extract_cam_params_from_img_inputs(img_inputs: tuple):
    """从 img_inputs 提取单样本相机投影参数.

    复现 BEVDet.prepare_inputs 的 sensor2keyegos 计算:
      sensor2keyegos = inv(ego2globals[0]) @ ego2globals @ sensor2egos
    确保投影参考帧与模型 OCC 网格完全一致 (key ego 帧).

    返回:
        dict 或 None, 字段:
          sensor2keyegos: (N, 4, 4)  cam→key_ego (与模型一致)
          intrins:        (N, 3, 3)
          post_rots:      (N, 3, 3)
          post_trans:     (N, 3)
    """
    if not isinstance(img_inputs, (list, tuple)) or len(img_inputs) < 6:
        return None

    def _pick_first_batch(x):
        if not isinstance(x, torch.Tensor):
            return None
        x = x.detach().cpu().float()
        if x.dim() == 4:   # (B,N,*,*)
            return x[0]
        if x.dim() == 3:   # (N,*,*)
            return x
        return None

    sensor2egos = _pick_first_batch(img_inputs[1])   # (N, 4, 4) cam→ego@cam
    ego2globals = _pick_first_batch(img_inputs[2])   # (N, 4, 4) ego@cam→global
    intrins = _pick_first_batch(img_inputs[3])       # (N, 3, 3)
    post_rots = _pick_first_batch(img_inputs[4])     # (N, 3, 3)

    post_trans_raw = img_inputs[5]
    post_trans = None
    if isinstance(post_trans_raw, torch.Tensor):
        t = post_trans_raw.detach().cpu().float()
        if t.dim() == 3:   # (B,N,3)
            post_trans = t[0]
        elif t.dim() == 2: # (N,3)
            post_trans = t

    if any(v is None for v in [sensor2egos, ego2globals, intrins, post_rots, post_trans]):
        return None

    # 复现 BEVDet.prepare_inputs: sensor2keyegos = inv(ego2globals[0]) @ ego2globals @ sensor2egos
    keyego2global = ego2globals[0:1].double()                     # (1, 4, 4)
    global2keyego = torch.linalg.inv(keyego2global)               # (1, 4, 4)
    sensor2keyegos = (global2keyego @ ego2globals.double()
                      @ sensor2egos.double()).float()              # (N, 4, 4)

    return {
        "sensor2keyegos": sensor2keyegos.numpy(),
        "intrins": intrins.numpy(),
        "post_rots": post_rots.numpy(),
        "post_trans": post_trans.numpy(),
    }


def _overlay_occ_on_camera_image(
    ax,
    img_rgb: np.ndarray,
    occ: np.ndarray,
    cam_params: dict | None,
    scene_info: dict | None,
    cam_name: str,
    cam_idx: int,
    voxel_step: int = 2,
    alpha: float = 0.28,
):
    """将 3D 体素按立方体(6面)投影并半透明叠加到图像.

    说明:
        - 每个体素都按完整六面体处理, 绘制全部6个面
        - 不做任何面剔除 (无朝向剔除/邻接剔除/近面裁剪)
        - 对 z<=0 的顶点做 clamp 防止除零
        - 统一收集所有面后一次渲染 (painter's algorithm)
    """
    ax.imshow(img_rgb)
    if cam_params is None:
        return

    step = max(int(voxel_step), 1)
    occ_sub = occ[::step, ::step, ::step]
    mask = (occ_sub != 17) & (occ_sub != 0)
    if not mask.any():
        return

    xi, yi, zi = np.where(mask)
    cls_ids = occ_sub[xi, yi, zi].astype(np.int32)

    # OCC 体素坐标系:
    #   模型 prepare_inputs 计算 sensor2keyegos = inv(ego2globals[0]) @ ego2globals @ sensor2egos
    #   view transformer 的 get_ego_coor 用 sensor2keyegos 将视锥投到 key ego 帧.
    #   OCC 网格坐标 (x,y,z) 直接在 key ego 帧下.
    #
    # 投影链 (key_ego → cam → pixel):
    #   ego2cam = inv(sensor2keyegos[cam_idx])   # key_ego → cam
    #   p_cam = ego2cam[:3,:3] @ p_ego + ego2cam[:3,3]
    #   (u,v) = K @ (p_cam / p_cam.z)
    #   (u',v') = post_rot @ (u,v,1) + post_trans

    if cam_params is None or cam_idx >= cam_params["sensor2keyegos"].shape[0]:
        return

    # key_ego → cam: 直接取 inv(sensor2keyegos[cam_idx])
    sensor2keyego = cam_params["sensor2keyegos"][cam_idx]       # (4,4) cam→key_ego
    ego2cam = np.linalg.inv(sensor2keyego.astype(np.float64)).astype(np.float32)
    K = cam_params["intrins"][cam_idx]        # (3,3)
    post_rot = cam_params["post_rots"][cam_idx]     # (3,3)
    post_tran = cam_params["post_trans"][cam_idx]   # (3,)
    voxel_size = _VS * step

    H, W = img_rgb.shape[:2]

    max_voxels = 22000
    if xi.size > max_voxels:
        keep = np.linspace(0, xi.size - 1, max_voxels, dtype=np.int32)
        xi = xi[keep]
        yi = yi[keep]
        zi = zi[keep]
        cls_ids = cls_ids[keep]

    # 体素下角点 (ego)
    x0 = _PCR[0] + xi.astype(np.float32) * voxel_size
    y0 = _PCR[1] + yi.astype(np.float32) * voxel_size
    z0 = _PCR[2] + zi.astype(np.float32) * voxel_size
    origins = np.stack([x0, y0, z0], axis=1)  # (N,3)

    # 8 corner offsets in [0,1]^3
    local_corners = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
    ], dtype=np.float32) * voxel_size

    # six faces (quad corner indices)
    face_indices = np.array([
        [0, 1, 2, 3],  # bottom
        [4, 5, 6, 7],  # top
        [0, 1, 5, 4],  # front-x
        [3, 2, 6, 7],  # back-x
        [0, 3, 7, 4],  # left-y
        [1, 2, 6, 5],  # right-y
    ], dtype=np.int32)
    face_shades = np.array([0.62, 0.95, 0.82, 0.74, 0.78, 0.70], dtype=np.float32)

    R_e2c = ego2cam[:3, :3]   # (3,3)
    t_e2c = ego2cam[:3, 3]    # (3,)

    # (N,8,3) corners in key ego frame, then to camera
    corners_ego = origins[:, None, :] + local_corners[None, :, :]
    corners_cam = (corners_ego.reshape(-1, 3) @ R_e2c.T + t_e2c[None, :]).reshape(corners_ego.shape)

    # ---- 所有面直接投影, 不做任何面剔除 ----
    # 对 z <= 0 的顶点做最小正值 clamp 防止除零, 不丢弃任何面.
    near_eps = 0.01

    def _project_face_cam_to_uv(face_pts_cam):
        """(M, 4, 3) 相机坐标 → (M, 4, 2) 增广像素坐标."""
        pts = face_pts_cam.copy()
        pts[..., 2] = np.maximum(pts[..., 2], near_eps)
        uvw = pts @ K.T
        u = uvw[..., 0] / uvw[..., 2]
        v = uvw[..., 1] / uvw[..., 2]
        uv1 = np.stack([u, v, np.ones_like(u)], axis=-1)
        uv_aug = uv1 @ post_rot.T + post_tran[None, None, :]
        return uv_aug[..., :2]

    polys_all = []
    cols_all = []
    dep_all = []

    for fidx in range(face_indices.shape[0]):
        idx4 = face_indices[fidx]
        face_cam = corners_cam[:, idx4, :]   # (N, 4, 3)

        # 至少一个顶点 z > 0 才投影 (完全在身后的面跳过)
        any_visible = face_cam[..., 2].max(axis=1) > near_eps
        if not any_visible.any():
            continue

        fc = face_cam[any_visible]   # (M, 4, 3)
        cf = cls_ids[any_visible]

        poly = _project_face_cam_to_uv(fc)  # (M, 4, 2)

        finite = np.isfinite(poly).all(axis=(1, 2))
        if not finite.any():
            continue
        poly = poly[finite]
        cf = cf[finite]
        fc = fc[finite]

        # 视口相交检测
        x_min = poly[..., 0].min(axis=1)
        x_max = poly[..., 0].max(axis=1)
        y_min = poly[..., 1].min(axis=1)
        y_max = poly[..., 1].max(axis=1)
        vis = (x_max >= 0) & (x_min < W) & (y_max >= 0) & (y_min < H)
        if not vis.any():
            continue

        poly = poly[vis]
        cf = cf[vis]
        fc = fc[vis]

        color = np.clip(_COLOR_LUT[np.clip(cf, 0, 255)] * face_shades[fidx], 0, 1)
        depth = fc[..., 2].mean(axis=1)

        polys_all.append(poly)
        cols_all.append(color)
        dep_all.append(depth)

    if not polys_all:
        return

    polys = np.concatenate(polys_all, axis=0)
    cols = np.concatenate(cols_all, axis=0)
    dep = np.concatenate(dep_all, axis=0)

    order = np.argsort(-dep)   # 远到近 painter
    polys = polys[order]
    cols = cols[order]

    # 关键: 先离屏渲染完整 overlay (不透明), 再在主图只做一次 alpha 叠加
    face_rgba = np.concatenate([
        cols,
        np.ones((cols.shape[0], 1), dtype=np.float32)
    ], axis=1)
    edge_rgba = np.concatenate([
        np.clip(cols * 0.55, 0.0, 1.0),
        np.ones((cols.shape[0], 1), dtype=np.float32)
    ], axis=1)

    dpi = 100
    fig_w = max(W / dpi, 1e-3)
    fig_h = max(H / dpi, 1e-3)
    fig_tmp = plt.figure(figsize=(fig_w, fig_h), dpi=dpi, facecolor=(0, 0, 0, 0))
    ax_tmp = fig_tmp.add_axes([0, 0, 1, 1])
    ax_tmp.set_axis_off()
    ax_tmp.set_xlim(-0.5, W - 0.5)
    ax_tmp.set_ylim(H - 0.5, -0.5)
    ax_tmp.set_facecolor((0, 0, 0, 0))

    pc = PolyCollection(
        polys,
        closed=True,
        facecolors=face_rgba,
        edgecolors=edge_rgba,
        linewidths=0.12,
    )
    ax_tmp.add_collection(pc)

    fig_tmp.canvas.draw()
    buf = np.asarray(fig_tmp.canvas.buffer_rgba()).astype(np.float32) / 255.0
    plt.close(fig_tmp)

    overlay_rgb = buf[..., :3]
    overlay_a = buf[..., 3]
    if overlay_a.max() <= 0:
        return

    # 单次全局 alpha 叠加
    ax.imshow(overlay_rgb, alpha=overlay_a * alpha)

    # 调试: 标注相机在 key ego 帧中的位置
    try:
        cam_pos_ego = sensor2keyego[:3, 3]   # cam→key_ego 的平移 = 相机在 key ego 中的位置
        z_txt = (f"cam@keyego=({cam_pos_ego[0]:+.1f},"
                 f"{cam_pos_ego[1]:+.1f},{cam_pos_ego[2]:+.1f})m")
        ax.text(
            0.01, 0.98, z_txt,
            transform=ax.transAxes, ha="left", va="top",
            fontsize=7, color="yellow",
            bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.5),
        )
    except Exception:
        pass


def visualize_combined(img_inputs: tuple, occ_gt, occ_pred: np.ndarray,
                       scene_info: dict | None,
                       save_path: str, voxel_step: int = 1,
                       title: str = "FlashOCC Combined Visualization"):
    """综合可视化: 6相机输入+体素半透明叠加 + 预测/真值 等轴测与BEV.

    布局 (3行):
      Row 0: 前方3相机输入 + 体素叠加 (FL, F, FR) — 1×3
      Row 1: 后方3相机输入 + 体素叠加 (BL, B, BR) — 1×3
      Row 2: 预测等轴测 | 预测BEV | 真值等轴测 | 真值BEV  (1×4)
    """
    has_gt = occ_gt is not None

    fig = plt.figure(figsize=(30, 22), facecolor="white")
    fig.suptitle(title, fontsize=17, fontweight="bold", y=0.998)

    gs_main = GridSpec(3, 1, figure=fig,
                       height_ratios=[0.48, 0.48, 0.94],
                       hspace=0.035,
                       top=0.976, bottom=0.03, left=0.02, right=0.98)

    # ---- 图像解码 ----
    imgs = _decode_input_images_to_bgr01(img_inputs)
    N = imgs.shape[0]
    cam_params = _extract_cam_params_from_img_inputs(img_inputs)

    front_cams = [("CAM_FRONT_LEFT", 0), ("CAM_FRONT", 1), ("CAM_FRONT_RIGHT", 2)]
    back_cams  = [("CAM_BACK_LEFT", 3), ("CAM_BACK", 4), ("CAM_BACK_RIGHT", 5)]

    def _draw_cam_row(spec, cam_list):
        """在给定 SubplotSpec 内绘制一行相机图像 + 体素叠加 (1×3)."""
        gs = GridSpecFromSubplotSpec(1, 3, subplot_spec=spec, wspace=0.03)
        for col_idx, (cam, cam_idx) in enumerate(cam_list):
            ax = fig.add_subplot(gs[0, col_idx])
            ax.axis("off")
            if cam_idx >= N:
                continue
            _overlay_occ_on_camera_image(
                ax,
                imgs[cam_idx],
                occ_pred,
                cam_params,
                scene_info,
                cam_name=cam,
                cam_idx=cam_idx,
                voxel_step=max(voxel_step, 2),
                alpha=0.5,
            )
            info = _CAM_INFO[cam]
            ax.set_title(
                f"{cam.replace('CAM_','')}  (hdg={info['heading']:+.0f}°, "
                f"HFoV={info['hfov']:.0f}°)  + OCC Overlay",
                fontsize=9, fontweight="bold", color=info["color"],
                bbox=dict(boxstyle="round", fc="white", ec=info["color"], alpha=0.7))

    # Row 0: 前方相机输入
    _draw_cam_row(gs_main[0], front_cams)
    # Row 1: 后方相机输入
    _draw_cam_row(gs_main[1], back_cams)

    # ================================================================
    #  Row 2: 预测等轴测 | 预测BEV | 真值等轴测 | 真值BEV (1×4)
    # ================================================================
    gs_bot = GridSpecFromSubplotSpec(1, 4, subplot_spec=gs_main[2],
                                     wspace=0.06)

    ax_pred_3d = fig.add_subplot(gs_bot[0])
    _render_isometric_panel(
        ax_pred_3d, occ_pred, voxel_step=max(voxel_step, 1),
        azim_deg=45.0, elev_deg=35.0, z_scale=2.5,
        title="Prediction — 3D Isometric", fontsize=11)

    ax_pred_bev = fig.add_subplot(gs_bot[1])
    _render_bev_panel(ax_pred_bev, occ_pred,
                      title="Prediction — BEV",
                      draw_helpers=True, fontsize=11)

    ax_gt_3d = fig.add_subplot(gs_bot[2])
    ax_gt_bev = fig.add_subplot(gs_bot[3])
    if has_gt:
        _render_occ_pair(
            ax_gt_3d, ax_gt_bev, occ_gt, voxel_step,
            iso_title="Ground Truth — 3D Isometric",
            bev_title="Ground Truth — BEV",
            fontsize=11,
        )
    else:
        ax_gt_3d.axis("off")
        ax_gt_3d.text(0.5, 0.5, "Ground Truth\nUnavailable",
                      transform=ax_gt_3d.transAxes, ha="center", va="center",
                      fontsize=14, fontweight="bold", color="gray")
        ax_gt_3d.set_title("Ground Truth — 3D Isometric",
                           fontsize=11, fontweight="bold")
        ax_gt_bev.axis("off")
        ax_gt_bev.text(0.5, 0.5, "Ground Truth\nUnavailable",
                       transform=ax_gt_bev.transAxes, ha="center", va="center",
                       fontsize=14, fontweight="bold", color="gray")
        ax_gt_bev.set_title("Ground Truth — BEV",
                            fontsize=11, fontweight="bold")

    # ---- 类别图例 ----
    all_cls: set = set(int(c) for c in occ_pred[(occ_pred != 17) & (occ_pred != 0)])
    if has_gt:
        all_cls |= set(int(c) for c in occ_gt[(occ_gt != 17) & (occ_gt != 0)])
    cls_patches = [
        Patch(facecolor=OCC_CLASS_COLORS.get(c, (0.5, 0.5, 0.5)),
              edgecolor="k", lw=0.4,
              label=f"{OCC_CLASS_NAMES[c] if c < len(OCC_CLASS_NAMES) else c}")
        for c in sorted(all_cls)
    ]
    if cls_patches:
        fig.legend(handles=cls_patches, loc="lower center",
                   ncol=min(10, len(cls_patches)), fontsize=8,
                   frameon=True, fancybox=True, bbox_to_anchor=(0.5, 0.005))

    plt.savefig(save_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  → 已保存: {save_path}")


def visualize_bev_diff_heatmap(occ_gt: np.ndarray, occ_pred: np.ndarray,
                                save_path: str,
                                title: str = "BEV Prediction vs Ground Truth Difference"):
    """BEV视角下预测与真值的差异热力图.

    对每个BEV位置(x,y), 统计沿Z轴的体素差异:
      - 正确 (TP): 预测=GT, 且非free/others (绿色)
      - 漏检 (FN): GT非free, 预测为free/others (蓝色)
      - 误检 (FP): 预测非free, GT为free/others (红色)
      - 混淆 (Confusion): 两者均非free但类别不同 (橙色)

    左: 分类差异图 (按最严重错误类型着色)
    右: 差异数量热力图 (沿Z轴不匹配体素数)
    """
    Dx, Dy, Dz = occ_gt.shape
    assert occ_pred.shape == occ_gt.shape, \
        f"Shape mismatch: GT {occ_gt.shape} vs Pred {occ_pred.shape}"

    gt_free = (occ_gt == 17) | (occ_gt == 0)
    pred_free = (occ_pred == 17) | (occ_pred == 0)

    tp   = (~gt_free) & (~pred_free) & (occ_gt == occ_pred)
    fn   = (~gt_free) & pred_free
    fp   = gt_free & (~pred_free)
    conf = (~gt_free) & (~pred_free) & (occ_gt != occ_pred)

    tp_z   = tp.sum(axis=2)
    fn_z   = fn.sum(axis=2)
    fp_z   = fp.sum(axis=2)
    conf_z = conf.sum(axis=2)
    total_err_z = fn_z + fp_z + conf_z

    # Categorical BEV: worst error per cell
    cat_bev = np.zeros((Dx, Dy), dtype=np.uint8)
    cat_bev[tp_z   > 0] = 1
    cat_bev[fp_z   > 0] = 2
    cat_bev[fn_z   > 0] = 3
    cat_bev[conf_z > 0] = 4

    cat_colors = {
        0: (0.95, 0.95, 0.95),
        1: (0.2,  0.8,  0.2),
        2: (0.9,  0.2,  0.2),
        3: (0.2,  0.4,  0.9),
        4: (1.0,  0.6,  0.0),
    }
    cat_rgb = np.zeros((Dx, Dy, 3))
    for k, color in cat_colors.items():
        cat_rgb[cat_bev == k] = color

    fig = plt.figure(figsize=(28, 13), facecolor="white")
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.99)
    gs = GridSpec(1, 3, figure=fig, wspace=0.12,
                  width_ratios=[1, 1, 0.08],
                  top=0.92, bottom=0.10, left=0.04, right=0.96)

    # Left: Category map
    ax_cat = fig.add_subplot(gs[0, 0])
    ax_cat.imshow(cat_rgb, origin="lower", aspect="equal")
    draw_camera_fovs(ax_cat, alpha_fill=0.06, alpha_line=0.5)
    draw_ego_vehicle(ax_cat, linewidth=1.5, color="#1E90FF")
    add_bev_annotations(ax_cat, title="Error Category Map (BEV)", fontsize=12)

    cat_patches = [
        Patch(facecolor=cat_colors[1], label=f"Correct (TP): {int(tp_z.sum()):,}"),
        Patch(facecolor=cat_colors[2], label=f"False Positive (FP): {int(fp_z.sum()):,}"),
        Patch(facecolor=cat_colors[3], label=f"False Negative (FN): {int(fn_z.sum()):,}"),
        Patch(facecolor=cat_colors[4], label=f"Confusion: {int(conf_z.sum()):,}"),
        Patch(facecolor=cat_colors[0], edgecolor="gray", label="Free / Empty"),
    ]
    ax_cat.legend(handles=cat_patches, loc="lower left", fontsize=8,
                  frameon=True, fancybox=True)

    # Right: Error count heatmap
    ax_heat = fig.add_subplot(gs[0, 1])
    max_err = max(int(total_err_z.max()), 1)
    im = ax_heat.imshow(total_err_z, origin="lower", aspect="equal",
                        cmap="YlOrRd", vmin=0, vmax=max_err)
    draw_camera_fovs(ax_heat, alpha_fill=0.06, alpha_line=0.5)
    draw_ego_vehicle(ax_heat, linewidth=1.5, color="#1E90FF")
    add_bev_annotations(ax_heat,
                        title="Error Count Heatmap (sum along Z)", fontsize=12)

    ax_cb = fig.add_subplot(gs[0, 2])
    plt.colorbar(im, cax=ax_cb, label="Mismatched voxels along Z")

    # Summary stats
    total_occupied_gt   = int((~gt_free).sum())
    total_occupied_pred = int((~pred_free).sum())
    tp_total = int(tp.sum())
    accuracy = tp_total / max(total_occupied_gt, 1)
    fig.text(
        0.5, 0.03,
        f"Total voxels: {occ_gt.size:,}  |  "
        f"GT occupied: {total_occupied_gt:,}  |  "
        f"Pred occupied: {total_occupied_pred:,}  |  "
        f"TP: {tp_total:,}  |  FP: {int(fp.sum()):,}  |  "
        f"FN: {int(fn.sum()):,}  |  Confusion: {int(conf.sum()):,}  |  "
        f"Occupied accuracy: {accuracy:.1%}",
        ha="center", fontsize=9, fontfamily="monospace",
        bbox=dict(boxstyle="round", fc="lightyellow", ec="gray", alpha=0.8))

    plt.savefig(save_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  → 已保存: {save_path}")


def _get_batch_by_index(data_loader, sample_idx: int):
    """从 dataloader 中按索引取 batch, 不存在则返回 None."""
    for i, batch in enumerate(data_loader):
        if i == sample_idx:
            return batch
    return None


def _decode_batch_if_needed(batch: dict, color_order: str) -> dict:
    """当 batch 含 jpeg_bytes 时执行 DALI 解码."""
    if "jpeg_bytes" not in batch:
        return batch

    from flashocc.datasets.dali_decode import dali_decode_batch

    for key in ("jpeg_bytes", "img_aug_params"):
        if key in batch and hasattr(batch[key], "data"):
            batch[key] = batch[key].data
    return dali_decode_batch(batch, color_order=color_order)


def _to_device_recursive(obj, device: torch.device):
    """将 Tensor / list / tuple 中的 Tensor 迁移到指定 device."""
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, list):
        return [_to_device_recursive(x, device) for x in obj]
    if isinstance(obj, tuple):
        return tuple(_to_device_recursive(x, device) for x in obj)
    return obj


def _normalize_img_metas(batch: dict) -> list[dict]:
    """统一 img_metas 为 list[dict] 结构."""
    img_metas = batch.get("img_metas", [{}])
    if hasattr(img_metas, "data"):
        img_metas = img_metas.data[0]
    if isinstance(img_metas, dict):
        return [img_metas]
    if isinstance(img_metas, list):
        return img_metas
    return [{}]


def _prepare_points(batch: dict, device: torch.device):
    """提取并迁移 points 到 device; 不存在时返回 None."""
    points = batch.get("points", None)
    if points is None:
        return None

    if hasattr(points, "data"):
        points = points.data[0]
    return _to_device_recursive(points, device)


def _extract_occ_prediction(results) -> np.ndarray | None:
    """从 forward_test 返回结果中解析占用预测 (Dx,Dy,Dz)."""
    occ_pred = None
    if isinstance(results, torch.Tensor):
        tensor = results[0] if results.dim() == 4 else results
        occ_pred = tensor.detach().cpu().numpy()
    elif isinstance(results, np.ndarray):
        occ_pred = results[0] if results.ndim == 4 else results
    elif isinstance(results, (list, tuple)) and results:
        item = results[0]
        if isinstance(item, np.ndarray):
            occ_pred = item
        elif isinstance(item, torch.Tensor):
            occ_pred = item.detach().cpu().numpy()
        elif isinstance(item, dict):
            for key in ["occ", "occ_pred", "occ_preds", "pts_bbox"]:
                if key not in item:
                    continue
                value = item[key]
                occ_pred = (value.detach().cpu().numpy()
                            if isinstance(value, torch.Tensor) else value)
                break

    if occ_pred is None:
        return None
    return occ_pred.astype(np.uint8)


def _extract_occ_gt(batch: dict) -> np.ndarray | None:
    """从 batch 中提取 voxel_semantics 作为 GT 占用体素."""
    if "voxel_semantics" not in batch:
        return None

    gt_data = batch["voxel_semantics"]
    if hasattr(gt_data, "data"):
        gt_data = gt_data.data[0]

    if isinstance(gt_data, np.ndarray):
        return gt_data.astype(np.uint8)

    if isinstance(gt_data, torch.Tensor):
        return gt_data[0].detach().cpu().numpy().astype(np.uint8)
    if isinstance(gt_data, list) and gt_data and isinstance(gt_data[0], torch.Tensor):
        return gt_data[0].detach().cpu().numpy().astype(np.uint8)
    return None


def _extract_occ_gt_from_dataset(dataset, sample_idx: int) -> np.ndarray | None:
    """从 dataset.data_infos[sample_idx]['occ_path']/labels.npz 回退加载 GT."""
    try:
        infos = getattr(dataset, "data_infos", None)
        if infos is None or sample_idx >= len(infos):
            return None
        occ_path = infos[sample_idx].get("occ_path", None)
        if not occ_path:
            return None
        labels_path = os.path.join(occ_path, "labels.npz")
        if not os.path.isfile(labels_path):
            return None
        labels = np.load(labels_path)
        if "semantics" not in labels:
            return None
        return labels["semantics"].astype(np.uint8)
    except Exception as e:
        print(f"  [GT回退加载失败] {e}")
        return None


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
    global _VIS_INPUT_COLOR_ORDER

    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)
    out_tag = build_output_tag(args.sample_idx, args.checkpoint)

    print("=" * 65)
    print("  FlashOCC 可视化工具  v2  (重新设计版)")
    print("=" * 65)

    print(f"\n[1/5] 加载配置: {args.config}")
    exp = load_experiment(args.config)
    _VIS_INPUT_COLOR_ORDER = str(getattr(exp, "image_color_order", "RGB")).upper()
    print(f"      image_color_order={_VIS_INPUT_COLOR_ORDER} (visualization auto-adapt)")

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
    batch = _get_batch_by_index(data_loader, args.sample_idx)
    if batch is None:
        print(f"  [错误] 无法获取样本 {args.sample_idx}"); return

    print("[5/5] 执行前向推理 ...")
    batch = _decode_batch_if_needed(batch, color_order=exp.image_color_order)

    img_inputs = batch["img_inputs"]
    img_inputs_gpu = _to_device_recursive(img_inputs, device)
    img_metas = _normalize_img_metas(batch)
    points = _prepare_points(batch, device)

    results = model.forward_test(
        points=points, img_inputs=img_inputs_gpu, img_metas=img_metas)

    vis_in = img_inputs_gpu if isinstance(img_inputs_gpu, (list,tuple)) else (img_inputs_gpu,)

    # ---- 可视化1: 中间层特征 (可选) ----
    if not args.no_feat:
        print("\n--- [可视化] 中间层特征 ---")
        try:
            for idx, (name, (layer_type, feat)) in enumerate(extractor.features.items()):
                print(f"  [{idx+1}/{len(extractor.features)}] {name}  "
                      f"type={layer_type}  shape={list(feat.shape)}")
                safe = (name.replace(" ","_").replace("/","_")
                           .replace("(","").replace(")","").replace(".","_"))
                sp = build_output_path(args.out_dir, out_tag, f"feat_{idx+1:02d}_{safe}")
                try:
                    if layer_type == "multicam":
                        visualize_multicam_features(feat, name, sp, n_cams=6)
                    elif layer_type == "occ_head":
                        visualize_occ_head(feat, name, sp)
                    else:
                        visualize_feature_map_bev(feat, name, sp)
                except Exception as e:
                    import traceback; traceback.print_exc(); print(f"    [跳过] {e}")

            # ---- LSS 深度预测可视化 ----
            if extractor.depth is not None:
                print(f"\n--- [可视化] LSS 深度预测 ---")
                print(f"  depth shape: {list(extractor.depth.shape)}")
                depth_cfg = (1.0, 45.0, 0.5)  # 默认 grid_config.depth
                try:
                    gc = getattr(exp, "grid_config", None)
                    if gc is not None:
                        depth_cfg = tuple(gc.depth)
                except Exception:
                    pass

                sp_d2d = build_output_path(args.out_dir, out_tag, "lss_depth_2d")
                try:
                    visualize_lss_depth_2d(
                        extractor.depth, vis_in, sp_d2d,
                        depth_cfg=depth_cfg, n_cams=6)
                except Exception as e:
                    import traceback; traceback.print_exc()
                    print(f"    [跳过] LSS depth 2D: {e}")

                sp_d3d = build_output_path(args.out_dir, out_tag, "lss_depth_3d")
                try:
                    visualize_lss_depth_3d_perspective(
                        extractor.depth, vis_in, sp_d3d,
                        depth_cfg=depth_cfg, n_cams=6)
                except Exception as e:
                    import traceback; traceback.print_exc()
                    print(f"    [跳过] LSS depth 3D: {e}")
            else:
                print("  [注意] 未捕获到 LSS 深度, 跳过深度可视化")
        finally:
            extractor.remove()

    # ---- 可视化2: 综合可视化 (输入相机 + 3D OCC + BEV) ----
    if not args.no_occ:
        print("\n--- [可视化] 综合可视化 (输入相机 + 3D占用预测 + BEV) ---")
        occ_pred = _extract_occ_prediction(results)
        if occ_pred is None:
            print("  [错误] 无法提取占用预测"); return

        print(f"  OCC shape: {occ_pred.shape}")
        print_occ_stats(occ_pred, "Prediction")

        occ_gt = _extract_occ_gt(batch)
        if occ_gt is None:
            print("  [诊断] batch 中无 voxel_semantics (当前 test_pipeline 仅收集 img_inputs/jpeg_bytes/img_aug_params)")
            occ_gt = _extract_occ_gt_from_dataset(dataset, args.sample_idx)
            if occ_gt is not None:
                print("  [诊断] 已从 dataset.data_infos[*].occ_path/labels.npz 回退加载 GT")
            else:
                print("  [诊断] 回退加载 GT 失败: 请检查 occ_path/labels.npz 是否存在")

        if occ_gt is not None:
            print(f"  GT shape: {occ_gt.shape}")
            print_occ_stats(occ_gt, "Ground Truth")

        # 生成单一综合可视化图
        combined_path = build_output_path(args.out_dir, out_tag, "combined")
        visualize_combined(
            img_inputs=vis_in,
            occ_gt=occ_gt,
            occ_pred=occ_pred,
            scene_info=dataset.data_infos[args.sample_idx],
            save_path=combined_path,
            voxel_step=args.voxel_step,
            title="FlashOCC — Input Cameras + 3D Occupancy Prediction",
        )

        # 生成 BEV 差异热力图 (需要GT)
        if occ_gt is not None:
            diff_path = build_output_path(args.out_dir, out_tag, "bev_diff")
            visualize_bev_diff_heatmap(
                occ_gt=occ_gt,
                occ_pred=occ_pred,
                save_path=diff_path,
                title="FlashOCC — BEV Prediction vs Ground Truth Difference",
            )

    print("\n" + "=" * 65)
    print(f"  可视化完成!  结果: {args.out_dir}/")
    print("=" * 65)
    for f in sorted(os.listdir(args.out_dir)):
        if os.path.isfile(os.path.join(args.out_dir, f)):
            print(f"    {f}")


if __name__ == "__main__":
    main()
