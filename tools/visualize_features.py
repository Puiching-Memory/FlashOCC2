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
import re
import sys
from collections import OrderedDict
from typing import Any

import numpy as np
import cv2
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

# ── 新的统一 3D 可视化引擎 ──────────────────────────────────────────
from flashocc.vis import (
    OccGrid,
    OccVoxelRenderer,
    OCC_CLASS_COLORS,
    COLOR_LUT as _COLOR_LUT,
    cls_to_rgb as _cls_to_rgb,
    build_class_legend,
    build_cam_legend_patches,
    CameraParams,
    PerspectiveProjection,
    CAM_ORDER as _CAM_ORDER,
    CAM_INFO as _CAM_INFO,
    ZOE_LENGTH as _ZOE_LENGTH,
    ZOE_WIDTH as _ZOE_WIDTH,
    ZOE_WHEELBASE as _ZOE_WB,
    draw_ego_vehicle as _draw_ego_vehicle_vis,
    draw_camera_fovs as _draw_camera_fovs_vis,
    add_bev_annotations as _add_bev_annotations_vis,
)


# =====================================================================
#  全局常量 — 从 constants / vis 统一引用, 不再重复定义
# =====================================================================

_PCR = POINT_CLOUD_RANGE    # [-40, -40, -1, 40, 40, 5.4]
_VS  = VOXEL_SIZE            # 0.4

# OCC格网维度
_DX = int((_PCR[3] - _PCR[0]) / _VS)  # 200
_DY = int((_PCR[4] - _PCR[1]) / _VS)  # 200
_DZ = int((_PCR[5] - _PCR[2]) / _VS)  # 16

# 默认 OccGrid 实例工厂 (用于 BEV 辅助绘制的坐标变换)
_DEFAULT_GRID = OccGrid(voxels=np.zeros((1, 1, 1), dtype=np.uint8))

# 输入图像在 img_inputs 中的通道顺序 (由配置 experiment.image_color_order 控制)
# 可视化时统一转换为 RGB 再显示
_VIS_INPUT_COLOR_ORDER = "RGB"


# =====================================================================
#  BEV坐标系约定 — 现在由 flashocc.vis.OccGrid 封装
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
    """世界坐标(x,y) → BEV像素(col=j, row=i). (兼容旧代码, 委托给 OccGrid)"""
    return _DEFAULT_GRID.world_to_bev_px(x, y)


def draw_ego_vehicle(ax, linewidth=2.0, color="#1E90FF", zorder=10):
    """绘制雷诺Zoé自车轮廓. (兼容旧代码, 委托给 vis.bev_helpers)"""
    _draw_ego_vehicle_vis(ax, _DEFAULT_GRID, linewidth=linewidth,
                          color=color, zorder=zorder)


def draw_camera_fovs(ax, alpha_fill=0.07, alpha_line=0.65, zorder=5):
    """绘制6相机视场角扇形辅助线. (兼容旧代码, 委托给 vis.bev_helpers)"""
    return _draw_camera_fovs_vis(ax, _DEFAULT_GRID, alpha_fill=alpha_fill,
                                 alpha_line=alpha_line, zorder=zorder)


def add_bev_annotations(ax, title="", fontsize=13, tick_m=10.0):
    """BEV轴标注. (兼容旧代码, 委托给 vis.bev_helpers)"""
    _add_bev_annotations_vis(ax, _DEFAULT_GRID, title=title,
                              fontsize=fontsize, tick_m=tick_m)


# =====================================================================
#  公共辅助函数 — 减少可视化代码重复
# =====================================================================

# 6 相机 2×3 网格布局 (cam_name, row, col)
_CAM_GRID = [
    ("CAM_FRONT_LEFT",  0, 0),
    ("CAM_FRONT",       0, 1),
    ("CAM_FRONT_RIGHT", 0, 2),
    ("CAM_BACK_LEFT",   1, 0),
    ("CAM_BACK",        1, 1),
    ("CAM_BACK_RIGHT",  1, 2),
]


def _short_cam(name: str) -> str:
    """'CAM_FRONT_LEFT' → 'FRONT_LEFT'."""
    return name.replace("CAM_", "")


def _cam_xticklabels(n_cams: int = 6) -> list[str]:
    """每相机柱状图的 x 轴标签."""
    return [_short_cam(c).replace("_", "\n") for c in _CAM_ORDER[:n_cams]]


def _save_figure(fig, path: str, dpi: int = 150, facecolor=None):
    """保存图表 → 关闭 → 打印路径."""
    kw: dict = {"dpi": dpi, "bbox_inches": "tight"}
    if facecolor is not None:
        kw["facecolor"] = facecolor
    plt.savefig(path, **kw)
    plt.close(fig)
    print(f"  → 已保存: {path}")


def _save_cv2_image(path: str, img_bgr: np.ndarray):
    """使用 OpenCV 直接保存图片并打印路径."""
    ok = cv2.imwrite(path, img_bgr)
    if not ok:
        raise RuntimeError(f"Failed to save image: {path}")
    print(f"  → 已保存: {path}")


def _rgb01_to_bgr8(img_rgb: np.ndarray) -> np.ndarray:
    """[0,1] RGB 或 uint8 RGB → uint8 BGR."""
    arr = np.asarray(img_rgb)
    if arr.dtype != np.uint8:
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return arr[..., ::-1].copy()


def _fit_to_panel(img_bgr: np.ndarray, panel_h: int, panel_w: int,
                  bg_color: tuple[int, int, int] = (245, 245, 245)) -> np.ndarray:
    """保持宽高比缩放并居中到固定 panel 尺寸."""
    canvas = np.full((panel_h, panel_w, 3), bg_color, dtype=np.uint8)
    h, w = img_bgr.shape[:2]
    if h <= 0 or w <= 0:
        return canvas

    scale = min(panel_w / w, panel_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=interp)

    x0 = (panel_w - new_w) // 2
    y0 = (panel_h - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


def _hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    """#RRGGBB -> (B,G,R)."""
    hc = hex_color.lstrip("#")
    if len(hc) != 6:
        return (220, 220, 220)
    r = int(hc[0:2], 16)
    g = int(hc[2:4], 16)
    b = int(hc[4:6], 16)
    return (b, g, r)


def _annotate_panel(img_bgr: np.ndarray, text: str,
                    color_bgr: tuple[int, int, int] = (20, 20, 20)):
    """在面板左上角绘制标题条."""
    out = img_bgr.copy()
    cv2.rectangle(out, (6, 6), (max(120, 14 + len(text) * 8), 28),
                  (255, 255, 255), thickness=-1)
    cv2.rectangle(out, (6, 6), (max(120, 14 + len(text) * 8), 28),
                  color_bgr, thickness=1)
    cv2.putText(out, text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                color_bgr, 1, cv2.LINE_AA)
    return out


def _unavailable_panel(h: int, w: int, title: str) -> np.ndarray:
    """GT 不可用占位图."""
    img = np.full((h, w, 3), (245, 245, 245), dtype=np.uint8)
    cv2.putText(img, title, (24, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (80, 80, 80), 2, cv2.LINE_AA)
    cv2.putText(img, "Ground Truth Unavailable", (24, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.95, (120, 120, 120), 2, cv2.LINE_AA)
    return img


def _gt_unavailable(ax, title: str, fontsize: int = 14, text_fontsize: int = 16):
    """GT 不可用时的占位面板."""
    ax.axis("off")
    ax.text(0.5, 0.5, "Ground Truth\nUnavailable",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=text_fontsize, fontweight="bold", color="gray")
    ax.set_title(title, fontsize=fontsize, fontweight="bold")


def _make_cls_patches(*occ_arrays, with_counts: bool = False,
                      count_src: np.ndarray | None = None) -> list[Patch]:
    """根据 OCC 数组收集所有非 free/others 类别, 生成图例色块."""
    all_cls: set[int] = set()
    for occ in occ_arrays:
        if occ is not None:
            all_cls |= set(int(c) for c in occ[(occ != 17) & (occ != 0)])
    patches = []
    for c in sorted(all_cls):
        label = OCC_CLASS_NAMES[c] if c < len(OCC_CLASS_NAMES) else f"cls{c}"
        if with_counts and count_src is not None:
            label += f" ({(count_src == c).sum():,})"
        patches.append(Patch(
            facecolor=OCC_CLASS_COLORS.get(c, (0.5, 0.5, 0.5)),
            edgecolor="k", lw=0.4, label=label))
    return patches


def _cam_bar(ax, values, title: str, ylabel: str,
             n_cams: int = 6, fontsize_ticks: int = 7, **bar_kw):
    """通用的每相机柱状图 (自动着色 + x 轴标签)."""
    cam_colors = [_CAM_INFO[c]["color"] for c in _CAM_ORDER[:n_cams]]
    ax.bar(range(n_cams), values, color=cam_colors, alpha=0.85, **bar_kw)
    ax.set_xticks(range(n_cams))
    ax.set_xticklabels(_cam_xticklabels(n_cams), fontsize=fontsize_ticks)
    ax.set_title(title, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=8)


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
            f"{_short_cam(cam_name)}\n"
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
    _cam_bar(ax_bar, mean_per_cam, "Per-Camera Mean Activation ± Std",
            "Channel Mean", n_cams=n_cams, fontsize_ticks=8,
            yerr=std_per_cam, capsize=5, ecolor="gray")
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
                     label=_short_cam(_CAM_ORDER[idx]),
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

    _save_figure(fig, save_path)


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
    title: str = "LSS Depth Prediction - 2D Heatmap",
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
        ax.set_title(f"{_short_cam(cam_name)} - E[depth]",
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
        ax.set_title(f"{_short_cam(cam_name)} - depth dist (row={mid_h})",
                     fontsize=9, fontweight="bold", color=cam_colors[idx])

        # 叠加 argmax 线
        peak_line = peak_depth[idx, mid_h, :]  # (fW,)
        ax.plot(np.arange(fW) + 0.5, peak_line, 'c-', linewidth=0.8,
                alpha=0.8, label="argmax depth")
        ax.legend(fontsize=6, loc="upper right")

    # ---- Row 2: 全局统计 ----
    ax_s = fig.add_subplot(gs_stats[0, 0])
    mean_depths = [float(expected_depth[i].mean()) for i in range(n_cams)]
    _cam_bar(ax_s, mean_depths, "Mean Expected Depth per Camera", "depth (m)",
            n_cams=n_cams)

    ax_e = fig.add_subplot(gs_stats[0, 1])
    # 深度分布的熵: H = -sum(p * log(p+eps))
    eps = 1e-8
    entropy = -(depth_np * np.log(depth_np + eps)).sum(axis=1).mean(axis=(1, 2))  # (N,)
    _cam_bar(ax_e, entropy, "Mean Depth Entropy per Camera", "entropy (nats)",
            n_cams=n_cams)

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

    _save_figure(fig, save_path)


def visualize_lss_depth_3d_perspective(
    depth: torch.Tensor,
    img_inputs: tuple,
    save_path: str,
    depth_cfg: tuple = (1.0, 45.0, 0.5),
    n_cams: int = 6,
    input_size: tuple = (256, 704),
    title: str = "LSS Depth - 3D Ego Space",
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

    # ------ 反投影到 ego 空间 (复用 _backproject_lss_to_ego) ------
    all_ego_pts = _backproject_lss_to_ego(
        depth, img_inputs, depth_cfg, n_cams, input_size)
    if not all_ego_pts:
        print("  [跳过] LSS depth 3D: 无法提取相机参数")
        return

    # ------ 绘图 ------
    bg_color = "#0d1117"
    fig = plt.figure(figsize=(16, 14), facecolor=bg_color)
    ax = fig.add_subplot(111, projection="3d", facecolor=bg_color)

    for cam_idx in range(n_cams):
        cam_name = _CAM_ORDER[cam_idx]
        pts = all_ego_pts[cam_idx]
        color = _CAM_INFO[cam_name]["color"]
        ax.scatter(
            pts[:, 0], pts[:, 1], pts[:, 2],
            c=color, s=3.0, alpha=0.7,
            label=_short_cam(cam_name),
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
    ax.set_xlabel("X / m  (front +)", color="white", fontsize=9, labelpad=8)
    ax.set_ylabel("Y / m  (left +)", color="white", fontsize=9, labelpad=8)
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

    _save_figure(fig, save_path, facecolor=fig.get_facecolor())


# =====================================================================
#  (1c) LSS 预测点云 vs LiDAR 真值点云对比可视化
# =====================================================================

def _load_lidar_points_ego(dataset, sample_idx: int) -> np.ndarray | None:
    """从 dataset.data_infos 加载 LiDAR 点云并变换到 key ego 坐标系.

    Returns:
        (N, 3) np.ndarray  ego坐标系下的点云, 或 None.
    """
    from pyquaternion import Quaternion as Quat
    try:
        infos = getattr(dataset, "data_infos", None)
        if infos is None or sample_idx >= len(infos):
            return None
        info = infos[sample_idx]

        # 1. 加载原始点云
        lidar_path = info.get("lidar_path", "")
        if not lidar_path or not os.path.isfile(lidar_path):
            return None
        pts = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)[:, :3]  # (N, 3)

        # 2. lidar → ego
        lidar2ego = np.eye(4, dtype=np.float64)
        lidar2ego[:3, :3] = Quat(info["lidar2ego_rotation"]).rotation_matrix
        lidar2ego[:3, 3] = np.array(info["lidar2ego_translation"])

        # 3. ego → global
        ego2global = np.eye(4, dtype=np.float64)
        ego2global[:3, :3] = Quat(info["ego2global_rotation"]).rotation_matrix
        ego2global[:3, 3] = np.array(info["ego2global_translation"])

        # lidar → global
        lidar2global = ego2global @ lidar2ego

        # 最终结果就是 ego 坐标系 (key ego = lidar ego 在单帧中相同)
        # 变换: p_ego = lidar2ego @ p_lidar
        pts_h = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float64)], axis=1)  # (N, 4)
        pts_ego = (lidar2ego @ pts_h.T).T[:, :3]  # (N, 3)
        return pts_ego.astype(np.float32)
    except Exception as e:
        print(f"  [加载LiDAR失败] {e}")
        return None


def _backproject_lss_to_ego(
    depth: torch.Tensor,
    img_inputs: tuple,
    depth_cfg: tuple = (1.0, 45.0, 0.5),
    n_cams: int = 6,
    input_size: tuple = (256, 704),
) -> list[np.ndarray]:
    """将 LSS 期望深度反投影到 ego 空间, 返回每相机点云列表.

    Returns:
        list of (M, 3) np.ndarray, 每个相机一份 ego 坐标点.
    """
    depth_np = depth.float().numpy()[:n_cams]
    D, fH, fW = depth_np.shape[1], depth_np.shape[2], depth_np.shape[3]
    bins = _make_depth_bins(depth_cfg)
    expected_depth = (depth_np * bins[None, :, None, None]).sum(axis=1)  # (N, fH, fW)

    cam_params = _extract_cam_params_from_img_inputs(img_inputs)
    if cam_params is None:
        return []

    H_in, W_in = input_size
    u_grid = np.linspace(0, W_in - 1, fW, dtype=np.float32)
    v_grid = np.linspace(0, H_in - 1, fH, dtype=np.float32)
    uu, vv = np.meshgrid(u_grid, v_grid)
    ones_hw = np.ones((fH, fW), dtype=np.float32)
    aug_pts = np.stack([uu, vv, ones_hw], axis=-1).reshape(-1, 3)

    all_ego_pts = []
    for cam_idx in range(n_cams):
        ed = expected_depth[cam_idx].flatten()
        K = cam_params["intrins"][cam_idx]
        post_rot = cam_params["post_rots"][cam_idx]
        post_tran = cam_params["post_trans"][cam_idx]
        s2ke = cam_params["sensor2keyegos"][cam_idx]

        inv_post_rot = np.linalg.inv(post_rot)
        deaug = (aug_pts - post_tran[None, :]) @ inv_post_rot.T
        uvd = np.stack([deaug[:, 0] * ed, deaug[:, 1] * ed, ed], axis=-1)
        K_inv = np.linalg.inv(K)
        R_c2e = s2ke[:3, :3]
        t_c2e = s2ke[:3, 3]
        combine = R_c2e @ K_inv
        p_ego = uvd @ combine.T + t_c2e[None, :]
        all_ego_pts.append(p_ego)
    return all_ego_pts


def _style_3d_ax(ax, title: str, lim: int = 48):
    """统一 3D 轴样式."""
    theta = np.linspace(0, 2 * np.pi, 200)
    for r in [10, 20, 30, 40]:
        cx = r * np.cos(theta)
        cy = r * np.sin(theta)
        cz = np.zeros_like(theta)
        ax.plot(cx, cy, cz, color="white", alpha=0.35, linewidth=0.8)
        ax.text(r + 0.5, 0, 0, f"{r}m", color="white", fontsize=7, alpha=0.7,
                ha="left", va="center")
    ax.scatter([0], [0], [0], c="white", s=80, marker="*",
               zorder=10, edgecolors="none")
    ax.plot([-lim, lim], [0, 0], [0, 0], color="white", alpha=0.15, lw=0.6)
    ax.plot([0, 0], [-lim, lim], [0, 0], color="white", alpha=0.15, lw=0.6)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-4, 8)
    ax.set_xlabel("X / m  (front +)", color="white", fontsize=8, labelpad=6)
    ax.set_ylabel("Y / m  (left +)", color="white", fontsize=8, labelpad=6)
    ax.set_zlabel("Z / m  (up)", color="white", fontsize=8, labelpad=6)
    ax.view_init(elev=55, azim=-90)
    ax.tick_params(colors="white", labelsize=6)
    for spine in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        spine.fill = False
        spine.set_edgecolor((1, 1, 1, 0.08))
    ax.xaxis.line.set_color((1, 1, 1, 0.15))
    ax.yaxis.line.set_color((1, 1, 1, 0.15))
    ax.zaxis.line.set_color((1, 1, 1, 0.15))
    ax.set_title(title, fontsize=11, fontweight="bold", color="white", pad=14)


# =====================================================================
#  (1c-helper) LiDAR → 相机投影, 构建射线方向真值深度
#
#  原理: 将 LiDAR 3D 点从 key ego 坐标系投影到每个相机的 2D 图像平面:
#    p_ego → inv(sensor2keyego) → p_cam → K → (u,v,z_cam) → post_aug → (u',v')
#  其中 z_cam 即为 **相机光轴方向深度**, 与 LSS 深度 bin 定义完全一致.
#  这样就可以在同一像素的同一射线方向上, 严格比较 pred 深度与 GT 深度.
# =====================================================================

def _project_lidar_to_cameras(
    lidar_pts_ego: np.ndarray,
    cam_params: dict,
    n_cams: int = 6,
    input_size: tuple = (256, 704),
    feat_size: tuple = (16, 44),
    depth_range: tuple = (1.0, 45.0),
) -> list[dict]:
    """将 LiDAR 3D 点投影到每个相机的特征网格, 获取射线方向真值深度.

    对每个相机:
        1. keyego → cam 坐标变换
        2. 过滤 z_cam > 0 (在相机前方的点)
        3. 针孔投影 + 数据增强映射到增强图像坐标 (u_aug, v_aug)
        4. 映射到特征网格 (feat_i, feat_j)
        5. 每个特征像素保留最近表面点 (最小 z_cam)

    Args:
        lidar_pts_ego: (N, 3) key ego 坐标系下的 LiDAR 点
        cam_params: dict from _extract_cam_params_from_img_inputs
        n_cams: 相机数
        input_size: 输入图像尺寸 (H_in, W_in)
        feat_size: 特征图尺寸 (fH, fW)
        depth_range: 有效深度范围 (min, max), 与 LSS depth bins 一致

    Returns:
        list of dict per camera:
            'u_aug': (M,) 增强图像 u 坐标 (所有投影点)
            'v_aug': (M,) 增强图像 v 坐标
            'z_cam': (M,) 相机光轴深度 = 射线方向 GT 深度
            'feat_i': (M,) 对应特征网格行号
            'feat_j': (M,) 对应特征网格列号
            'gt_depth_map': (fH, fW) 稀疏 GT 深度图 (-1 = 无值)
            'valid_mask': (fH, fW) bool, True = 该像素有 GT 深度
    """
    H_in, W_in = input_size
    fH, fW = feat_size
    d_min, d_max = depth_range

    # 预计算齐次点
    pts_h = np.concatenate(
        [lidar_pts_ego, np.ones((len(lidar_pts_ego), 1), dtype=np.float32)],
        axis=1)  # (N, 4)

    results = []
    for cam_idx in range(n_cams):
        s2ke = cam_params["sensor2keyegos"][cam_idx]  # (4,4) cam→keyego
        K = cam_params["intrins"][cam_idx]             # (3,3)
        post_rot = cam_params["post_rots"][cam_idx]    # (3,3)
        post_tran = cam_params["post_trans"][cam_idx]  # (3,)

        # keyego → camera
        ke2cam = np.linalg.inv(s2ke.astype(np.float64)).astype(np.float32)
        pts_cam = (ke2cam @ pts_h.T).T[:, :3]  # (N, 3)

        z_cam = pts_cam[:, 2].copy()

        # 过滤: 在相机前方
        front = z_cam > 0.5
        pts_cam_f = pts_cam[front]
        z_cam_f = z_cam[front]

        # 针孔投影
        pts_img = (K @ pts_cam_f.T).T  # (M, 3)
        u = pts_img[:, 0] / pts_img[:, 2]
        v = pts_img[:, 1] / pts_img[:, 2]

        # 数据增强映射
        uv1 = np.stack([u, v, np.ones_like(u)], axis=-1)  # (M, 3)
        uv_aug = (post_rot @ uv1.T).T + post_tran[None, :]
        u_aug = uv_aug[:, 0]
        v_aug = uv_aug[:, 1]

        # 过滤: 图像边界内
        in_bounds = ((u_aug >= 0) & (u_aug < W_in) &
                     (v_aug >= 0) & (v_aug < H_in))
        u_aug = u_aug[in_bounds]
        v_aug = v_aug[in_bounds]
        z_cam_f = z_cam_f[in_bounds]

        # 过滤: 深度范围内 (与 LSS bins 一致)
        in_depth = (z_cam_f >= d_min) & (z_cam_f <= d_max)
        u_aug = u_aug[in_depth]
        v_aug = v_aug[in_depth]
        z_cam_f = z_cam_f[in_depth]

        # 映射到特征网格坐标
        feat_j = np.clip(
            np.round(u_aug * (fW - 1) / max(W_in - 1, 1)).astype(np.intp),
            0, fW - 1)
        feat_i = np.clip(
            np.round(v_aug * (fH - 1) / max(H_in - 1, 1)).astype(np.intp),
            0, fH - 1)

        # 构建稀疏 GT 深度图 (每像素保留最近表面点)
        # 策略: 按深度降序排列后直接赋值, 近处覆盖远处
        gt_depth_map = np.full((fH, fW), -1.0, dtype=np.float32)
        if len(z_cam_f) > 0:
            order = np.argsort(-z_cam_f)  # 降序: 远→近
            z_sorted = z_cam_f[order]
            fi_sorted = feat_i[order]
            fj_sorted = feat_j[order]
            gt_depth_map[fi_sorted, fj_sorted] = z_sorted  # 近处最后赋值, 覆盖远处

        valid_mask = gt_depth_map > 0

        results.append({
            'u_aug': u_aug,
            'v_aug': v_aug,
            'z_cam': z_cam_f,
            'feat_i': feat_i,
            'feat_j': feat_j,
            'gt_depth_map': gt_depth_map,
            'valid_mask': valid_mask,
        })

    return results



def visualize_lss_depth_3d(
    depth: torch.Tensor,
    img_inputs: tuple,
    lidar_pts_ego: np.ndarray,
    save_path: str,
    depth_cfg: tuple = (1.0, 45.0, 0.5),
    n_cams: int = 6,
    input_size: tuple = (256, 704),
):
    """LSS 射线方向深度误差分析可视化 (重新设计版).

    核心方法改进 (相比旧版 3D KNN):
    ─────────────────────────────────────────────────────────────────
    旧版: 将 LSS 预测点反投影到 3D ego 空间, 用 cKDTree 找最近 LiDAR 点
          的 3D 欧式距离作为 "depth error" — 这会匹配到错误物体上,
          严重低估真实深度误差.

    新版: 将 LiDAR 3D 点正向投影到每个相机的 2D 图像平面, 在同一像素的
          同一射线方向上严格比较 LSS 期望深度 E[d] 与 LiDAR 真值深度
          z_cam — 这是深度估计任务的标准评估方式.
    ─────────────────────────────────────────────────────────────────

    布局:
        Row 0-1: 6 相机面板 (2×3), 输入图像 + LiDAR 投影点射线深度误差散点
        Row 2:   [Pred vs GT 散点图] [每相机误差柱状图] [误差-距离曲线]

    评估指标:
        · Mean/Median/P90 |pred - gt|  (绝对误差)
        · RMSE = sqrt(mean((pred-gt)²))
        · Bias = mean(pred - gt)  (+偏表示预测偏远, -偏表示预测偏近)
        · AbsRel = mean(|pred-gt| / gt)
        · δ<1.25^k (k=1,2,3): max(pred/gt, gt/pred) < 阈值 的像素占比

    Args:
        depth: (B*N, D, fH, fW) softmax 深度分布
        img_inputs: img_inputs tuple
        lidar_pts_ego: (N_lidar, 3) LiDAR 真值点 (key ego 坐标系)
        save_path: 输出路径
        depth_cfg: depth bins 配置 (min, max, step)
        n_cams: 相机数
        input_size: 输入图像尺寸 (H, W)
    """
    # --- 计算 LSS 期望深度 ---
    depth_np = depth.float().numpy()[:n_cams]  # (N, D, fH, fW)
    D, fH, fW = depth_np.shape[1], depth_np.shape[2], depth_np.shape[3]
    bins = _make_depth_bins(depth_cfg)  # (D,)
    expected_depth = (depth_np * bins[None, :, None, None]).sum(axis=1)  # (N, fH, fW)

    # --- 提取相机参数 ---
    cam_params = _extract_cam_params_from_img_inputs(img_inputs)
    if cam_params is None:
        print("  [跳过] LSS depth 3D: 无法提取相机参数")
        return

    # --- 过滤 LiDAR 到感知范围 ---
    pcr = _PCR  # [-40, -40, -1, 40, 40, 5.4]
    mask_lidar = (
        (lidar_pts_ego[:, 0] >= pcr[0]) & (lidar_pts_ego[:, 0] <= pcr[3]) &
        (lidar_pts_ego[:, 1] >= pcr[1]) & (lidar_pts_ego[:, 1] <= pcr[4]) &
        (lidar_pts_ego[:, 2] >= pcr[2]) & (lidar_pts_ego[:, 2] <= pcr[5])
    )
    lidar_filtered = lidar_pts_ego[mask_lidar]

    # --- 投影 LiDAR 到每个相机的特征网格 ---
    projections = _project_lidar_to_cameras(
        lidar_filtered, cam_params,
        n_cams=n_cams,
        input_size=input_size,
        feat_size=(fH, fW),
        depth_range=(depth_cfg[0], depth_cfg[1]),
    )

    # --- 计算每相机射线方向深度误差 ---
    cam_errors_signed: list[np.ndarray] = []   # pred - gt (有符号)
    cam_errors_abs: list[np.ndarray] = []      # |pred - gt|
    cam_gt_depths: list[np.ndarray] = []       # GT 深度
    cam_pred_depths: list[np.ndarray] = []     # pred 深度
    cam_n_matched: list[int] = []              # 匹配像素数

    for cam_idx in range(n_cams):
        proj = projections[cam_idx]
        mask = proj['valid_mask']  # (fH, fW)
        if mask.sum() == 0:
            cam_errors_signed.append(np.array([]))
            cam_errors_abs.append(np.array([]))
            cam_gt_depths.append(np.array([]))
            cam_pred_depths.append(np.array([]))
            cam_n_matched.append(0)
            continue

        gt_d = proj['gt_depth_map'][mask]           # (M,) GT depth along ray
        pred_d = expected_depth[cam_idx][mask]       # (M,) LSS E[d]
        signed_err = pred_d - gt_d                   # + = predict too far
        abs_err = np.abs(signed_err)

        cam_errors_signed.append(signed_err)
        cam_errors_abs.append(abs_err)
        cam_gt_depths.append(gt_d)
        cam_pred_depths.append(pred_d)
        cam_n_matched.append(int(mask.sum()))

    # --- 合并所有相机的统计量 ---
    all_gt = np.concatenate([d for d in cam_gt_depths if len(d) > 0])
    all_pred = np.concatenate([d for d in cam_pred_depths if len(d) > 0])
    all_abs_err = np.concatenate([e for e in cam_errors_abs if len(e) > 0])
    all_signed_err = np.concatenate([e for e in cam_errors_signed if len(e) > 0])
    total_matched = len(all_gt)

    if total_matched == 0:
        print("  [跳过] LSS depth 3D: 无射线匹配点, 无法计算深度误差")
        return

    # --- 全局评估指标 ---
    mean_abs = float(all_abs_err.mean())
    median_abs = float(np.median(all_abs_err))
    p90_abs = float(np.percentile(all_abs_err, 90))
    mean_signed = float(all_signed_err.mean())
    rmse = float(np.sqrt((all_signed_err ** 2).mean()))

    # 深度估计标准指标
    gt_safe = np.maximum(all_gt, 1e-6)
    pred_safe = np.maximum(all_pred, 1e-6)
    ratios = np.maximum(pred_safe / gt_safe, gt_safe / pred_safe)
    delta1 = float((ratios < 1.25).mean() * 100)
    delta2 = float((ratios < 1.25**2).mean() * 100)
    delta3 = float((ratios < 1.25**3).mean() * 100)
    abs_rel = float((all_abs_err / gt_safe).mean())

    # --- 解码输入图像 ---
    imgs = _decode_input_images_to_bgr01(img_inputs)

    # =====================================================================
    #  绘图
    # =====================================================================
    bg_color = "#0d1117"
    panel_face = "#131924"
    fig = plt.figure(figsize=(42, 28), facecolor=bg_color)
    fig.suptitle(
        "LSS Ray-Aligned Depth Error Analysis\n"
        "Method: LiDAR → camera projection → per-pixel depth comparison "
        "along camera ray  (replaces incorrect 3D KNN)\n"
        f"Depth bins: {depth_cfg[0]:.1f}→{depth_cfg[1]:.1f}m, "
        f"step={depth_cfg[2]:.1f}m, D={D}, "
        f"feature=({fH}×{fW}), "
        f"matched={total_matched} pixels across {n_cams} cameras",
        fontsize=14, fontweight="bold", color="white", y=0.98,
    )

    # 布局:  top = 2×3 cameras;  bottom = 1×3 analysis panels
    gs_top = GridSpec(2, 3, figure=fig, top=0.93, bottom=0.42,
                      hspace=0.22, wspace=0.08,
                      left=0.03, right=0.97)
    gs_bot = GridSpec(1, 3, figure=fig, top=0.37, bottom=0.05,
                      hspace=0.05, wspace=0.18,
                      left=0.05, right=0.95)

    err_vmax = min(8.0, float(np.percentile(all_abs_err, 98)))

    # ====== Row 0-1: 6 相机 — 输入图像 + 射线深度误差散点叠加 ======
    for idx in range(n_cams):
        cam_name, grid_r, grid_c = _CAM_GRID[idx]
        ax = fig.add_subplot(gs_top[grid_r, grid_c])

        # 输入图像
        img = imgs[idx]  # (H, W, 3) RGB 0~1
        ax.imshow(img)

        proj = projections[idx]
        u = proj['u_aug']
        v = proj['v_aug']
        z_gt = proj['z_cam']

        if len(u) > 0 and cam_n_matched[idx] > 0:
            # 对每个 LiDAR 投影点, 查找其所在特征像素的 LSS 预测深度
            fj = np.clip(np.round(u * (fW - 1) / max(input_size[1] - 1, 1)
                                  ).astype(np.intp), 0, fW - 1)
            fi = np.clip(np.round(v * (fH - 1) / max(input_size[0] - 1, 1)
                                  ).astype(np.intp), 0, fH - 1)
            pred_at_pts = expected_depth[idx][fi, fj]
            err_at_pts = np.abs(pred_at_pts - z_gt)

            sc = ax.scatter(
                u, v, c=err_at_pts, cmap="jet", s=3.0, alpha=0.85,
                vmin=0, vmax=err_vmax, edgecolors="none", rasterized=True)

            # 右上角的 colorbar (仅在第一行最右相机上显示)
            if idx == 2:
                cbar = plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.01,
                                    shrink=0.8)
                cbar.set_label("Ray depth error |pred-gt| (m)",
                               fontsize=8, color="white")
                cbar.ax.tick_params(colors="white", labelsize=6)

        mean_e = (float(cam_errors_abs[idx].mean())
                  if len(cam_errors_abs[idx]) > 0 else 0)
        bias_e = (float(cam_errors_signed[idx].mean())
                  if len(cam_errors_signed[idx]) > 0 else 0)
        n_m = cam_n_matched[idx]
        cam_color = _CAM_INFO[cam_name]["color"]
        ax.set_title(
            f"{_short_cam(cam_name)}  |  "
            f"n={n_m}  mean={mean_e:.2f}m  bias={bias_e:+.2f}m",
            fontsize=10, fontweight="bold", color=cam_color, pad=4)
        ax.axis("off")

    # ====== Bottom-left: Pred vs GT 深度散点图 ======
    ax_scatter = fig.add_subplot(gs_bot[0, 0], facecolor=panel_face)

    # 密度着色 (避免过多点重叠)
    max_scatter_pts = 15000
    if total_matched > max_scatter_pts:
        idx_sub = np.random.default_rng(42).choice(
            total_matched, max_scatter_pts, replace=False)
    else:
        idx_sub = np.arange(total_matched)

    try:
        from scipy.stats import gaussian_kde
        xy = np.vstack([all_gt[idx_sub], all_pred[idx_sub]])
        kde = gaussian_kde(xy)
        density = kde(xy)
    except Exception:
        density = np.ones(len(idx_sub))

    ax_scatter.scatter(
        all_gt[idx_sub], all_pred[idx_sub],
        c=density, cmap="inferno", s=4, alpha=0.6,
        edgecolors="none", rasterized=True)

    # y = x 参考线
    d_lo, d_hi = float(depth_cfg[0]), float(depth_cfg[1])
    ax_scatter.plot([d_lo, d_hi], [d_lo, d_hi], '--', color="cyan",
                    linewidth=2, alpha=0.8, label="y = x (perfect)")
    # ±20% 误差带
    ax_scatter.fill_between(
        [d_lo, d_hi], [d_lo * 0.8, d_hi * 0.8], [d_lo * 1.2, d_hi * 1.2],
        color="cyan", alpha=0.06, label="±20% band")

    ax_scatter.set_xlabel("GT Depth (m) - LiDAR z_cam along camera ray",
                          fontsize=10, color="white")
    ax_scatter.set_ylabel("Predicted Depth (m) - LSS E[d]",
                          fontsize=10, color="white")
    ax_scatter.set_title("Predicted vs Ground Truth Depth (ray-aligned)",
                         fontsize=12, fontweight="bold", color="white", pad=10)
    ax_scatter.set_xlim(d_lo, d_hi)
    ax_scatter.set_ylim(d_lo, d_hi)
    ax_scatter.set_aspect("equal")
    ax_scatter.tick_params(colors="white", labelsize=8)
    for sp in ax_scatter.spines.values():
        sp.set_color("white")
        sp.set_alpha(0.3)
    ax_scatter.grid(color="white", alpha=0.1, linewidth=0.5)

    # 指标标注 (左上角)
    metrics_text = (
        f"AbsRel: {abs_rel:.4f}\n"
        f"RMSE:   {rmse:.2f}m\n"
        f"δ<1.25:  {delta1:.1f}%\n"
        f"δ<1.25²: {delta2:.1f}%\n"
        f"δ<1.25³: {delta3:.1f}%\n"
        f"Bias:   {mean_signed:+.2f}m"
    )
    ax_scatter.text(
        0.03, 0.97, metrics_text,
        transform=ax_scatter.transAxes, fontsize=9, va="top",
        fontfamily="monospace", color="white",
        bbox=dict(boxstyle="round,pad=0.3", fc="#1a1a2e",
                  ec="white", alpha=0.7))

    leg_s = ax_scatter.legend(fontsize=8, loc="lower right", framealpha=0.3,
                              edgecolor="white", facecolor="#1a1a2e")
    for t in leg_s.get_texts():
        t.set_color("white")

    # ====== Bottom-center: 每相机射线深度误差柱状图 ======
    ax_bar = fig.add_subplot(gs_bot[0, 1], facecolor=panel_face)

    cam_labels = [_short_cam(_CAM_ORDER[i]).replace("_", "\n")
                  for i in range(n_cams)]
    cam_colors = [_CAM_INFO[_CAM_ORDER[i]]["color"] for i in range(n_cams)]
    cam_means = [float(e.mean()) if len(e) > 0 else 0.0
                 for e in cam_errors_abs]
    cam_medians = [float(np.median(e)) if len(e) > 0 else 0.0
                   for e in cam_errors_abs]
    cam_p90s = [float(np.percentile(e, 90)) if len(e) > 0 else 0.0
                for e in cam_errors_abs]

    x = np.arange(n_cams)
    w = 0.22
    bars1 = ax_bar.bar(x - w, cam_means, w, color=cam_colors, alpha=0.85,
                       edgecolor="white", linewidth=0.5, label="Mean")
    bars2 = ax_bar.bar(x, cam_medians, w, color=cam_colors, alpha=0.55,
                       edgecolor="white", linewidth=0.5, label="Median",
                       hatch="//")
    bars3 = ax_bar.bar(x + w, cam_p90s, w, color=cam_colors, alpha=0.35,
                       edgecolor="white", linewidth=0.5, label="P90",
                       hatch="xx")

    # 柱顶标注数值
    for bar_group in [bars1, bars2, bars3]:
        for bar in bar_group:
            h = bar.get_height()
            if h > 0:
                ax_bar.text(bar.get_x() + bar.get_width() / 2, h + 0.05,
                            f"{h:.2f}", ha="center", va="bottom",
                            fontsize=7, color="white", fontweight="bold")

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(cam_labels, fontsize=8, color="white")
    ax_bar.set_ylabel("Ray-Aligned Depth Error (m)", fontsize=9, color="white")
    ax_bar.set_title("Per-Camera Ray-Aligned Depth Error\n"
                     "(|LSS E[d] - LiDAR z_cam| at matched pixels)",
                     fontsize=11, fontweight="bold", color="white", pad=10)
    ax_bar.tick_params(colors="white", labelsize=7)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    for sp in ax_bar.spines.values():
        sp.set_color("white")
        sp.set_alpha(0.3)
    ax_bar.set_facecolor(panel_face)
    ax_bar.grid(axis="y", color="white", alpha=0.1, linewidth=0.5)

    leg_bar = ax_bar.legend(fontsize=8, loc="upper right", framealpha=0.3,
                            edgecolor="white", facecolor="#1a1a2e")
    for t in leg_bar.get_texts():
        t.set_color("white")

    # 柱状图下方标注匹配点数
    for i in range(n_cams):
        ax_bar.text(i, -0.04, f"n={cam_n_matched[i]}", ha="center",
                    transform=ax_bar.get_xaxis_transform(),
                    fontsize=7, color="white", alpha=0.7)

    # ====== Bottom-right: 射线深度误差 vs GT 深度距离曲线 ======
    ax_curve = fig.add_subplot(gs_bot[0, 2], facecolor=panel_face)

    dist_bins = np.arange(depth_cfg[0], depth_cfg[1] + 1, 2.0)
    bin_centers = (dist_bins[:-1] + dist_bins[1:]) / 2

    for cam_idx in range(n_cams):
        gt_d = cam_gt_depths[cam_idx]
        abs_e = cam_errors_abs[cam_idx]
        if len(gt_d) == 0:
            continue

        cam_name = _CAM_ORDER[cam_idx]
        color = _CAM_INFO[cam_name]["color"]

        bin_means = []
        bin_stds = []
        for bi in range(len(dist_bins) - 1):
            mask = (gt_d >= dist_bins[bi]) & (gt_d < dist_bins[bi + 1])
            if mask.sum() > 3:
                bin_means.append(float(abs_e[mask].mean()))
                bin_stds.append(float(abs_e[mask].std()))
            else:
                bin_means.append(np.nan)
                bin_stds.append(np.nan)

        means_arr = np.array(bin_means)
        stds_arr = np.array(bin_stds)
        valid = ~np.isnan(means_arr)

        if valid.sum() > 0:
            ax_curve.plot(
                bin_centers[valid], means_arr[valid],
                color=color, linewidth=2.0, alpha=0.85,
                marker="o", markersize=3,
                label=_short_cam(cam_name))
            if valid.sum() > 1:
                ax_curve.fill_between(
                    bin_centers[valid],
                    np.maximum(0, (means_arr - stds_arr)[valid]),
                    (means_arr + stds_arr)[valid],
                    color=color, alpha=0.08)

    ax_curve.set_xlabel("GT Depth (m) - distance along camera ray",
                        fontsize=9, color="white")
    ax_curve.set_ylabel("Mean |Pred - GT| Depth Error (m)",
                        fontsize=9, color="white")
    ax_curve.set_title("Ray-Aligned Depth Error vs. GT Distance\n"
                       "(X = LiDAR z_cam,  Y = |E[d] - z_cam|)",
                       fontsize=11, fontweight="bold", color="white", pad=10)
    ax_curve.tick_params(colors="white", labelsize=7)
    ax_curve.spines["top"].set_visible(False)
    ax_curve.spines["right"].set_visible(False)
    for sp in ax_curve.spines.values():
        sp.set_color("white")
        sp.set_alpha(0.3)
    ax_curve.set_facecolor(panel_face)
    ax_curve.grid(color="white", alpha=0.1, linewidth=0.5)
    ax_curve.set_xlim(depth_cfg[0], depth_cfg[1])
    ax_curve.set_ylim(bottom=0)

    leg_curve = ax_curve.legend(fontsize=7, loc="upper left", framealpha=0.3,
                                edgecolor="white", facecolor="#1a1a2e",
                                ncol=2)
    for t in leg_curve.get_texts():
        t.set_color("white")

    # --- 统计信息文字 (右下角) ---
    stats_text = (
        f"Ray-Aligned Depth Error Statistics\n"
        f"{'=' * 38}\n"
        f"Mean |err|:   {mean_abs:.3f} m\n"
        f"Median |err|: {median_abs:.3f} m\n"
        f"P90 |err|:    {p90_abs:.3f} m\n"
        f"RMSE:         {rmse:.3f} m\n"
        f"Mean bias:    {mean_signed:+.3f} m\n"
        f"  (+ = predict too far)\n"
        f"{'=' * 38}\n"
        f"AbsRel:  {abs_rel:.4f}\n"
        f"δ<1.25:  {delta1:.1f}%\n"
        f"δ<1.25²: {delta2:.1f}%\n"
        f"δ<1.25³: {delta3:.1f}%\n"
        f"{'=' * 38}\n"
        f"Matched pixels: {total_matched}\n"
        f"LiDAR pts (in range): {len(lidar_filtered)}\n"
        f"Feature grid: {fH}×{fW}\n"
        f"Method: LiDAR→cam project\n"
        f"  (ray-aligned, NOT 3D KNN)"
    )
    fig.text(0.97, 0.01, stats_text, fontsize=9, fontfamily="monospace",
             color="white", ha="right", va="bottom",
             bbox=dict(boxstyle="round,pad=0.5", fc="#1a1a2e",
                       ec="white", alpha=0.8))

    _save_figure(fig, save_path, facecolor=fig.get_facecolor())


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
    _save_figure(fig, save_path)


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
    """(Dx,Dy,Dz) → BEV (Dx,Dy): 委托给 OccGrid.bev_projection."""
    return OccGrid.from_numpy(pred).bev_projection()


def _side_argmax_proj(pred: np.ndarray) -> np.ndarray:
    """(Dx,Dy,Dz) → Side (Dx,Dz): 委托给 OccGrid.side_projection."""
    return OccGrid.from_numpy(pred).side_projection()


def _front_argmax_proj(pred: np.ndarray) -> np.ndarray:
    """(Dx,Dy,Dz) → Front (Dy,Dz): 委托给 OccGrid.front_projection."""
    return OccGrid.from_numpy(pred).front_projection()


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
    patches = _make_cls_patches(pred_cls)
    if patches:
        fig.legend(handles=patches, loc="lower center",
                   ncol=min(9, len(patches)), fontsize=7,
                   frameon=True, bbox_to_anchor=(0.5, 0.005))

    _save_figure(fig, save_path)


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
    """渲染BEV面板 — 委托给 OccVoxelRenderer."""
    grid = OccGrid.from_numpy(occ)
    renderer = OccVoxelRenderer(grid, voxel_step=1)
    renderer.render_bev(ax, draw_helpers=draw_helpers, title=title, fontsize=fontsize)


def _render_isometric_panel(ax, occ: np.ndarray, voxel_step=2,
                             azim_deg=45.0, elev_deg=35.0, z_scale=2.5,
                             title="3D Isometric", fontsize=11):
    """等轴测体素渲染 — 委托给 OccVoxelRenderer (邻接剔除 + Painter's)."""
    grid = OccGrid.from_numpy(occ)
    renderer = OccVoxelRenderer(grid, voxel_step=voxel_step)
    renderer.render_isometric(
        ax, azim_deg=azim_deg, elev_deg=elev_deg, z_scale=z_scale,
        title=title, fontsize=fontsize,
    )


def _render_perspective_panel(ax, occ: np.ndarray, voxel_step=2,
                              heading_deg=0.0, elev_deg=8.0,
                              z_scale=1.5, fov_deg=90.0,
                              ego_height_m=1.5, eye_back_m=0.0,
                              title="3D Perspective", fontsize=11):
    """透视体素渲染 — 委托给 OccVoxelRenderer."""
    grid = OccGrid.from_numpy(occ)
    renderer = OccVoxelRenderer(grid, voxel_step=voxel_step)
    renderer.render_perspective(
        ax, heading_deg=heading_deg, elev_deg=elev_deg,
        z_scale=z_scale, fov_deg=fov_deg,
        ego_height_m=ego_height_m, eye_back_m=eye_back_m,
        title=title, fontsize=fontsize,
    )


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
              label=f"{_short_cam(name).replace('_',' ')} "
                    f"[hdg={info['heading']:+.0f}° HFoV={info['hfov']:.0f}°]")
        for name, info in _CAM_INFO.items()
    ]
    cam_patches.append(Patch(facecolor="#1E90FF", alpha=0.35, edgecolor="#1E90FF",
                             label="Ego - Renault Zoe (4.084m x 1.73m, wb=2.588m)"))

    all_cls: set = set(int(c) for c in occ_pred[(occ_pred != 17) & (occ_pred != 0)])
    if has_gt:
        all_cls |= set(int(c) for c in occ_gt[(occ_gt != 17) & (occ_gt != 0)])
    cls_patches = _make_cls_patches(occ_pred, occ_gt, with_counts=True,
                                    count_src=occ_pred)

    leg1 = fig.legend(handles=cam_patches, loc="lower center",
                      ncol=len(cam_patches), fontsize=8, frameon=True,
                      fancybox=True, bbox_to_anchor=(0.5, 0.068),
                      title="Camera FoV Lines & Ego Vehicle", title_fontsize=8)
    fig.legend(handles=cls_patches, loc="lower center",
               ncol=min(9, len(cls_patches)), fontsize=8,
               frameon=True, fancybox=True, bbox_to_anchor=(0.5, 0.005))
    fig.add_artist(leg1)

    _save_figure(fig, save_path, dpi=200, facecolor="white")


# =====================================================================
#  输入图像可视化
# =====================================================================

def visualize_input_images(img_inputs: tuple, save_path: str):
    """可视化6相机输入图像 + 右侧BEV相机布局示意."""
    imgs = _decode_input_images_to_bgr01(img_inputs)
    N = imgs.shape[0]

    fig = plt.figure(figsize=(22, 9))
    fig.suptitle("Input Camera Images - 6-Camera Surround View (after augmentation)",
                 fontsize=14, fontweight="bold")

    gs = GridSpec(2, 3, figure=fig, left=0.01, right=0.83,
                  hspace=0.22, wspace=0.05)

    for i, (cam, row, col) in enumerate(_CAM_GRID):
        if i >= N: continue
        img = imgs[i]
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(img)
        info = _CAM_INFO[cam]
        ax.set_title(
            f"{_short_cam(cam)}\n"
            f"hdg={info['heading']:+.0f}°  HFoV={info['hfov']:.0f}°",
            fontsize=10, fontweight="bold", color=info["color"],
            bbox=dict(boxstyle="round", fc="white", ec=info["color"], alpha=0.7))
        ax.axis("off")
        h, w = img.shape[:2]
        ax.text(4, h-4, f"{w}x{h}px", fontsize=7, color="white", va="bottom",
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
    ax_l.text(cx, _DX-3, "FRONT", ha="center", va="top",
              color="white", fontsize=7, fontweight="bold")
    ax_l.text(cx, 3, "REAR", ha="center", va="bottom",
              color="white", fontsize=7, fontweight="bold")

    _save_figure(fig, save_path)


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


def build_output_tag(checkpoint_path: str, config_path: str) -> str:
    """统一输出命名标签: {模型型号}[_ema]_{epoch}.

    模型型号 从 config 文件名提取, ema/epoch 从 checkpoint 文件名解析.
    例: config='flashocc_convnext_tiny_dinov3.py', ckpt='epoch_100_ema.pth'
        → 'flashocc_convnext_tiny_dinov3_ema_100'
    """
    config_name = os.path.splitext(os.path.basename(config_path))[0]
    ckpt_name = os.path.splitext(os.path.basename(checkpoint_path))[0]

    model_type = config_name
    is_ema = "_ema" in ckpt_name.lower()

    epoch_match = re.search(r'epoch[_-]?(\d+)', ckpt_name)
    epoch = int(epoch_match.group(1)) if epoch_match else 0

    tag = model_type
    if is_ema:
        tag += "_ema"
    tag += f"_{epoch}"
    return tag


def build_output_path(out_dir: str, tag: str, name: str,
                      layer_idx: int | None = None) -> str:
    """统一输出文件路径.

    文件名格式: {tag}_{name}[_{layer_idx:02d}].png
    即: {模型型号/ema}_{epoch}_{name}[_{number of layer}].png
    """
    if layer_idx is not None:
        return os.path.join(out_dir, f"{tag}_{name}_{layer_idx:02d}.png")
    return os.path.join(out_dir, f"{tag}_{name}.png")


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
            _gt_unavailable(ax, ttl, fontsize=fontsize, text_fontsize=16)
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

    for i, (cam, row, col) in enumerate(_CAM_GRID):
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
            f"{_short_cam(cam)}  (hdg={info['heading']:+.0f}°, HFoV={info['hfov']:.0f}°)",
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
    _save_figure(fig, save_path, dpi=180, facecolor="white")


def _draw_input_cameras_compact(fig, spec, img_inputs: tuple):
    """在给定 SubplotSpec 内紧凑绘制 6 相机输入 (2×3), 不含BEV布局示意."""
    imgs = _decode_input_images_to_bgr01(img_inputs)
    N = imgs.shape[0]

    sub = GridSpecFromSubplotSpec(2, 3, subplot_spec=spec, wspace=0.04, hspace=0.12)

    for i, (cam, row, col) in enumerate(_CAM_GRID):
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
            f"{_short_cam(cam)}",
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

    委托给 OccVoxelRenderer.render_camera_overlay —
    统一的投影链 + Painter's Algorithm + 离屏渲染 + 单次 alpha 叠加.
    """
    ax.imshow(img_rgb)
    if cam_params is None:
        return
    if cam_idx >= cam_params["sensor2keyegos"].shape[0]:
        return

    cam = CameraParams(
        sensor2keyego=cam_params["sensor2keyegos"][cam_idx],
        intrinsics=cam_params["intrins"][cam_idx],
        post_rot=cam_params["post_rots"][cam_idx],
        post_trans=cam_params["post_trans"][cam_idx],
    )
    grid = OccGrid.from_numpy(occ)
    renderer = OccVoxelRenderer(grid, voxel_step=max(int(voxel_step), 2))
    renderer.render_camera_overlay(ax, img_rgb, cam, alpha=alpha, cam_name=cam_name)


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
    imgs = _decode_input_images_to_bgr01(img_inputs)
    N = imgs.shape[0]
    cam_params = _extract_cam_params_from_img_inputs(img_inputs)

    pred_renderer = OccVoxelRenderer(
        OccGrid.from_numpy(occ_pred), voxel_step=max(int(voxel_step), 1))

    H_img, W_img = imgs.shape[1], imgs.shape[2]
    gap = 16
    margin = 22
    row_h = H_img
    row_w = W_img * 3 + gap * 2
    canvas_w = row_w + margin * 2

    bot_panel_w = (canvas_w - margin * 2 - gap * 3) // 4
    bot_panel_h = int(bot_panel_w * 0.75)

    title_h = 46
    canvas_h = margin * 2 + title_h + row_h * 2 + gap * 2 + bot_panel_h
    canvas = np.full((canvas_h, canvas_w, 3), (246, 246, 246), dtype=np.uint8)

    cv2.putText(canvas, title, (margin, margin + 30), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (20, 20, 20), 2, cv2.LINE_AA)

    def _cam_overlay(cam_name: str, cam_idx: int) -> np.ndarray:
        if cam_idx >= N:
            return np.zeros((H_img, W_img, 3), dtype=np.uint8)
        if cam_params is None or cam_idx >= cam_params["sensor2keyegos"].shape[0]:
            return _rgb01_to_bgr8(imgs[cam_idx])
        cam = CameraParams(
            sensor2keyego=cam_params["sensor2keyegos"][cam_idx],
            intrinsics=cam_params["intrins"][cam_idx],
            post_rot=cam_params["post_rots"][cam_idx],
            post_trans=cam_params["post_trans"][cam_idx],
        )
        img = pred_renderer.render_camera_overlay_image(
            imgs[cam_idx], cam, alpha=0.5, cam_name="")
        info = _CAM_INFO[cam_name]
        return _annotate_panel(img, _short_cam(cam_name), _hex_to_bgr(info["color"]))

    cam_rows = [
        [("CAM_FRONT_LEFT", 0), ("CAM_FRONT", 1), ("CAM_FRONT_RIGHT", 2)],
        [("CAM_BACK_LEFT", 3), ("CAM_BACK", 4), ("CAM_BACK_RIGHT", 5)],
    ]

    y = margin + title_h
    for row in cam_rows:
        x = margin
        for cam_name, cam_idx in row:
            tile = _cam_overlay(cam_name, cam_idx)
            canvas[y:y + row_h, x:x + W_img] = tile
            x += W_img + gap
        y += row_h + gap

    pred_iso = pred_renderer.render_isometric_image(
        azim_deg=45.0, elev_deg=35.0, z_scale=2.5,
        width=max(bot_panel_w, 640), height=max(bot_panel_h, 480),
    )
    pred_iso = _annotate_panel(_fit_to_panel(pred_iso, bot_panel_h, bot_panel_w),
                               "Prediction - 3D Isometric")

    pred_bev = pred_renderer.render_bev_image(draw_helpers=True, scale=4)
    pred_bev = _annotate_panel(_fit_to_panel(pred_bev, bot_panel_h, bot_panel_w),
                               "Prediction - BEV")

    if has_gt:
        gt_renderer = OccVoxelRenderer(
            OccGrid.from_numpy(occ_gt), voxel_step=max(int(voxel_step), 1))
        gt_iso = gt_renderer.render_isometric_image(
            azim_deg=45.0, elev_deg=35.0, z_scale=2.5,
            width=max(bot_panel_w, 640), height=max(bot_panel_h, 480),
        )
        gt_bev = gt_renderer.render_bev_image(draw_helpers=True, scale=4)
        gt_iso = _annotate_panel(_fit_to_panel(gt_iso, bot_panel_h, bot_panel_w),
                                 "Ground Truth - 3D Isometric")
        gt_bev = _annotate_panel(_fit_to_panel(gt_bev, bot_panel_h, bot_panel_w),
                                 "Ground Truth - BEV")
    else:
        gt_iso = _unavailable_panel(bot_panel_h, bot_panel_w, "Ground Truth - 3D")
        gt_bev = _unavailable_panel(bot_panel_h, bot_panel_w, "Ground Truth - BEV")

    panels = [pred_iso, pred_bev, gt_iso, gt_bev]
    x = margin
    y_bot = canvas_h - margin - bot_panel_h
    for p in panels:
        canvas[y_bot:y_bot + bot_panel_h, x:x + bot_panel_w] = p
        x += bot_panel_w + gap

    _save_cv2_image(save_path, canvas)


def _draw_ego_3d_box_perspective(ax, grid_shape, heading_deg, elev_deg,
                                  z_scale, fov_deg, ego_height_m,
                                  eye_back_m, pcr_z_min, pcr_z_max):
    """在透视图上绘制自车 3D 线框包围盒."""
    vs = _VS
    # 自车在世界坐标 (x=前, y=左) 中的 8 个角点
    # 后轴在车身中心偏后 ~0.748m, 这里简化为中心对齐
    half_l = _ZOE_LENGTH / 2.0
    half_w = _ZOE_WIDTH / 2.0
    z_bot = 0.0           # 地面
    z_top = 1.56          # 车顶约 1.56m

    # 世界坐标 8 角点
    corners_world = np.array([
        [-half_l, -half_w, z_bot],
        [ half_l, -half_w, z_bot],
        [ half_l,  half_w, z_bot],
        [-half_l,  half_w, z_bot],
        [-half_l, -half_w, z_top],
        [ half_l, -half_w, z_top],
        [ half_l,  half_w, z_top],
        [-half_l,  half_w, z_top],
    ])  # (8, 3)

    # 世界 → 体素坐标
    corners_vox = np.empty_like(corners_world)
    corners_vox[:, 0] = (corners_world[:, 0] - _PCR[0]) / vs
    corners_vox[:, 1] = (corners_world[:, 1] - _PCR[1]) / vs
    corners_vox[:, 2] = (corners_world[:, 2] - pcr_z_min) / vs * z_scale

    proj = PerspectiveProjection(
        grid_shape=grid_shape,
        heading_deg=heading_deg,
        elev_deg=elev_deg,
        z_scale=z_scale,
        fov_deg=fov_deg,
        ego_height_m=ego_height_m,
        eye_back_m=eye_back_m,
        pcr_z_min=pcr_z_min,
        pcr_z_max=pcr_z_max,
    )

    # 投影: project 期望 (N_faces, N_verts, 3) → (N_faces, N_verts, 2)
    # 包装成 (1, 8, 3) 来投影全部角点
    pts_2d = proj.project(corners_vox[None, :, :])  # (1, 8, 2)
    pts_2d = pts_2d[0]  # (8, 2)

    if not np.isfinite(pts_2d).all():
        return

    # 12 条边
    edges = [
        (0,1),(1,2),(2,3),(3,0),  # 底面
        (4,5),(5,6),(6,7),(7,4),  # 顶面
        (0,4),(1,5),(2,6),(3,7),  # 垂直边
    ]
    for i, j in edges:
        ax.plot([pts_2d[i,0], pts_2d[j,0]],
                [pts_2d[i,1], pts_2d[j,1]],
                color="#1E90FF", linewidth=2.0, alpha=0.9, zorder=50)

    # 标注
    cx = pts_2d[:, 0].mean()
    cy = pts_2d[:, 1].min() - 0.05
    ax.text(cx, cy, "EGO", ha="center", va="bottom",
            fontsize=8, fontweight="bold", color="#1E90FF",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="#1E90FF", alpha=0.8),
            zorder=51)


def visualize_poster(img_inputs: tuple, occ_pred: np.ndarray,
                     cam_params: dict | None,
                     save_path: str, voxel_step: int = 1,
                     title: str = "FlashOCC - Poster Overview"):
    """海报风格 2×2 可视化.

    布局:
        ┌─────────────────┬─────────────────┐
        │  6相机原图 (2×3)  │ 6相机OCC体素(2×3) │
        ├─────────────────┼─────────────────┤
        │  3D透视(自车上方) │  BEV鸟瞰图       │
        └─────────────────┴─────────────────┘
    """
    imgs = _decode_input_images_to_bgr01(img_inputs)
    N = imgs.shape[0]
    H_img, W_img = imgs.shape[1], imgs.shape[2]

    renderer = OccVoxelRenderer(
        OccGrid.from_numpy(occ_pred), voxel_step=max(int(voxel_step), 1))

    gap = 16
    margin = 24
    tile_h = H_img
    tile_w = W_img
    half_w = tile_w * 3 + gap * 2
    top_h = tile_h * 2 + gap

    bottom_h = int(half_w * 0.62)
    title_h = 50
    canvas_w = margin * 2 + half_w * 2 + gap
    canvas_h = margin * 2 + title_h + top_h + gap + bottom_h
    canvas = np.full((canvas_h, canvas_w, 3), (246, 246, 246), dtype=np.uint8)

    cv2.putText(canvas, title, (margin, margin + 32), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (20, 20, 20), 2, cv2.LINE_AA)

    def _camera_grid(overlay_only: bool) -> np.ndarray:
        out = np.full((top_h, half_w, 3), (255, 255, 255) if overlay_only else (245, 245, 245), dtype=np.uint8)
        white_rgb = np.ones((H_img, W_img, 3), dtype=np.float32)
        for idx, (cam, row, col) in enumerate(_CAM_GRID):
            x = col * (tile_w + gap)
            y = row * (tile_h + gap)
            if idx >= N:
                continue

            if not overlay_only:
                tile = _rgb01_to_bgr8(imgs[idx])
            else:
                if cam_params is None or idx >= cam_params["sensor2keyegos"].shape[0]:
                    tile = np.zeros((H_img, W_img, 3), dtype=np.uint8)
                else:
                    cam_obj = CameraParams(
                        sensor2keyego=cam_params["sensor2keyegos"][idx],
                        intrinsics=cam_params["intrins"][idx],
                        post_rot=cam_params["post_rots"][idx],
                        post_trans=cam_params["post_trans"][idx],
                    )
                    tile = renderer.render_camera_overlay_image(
                        white_rgb, cam_obj, alpha=1.0, cam_name="")

            info = _CAM_INFO[cam]
            tile = _annotate_panel(tile, _short_cam(cam), _hex_to_bgr(info["color"]))
            out[y:y + tile_h, x:x + tile_w] = tile
        return out

    grid_raw = _camera_grid(overlay_only=False)
    grid_occ = _camera_grid(overlay_only=True)

    # 左下直接走体素透视渲染引擎, 使用更高机位+更广FOV, 获得广视野俯视图.
    topdown_wide = renderer.render_perspective_image(
        heading_deg=0.0,
        elev_deg=-72.0,
        z_scale=2.0,
        fov_deg=125.0,
        ego_height_m=20.0,
        eye_back_m=10.0,
        width=max(half_w, 960),
        height=max(bottom_h, 680),
        bg_bgr=(255, 255, 255),
    )
    # Align with camera correspondence convention.
    topdown_wide = _annotate_panel(
        _fit_to_panel(topdown_wide, bottom_h, half_w, bg_color=(255, 255, 255)),
        "Top-Down Wide View (renderer perspective)")

    bev = renderer.render_bev_image(draw_helpers=True, scale=6)
    bev = cv2.flip(bev, 1)
    bev = _annotate_panel(_fit_to_panel(bev, bottom_h, half_w),
                          "Bird's Eye View (BEV)")

    x0 = margin
    y0 = margin + title_h
    canvas[y0:y0 + top_h, x0:x0 + half_w] = grid_raw
    canvas[y0:y0 + top_h, x0 + half_w + gap:x0 + half_w + gap + half_w] = grid_occ

    y1 = y0 + top_h + gap
    canvas[y1:y1 + bottom_h, x0:x0 + half_w] = topdown_wide
    canvas[y1:y1 + bottom_h, x0 + half_w + gap:x0 + half_w + gap + half_w] = bev

    _save_cv2_image(save_path, canvas)


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

    _save_figure(fig, save_path, dpi=180, facecolor="white")


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
    out_tag = build_output_tag(args.checkpoint, args.config)

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
                sp = build_output_path(args.out_dir, out_tag, safe, layer_idx=idx+1)
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

                # ---- LSS 深度 3D 综合可视化 (点云 + 误差 + 曲面) ----
                lidar_ego = _load_lidar_points_ego(dataset, args.sample_idx)
                if lidar_ego is not None:
                    sp_d3d = build_output_path(args.out_dir, out_tag, "lss_depth_3d")
                    try:
                        visualize_lss_depth_3d(
                            extractor.depth, vis_in, lidar_ego, sp_d3d,
                            depth_cfg=depth_cfg, n_cams=6)
                    except Exception as e:
                        import traceback; traceback.print_exc()
                        print(f"    [跳过] LSS depth 3D: {e}")
                else:
                    print("  [注意] 无法加载 LiDAR 点云, 跳过 LSS depth 3D")
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
            title="FlashOCC - Input Cameras + 3D Occupancy Prediction",
        )

        # 生成海报风格 2×2 总览图
        poster_path = build_output_path(args.out_dir, out_tag, "poster")
        cam_params = _extract_cam_params_from_img_inputs(vis_in)
        visualize_poster(
            img_inputs=vis_in,
            occ_pred=occ_pred,
            cam_params=cam_params,
            save_path=poster_path,
            voxel_step=args.voxel_step,
            title="FlashOCC - Poster Overview",
        )

        # 生成 BEV 差异热力图 (需要GT)
        if occ_gt is not None:
            diff_path = build_output_path(args.out_dir, out_tag, "bev_diff")
            visualize_bev_diff_heatmap(
                occ_gt=occ_gt,
                occ_pred=occ_pred,
                save_path=diff_path,
                title="FlashOCC - BEV Prediction vs Ground Truth Difference",
            )

    print("\n" + "=" * 65)
    print(f"  可视化完成!  结果: {args.out_dir}/")
    print("=" * 65)
    for f in sorted(os.listdir(args.out_dir)):
        if os.path.isfile(os.path.join(args.out_dir, f)):
            print(f"    {f}")


if __name__ == "__main__":
    main()
