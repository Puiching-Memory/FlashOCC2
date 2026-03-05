"""颜色管理 — OCC 语义类别颜色表 / LUT / 图例生成.

集中管理所有颜色映射, 消除散布各处的颜色定义和转换代码.
"""
from __future__ import annotations

import numpy as np
from matplotlib.patches import Patch

from flashocc.constants import OCC_CLASS_NAMES

# =====================================================================
#  18 类语义颜色表 (RGB, 0-1 浮点)
# =====================================================================

OCC_CLASS_COLORS: dict[int, tuple[float, float, float]] = {
    # 使用 vis_occ.py 同源配色 (RGB / 255)
    0:  (0 / 255, 0 / 255, 0 / 255),        # others
    1:  (255 / 255, 120 / 255, 50 / 255),   # barrier
    2:  (255 / 255, 192 / 255, 203 / 255),  # bicycle
    3:  (255 / 255, 255 / 255, 0 / 255),    # bus
    4:  (0 / 255, 150 / 255, 245 / 255),    # car
    5:  (0 / 255, 255 / 255, 255 / 255),    # construction_vehicle
    6:  (200 / 255, 180 / 255, 0 / 255),    # motorcycle
    7:  (255 / 255, 0 / 255, 0 / 255),      # pedestrian
    8:  (255 / 255, 240 / 255, 150 / 255),  # traffic_cone
    9:  (135 / 255, 60 / 255, 0 / 255),     # trailer
    10: (160 / 255, 32 / 255, 240 / 255),   # truck
    11: (255 / 255, 0 / 255, 255 / 255),    # driveable_surface
    12: (175 / 255, 0 / 255, 75 / 255),     # other_flat
    13: (75 / 255, 0 / 255, 75 / 255),      # sidewalk
    14: (150 / 255, 240 / 255, 80 / 255),   # terrain
    15: (230 / 255, 230 / 255, 250 / 255),  # manmade
    16: (0 / 255, 175 / 255, 0 / 255),      # vegetation
    17: (255 / 255, 255 / 255, 255 / 255),  # free
}


def _build_color_lut() -> np.ndarray:
    """构建 (256, 3) 颜色查找表, 索引即类别 ID."""
    lut = np.full((256, 3), 0.5, dtype=np.float64)
    for cls_id, rgb in OCC_CLASS_COLORS.items():
        if cls_id < 256:
            lut[cls_id] = rgb
    return lut


COLOR_LUT: np.ndarray = _build_color_lut()
"""(256, 3) 颜色查找表: COLOR_LUT[class_id] → (R, G, B) ∈ [0,1]."""


def cls_to_rgb(cls_2d: np.ndarray) -> np.ndarray:
    """类别 ID 数组 → RGB 图像.

    Args:
        cls_2d: (...) uint8 类别 ID

    Returns:
        (..., 3) float64 RGB ∈ [0, 1]
    """
    flat = cls_2d.flatten().astype(np.intp)
    rgb = COLOR_LUT[np.clip(flat, 0, 255)]
    return rgb.reshape(*cls_2d.shape, 3)


def cls_to_rgba(cls_ids: np.ndarray, shade: float = 1.0, alpha: float = 1.0) -> np.ndarray:
    """类别 ID 数组 → RGBA 颜色, 可选明度缩放.

    Args:
        cls_ids: (N,) uint8/int 类别 ID
        shade:   明度乘数 (0-1)
        alpha:   透明度 (0-1)

    Returns:
        (N, 4) float64 RGBA
    """
    rgb = COLOR_LUT[np.clip(cls_ids.flatten().astype(np.intp), 0, 255)]
    rgb = np.clip(rgb * shade, 0.0, 1.0)
    a = np.full((rgb.shape[0], 1), alpha, dtype=np.float64)
    return np.concatenate([rgb, a], axis=1)


def build_class_legend(
    present_classes: set[int],
    counts: dict[int, int] | None = None,
    show_count: bool = True,
) -> list[Patch]:
    """为出现的类别生成 matplotlib 图例 Patch 列表.

    Args:
        present_classes: 出现的类别 ID 集合
        counts:          各类别体素数 {cls_id: count}
        show_count:      是否在标签中显示数量
    """
    patches = []
    for c in sorted(present_classes):
        name = OCC_CLASS_NAMES[c] if c < len(OCC_CLASS_NAMES) else f"cls{c}"
        label = name
        if show_count and counts and c in counts:
            label = f"{name} ({counts[c]:,})"
        patches.append(
            Patch(
                facecolor=OCC_CLASS_COLORS.get(c, (0.5, 0.5, 0.5)),
                edgecolor="k",
                lw=0.4,
                label=label,
            )
        )
    return patches
