"""BEV 辅助绘制 — 自车轮廓、相机视场角、网格刻度.

所有辅助绘制函数都接受 OccGrid 作为参数,
从中获取空间范围和体素尺寸, 消除重复的全局常量引用.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from .occ_grid import OccGrid


# =====================================================================
#  NuScenes 6 相机布局
# =====================================================================

CAM_ORDER: list[str] = [
    "CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT",  "CAM_BACK",  "CAM_BACK_RIGHT",
]

CAM_INFO: dict[str, dict] = {
    "CAM_FRONT_LEFT":  dict(heading=+55.0,  hfov=70.0,  range_m=50.0, color="#FF6B6B"),
    "CAM_FRONT":       dict(heading=  0.0,  hfov=70.0,  range_m=50.0, color="#FFD93D"),
    "CAM_FRONT_RIGHT": dict(heading=-55.0,  hfov=70.0,  range_m=50.0, color="#6BCB77"),
    "CAM_BACK_LEFT":   dict(heading=+110.0, hfov=70.0,  range_m=50.0, color="#4D96FF"),
    "CAM_BACK":        dict(heading= 180.0, hfov=110.0, range_m=50.0, color="#C77DFF"),
    "CAM_BACK_RIGHT":  dict(heading=-110.0, hfov=70.0,  range_m=50.0, color="#FF9F1C"),
}

# 雷诺 Zoé 尺寸 (m)
ZOE_LENGTH: float = 4.084
ZOE_WIDTH: float = 1.730
ZOE_WHEELBASE: float = 2.588


# =====================================================================
#  自车轮廓绘制
# =====================================================================

def draw_ego_vehicle(
    ax: plt.Axes,
    grid: OccGrid,
    *,
    linewidth: float = 2.0,
    color: str = "#1E90FF",
    zorder: int = 10,
) -> None:
    """绘制雷诺 Zoé 自车轮廓 + 前进方向箭头."""
    cx, cr = grid.world_to_bev_px(0, 0)
    vs = grid.vs

    corners_w = [
        (-ZOE_LENGTH / 2, -ZOE_WIDTH / 2),
        (+ZOE_LENGTH / 2, -ZOE_WIDTH / 2),
        (+ZOE_LENGTH / 2, +ZOE_WIDTH / 2),
        (-ZOE_LENGTH / 2, +ZOE_WIDTH / 2),
        (-ZOE_LENGTH / 2, -ZOE_WIDTH / 2),
    ]
    cols = [(y - grid.pcr[1]) / vs for (_, y) in corners_w]
    rows = [(x - grid.pcr[0]) / vs for (x, _) in corners_w]

    ax.fill(cols, rows, color=color, alpha=0.30, zorder=zorder)
    ax.plot(cols, rows, color=color, linewidth=linewidth, zorder=zorder)

    # 前进方向箭头 (+X = 向上 in BEV)
    arrow_end = cr + (ZOE_LENGTH / 2 + 2.5) / vs
    ax.annotate("", xy=(cx, arrow_end), xytext=(cx, cr),
                arrowprops=dict(arrowstyle="->", color=color, lw=2.0),
                zorder=zorder + 1)

    # 轴距虚线
    wb_px = ZOE_WHEELBASE / vs
    ax.plot([cx, cx], [cr - wb_px / 2, cr + wb_px / 2],
            color=color, linewidth=1.0, linestyle=":", alpha=0.7, zorder=zorder)

    # 车型标注
    ax.text(cx + 2, cr, f"Renault Zoé\n{ZOE_LENGTH}m×{ZOE_WIDTH}m",
            fontsize=6, color=color, va="center",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color,
                      alpha=0.75, linewidth=0.5))


# =====================================================================
#  相机视场角绘制
# =====================================================================

def draw_camera_fovs(
    ax: plt.Axes,
    grid: OccGrid,
    *,
    alpha_fill: float = 0.07,
    alpha_line: float = 0.65,
    zorder: int = 5,
) -> list[Patch]:
    """绘制 6 相机视场角扇形辅助线."""
    ego_col, ego_row = grid.world_to_bev_px(0, 0)
    legend_patches = []

    for cam_name in CAM_ORDER:
        info = CAM_INFO[cam_name]
        heading_rad = np.radians(info["heading"])
        half_fov_rad = np.radians(info["hfov"] / 2.0)
        color = info["color"]

        n_pts = 40
        angles = np.linspace(
            heading_rad - half_fov_rad,
            heading_rad + half_fov_rad,
            n_pts,
        )
        wx = np.cos(angles) * info["range_m"]
        wy = np.sin(angles) * info["range_m"]
        arc_cols = (wy - grid.pcr[1]) / grid.vs
        arc_rows = (wx - grid.pcr[0]) / grid.vs

        fan_cols = np.concatenate([[ego_col], arc_cols, [ego_col]])
        fan_rows = np.concatenate([[ego_row], arc_rows, [ego_row]])

        ax.fill(fan_cols, fan_rows, color=color, alpha=alpha_fill, zorder=zorder)
        ax.plot(fan_cols, fan_rows, color=color, linewidth=1.0,
                alpha=alpha_line, zorder=zorder)

        # 标签
        f_col = arc_cols[n_pts // 2] * 0.6 + ego_col * 0.4
        f_row = arc_rows[n_pts // 2] * 0.6 + ego_row * 0.4
        short = cam_name.replace("CAM_", "").replace("_", "\n")
        ax.text(f_col, f_row, short, fontsize=5.5, ha="center", va="center",
                color=color, fontweight="bold", zorder=zorder + 1,
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec=color,
                          alpha=0.75, linewidth=0.5))

        legend_patches.append(
            Patch(facecolor=color, alpha=0.6, edgecolor=color,
                  label=f"{cam_name.replace('CAM_', '').replace('_', ' ')} "
                        f"(hdg={info['heading']:+.0f}°, HFoV={info['hfov']:.0f}°)"))

    return legend_patches


# =====================================================================
#  BEV 轴标注
# =====================================================================

def add_bev_annotations(
    ax: plt.Axes,
    grid: OccGrid,
    *,
    title: str = "",
    fontsize: int = 13,
    tick_m: float = 10.0,
) -> None:
    """BEV 轴: 刻度、网格、方位文字."""
    vs = grid.vs
    ticks_m = np.arange(grid.pcr[0], grid.pcr[3] + 1, tick_m)

    ax.set_xticks([(t - grid.pcr[1]) / vs for t in ticks_m])
    ax.set_xticklabels([f"{int(t)}" for t in ticks_m], fontsize=6)
    ax.set_yticks([(t - grid.pcr[0]) / vs for t in ticks_m])
    ax.set_yticklabels([f"{int(t)}" for t in ticks_m], fontsize=6)
    ax.set_xlabel("Y / m  (right ← | → left )", fontsize=8)
    ax.set_ylabel("X / m  (rear  ↓ | ↑  front)", fontsize=8)

    for t in ticks_m:
        ax.axvline((t - grid.pcr[1]) / vs, color="gray", lw=0.3, alpha=0.4)
        ax.axhline((t - grid.pcr[0]) / vs, color="gray", lw=0.3, alpha=0.4)

    cx, _ = grid.world_to_bev_px(0, 0)
    kw = dict(ha="center", fontsize=7, color="white", fontweight="bold",
              bbox=dict(boxstyle="round,pad=0.2", fc="#333", alpha=0.7))
    ax.text(cx, grid.Dx - 2, "FRONT ▲", va="top", **kw)
    ax.text(cx, 2, "REAR  ▼", va="bottom", **kw)

    if title:
        ax.set_title(title, fontsize=fontsize, fontweight="bold", pad=8)


# =====================================================================
#  相机图例
# =====================================================================

def build_cam_legend_patches() -> list[Patch]:
    """构建 6 相机 + 自车图例 Patch 列表."""
    patches = [
        Patch(facecolor=info["color"], edgecolor=info["color"], alpha=0.7,
              label=f"{name.replace('CAM_', '').replace('_', ' ')} "
                    f"[hdg={info['heading']:+.0f}° HFoV={info['hfov']:.0f}°]")
        for name, info in CAM_INFO.items()
    ]
    patches.append(
        Patch(facecolor="#1E90FF", alpha=0.35, edgecolor="#1E90FF",
              label=f"Ego — Renault Zoé ({ZOE_LENGTH}m×{ZOE_WIDTH}m, wb={ZOE_WHEELBASE}m)")
    )
    return patches
