"""OccVoxelRenderer — 统一体素渲染引擎.

将面片生成、投影变换、Painter's Algorithm、PolyCollection 渲染
整合为一个干净的入口, 供上层可视化函数调用.

用法示例:
    from flashocc.vis import OccGrid, OccVoxelRenderer

    grid = OccGrid.from_numpy(occ_pred)
    renderer = OccVoxelRenderer(grid, voxel_step=2)

    # 等轴测渲染
    renderer.render_isometric(ax, azim_deg=45, elev_deg=35, z_scale=2.5)

    # BEV 渲染 (带辅助线)
    renderer.render_bev(ax, draw_helpers=True)

    # 相机叠加渲染
    renderer.render_camera_overlay(ax, img_rgb, cam_params)
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

from .occ_grid import OccGrid, FREE_CLASS, OTHERS_CLASS
from .colors import COLOR_LUT, OCC_CLASS_COLORS, cls_to_rgb, cls_to_rgba
from .faces import FaceData, generate_faces, generate_faces_camera
from .projection import (
    Projection,
    IsometricProjection,
    PerspectiveProjection,
    CameraProjection,
    CameraParams,
)


def painters_sort(
    polys_2d: np.ndarray,
    face_colors: np.ndarray,
    depth: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Painter's Algorithm: 按深度从远到近排序.

    Args:
        polys_2d:    (M, 4, 2) 屏幕坐标
        face_colors: (M, 4) RGBA 面颜色
        depth:       (M,) 深度值

    Returns:
        (sorted_polys, sorted_colors)
    """
    order = np.argsort(-depth)  # 远到近 (最远先画, 最近最后覆盖)
    return polys_2d[order], face_colors[order]


def _make_edge_colors(face_colors: np.ndarray, darken: float = 0.5, alpha: float = 0.3) -> np.ndarray:
    """由面颜色生成边颜色 (变暗 + 半透明)."""
    ec = face_colors.copy()
    ec[:, :3] = np.clip(ec[:, :3] * darken, 0.0, 1.0)
    ec[:, 3] = alpha
    return ec


class OccVoxelRenderer:
    """统一体素渲染引擎.

    核心设计:
        1. 接受 OccGrid 作为数据源
        2. 按需生成面片 (带邻接剔除缓存)
        3. 通过不同 Projection 渲染到 matplotlib Axes
        4. 统一使用 Painter's Algorithm 处理遮挡
    """

    def __init__(self, grid: OccGrid, voxel_step: int = 1):
        """
        Args:
            grid:       原始体素网格
            voxel_step: 下采样步长 (越大越快, 细节越少)
        """
        self._grid_full = grid
        self._grid = grid.subsample(voxel_step)
        self._voxel_step = max(1, int(voxel_step))
        # 缓存
        self._faces_iso: dict[float, FaceData] = {}
        self._faces_cam: FaceData | None = None

    @property
    def grid(self) -> OccGrid:
        """下采样后的工作网格."""
        return self._grid

    @property
    def grid_full(self) -> OccGrid:
        """原始分辨率网格."""
        return self._grid_full

    # ── 面片生成 (带缓存) ────────────────────────────────────

    def _get_faces_iso(self, z_scale: float) -> FaceData:
        if z_scale not in self._faces_iso:
            self._faces_iso[z_scale] = generate_faces(
                self._grid, adjacency_cull=True, z_scale=z_scale
            )
        return self._faces_iso[z_scale]

    def _get_faces_cam(self) -> FaceData:
        if self._faces_cam is None:
            self._faces_cam = generate_faces_camera(
                self._grid_full.subsample(max(self._voxel_step, 2)),
            )
        return self._faces_cam

    # ── 通用渲染管线 ─────────────────────────────────────────

    def _render_faces_on_ax(
        self,
        ax: plt.Axes,
        faces: FaceData,
        proj: Projection,
        *,
        bg_color: str = "#EEEEEE",
        edge_lw: float = 0.15,
    ) -> bool:
        """通用渲染: 面片 → 投影 → 排序 → PolyCollection.

        Returns:
            True 如果渲染成功 (有可见面片), False 否则.
        """
        if faces.empty:
            return False

        # 投影到 2D
        polys_2d = proj.project(faces.verts_3d)
        depth = proj.compute_depth(faces.centers)

        # 可见性过滤
        vis_mask = proj.filter_visible(polys_2d, depth)
        if not vis_mask.any():
            return False

        polys_2d = polys_2d[vis_mask]
        depth = depth[vis_mask]
        cls_ids = faces.cls_ids[vis_mask]
        shades = faces.shades[vis_mask]

        # 着色
        face_colors = np.ones((len(cls_ids), 4), dtype=np.float64)
        face_colors[:, :3] = np.clip(
            COLOR_LUT[np.clip(cls_ids, 0, 255)] * shades[:, None], 0.0, 1.0
        )

        # Painter's sort
        polys_sorted, fc_sorted = painters_sort(polys_2d, face_colors, depth)
        ec_sorted = _make_edge_colors(fc_sorted)

        # 渲染
        ax.set_facecolor(bg_color)
        pc = PolyCollection(polys_sorted, closed=True)
        pc.set_facecolor(fc_sorted)
        pc.set_edgecolor(ec_sorted)
        pc.set_linewidth(edge_lw)
        ax.add_collection(pc)
        ax.autoscale()
        ax.set_aspect("equal")
        return True

    # ── 公开渲染接口 ─────────────────────────────────────────

    def render_isometric(
        self,
        ax: plt.Axes,
        *,
        azim_deg: float = 45.0,
        elev_deg: float = 35.0,
        z_scale: float = 2.5,
        title: str = "3D Isometric",
        fontsize: int = 11,
        show_axes: bool = False,
    ) -> None:
        """等轴测体素渲染 (软件光照 + Painter's Algorithm + 邻接剔除)."""
        faces = self._get_faces_iso(z_scale)

        if faces.empty:
            ax.text(0.5, 0.5, "No occupied voxels", transform=ax.transAxes,
                    ha="center", va="center", fontsize=14)
            ax.axis("off")
            ax.set_title(title, fontsize=fontsize, fontweight="bold")
            return

        proj = IsometricProjection(azim_deg=azim_deg, elev_deg=elev_deg)
        ok = self._render_faces_on_ax(ax, faces, proj)

        if not ok:
            ax.text(0.5, 0.5, "No exposed faces", transform=ax.transAxes,
                    ha="center", va="center")

        if not show_axes:
            ax.axis("off")

        ax.text(0.01, 0.03,
                f"↑X(front) ←Y(left) Z↑(height×{z_scale})",
                transform=ax.transAxes, fontsize=7, color="#333",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))
        ax.set_title(title, fontsize=fontsize, fontweight="bold", pad=8)

    def render_perspective(
        self,
        ax: plt.Axes,
        *,
        heading_deg: float = 0.0,
        elev_deg: float = 8.0,
        z_scale: float = 1.5,
        fov_deg: float = 90.0,
        title: str = "3D Perspective",
        fontsize: int = 11,
    ) -> None:
        """从自车位置的透视渲染."""
        faces = self._get_faces_iso(z_scale)

        if faces.empty:
            ax.text(0.5, 0.5, "No occupied voxels", transform=ax.transAxes,
                    ha="center", va="center", fontsize=14)
            ax.axis("off")
            ax.set_title(title, fontsize=fontsize, fontweight="bold")
            return

        proj = PerspectiveProjection(
            grid_shape=self._grid.shape,
            heading_deg=heading_deg,
            elev_deg=elev_deg,
            z_scale=z_scale,
            fov_deg=fov_deg,
            pcr_z_min=self._grid.pcr[2],
            pcr_z_max=self._grid.pcr[5],
        )
        ok = self._render_faces_on_ax(ax, faces, proj)

        if not ok:
            ax.text(0.5, 0.5, "No visible faces", transform=ax.transAxes,
                    ha="center", va="center")

        ax.axis("off")
        ax.set_title(title, fontsize=fontsize, fontweight="bold", pad=8)

    def render_bev(
        self,
        ax: plt.Axes,
        *,
        draw_helpers: bool = True,
        title: str = "BEV",
        fontsize: int = 11,
    ) -> None:
        """BEV (鸟瞰视角) 像素化渲染.

        使用 Z 轴投影 (非 3D 面片), 直接生成 (Dx, Dy, 3) 的 RGB 图像.
        """
        # 直接使用原始网格的投影 (不用下采样)
        grid = self._grid_full
        bev_cls = grid.bev_projection()
        bev_rgb = cls_to_rgb(bev_cls)

        ax.imshow(bev_rgb, origin="lower", aspect="equal")

        if draw_helpers:
            from .bev_helpers import draw_camera_fovs, draw_ego_vehicle, add_bev_annotations
            draw_camera_fovs(ax, grid)
            draw_ego_vehicle(ax, grid)
            add_bev_annotations(ax, grid, title=title, fontsize=fontsize)
        else:
            ax.axis("off")
            ax.set_title(title, fontsize=fontsize, fontweight="bold", pad=8)

    def render_camera_overlay(
        self,
        ax: plt.Axes,
        img_rgb: np.ndarray,
        cam: CameraParams,
        *,
        alpha: float = 0.28,
        cam_name: str = "",
    ) -> None:
        """将体素半透明叠加到相机图像上.

        投影链: ego → cam → pixel → post_aug
        使用离屏渲染 + 单次 alpha 混合避免多层透明度累积错误.
        """
        H, W = img_rgb.shape[:2]
        ax.imshow(img_rgb)

        faces = self._get_faces_cam()
        if faces.empty:
            return

        proj = CameraProjection(cam=cam, img_hw=(H, W))

        # 投影
        polys_2d = proj.project(faces.verts_3d)
        depth = proj.compute_depth(faces.centers)

        # 可见性过滤
        vis = proj.filter_visible(polys_2d, depth)
        if not vis.any():
            return

        polys_2d = polys_2d[vis]
        depth = depth[vis]
        cls_ids = faces.cls_ids[vis]
        shades = faces.shades[vis]

        # 着色
        face_colors = np.ones((len(cls_ids), 4), dtype=np.float64)
        face_colors[:, :3] = np.clip(
            COLOR_LUT[np.clip(cls_ids, 0, 255)] * shades[:, None], 0.0, 1.0
        )

        # Painter's sort
        order = np.argsort(-depth)
        polys_2d = polys_2d[order]
        face_colors = face_colors[order]

        edge_colors = _make_edge_colors(face_colors)

        # 离屏渲染: 先绘制完整的不透明 overlay, 再单次 alpha 叠加
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
            polys_2d, closed=True,
            facecolors=face_colors,
            edgecolors=edge_colors,
            linewidths=0.12,
        )
        ax_tmp.add_collection(pc)
        fig_tmp.canvas.draw()

        buf = np.asarray(fig_tmp.canvas.buffer_rgba()).astype(np.float32) / 255.0
        plt.close(fig_tmp)

        overlay_rgb = buf[..., :3]
        overlay_a = buf[..., 3]
        if overlay_a.max() > 0:
            ax.imshow(overlay_rgb, alpha=overlay_a * alpha)

        # 标注相机在 key ego 帧中的位置
        try:
            cam_pos = cam.sensor2keyego[:3, 3]
            txt = f"cam@ego=({cam_pos[0]:+.1f},{cam_pos[1]:+.1f},{cam_pos[2]:+.1f})m"
            ax.text(0.01, 0.98, txt,
                    transform=ax.transAxes, ha="left", va="top",
                    fontsize=7, color="yellow",
                    bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.5))
        except Exception:
            pass
