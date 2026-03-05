"""OccVoxelRenderer — 基于 numpy + OpenCV 的统一体素渲染引擎.

实现目标:
1. 体素面片计算保持 numpy 向量化.
2. 2D 光栅化使用 OpenCV (fillConvexPoly/polylines).
3. 支持直接保存图片 (cv2.imwrite), 不依赖 matplotlib savefig.
4. 保持原有 matplotlib Axes API 兼容, 便于旧代码平滑迁移.
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .occ_grid import OccGrid
from .colors import COLOR_LUT, cls_to_rgb
from .faces import FaceData, generate_faces
from .projection import (
    Projection,
    IsometricProjection,
    PerspectiveProjection,
    CameraProjection,
    CameraParams,
)
from .bev_helpers import CAM_ORDER, CAM_INFO, ZOE_LENGTH, ZOE_WIDTH

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - matplotlib is optional at runtime
    plt = None


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

    @staticmethod
    def _as_bgr_uint8(rgb: np.ndarray) -> np.ndarray:
        arr = np.asarray(rgb)
        if arr.dtype != np.uint8:
            arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        return arr[..., ::-1].copy()

    @staticmethod
    def _bgr_to_rgb01(bgr: np.ndarray) -> np.ndarray:
        return bgr[..., ::-1].astype(np.float32) / 255.0

    @staticmethod
    def _fit_to_canvas(polys_2d: np.ndarray, width: int, height: int, pad: int = 24) -> np.ndarray:
        finite = np.isfinite(polys_2d).all(axis=(1, 2))
        if not finite.any():
            return np.empty((0, 4, 2), dtype=np.int32)

        p = polys_2d[finite]
        x_min = float(p[..., 0].min())
        x_max = float(p[..., 0].max())
        y_min = float(p[..., 1].min())
        y_max = float(p[..., 1].max())
        span_x = max(x_max - x_min, 1e-6)
        span_y = max(y_max - y_min, 1e-6)

        draw_w = max(width - 2 * pad, 1)
        draw_h = max(height - 2 * pad, 1)
        scale = min(draw_w / span_x, draw_h / span_y)

        x = (p[..., 0] - x_min) * scale + pad
        y = height - 1 - ((p[..., 1] - y_min) * scale + pad)
        out = np.stack([x, y], axis=-1)
        out = np.round(out).astype(np.int32)
        return out

    @staticmethod
    def _draw_polygons_bgr(
        canvas: np.ndarray,
        polys_px: np.ndarray,
        face_colors: np.ndarray,
        *,
        edge_lw: int = 1,
        edge_alpha: float = 0.35,
    ) -> None:
        if len(polys_px) == 0:
            return

        colors_rgb = np.clip(face_colors[:, :3], 0.0, 1.0)
        colors_bgr = np.clip(colors_rgb[:, ::-1] * 255.0, 0, 255).astype(np.uint8)
        edge_bgr = np.clip(colors_bgr.astype(np.float32) * 0.5, 0, 255).astype(np.uint8)

        for poly, c_fill, c_edge in zip(polys_px, colors_bgr, edge_bgr):
            cv2.fillConvexPoly(canvas, poly, color=(int(c_fill[0]), int(c_fill[1]), int(c_fill[2])), lineType=cv2.LINE_AA)
            if edge_lw > 0:
                cv2.polylines(
                    canvas,
                    [poly],
                    isClosed=True,
                    color=(int(c_edge[0]), int(c_edge[1]), int(c_edge[2])),
                    thickness=edge_lw,
                    lineType=cv2.LINE_AA,
                )

        if edge_alpha < 1.0:
            canvas[:] = np.clip(canvas.astype(np.float32), 0, 255).astype(np.uint8)

    def _render_faces_to_image(
        self,
        faces: FaceData,
        proj: Projection,
        *,
        width: int,
        height: int,
        bg_bgr: tuple[int, int, int],
        edge_lw: int = 1,
    ) -> np.ndarray:
        canvas = np.full((height, width, 3), bg_bgr, dtype=np.uint8)
        if faces.empty:
            return canvas

        polys_2d = proj.project(faces.verts_3d)
        depth = proj.compute_depth(faces.centers)

        vis = proj.filter_visible(polys_2d, depth)
        if not vis.any():
            return canvas

        polys_2d = polys_2d[vis]
        depth = depth[vis]
        cls_ids = faces.cls_ids[vis]
        shades = faces.shades[vis]

        face_colors = np.ones((len(cls_ids), 4), dtype=np.float64)
        face_colors[:, :3] = np.clip(COLOR_LUT[np.clip(cls_ids, 0, 255)] * shades[:, None], 0.0, 1.0)

        polys_sorted, fc_sorted = painters_sort(polys_2d, face_colors, depth)
        polys_px = self._fit_to_canvas(polys_sorted, width=width, height=height)
        self._draw_polygons_bgr(canvas, polys_px, fc_sorted, edge_lw=edge_lw)
        return canvas

    def _draw_bev_helpers_cv2(self, img_bgr: np.ndarray, scale: int) -> None:
        pcr = self._grid_full.pcr
        vs = self._grid_full.vs
        H, W = img_bgr.shape[:2]

        def world_to_px(x: float, y: float) -> tuple[int, int]:
            col = (y - pcr[1]) / vs * scale
            row = (x - pcr[0]) / vs * scale
            py = int(np.clip(H - 1 - row, 0, H - 1))
            px = int(np.clip(col, 0, W - 1))
            return px, py

        tick_m = 10.0
        for t in np.arange(pcr[0], pcr[3] + 1e-6, tick_m):
            _, y = world_to_px(t, pcr[1])
            cv2.line(img_bgr, (0, y), (W - 1, y), (170, 170, 170), 1, cv2.LINE_AA)
        for t in np.arange(pcr[1], pcr[4] + 1e-6, tick_m):
            x, _ = world_to_px(pcr[0], t)
            cv2.line(img_bgr, (x, 0), (x, H - 1), (170, 170, 170), 1, cv2.LINE_AA)

        ego_poly_world = np.array([
            [-ZOE_LENGTH / 2, -ZOE_WIDTH / 2],
            [ZOE_LENGTH / 2, -ZOE_WIDTH / 2],
            [ZOE_LENGTH / 2, ZOE_WIDTH / 2],
            [-ZOE_LENGTH / 2, ZOE_WIDTH / 2],
        ], dtype=np.float32)
        ego_poly = np.array([world_to_px(float(x), float(y)) for x, y in ego_poly_world], dtype=np.int32)
        cv2.fillConvexPoly(img_bgr, ego_poly, (255, 110, 30), lineType=cv2.LINE_AA)
        cv2.polylines(img_bgr, [ego_poly], isClosed=True, color=(255, 220, 180), thickness=2, lineType=cv2.LINE_AA)

        ego_px = world_to_px(0.0, 0.0)
        for cam_name in CAM_ORDER:
            info = CAM_INFO[cam_name]
            color_hex = info["color"].lstrip("#")
            rgb = tuple(int(color_hex[i:i + 2], 16) for i in (0, 2, 4))
            bgr = (rgb[2], rgb[1], rgb[0])

            heading = np.radians(info["heading"])
            half = np.radians(info["hfov"] / 2.0)
            angles = np.linspace(heading - half, heading + half, 36)
            arc_pts = np.array(
                [world_to_px(float(np.cos(a) * info["range_m"]), float(np.sin(a) * info["range_m"])) for a in angles],
                dtype=np.int32,
            )
            fan = np.vstack([np.array([ego_px], dtype=np.int32), arc_pts])
            overlay = img_bgr.copy()
            cv2.fillPoly(overlay, [fan], color=bgr, lineType=cv2.LINE_AA)
            cv2.addWeighted(overlay, 0.10, img_bgr, 0.90, 0, img_bgr)
            cv2.polylines(img_bgr, [fan], isClosed=True, color=bgr, thickness=1, lineType=cv2.LINE_AA)

        cv2.putText(img_bgr, "FRONT", (max(6, ego_px[0] - 26), 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 1, cv2.LINE_AA)
        cv2.putText(img_bgr, "REAR", (max(6, ego_px[0] - 22), H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 1, cv2.LINE_AA)

    @staticmethod
    def _save_bgr(path: str, image_bgr: np.ndarray) -> None:
        ok = cv2.imwrite(path, image_bgr)
        if not ok:
            raise RuntimeError(f"Failed to save image: {path}")

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
            # 相机叠加仅渲染暴露面，避免内部面导致的棋盘/网格伪影。
            sub_grid = self._grid_full.subsample(max(self._voxel_step, 2))
            faces_vox = generate_faces(sub_grid, adjacency_cull=True, z_scale=1.0)
            if faces_vox.empty:
                self._faces_cam = faces_vox
            else:
                vs = float(sub_grid.vs)
                pcr = sub_grid.pcr

                verts_world = faces_vox.verts_3d.copy()
                verts_world[..., 0] = pcr[0] + verts_world[..., 0] * vs
                verts_world[..., 1] = pcr[1] + verts_world[..., 1] * vs
                verts_world[..., 2] = pcr[2] + verts_world[..., 2] * vs

                centers_world = faces_vox.centers.copy()
                centers_world[:, 0] = pcr[0] + centers_world[:, 0] * vs
                centers_world[:, 1] = pcr[1] + centers_world[:, 1] * vs
                centers_world[:, 2] = pcr[2] + centers_world[:, 2] * vs

                self._faces_cam = FaceData(
                    verts_3d=verts_world,
                    cls_ids=faces_vox.cls_ids,
                    shades=faces_vox.shades,
                    centers=centers_world,
                )
        return self._faces_cam

    # ── 通用渲染管线 ─────────────────────────────────────────

    def _render_to_ax(self, ax: Any, image_bgr: np.ndarray, title: str, fontsize: int) -> None:
        ax.imshow(self._bgr_to_rgb01(image_bgr))
        ax.axis("off")
        ax.set_title(title, fontsize=fontsize, fontweight="bold", pad=8)

    def render_isometric_image(
        self,
        *,
        azim_deg: float = 45.0,
        elev_deg: float = 35.0,
        z_scale: float = 2.5,
        width: int = 1280,
        height: int = 960,
        bg_bgr: tuple[int, int, int] = (238, 238, 238),
    ) -> np.ndarray:
        faces = self._get_faces_iso(z_scale)
        proj = IsometricProjection(azim_deg=azim_deg, elev_deg=elev_deg)
        return self._render_faces_to_image(
            faces,
            proj,
            width=width,
            height=height,
            bg_bgr=bg_bgr,
            edge_lw=1,
        )

    def render_perspective_image(
        self,
        *,
        heading_deg: float = 0.0,
        elev_deg: float = 8.0,
        z_scale: float = 1.5,
        fov_deg: float = 90.0,
        ego_height_m: float = 1.5,
        eye_back_m: float = 0.0,
        width: int = 1280,
        height: int = 960,
        bg_bgr: tuple[int, int, int] = (238, 238, 238),
    ) -> np.ndarray:
        faces = self._get_faces_iso(z_scale)
        proj = PerspectiveProjection(
            grid_shape=self._grid.shape,
            heading_deg=heading_deg,
            elev_deg=elev_deg,
            z_scale=z_scale,
            fov_deg=fov_deg,
            ego_height_m=ego_height_m,
            eye_back_m=eye_back_m,
            pcr_z_min=self._grid.pcr[2],
            pcr_z_max=self._grid.pcr[5],
        )
        return self._render_faces_to_image(
            faces,
            proj,
            width=width,
            height=height,
            bg_bgr=bg_bgr,
            edge_lw=1,
        )

    def render_bev_image(
        self,
        *,
        draw_helpers: bool = True,
        scale: int = 4,
    ) -> np.ndarray:
        grid = self._grid_full
        bev_cls = grid.bev_projection()
        bev_rgb = cls_to_rgb(bev_cls)
        bev_bgr = self._as_bgr_uint8(np.flipud(bev_rgb))
        if scale > 1:
            bev_bgr = cv2.resize(
                bev_bgr,
                (grid.Dy * scale, grid.Dx * scale),
                interpolation=cv2.INTER_NEAREST,
            )
        if draw_helpers:
            self._draw_bev_helpers_cv2(bev_bgr, scale=scale)
        return bev_bgr

    def render_camera_overlay_image(
        self,
        img_rgb: np.ndarray,
        cam: CameraParams,
        *,
        alpha: float = 0.28,
        cam_name: str = "",
    ) -> np.ndarray:
        base = self._as_bgr_uint8(img_rgb)
        H, W = base.shape[:2]

        faces = self._get_faces_cam()
        if faces.empty:
            return base

        proj = CameraProjection(cam=cam, img_hw=(H, W))
        polys_2d = proj.project(faces.verts_3d)
        depth = proj.compute_depth(faces.centers)
        vis = proj.filter_visible(polys_2d, depth)
        if not vis.any():
            return base

        polys_2d = polys_2d[vis]
        depth = depth[vis]
        cls_ids = faces.cls_ids[vis]
        shades = faces.shades[vis]

        face_colors = np.ones((len(cls_ids), 4), dtype=np.float64)
        face_colors[:, :3] = np.clip(COLOR_LUT[np.clip(cls_ids, 0, 255)] * shades[:, None], 0.0, 1.0)

        polys_sorted, fc_sorted = painters_sort(polys_2d, face_colors, depth)
        polys_px = np.round(polys_sorted).astype(np.int32)

        overlay = np.zeros_like(base)
        self._draw_polygons_bgr(overlay, polys_px, fc_sorted, edge_lw=1)

        mask = np.any(overlay > 0, axis=-1)
        if mask.any():
            out = base.copy()
            out_f = out.astype(np.float32)
            over_f = overlay.astype(np.float32)
            out_f[mask] = out_f[mask] * (1.0 - alpha) + over_f[mask] * alpha
            out = np.clip(out_f, 0, 255).astype(np.uint8)
        else:
            out = base

        if cam_name:
            cv2.putText(
                out,
                cam_name,
                (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (245, 245, 245),
                1,
                cv2.LINE_AA,
            )
        return out

    def save_isometric(self, path: str, **kwargs) -> str:
        img = self.render_isometric_image(**kwargs)
        self._save_bgr(path, img)
        return path

    def save_perspective(self, path: str, **kwargs) -> str:
        img = self.render_perspective_image(**kwargs)
        self._save_bgr(path, img)
        return path

    def save_bev(self, path: str, **kwargs) -> str:
        img = self.render_bev_image(**kwargs)
        self._save_bgr(path, img)
        return path

    def save_camera_overlay(self, path: str, img_rgb: np.ndarray, cam: CameraParams, **kwargs) -> str:
        img = self.render_camera_overlay_image(img_rgb, cam, **kwargs)
        self._save_bgr(path, img)
        return path

    # ── 公开渲染接口 ─────────────────────────────────────────

    def render_isometric(
        self,
        ax: Any,
        *,
        azim_deg: float = 45.0,
        elev_deg: float = 35.0,
        z_scale: float = 2.5,
        title: str = "3D Isometric",
        fontsize: int = 11,
        show_axes: bool = False,
    ) -> None:
        """兼容接口: 渲染到 matplotlib Axes (底层由 OpenCV 完成)."""
        img = self.render_isometric_image(
            azim_deg=azim_deg,
            elev_deg=elev_deg,
            z_scale=z_scale,
        )
        self._render_to_ax(ax, img, title, fontsize)
        if show_axes:
            ax.axis("on")

    def render_perspective(
        self,
        ax: Any,
        *,
        heading_deg: float = 0.0,
        elev_deg: float = 8.0,
        z_scale: float = 1.5,
        fov_deg: float = 90.0,
        ego_height_m: float = 1.5,
        eye_back_m: float = 0.0,
        title: str = "3D Perspective",
        fontsize: int = 11,
    ) -> None:
        """兼容接口: 渲染到 matplotlib Axes (底层由 OpenCV 完成)."""
        img = self.render_perspective_image(
            heading_deg=heading_deg,
            elev_deg=elev_deg,
            z_scale=z_scale,
            fov_deg=fov_deg,
            ego_height_m=ego_height_m,
            eye_back_m=eye_back_m,
        )
        self._render_to_ax(ax, img, title, fontsize)

    def render_bev(
        self,
        ax: Any,
        *,
        draw_helpers: bool = True,
        title: str = "BEV",
        fontsize: int = 11,
    ) -> None:
        """兼容接口: 渲染到 matplotlib Axes (底层由 OpenCV 完成)."""
        img = self.render_bev_image(draw_helpers=draw_helpers)
        self._render_to_ax(ax, img, title, fontsize)

    def render_camera_overlay(
        self,
        ax: Any,
        img_rgb: np.ndarray,
        cam: CameraParams,
        *,
        alpha: float = 0.28,
        cam_name: str = "",
    ) -> None:
        """兼容接口: 相机叠加渲染到 matplotlib Axes (底层由 OpenCV 完成)."""
        img = self.render_camera_overlay_image(
            img_rgb,
            cam,
            alpha=alpha,
            cam_name=cam_name,
        )
        self._render_to_ax(ax, img, title="", fontsize=8)
