"""投影变换 — 将 3D 面片顶点投影到 2D 屏幕坐标.

提供三种投影模式的统一接口:
    1. IsometricProjection:  等轴测投影 (平行光, 无透视缩小)
    2. PerspectiveProjection: 自车中心透视投影
    3. CameraProjection:      真实相机内外参投影

每种投影都实现:
    - project(verts_3d)    → (M, 4, 2) 2D 屏幕坐标
    - compute_depth(centers) → (M,) 深度值 (用于 painter's algorithm)
    - filter_visible(...)  → 可见性掩码

坐标系:
    体素网格坐标: (i, j, k) 整数索引, axes = (X, Y, Z)
    世界坐标: (x, y, z) 米, 与体素网格通过 OccGrid 转换
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from .faces import FaceData


class Projection(ABC):
    """投影基类."""

    @abstractmethod
    def project(self, verts_3d: np.ndarray) -> np.ndarray:
        """(M, 4, 3) → (M, 4, 2) 2D 屏幕坐标."""
        ...

    @abstractmethod
    def compute_depth(self, centers: np.ndarray) -> np.ndarray:
        """(M, 3) → (M,) 沿视线方向的深度, 越大越远."""
        ...

    def filter_visible(
        self,
        polys_2d: np.ndarray,
        depth: np.ndarray,
    ) -> np.ndarray:
        """返回可见面片的 bool 掩码.

        默认: 过滤 NaN/Inf.
        """
        finite = np.isfinite(polys_2d).all(axis=(1, 2))
        return finite


# =====================================================================
#  等轴测投影
# =====================================================================

@dataclass
class IsometricProjection(Projection):
    """等轴测投影 (无透视缩小, 适用于全局概览).

    参数:
        azim_deg: 水平旋转角 (°), 从+X轴看, 逆时针为正
        elev_deg: 俯仰角 (°), 正=从上方俯视
    """

    azim_deg: float = 45.0
    elev_deg: float = 35.0

    def project(self, verts_3d: np.ndarray) -> np.ndarray:
        azim = np.radians(self.azim_deg)
        elev = np.radians(self.elev_deg)
        cos_a, sin_a = np.cos(azim), np.sin(azim)
        cos_e, sin_e = np.cos(elev), np.sin(elev)

        vx, vy, vz = verts_3d[..., 0], verts_3d[..., 1], verts_3d[..., 2]

        # 水平旋转
        xr = vx * cos_a - vy * sin_a
        yr = vx * sin_a + vy * cos_a

        # 俯仰: y → depth, z → screen-y
        screen_x = xr
        screen_y = yr * sin_e + vz * cos_e

        return np.stack([screen_x, screen_y], axis=-1)

    def compute_depth(self, centers: np.ndarray) -> np.ndarray:
        azim = np.radians(self.azim_deg)
        elev = np.radians(self.elev_deg)
        cos_a, sin_a = np.cos(azim), np.sin(azim)
        cos_e, sin_e = np.cos(elev), np.sin(elev)

        # 视线方向深度: 物体沿 (sin_a, cos_a, 0) 方向的投影 + 高度分量
        return (
            (centers[:, 0] * sin_a + centers[:, 1] * cos_a) * cos_e
            - centers[:, 2] * sin_e
        )


# =====================================================================
#  透视投影
# =====================================================================

@dataclass
class PerspectiveProjection(Projection):
    """从自车位置的透视投影.

    参数:
        grid_shape: (Dx, Dy, Dz) 体素网格尺寸
        heading_deg: 朝向角 (°), 0=前方(+X), 逆时针为正
        elev_deg:    俯仰角 (°), 正=向上看
        z_scale:     Z 方向拉伸系数
        fov_deg:     视场角 (°)
        ego_height_m: 自车相机高度 (米)
        near_clip:   近裁剪距离 (体素单位)
        proj_limit:  投影坐标范围限制
    """

    grid_shape: tuple[int, int, int] = (200, 200, 16)
    heading_deg: float = 0.0
    elev_deg: float = 8.0
    z_scale: float = 1.5
    fov_deg: float = 90.0
    ego_height_m: float = 1.5
    eye_back_m: float = 0.0
    pcr_z_min: float = -1.0
    pcr_z_max: float = 5.4
    near_clip: float = 0.8
    proj_limit: float = 8.0

    def _build_camera(self) -> tuple[np.ndarray, np.ndarray, float]:
        """构建相机位置、旋转矩阵、focal."""
        Dx, Dy, Dz = self.grid_shape
        ego_z_frac = (self.ego_height_m - self.pcr_z_min) / (self.pcr_z_max - self.pcr_z_min)
        ego = np.array([Dx / 2.0, Dy / 2.0, ego_z_frac * Dz * self.z_scale])

        # 沿 heading 反方向平移 eye_back_m (使相机后退, 可看到自车)
        if abs(self.eye_back_m) > 1e-6:
            heading_rad = np.radians(self.heading_deg)
            voxel_size_xy = (self.pcr_z_max - self.pcr_z_min) / Dz  # 各向同性
            back_vox = self.eye_back_m / voxel_size_xy
            ego[0] -= np.cos(heading_rad) * back_vox
            ego[1] -= np.sin(heading_rad) * back_vox

        heading = np.radians(self.heading_deg)
        elev = np.radians(self.elev_deg)
        cos_h, sin_h = np.cos(heading), np.sin(heading)
        cos_e, sin_e = np.cos(elev), np.sin(elev)

        cam_fwd = np.array([cos_h * cos_e, sin_h * cos_e, sin_e])
        cam_fwd /= np.linalg.norm(cam_fwd)
        cam_right = np.array([sin_h, -cos_h, 0.0])
        cam_right /= np.linalg.norm(cam_right)
        cam_up = np.cross(cam_right, cam_fwd)
        cam_up /= np.linalg.norm(cam_up)

        R = np.stack([cam_right, cam_up, cam_fwd])  # 3×3
        focal = 1.0 / np.tan(np.radians(self.fov_deg / 2.0))

        return ego, R, focal

    def project(self, verts_3d: np.ndarray) -> np.ndarray:
        ego, R, focal = self._build_camera()

        # 相机空间: v_cam = R @ (v3d - ego)
        v_rel = verts_3d - ego[None, None, :]
        v_cam = np.einsum("nvi,ji->nvj", v_rel, R)

        # 透视除法: 对 near plane 后方顶点直接置 NaN, 避免拉伸伪影.
        depth = v_cam[..., 2]
        valid = depth > self.near_clip
        depth_safe = np.where(valid, depth, np.nan)
        x_proj = v_cam[..., 0] * focal / depth_safe
        y_proj = v_cam[..., 1] * focal / depth_safe

        return np.stack([x_proj, y_proj], axis=-1)

    def compute_depth(self, centers: np.ndarray) -> np.ndarray:
        ego, R, _ = self._build_camera()
        ctr_rel = centers - ego[None, :]
        return ctr_rel @ R[2]  # 沿 cam_fwd 方向

    def filter_visible(
        self,
        polys_2d: np.ndarray,
        depth: np.ndarray,
    ) -> np.ndarray:
        finite = np.isfinite(polys_2d).all(axis=(1, 2))
        bounded = np.abs(polys_2d).max(axis=(1, 2)) < self.proj_limit
        positive_depth = depth > self.near_clip
        return finite & bounded & positive_depth


# =====================================================================
#  相机投影 (真实内外参)
# =====================================================================

@dataclass
class CameraParams:
    """单相机投影参数."""

    sensor2keyego: np.ndarray   # (4, 4) cam → key_ego
    intrinsics: np.ndarray      # (3, 3)
    post_rot: np.ndarray        # (3, 3) 数据增强旋转
    post_trans: np.ndarray      # (3,) 数据增强平移

    @property
    def ego2cam(self) -> np.ndarray:
        """key_ego → cam 变换矩阵 (4×4)."""
        return np.linalg.inv(self.sensor2keyego.astype(np.float64)).astype(np.float32)


@dataclass
class CameraProjection(Projection):
    """通过真实相机参数的投影 (用于体素叠加到图像上).

    参数:
        cam:    相机内外参
        img_hw: 图像尺寸 (H, W)
        near_eps: 近裁剪 epsilon
    """

    cam: CameraParams
    img_hw: tuple[int, int] = (256, 704)
    near_eps: float = 0.01

    def project(self, verts_3d: np.ndarray) -> np.ndarray:
        """世界坐标面片 → 像素坐标.

        投影链: ego → cam → pixel → post_aug
        """
        e2c = self.cam.ego2cam
        R = e2c[:3, :3]
        t = e2c[:3, 3]
        K = self.cam.intrinsics
        pr = self.cam.post_rot
        pt = self.cam.post_trans

        # (M, 4, 3) → 相机坐标
        pts_flat = verts_3d.reshape(-1, 3)
        pts_cam = (pts_flat @ R.T + t[None, :]).reshape(verts_3d.shape)

        # 透视除法 (clamp near)
        pts_safe = pts_cam.copy()
        pts_safe[..., 2] = np.maximum(pts_safe[..., 2], self.near_eps)
        uvw = pts_safe @ K.T
        u = uvw[..., 0] / uvw[..., 2]
        v = uvw[..., 1] / uvw[..., 2]

        # 数据增强变换
        uv1 = np.stack([u, v, np.ones_like(u)], axis=-1)
        uv_aug = uv1 @ pr.T + pt[None, None, :]

        return uv_aug[..., :2]

    def compute_depth(self, centers: np.ndarray) -> np.ndarray:
        e2c = self.cam.ego2cam
        R = e2c[:3, :3]
        t = e2c[:3, 3]
        pts_cam = centers @ R.T + t[None, :]
        return pts_cam[:, 2]  # z_cam

    def filter_visible(
        self,
        polys_2d: np.ndarray,
        depth: np.ndarray,
    ) -> np.ndarray:
        H, W = self.img_hw
        finite = np.isfinite(polys_2d).all(axis=(1, 2))

        # 至少部分与视口重叠
        x_min = polys_2d[..., 0].min(axis=1)
        x_max = polys_2d[..., 0].max(axis=1)
        y_min = polys_2d[..., 1].min(axis=1)
        y_max = polys_2d[..., 1].max(axis=1)
        in_view = (x_max >= 0) & (x_min < W) & (y_max >= 0) & (y_min < H)

        # 面片的中心 z_cam > 0 (在相机前方)
        positive_depth = depth > self.near_eps

        return finite & in_view & positive_depth
