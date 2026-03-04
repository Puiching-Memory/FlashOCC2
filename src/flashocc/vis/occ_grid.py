"""OccGrid — 语义占用体素网格的核心数据容器.

将 (Dx, Dy, Dz) 的 uint8 类别数组与空间元数据绑定,
提供坐标变换、投影和过滤等基础操作, 消除散落在各处的坐标计算错误.

坐标系约定 (NuScenes ego frame):
    X: 前后方向 (前=+X), 对应体素索引 i (axis=0)
    Y: 左右方向 (左=+Y), 对应体素索引 j (axis=1)
    Z: 高度方向 (上=+Z), 对应体素索引 k (axis=2)

BEV 显示约定 (imshow origin='lower'):
    水平轴 col = j  →  世界 Y 方向
    垂直轴 row = i  →  世界 X 方向 (origin=lower: i=0 在底=后方)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from flashocc.constants import (
    OCC_CLASS_NAMES,
    POINT_CLOUD_RANGE,
    VOXEL_SIZE,
    OCC_GRID_SHAPE,
)

# 语义类别常量
FREE_CLASS: int = 17
OTHERS_CLASS: int = 0


@dataclass(frozen=True)
class OccGrid:
    """语义占用体素网格 + 空间元数据.

    Parameters:
        voxels: (Dx, Dy, Dz) uint8 类别 ID
        pcr:    [x_min, y_min, z_min, x_max, y_max, z_max]
        vs:     体素边长 (米)
    """

    voxels: np.ndarray
    pcr: Sequence[float] = field(default_factory=lambda: list(POINT_CLOUD_RANGE))
    vs: float = VOXEL_SIZE

    # ── 空间属性 ──────────────────────────────────────────────

    @property
    def Dx(self) -> int:
        return self.voxels.shape[0]

    @property
    def Dy(self) -> int:
        return self.voxels.shape[1]

    @property
    def Dz(self) -> int:
        return self.voxels.shape[2]

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.Dx, self.Dy, self.Dz)

    @property
    def x_range(self) -> tuple[float, float]:
        return (float(self.pcr[0]), float(self.pcr[3]))

    @property
    def y_range(self) -> tuple[float, float]:
        return (float(self.pcr[1]), float(self.pcr[4]))

    @property
    def z_range(self) -> tuple[float, float]:
        return (float(self.pcr[2]), float(self.pcr[5]))

    @property
    def bev_extent(self) -> list[float]:
        """[y_min, y_max, x_min, x_max] — 用于 imshow extent (origin='lower')."""
        return [self.pcr[1], self.pcr[4], self.pcr[0], self.pcr[3]]

    # ── 坐标变换 ──────────────────────────────────────────────

    def voxel_to_world(self, i: float, j: float, k: float) -> tuple[float, float, float]:
        """体素索引 (i,j,k) → 世界坐标 (x,y,z), 返回体素左下角."""
        return (
            self.pcr[0] + i * self.vs,
            self.pcr[1] + j * self.vs,
            self.pcr[2] + k * self.vs,
        )

    def voxel_center_to_world(self, i: float, j: float, k: float) -> tuple[float, float, float]:
        """体素索引 (i,j,k) → 世界坐标中心 (x,y,z)."""
        return (
            self.pcr[0] + (i + 0.5) * self.vs,
            self.pcr[1] + (j + 0.5) * self.vs,
            self.pcr[2] + (k + 0.5) * self.vs,
        )

    def world_to_voxel(self, x: float, y: float, z: float) -> tuple[float, float, float]:
        """世界坐标 (x,y,z) → 体素索引 (i,j,k)."""
        return (
            (x - self.pcr[0]) / self.vs,
            (y - self.pcr[1]) / self.vs,
            (z - self.pcr[2]) / self.vs,
        )

    def world_to_bev_px(self, x: float, y: float) -> tuple[float, float]:
        """世界坐标 (x,y) → BEV 像素 (col=j, row=i).

        与 imshow(origin='lower') 配合: 水平轴=Y, 垂直轴=X.
        """
        col = (y - self.pcr[1]) / self.vs
        row = (x - self.pcr[0]) / self.vs
        return col, row

    # ── 掩码 & 查询 ──────────────────────────────────────────

    def occupied_mask(
        self,
        exclude_free: bool = True,
        exclude_others: bool = True,
    ) -> np.ndarray:
        """返回 (Dx, Dy, Dz) 的 bool 掩码, True=占用."""
        mask = np.ones(self.shape, dtype=bool)
        if exclude_free:
            mask &= self.voxels != FREE_CLASS
        if exclude_others:
            mask &= self.voxels != OTHERS_CLASS
        return mask

    def subsample(self, step: int) -> "OccGrid":
        """按步长下采样, 返回新 OccGrid (空间范围不变, vs 变大)."""
        step = max(1, int(step))
        if step == 1:
            return self
        sub = self.voxels[::step, ::step, ::step]
        return OccGrid(voxels=sub, pcr=list(self.pcr), vs=self.vs * step)

    def present_classes(self, exclude_free: bool = True, exclude_others: bool = True) -> set[int]:
        """返回出现的类别 ID 集合."""
        mask = self.occupied_mask(exclude_free, exclude_others)
        return set(int(c) for c in np.unique(self.voxels[mask]))

    # ── 三视图投影 ──────────────────────────────────────────

    def bev_projection(self) -> np.ndarray:
        """(Dx, Dy, Dz) → BEV (Dx, Dy): 沿 Z 从高到低取首个非 free/others 类."""
        result = np.full((self.Dx, self.Dy), FREE_CLASS, dtype=np.uint8)
        for z in range(self.Dz - 1, -1, -1):
            layer = self.voxels[:, :, z]
            valid = (layer != FREE_CLASS) & (layer != OTHERS_CLASS) & (result == FREE_CLASS)
            result[valid] = layer[valid]
        return result

    def side_projection(self) -> np.ndarray:
        """(Dx, Dy, Dz) → Side (Dx, Dz): 沿 Y 取首个非 free/others 类."""
        result = np.full((self.Dx, self.Dz), FREE_CLASS, dtype=np.uint8)
        for y in range(self.Dy):
            layer = self.voxels[:, y, :]
            valid = (layer != FREE_CLASS) & (layer != OTHERS_CLASS) & (result == FREE_CLASS)
            result[valid] = layer[valid]
        return result

    def front_projection(self) -> np.ndarray:
        """(Dx, Dy, Dz) → Front (Dy, Dz): 沿 X 取首个非 free/others 类."""
        result = np.full((self.Dy, self.Dz), FREE_CLASS, dtype=np.uint8)
        for x in range(self.Dx):
            layer = self.voxels[x, :, :]
            valid = (layer != FREE_CLASS) & (layer != OTHERS_CLASS) & (result == FREE_CLASS)
            result[valid] = layer[valid]
        return result

    # ── 工厂方法 ──────────────────────────────────────────────

    @staticmethod
    def from_numpy(voxels: np.ndarray, **kwargs) -> "OccGrid":
        """从 numpy 数组构建, 自动处理 dtype."""
        return OccGrid(voxels=voxels.astype(np.uint8), **kwargs)

    def __repr__(self) -> str:
        occ = self.occupied_mask()
        return (
            f"OccGrid(shape={self.shape}, vs={self.vs}m, "
            f"occupied={int(occ.sum()):,}/{self.voxels.size:,}, "
            f"classes={sorted(self.present_classes())})"
        )
