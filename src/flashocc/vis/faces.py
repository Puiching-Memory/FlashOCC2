"""体素面片生成引擎 — 从 OccGrid 提取可渲染的四边形面片.

核心优化:
    1. 邻接剔除: 仅生成暴露在空气中的面 (相邻体素为 free/others 或越界)
    2. 统一面定义: 6 个面各自的顶点模板、法线方向、明度系数
    3. 向量化: 全程 numpy 批量操作, 无 Python for-loop 遍历体素

面片数据格式 (FaceData):
    verts_3d:  (M, 4, 3) 四边形的 4 个 3D 顶点 (体素网格坐标系)
    cls_ids:   (M,) 每个面所属体素的类别 ID
    shades:    (M,) 每个面的明度系数 (模拟光照)
    centers:   (M, 3) 每个面所属体素的中心坐标 (用于深度排序)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .occ_grid import OccGrid, FREE_CLASS, OTHERS_CLASS


@dataclass(frozen=True)
class FaceData:
    """一批可渲染的四边形面片."""

    verts_3d: np.ndarray   # (M, 4, 3) 顶点坐标 (体素网格坐标)
    cls_ids: np.ndarray    # (M,) 类别 ID
    shades: np.ndarray     # (M,) 明度系数 ∈ (0, 1]
    centers: np.ndarray    # (M, 3) 体素中心坐标

    @property
    def count(self) -> int:
        return self.verts_3d.shape[0]

    @property
    def empty(self) -> bool:
        return self.count == 0

    def concat(self, other: "FaceData") -> "FaceData":
        """合并两组面片."""
        if self.empty:
            return other
        if other.empty:
            return self
        return FaceData(
            verts_3d=np.concatenate([self.verts_3d, other.verts_3d]),
            cls_ids=np.concatenate([self.cls_ids, other.cls_ids]),
            shades=np.concatenate([self.shades, other.shades]),
            centers=np.concatenate([self.centers, other.centers]),
        )

    @staticmethod
    def empty_data() -> "FaceData":
        return FaceData(
            verts_3d=np.empty((0, 4, 3), dtype=np.float64),
            cls_ids=np.empty(0, dtype=np.int32),
            shades=np.empty(0, dtype=np.float64),
            centers=np.empty((0, 3), dtype=np.float64),
        )


# =====================================================================
#  六面定义 (单位立方体 [0,1]^3 内的顶点模板)
#
#  法线方向与明度:
#    +X (前): 0.80    -X (后): 0.70
#    +Y (左): 0.65    -Y (右): 0.75
#    +Z (上): 1.00    -Z (下): 0.55
# =====================================================================

# (normal_dx, normal_dy, normal_dz, 4_corner_offsets, shade)
_FACE_DEFS: list[tuple[int, int, int, np.ndarray, float]] = [
    (+1, 0, 0,
     np.array([[1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 0, 1]], dtype=np.float64),
     0.80),
    (-1, 0, 0,
     np.array([[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]], dtype=np.float64),
     0.70),
    (0, +1, 0,
     np.array([[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]], dtype=np.float64),
     0.65),
    (0, -1, 0,
     np.array([[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]], dtype=np.float64),
     0.75),
    (0, 0, +1,
     np.array([[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=np.float64),
     1.00),
    (0, 0, -1,
     np.array([[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]], dtype=np.float64),
     0.55),
]


def generate_faces(
    grid: OccGrid,
    *,
    adjacency_cull: bool = True,
    z_scale: float = 1.0,
) -> FaceData:
    """从 OccGrid 提取所有暴露的四边形面片.

    Args:
        grid:           体素网格
        adjacency_cull: 是否进行邻接剔除 (True=仅保留暴露面, 大幅减少面数)
        z_scale:        Z 方向拉伸系数 (用于等轴测/透视夸大高度)

    Returns:
        FaceData 包含所有可渲染面片
    """
    voxels = grid.voxels
    occ_mask = grid.occupied_mask()

    if not occ_mask.any():
        return FaceData.empty_data()

    Dx, Dy, Dz = grid.shape
    xi, yi, zi = np.where(occ_mask)
    n_voxels = len(xi)

    # 所有占用体素的类别和中心坐标
    cls_ids_all = voxels[xi, yi, zi].astype(np.int32)
    centers_all = np.stack([xi + 0.5, yi + 0.5, (zi + 0.5) * z_scale], axis=-1)

    all_verts = []
    all_cls = []
    all_shades = []
    all_centers = []

    for ndx, ndy, ndz, vtpl, shade in _FACE_DEFS:
        if adjacency_cull:
            # 邻接体素位置
            ni = xi + ndx
            nj = yi + ndy
            nk = zi + ndz
            # 越界 → 暴露; 邻接为 free/others → 暴露
            out_of_bounds = (ni < 0) | (ni >= Dx) | (nj < 0) | (nj >= Dy) | (nk < 0) | (nk >= Dz)
            neighbor_free = np.zeros(n_voxels, dtype=bool)

            in_bounds = ~out_of_bounds
            if in_bounds.any():
                nb_cls = voxels[ni[in_bounds], nj[in_bounds], nk[in_bounds]]
                neighbor_free[in_bounds] = (nb_cls == FREE_CLASS) | (nb_cls == OTHERS_CLASS)

            exposed = out_of_bounds | neighbor_free
            if not exposed.any():
                continue

            sel_xi = xi[exposed]
            sel_yi = yi[exposed]
            sel_zi = zi[exposed]
            sel_cls = cls_ids_all[exposed]
            sel_centers = centers_all[exposed]
        else:
            sel_xi = xi
            sel_yi = yi
            sel_zi = zi
            sel_cls = cls_ids_all
            sel_centers = centers_all

        n_faces = len(sel_xi)
        # 构建每个面的 4 个顶点: origin + vtpl * [1,1,z_scale]
        origins = np.stack([sel_xi, sel_yi, sel_zi], axis=-1).astype(np.float64)
        scale = np.array([1.0, 1.0, z_scale], dtype=np.float64)
        verts = origins[:, None, :] + vtpl[None, :, :] * scale[None, None, :]

        all_verts.append(verts)
        all_cls.append(sel_cls)
        all_shades.append(np.full(n_faces, shade, dtype=np.float64))
        all_centers.append(sel_centers)

    if not all_verts:
        return FaceData.empty_data()

    return FaceData(
        verts_3d=np.concatenate(all_verts),
        cls_ids=np.concatenate(all_cls),
        shades=np.concatenate(all_shades),
        centers=np.concatenate(all_centers),
    )


def generate_faces_camera(
    grid: OccGrid,
    *,
    voxel_step: int = 1,
    max_voxels: int = 22000,
) -> FaceData:
    """为相机投影叠加生成面片 (不做邻接剔除, 带体素数限制).

    与 generate_faces 的区别:
      - 不做邻接剔除 (因相机投影需要完整立方体)
      - 支持 max_voxels 限制以控制渲染开销
      - z_scale 固定为 1.0 (世界坐标系)
      - 顶点坐标换算为世界坐标 (米)
    """
    occ_mask = grid.occupied_mask()
    if not occ_mask.any():
        return FaceData.empty_data()

    xi, yi, zi = np.where(occ_mask)
    cls_ids = grid.voxels[xi, yi, zi].astype(np.int32)

    # 体素数量限制
    if xi.size > max_voxels:
        keep = np.linspace(0, xi.size - 1, max_voxels, dtype=np.intp)
        xi, yi, zi = xi[keep], yi[keep], zi[keep]
        cls_ids = cls_ids[keep]

    n = len(xi)
    vs = grid.vs

    # 世界坐标下的体素左下角
    x0 = grid.pcr[0] + xi.astype(np.float64) * vs
    y0 = grid.pcr[1] + yi.astype(np.float64) * vs
    z0 = grid.pcr[2] + zi.astype(np.float64) * vs
    origins = np.stack([x0, y0, z0], axis=-1)
    centers = origins + vs * 0.5

    # 立方体 8 角偏移 (世界坐标尺度)
    local_corners = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
    ], dtype=np.float64) * vs

    face_indices = np.array([
        [0, 1, 2, 3],  # bottom  (-Z)
        [4, 5, 6, 7],  # top     (+Z)
        [0, 1, 5, 4],  # front-x (+X)
        [3, 2, 6, 7],  # back-x  (-X)
        [0, 3, 7, 4],  # left-y  (-Y)
        [1, 2, 6, 5],  # right-y (+Y)
    ], dtype=np.intp)
    face_shades = np.array([0.55, 1.00, 0.80, 0.70, 0.75, 0.65], dtype=np.float64)

    # 所有顶点: (n, 8, 3)
    all_corners = origins[:, None, :] + local_corners[None, :, :]

    all_verts = []
    all_cls_list = []
    all_shades_list = []
    all_centers_list = []

    for fidx in range(6):
        face_verts = all_corners[:, face_indices[fidx], :]  # (n, 4, 3)
        all_verts.append(face_verts)
        all_cls_list.append(cls_ids)
        all_shades_list.append(np.full(n, face_shades[fidx]))
        all_centers_list.append(centers)

    return FaceData(
        verts_3d=np.concatenate(all_verts),
        cls_ids=np.concatenate(all_cls_list),
        shades=np.concatenate(all_shades_list),
        centers=np.concatenate(all_centers_list),
    )
