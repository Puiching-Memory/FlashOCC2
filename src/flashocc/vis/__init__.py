"""flashocc.vis — OCC 占用预测体素化 3D 可视化引擎.

模块结构:
    occ_grid.py     OccGrid 数据容器 (体素网格 + 空间元数据 + 坐标变换)
    colors.py       颜色管理 (类别颜色 / LUT / 图例)
    faces.py        面片生成引擎 (邻接剔除 / 体素→四边形)
    projection.py   投影变换 (等轴测 / 透视 / 相机)
    renderer.py     OccVoxelRenderer 统一渲染器
    bev_helpers.py  BEV 辅助绘制 (自车 / 相机FOV / 刻度)

快速入门:
    >>> from flashocc.vis import OccGrid, OccVoxelRenderer
    >>>
    >>> grid = OccGrid.from_numpy(occ_pred)  # (200, 200, 16) uint8
    >>> renderer = OccVoxelRenderer(grid, voxel_step=2)
    >>>
    >>> fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    >>> renderer.render_isometric(axes[0], title="Prediction 3D")
    >>> renderer.render_bev(axes[1], title="Prediction BEV")
"""
from __future__ import annotations

from .occ_grid import OccGrid, FREE_CLASS, OTHERS_CLASS
from .colors import (
    OCC_CLASS_COLORS,
    COLOR_LUT,
    cls_to_rgb,
    cls_to_rgba,
    build_class_legend,
)
from .faces import FaceData, generate_faces, generate_faces_camera
from .projection import (
    Projection,
    IsometricProjection,
    PerspectiveProjection,
    CameraProjection,
    CameraParams,
)
from .renderer import OccVoxelRenderer, painters_sort
from .bev_helpers import (
    CAM_ORDER,
    CAM_INFO,
    ZOE_LENGTH,
    ZOE_WIDTH,
    ZOE_WHEELBASE,
    draw_ego_vehicle,
    draw_camera_fovs,
    add_bev_annotations,
    build_cam_legend_patches,
)

__all__ = [
    # 数据容器
    "OccGrid",
    "FREE_CLASS",
    "OTHERS_CLASS",
    # 颜色
    "OCC_CLASS_COLORS",
    "COLOR_LUT",
    "cls_to_rgb",
    "cls_to_rgba",
    "build_class_legend",
    # 面片
    "FaceData",
    "generate_faces",
    "generate_faces_camera",
    # 投影
    "Projection",
    "IsometricProjection",
    "PerspectiveProjection",
    "CameraProjection",
    "CameraParams",
    # 渲染器
    "OccVoxelRenderer",
    "painters_sort",
    # BEV 辅助
    "CAM_ORDER",
    "CAM_INFO",
    "ZOE_LENGTH",
    "ZOE_WIDTH",
    "ZOE_WHEELBASE",
    "draw_ego_vehicle",
    "draw_camera_fovs",
    "add_bev_annotations",
    "build_cam_legend_patches",
]
