"""FlashOCC 全局常量 — NuScenes 占用预测."""

from __future__ import annotations

import numpy as np

# =====================================================================
#  NuScenes 占用语义类别 (18 类, 含 free)
# =====================================================================

OCC_CLASS_NAMES: list[str] = [
    "others",
    "barrier",
    "bicycle",
    "bus",
    "car",
    "construction_vehicle",
    "motorcycle",
    "pedestrian",
    "traffic_cone",
    "trailer",
    "truck",
    "driveable_surface",
    "other_flat",
    "sidewalk",
    "terrain",
    "manmade",
    "vegetation",
    "free",
]

NUM_OCC_CLASSES: int = len(OCC_CLASS_NAMES)  # 18

# =====================================================================
#  空间范围 & 体素 — NuScenes Occ3D 默认设置
# =====================================================================

POINT_CLOUD_RANGE: list[float] = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
VOXEL_SIZE: float = 0.4
OCCUPANCY_SIZE: list[float] = [VOXEL_SIZE, VOXEL_SIZE, VOXEL_SIZE]

# 体素网格维度 (200, 200, 16)
OCC_GRID_SHAPE: tuple[int, int, int] = (
    int((POINT_CLOUD_RANGE[3] - POINT_CLOUD_RANGE[0]) / VOXEL_SIZE),
    int((POINT_CLOUD_RANGE[4] - POINT_CLOUD_RANGE[1]) / VOXEL_SIZE),
    int((POINT_CLOUD_RANGE[5] - POINT_CLOUD_RANGE[2]) / VOXEL_SIZE),
)

# =====================================================================
#  ImageNet 归一化常量 (BGR → RGB)
# =====================================================================

IMAGENET_MEAN: np.ndarray = np.array([123.675, 116.28, 103.53], dtype=np.float32)
IMAGENET_STD: np.ndarray = np.array([58.395, 57.12, 57.375], dtype=np.float32)

# =====================================================================
#  NuScenes 占用类别频率 — 用于 class-balanced loss
# =====================================================================

NUSC_CLASS_FREQUENCIES: np.ndarray = np.array([
    944004, 1897170, 152386, 2391677, 16957802, 724139, 189027,
    2074468, 413451, 2384460, 5916653, 175883646, 4275424,
    51393615, 61411620, 105975596, 116424404, 1892500630,
], dtype=np.float64)

__all__ = [
    "OCC_CLASS_NAMES",
    "NUM_OCC_CLASSES",
    "POINT_CLOUD_RANGE",
    "VOXEL_SIZE",
    "OCCUPANCY_SIZE",
    "OCC_GRID_SHAPE",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "NUSC_CLASS_FREQUENCIES",
]
