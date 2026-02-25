"""配置类型 — 消除字典, 提供完整的 IDE 类型提示.

所有"场景参数"用 dataclass 表达, 属性访问替代 dict['key']:
  - 编写时 IDE 自动补全每个字段
  - 拼写错误立即 AttributeError, 而非运行到一半 KeyError
  - 类型注解让 mypy / pyright 一目了然
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class DataConfig:
    """多视角相机数据配置."""

    cams: List[str]
    Ncams: int
    input_size: Tuple[int, int]
    src_size: Tuple[int, int]

    # 数据增广
    resize: Tuple[float, float]
    rot: Tuple[float, float]
    flip: bool
    crop_h: Tuple[float, float]

    # 测试 resize 偏移
    resize_test: float = 0.0


@dataclass
class GridConfig:
    """BEV 网格配置 — 每个轴 (下界, 上界, 步长)."""

    x: Tuple[float, float, float]
    y: Tuple[float, float, float]
    z: Tuple[float, float, float]
    depth: Tuple[float, float, float]


@dataclass
class BDAAugConfig:
    """BEV 数据增广配置."""

    rot_lim: Tuple[float, float]
    scale_lim: Tuple[float, float]
    flip_dx_ratio: float
    flip_dy_ratio: float


__all__ = ["DataConfig", "GridConfig", "BDAAugConfig"]
