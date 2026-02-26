"""配置类型 — 消除字典, 提供完整的 IDE 类型提示.

所有"场景参数"用 pydantic BaseModel 表达, 属性访问替代 dict['key']:
  - 编写时 IDE 自动补全每个字段
  - 拼写错误立即 ValidationError, 而非运行到一半 KeyError
  - 类型注解 + pydantic 验证, 启动即校验
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, field_validator


class DataConfig(BaseModel):
    """多视角相机数据配置."""

    model_config = ConfigDict(frozen=True)

    cams: list[str]
    Ncams: int
    input_size: tuple[int, int]
    src_size: tuple[int, int]

    # 数据增广
    resize: tuple[float, float]
    rot: tuple[float, float]
    flip: bool
    crop_h: tuple[float, float]

    # 测试 resize 偏移
    resize_test: float = 0.0

    @field_validator("Ncams")
    @classmethod
    def _ncams_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"Ncams 必须为正整数, 收到 {v}")
        return v


class GridConfig(BaseModel):
    """BEV 网格配置 — 每个轴 (下界, 上界, 步长)."""

    model_config = ConfigDict(frozen=True)

    x: tuple[float, float, float]
    y: tuple[float, float, float]
    z: tuple[float, float, float]
    depth: tuple[float, float, float]

    @field_validator("x", "y", "z", "depth")
    @classmethod
    def _range_ordered(cls, v: tuple[float, float, float]) -> tuple[float, float, float]:
        lo, hi, step = v
        if lo >= hi:
            raise ValueError(f"下界 ({lo}) 必须小于上界 ({hi})")
        if step <= 0:
            raise ValueError(f"步长必须为正数, 收到 {step}")
        return v


class BDAAugConfig(BaseModel):
    """BEV 数据增广配置."""

    model_config = ConfigDict(frozen=True)

    rot_lim: tuple[float, float]
    scale_lim: tuple[float, float]
    flip_dx_ratio: float
    flip_dy_ratio: float

    @field_validator("flip_dx_ratio", "flip_dy_ratio")
    @classmethod
    def _ratio_in_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"翻转概率应在 [0, 1], 收到 {v}")
        return v


__all__ = ["DataConfig", "GridConfig", "BDAAugConfig"]
