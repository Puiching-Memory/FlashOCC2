"""数据集和数据加载模块."""

from .builder import DATASETS, PIPELINES  # noqa

# 导入子模块以触发 @register 装饰器
from . import nuscenes_bevdet  # noqa
from . import nuscenes_occ  # noqa
from . import pipelines  # noqa

__all__ = ["DATASETS", "PIPELINES"]
