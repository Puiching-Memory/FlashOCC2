"""数据预处理管线."""
from flashocc.datasets.builder import PIPELINES
from .compose import Compose
from .base import to_tensor, LoadImageFromFile, LoadAnnotations

# 导入子模块以触发 @register 装饰器
# _compat_pipelines 必须在 formatting 之前导入，因为 formatting 用 force=True 覆盖重复注册
from . import _compat_pipelines  # noqa: LoadPointsFromFile, etc.
from . import loading  # noqa: PrepareImageInputs, LoadAnnotationsBEVDepth, etc.
from . import formatting  # noqa: DefaultFormatBundle3D, Collect3D

__all__ = ["PIPELINES", "Compose", "to_tensor", "LoadImageFromFile", "LoadAnnotations"]
