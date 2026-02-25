"""FlashOCC — 快速且内存高效的占用预测."""

__version__ = "2.0.0"

# 向后兼容: 暴露常用的 io 函数到顶层命名空间
from flashocc.core.io import load, dump  # noqa
