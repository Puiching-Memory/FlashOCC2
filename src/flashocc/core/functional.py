"""训练/推理核心工具 (向后兼容入口, 实际实现已移至 dist.py)."""
from .dist import reduce_mean

__all__ = ["reduce_mean"]
