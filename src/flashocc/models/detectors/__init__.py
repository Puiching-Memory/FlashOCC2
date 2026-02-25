"""检测器模块."""

from .base import MVXTwoStageDetector
from .bevdet import *  # noqa: F401,F403
from .bevdet_occ import *  # noqa: F401,F403

__all__ = ["MVXTwoStageDetector"]
