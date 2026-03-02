"""检测器模块."""

from .base import MVXTwoStageDetector
from .bevdet import BEVDet
from .bevdet_occ import BEVDetOCC

__all__ = ["MVXTwoStageDetector", "BEVDet", "BEVDetOCC"]
