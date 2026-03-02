"""组合式 Python 配置系统.

::

    from flashocc.config import Lazy, Experiment
    model = Lazy(BEVDetOCC, img_backbone=Lazy(ResNet, depth=50), ...)
    experiment = Experiment(model=model, ...)
"""

from .lazy import Lazy, Experiment, load_experiment
from .types import DataConfig, GridConfig, BDAAugConfig

__all__ = ["Lazy", "Experiment", "load_experiment", "DataConfig", "GridConfig", "BDAAugConfig"]
