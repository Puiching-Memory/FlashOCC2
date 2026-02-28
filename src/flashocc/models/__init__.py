"""模型注册表."""

from flashocc.core.registry import Registry

MODELS = Registry("models")
DETECTORS = Registry("detectors")
BACKBONES = Registry("backbones")
NECKS = Registry("necks")
HEADS = Registry("heads")
LOSSES = Registry("losses")


# --- 触发 @register 装饰器 ---
from . import backbones  # noqa: F401,E402
from . import necks  # noqa: F401,E402
from . import heads  # noqa: F401,E402
from . import losses  # noqa: F401,E402
from . import detectors  # noqa: F401,E402
