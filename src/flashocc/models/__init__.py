"""模型注册表与构建器.

使用 plum-dispatch 实现类型安全的 passthrough/build 分派:
  - nn.Module 直接透传
  - dict 通过 Registry.build() 构建
  - Lazy 通过 Lazy.build() 构建
"""

from __future__ import annotations

import torch.nn as nn
from plum import dispatch

from flashocc.core.registry import Registry

MODELS = Registry("models")
DETECTORS = Registry("detectors")
BACKBONES = Registry("backbones")
NECKS = Registry("necks")
HEADS = Registry("heads")
LOSSES = Registry("losses")


# =====================================================================
#  plum-dispatch 构建器 — 按类型自动分派
# =====================================================================

@dispatch
def _build_from(registry: Registry, cfg: nn.Module, **kwargs) -> nn.Module:
    """已构建的 nn.Module → 直通."""
    return cfg


@dispatch
def _build_from(registry: Registry, cfg: dict, **kwargs) -> nn.Module:
    """dict 配置 → 通过 Registry 构建."""
    return registry.build(cfg, default_args=kwargs if kwargs else None)


def build_backbone(cfg, **kwargs):
    return _build_from(BACKBONES, cfg, **kwargs)


def build_neck(cfg, **kwargs):
    return _build_from(NECKS, cfg, **kwargs)


def build_head(cfg, **kwargs):
    return _build_from(HEADS, cfg, **kwargs)


def build_loss(cfg, **kwargs):
    return _build_from(LOSSES, cfg, **kwargs)


def build_detector(cfg, train_cfg=None, test_cfg=None):
    if isinstance(cfg, nn.Module):
        return cfg
    default_args = {}
    if train_cfg is not None:
        default_args["train_cfg"] = train_cfg
    if test_cfg is not None:
        default_args["test_cfg"] = test_cfg
    return DETECTORS.build(cfg, default_args=default_args or None)


def build_model(cfg, train_cfg=None, test_cfg=None):
    return build_detector(cfg, train_cfg, test_cfg)


# --- 触发 @register_module() 注册 ---
from . import backbones  # noqa: F401,E402
from . import necks  # noqa: F401,E402
from . import heads  # noqa: F401,E402
from . import losses  # noqa: F401,E402
from . import detectors  # noqa: F401,E402
