"""模型注册表与构建器.

Lazy 路径: Lazy 递归 build() 时, 子组件已是 nn.Module, build_* 直接返回。
"""

from __future__ import annotations

import torch.nn as nn

from flashocc.core.registry import Registry

MODELS = Registry("models")
DETECTORS = Registry("detectors")
BACKBONES = Registry("backbones")
NECKS = Registry("necks")
HEADS = Registry("heads")
LOSSES = Registry("losses")


def _passthrough_or_build(registry: Registry, cfg):
    """如果 cfg 是已构建的 nn.Module 则直通, 否则通过 Registry 构建."""
    if isinstance(cfg, nn.Module):
        return cfg
    return registry.build(cfg)


def build_backbone(cfg, **kwargs):
    return _passthrough_or_build(BACKBONES, cfg)


def build_neck(cfg, **kwargs):
    return _passthrough_or_build(NECKS, cfg)


def build_head(cfg, **kwargs):
    if isinstance(cfg, nn.Module):
        return cfg
    return HEADS.build(cfg, default_args=kwargs if kwargs else None)


def build_loss(cfg, **kwargs):
    return _passthrough_or_build(LOSSES, cfg)


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
