"""BaseModule — 支持 init_weights 的 nn.Module 基类.

使用 pydantic 定义类型安全的初始化配置, 用 plum-dispatch 替代 if/elif 链。
"""

from __future__ import annotations

import copy
import warnings
from typing import Literal, Optional, Union

import torch.nn as nn
from plum import dispatch
from pydantic import BaseModel, ConfigDict


# =====================================================================
#  Pydantic 初始化配置模型
# =====================================================================

class PretrainedInit(BaseModel):
    """预训练权重初始化配置."""
    model_config = ConfigDict(extra="allow")
    type: Literal["Pretrained"] = "Pretrained"
    checkpoint: str


class XavierInit(BaseModel):
    """Xavier 初始化配置."""
    model_config = ConfigDict(extra="allow")
    type: Literal["Xavier"] = "Xavier"
    layer: Optional[Union[str, list[str]]] = None
    distribution: Literal["uniform", "normal"] = "uniform"


class KaimingInit(BaseModel):
    """Kaiming 初始化配置."""
    model_config = ConfigDict(extra="allow")
    type: Literal["Kaiming"] = "Kaiming"
    layer: Optional[Union[str, list[str]]] = None


class ConstantInit(BaseModel):
    """常数初始化配置."""
    model_config = ConfigDict(extra="allow")
    type: Literal["Constant"] = "Constant"
    layer: Optional[Union[str, list[str]]] = None
    val: float = 0.0


class NormalInit(BaseModel):
    """正态初始化配置."""
    model_config = ConfigDict(extra="allow")
    type: Literal["Normal"] = "Normal"
    layer: Optional[Union[str, list[str]]] = None
    std: float = 0.01


InitConfig = Union[PretrainedInit, XavierInit, KaimingInit, ConstantInit, NormalInit]


# =====================================================================
#  plum-dispatch 初始化策略
# =====================================================================

@dispatch
def _apply_init(cfg: PretrainedInit, module: nn.Module) -> None:
    """加载预训练权重."""
    from .checkpoint import load_checkpoint
    load_checkpoint(module, cfg.checkpoint, strict=False)


@dispatch
def _apply_init(cfg: XavierInit, module: nn.Module) -> None:
    """Xavier 初始化."""
    for m in module.modules():
        if _match_layer(m, cfg.layer):
            if cfg.distribution == "uniform":
                nn.init.xavier_uniform_(m.weight)
            else:
                nn.init.xavier_normal_(m.weight)
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias, 0)


@dispatch
def _apply_init(cfg: KaimingInit, module: nn.Module) -> None:
    """Kaiming 初始化."""
    for m in module.modules():
        if _match_layer(m, cfg.layer):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias, 0)


@dispatch
def _apply_init(cfg: ConstantInit, module: nn.Module) -> None:
    """常数初始化."""
    for m in module.modules():
        if _match_layer(m, cfg.layer):
            nn.init.constant_(m.weight, cfg.val)


@dispatch
def _apply_init(cfg: NormalInit, module: nn.Module) -> None:
    """正态初始化."""
    for m in module.modules():
        if _match_layer(m, cfg.layer):
            nn.init.normal_(m.weight, 0, cfg.std)


# =====================================================================
#  BaseModule
# =====================================================================

def _parse_init_cfg(raw) -> list[InitConfig]:
    """将原始 init_cfg (dict/list[dict]) 解析为 pydantic 模型列表."""
    if raw is None:
        return []
    items = raw if isinstance(raw, list) else [raw]
    result = []
    for item in items:
        if isinstance(item, BaseModel):
            result.append(item)
        elif isinstance(item, dict):
            type_name = item.get("type", "")
            _TYPE_MAP = {
                "Pretrained": PretrainedInit,
                "Xavier": XavierInit,
                "Kaiming": KaimingInit,
                "Constant": ConstantInit,
                "Normal": NormalInit,
            }
            model_cls = _TYPE_MAP.get(type_name)
            if model_cls is None:
                warnings.warn(f"未知的初始化类型: {type_name}, 跳过")
                continue
            result.append(model_cls.model_validate(item))
        else:
            warnings.warn(f"不支持的 init_cfg 项类型: {type(item)}, 跳过")
    return result


class BaseModule(nn.Module):
    """所有 FlashOCC 模型的基类.

    支持 pydantic 类型安全的 ``init_cfg`` 配置式权重初始化,
    使用 plum-dispatch 按类型分派初始化策略。
    """

    def __init__(self, init_cfg=None):
        super().__init__()
        self._raw_init_cfg = copy.deepcopy(init_cfg) if init_cfg else None
        # 兼容旧代码读取
        self.init_cfg = self._raw_init_cfg
        self._is_init = False

    def init_weights(self):
        """根据 init_cfg 初始化权重 — plum-dispatch 按类型自动分派."""
        if self._raw_init_cfg is None:
            return
        if self._is_init:
            warnings.warn(f"{self.__class__.__name__} 已经初始化, 跳过.")
            return

        configs = _parse_init_cfg(self._raw_init_cfg)
        for cfg in configs:
            _apply_init(cfg, self)

        self._is_init = True


# =====================================================================
#  layer 匹配工具
# =====================================================================

_LAYER_MAP = {
    "Conv2d": nn.Conv2d,
    "Conv3d": nn.Conv3d,
    "Linear": nn.Linear,
    "BatchNorm2d": nn.BatchNorm2d,
    "BatchNorm3d": nn.BatchNorm3d,
}

_DEFAULT_LAYER_NAMES = frozenset({"Conv2d", "Conv3d", "Linear"})


def _match_layer(module, layer_spec):
    """判断 module 是否匹配指定的 layer 类型."""
    module_name = type(module).__name__
    if layer_spec is None:
        return module_name in _DEFAULT_LAYER_NAMES
    specs = [layer_spec] if isinstance(layer_spec, str) else layer_spec
    return module_name in {ls for ls in specs if ls in _LAYER_MAP}


__all__ = [
    "BaseModule",
    "PretrainedInit", "XavierInit", "KaimingInit", "ConstantInit", "NormalInit",
    "InitConfig",
]
