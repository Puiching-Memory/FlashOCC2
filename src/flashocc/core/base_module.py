"""BaseModule — 支持 init_weights 的 nn.Module 基类."""

from __future__ import annotations

import copy
import warnings

import torch.nn as nn


class BaseModule(nn.Module):
    """所有 FlashOCC 模型的基类.

    支持 ``init_cfg`` 配置式权重初始化.
    """

    def __init__(self, init_cfg=None):
        super().__init__()
        self.init_cfg = copy.deepcopy(init_cfg) if init_cfg else None
        self._is_init = False

    def init_weights(self):
        """根据 init_cfg 初始化权重."""
        if self.init_cfg is None:
            return
        if self._is_init:
            warnings.warn(f"{self.__class__.__name__} 已经初始化, 跳过.")
            return

        cfgs = self.init_cfg if isinstance(self.init_cfg, list) else [self.init_cfg]

        for cfg in cfgs:
            cfg = cfg.copy()
            init_type = cfg.pop("type", None)
            layer = cfg.pop("layer", None)

            if init_type == "Pretrained":
                pretrained = cfg.get("checkpoint")
                if pretrained:
                    from .checkpoint import load_checkpoint
                    load_checkpoint(self, pretrained, strict=False)

            elif init_type == "Xavier":
                distribution = cfg.get("distribution", "uniform")
                for m in self.modules():
                    if _match_layer(m, layer):
                        if distribution == "uniform":
                            nn.init.xavier_uniform_(m.weight)
                        else:
                            nn.init.xavier_normal_(m.weight)
                        if hasattr(m, "bias") and m.bias is not None:
                            nn.init.constant_(m.bias, 0)

            elif init_type == "Kaiming":
                for m in self.modules():
                    if _match_layer(m, layer):
                        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                        if hasattr(m, "bias") and m.bias is not None:
                            nn.init.constant_(m.bias, 0)

            elif init_type == "Constant":
                val = cfg.get("val", 0)
                for m in self.modules():
                    if _match_layer(m, layer):
                        nn.init.constant_(m.weight, val)

            elif init_type == "Normal":
                std = cfg.get("std", 0.01)
                for m in self.modules():
                    if _match_layer(m, layer):
                        nn.init.normal_(m.weight, 0, std)

        self._is_init = True


_LAYER_MAP = {
    "Conv2d": nn.Conv2d,
    "Conv3d": nn.Conv3d,
    "Linear": nn.Linear,
    "BatchNorm2d": nn.BatchNorm2d,
    "BatchNorm3d": nn.BatchNorm3d,
}


def _match_layer(module, layer_spec):
    """判断 module 是否匹配指定的 layer 类型."""
    if layer_spec is None:
        return isinstance(module, (nn.Conv2d, nn.Conv3d, nn.Linear))
    if isinstance(layer_spec, str):
        layer_spec = [layer_spec]
    return any(isinstance(module, _LAYER_MAP[ls]) for ls in layer_spec if ls in _LAYER_MAP)


__all__ = ["BaseModule"]
