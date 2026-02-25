"""神经网络构建模块 — Conv / Norm / Init 工具."""

from __future__ import annotations

import torch.nn as nn

# =====================================================================
#  查找表
# =====================================================================

_CONV = {
    "Conv1d": nn.Conv1d,
    "Conv2d": nn.Conv2d,
    "Conv3d": nn.Conv3d,
    "Conv": nn.Conv2d,
}

_NORM = {
    "BN": nn.BatchNorm2d, "BN1d": nn.BatchNorm1d, "BN2d": nn.BatchNorm2d,
    "BN3d": nn.BatchNorm3d, "SyncBN": nn.SyncBatchNorm,
    "GN": nn.GroupNorm, "LN": nn.LayerNorm,
    "IN": nn.InstanceNorm2d, "IN3d": nn.InstanceNorm3d,
}

_ACT = {
    "ReLU": nn.ReLU, "LeakyReLU": nn.LeakyReLU, "PReLU": nn.PReLU,
    "RReLU": nn.RReLU, "ReLU6": nn.ReLU6, "ELU": nn.ELU,
    "Sigmoid": nn.Sigmoid, "Tanh": nn.Tanh, "GELU": nn.GELU, "SiLU": nn.SiLU,
}

# =====================================================================
#  Conv
# =====================================================================


def build_conv_layer(cfg, *args, **kwargs) -> nn.Module:
    """根据 cfg 构建卷积层, cfg=None 默认 Conv2d."""
    if cfg is None:
        return nn.Conv2d(*args, **kwargs)
    cfg = cfg.copy()
    cls = _CONV[cfg.pop("type")]
    return cls(*args, **kwargs)


def _build_norm(norm_cfg, num_features):
    if norm_cfg is None:
        return None
    cfg = norm_cfg.copy()
    tp = cfg.pop("type")
    if tp == "GN":
        return nn.GroupNorm(cfg.pop("num_groups", 32), num_features, **cfg)
    if tp == "LN":
        return nn.LayerNorm(num_features, **cfg)
    return _NORM[tp](num_features, **cfg)


def _build_act(act_cfg, inplace=True):
    if act_cfg is None:
        return None
    cfg = act_cfg.copy()
    tp = cfg.pop("type")
    cls = _ACT[tp]
    if "inplace" not in cfg and hasattr(cls(), "inplace"):
        cfg["inplace"] = inplace
    return cls(**cfg)


class ConvModule(nn.Module):
    """Conv + Norm + Act 捆绑模块."""

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1,
                 bias="auto", conv_cfg=None, norm_cfg=None,
                 act_cfg=dict(type="ReLU"), inplace=True,
                 order=("conv", "norm", "act"), **kwargs):
        super().__init__()
        self.order = order
        self.with_norm = norm_cfg is not None
        self.with_activation = act_cfg is not None
        if bias == "auto":
            bias = not self.with_norm

        self.conv = build_conv_layer(
            conv_cfg, in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias)
        self.bn = _build_norm(norm_cfg, out_channels) if self.with_norm else None
        self.activate = _build_act(act_cfg, inplace) if self.with_activation else None

    def forward(self, x):
        for name in self.order:
            if name == "conv":
                x = self.conv(x)
            elif name == "norm" and self.bn is not None:
                x = self.bn(x)
            elif name == "act" and self.activate is not None:
                x = self.activate(x)
        return x


# =====================================================================
#  Norm (独立构建器)
# =====================================================================


def build_norm_layer(cfg: dict, num_features: int, postfix="") -> tuple[str, nn.Module]:
    """根据 cfg 构建归一化层.

    Returns:
        (name, layer) 元组.
    """
    if cfg is None:
        raise TypeError("cfg 不能为 None")
    cfg = cfg.copy()
    tp = cfg.pop("type")
    name = tp.lower() + str(postfix)

    if tp == "GN":
        layer = nn.GroupNorm(cfg.pop("num_groups", 32), num_features, **cfg)
    elif tp == "LN":
        layer = nn.LayerNorm(num_features, **cfg)
    else:
        layer = _NORM[tp](num_features, **cfg)

    return name, layer


# =====================================================================
#  权重初始化
# =====================================================================


def trunc_normal_init(module, mean=0.0, std=1.0, a=-2.0, b=2.0, bias=0.0):
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.trunc_normal_(module.weight, mean=mean, std=std, a=a, b=b)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def constant_init(module, val, bias=0.0):
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def kaiming_init(module, a=0, mode="fan_out", nonlinearity="relu", bias=0.0):
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.kaiming_normal_(module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def xavier_init(module, gain=1, bias=0.0, distribution="uniform"):
    if hasattr(module, "weight") and module.weight is not None:
        if distribution == "uniform":
            nn.init.xavier_uniform_(module.weight, gain=gain)
        else:
            nn.init.xavier_normal_(module.weight, gain=gain)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


__all__ = [
    "build_conv_layer", "ConvModule",
    "build_norm_layer",
    "trunc_normal_init", "constant_init", "kaiming_init", "xavier_init",
]
