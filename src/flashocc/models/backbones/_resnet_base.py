"""ResNet backbone — 基于 timm 实现.

用 ``timm.create_model`` 替换原自定义的 ~250 行实现。
保留 BasicBlock / Bottleneck 导出 (供 depthnet.py 等使用)。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp

try:
    import timm
except ImportError:
    raise ImportError("请安装 timm: pip install timm")

from flashocc.models import BACKBONES

# torchvision BasicBlock/Bottleneck 用于 timm ResNet
from torchvision.models.resnet import BasicBlock as _TvBasicBlock
from torchvision.models.resnet import Bottleneck as _TvBottleneck

_NORM_MAP = {
    "BN": nn.BatchNorm2d, "BN2d": nn.BatchNorm2d,
    "SyncBN": nn.SyncBatchNorm, "GN": None,
}


def _norm_cfg_to_layer(norm_cfg):
    """将 norm_cfg dict 转为 norm_layer callable."""
    if norm_cfg is None:
        return nn.BatchNorm2d
    tp = norm_cfg.get("type", "BN") if hasattr(norm_cfg, 'get') else "BN"
    return _NORM_MAP.get(tp, nn.BatchNorm2d) or nn.BatchNorm2d


class BasicBlock(_TvBasicBlock):
    """兼容 norm_cfg 参数的 BasicBlock."""

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 norm_cfg=None, **kwargs):
        norm_layer = _norm_cfg_to_layer(norm_cfg)
        super().__init__(inplanes, planes, stride=stride,
                         downsample=downsample, norm_layer=norm_layer)


class Bottleneck(_TvBottleneck):
    """兼容 norm_cfg 参数的 Bottleneck."""

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 norm_cfg=None, **kwargs):
        norm_layer = _norm_cfg_to_layer(norm_cfg)
        super().__init__(inplanes, planes, stride=stride,
                         downsample=downsample, norm_layer=norm_layer)

_TIMM_MAP = {
    18: "resnet18",
    34: "resnet34",
    50: "resnet50",
    101: "resnet101",
    152: "resnet152",
}


@BACKBONES.register
class ResNet(nn.Module):
    """ResNet backbone (timm wrapper).

    Args:
        depth: ResNet 深度 (18, 34, 50, 101, 152).
        num_stages: 阶段数 (1-4).
        out_indices: 输出阶段索引.
        frozen_stages: 冻结阶段数 (-1 表示不冻结).
        norm_cfg: 归一化层配置 (忽略, timm 内部管理).
        norm_eval: 是否在训练时将 BN 设为 eval 模式.
        with_cp: 使用梯度检查点节省显存.
        pretrained: 预训练模型路径或 True.
    """

    def __init__(self, depth=50, num_stages=4, out_indices=(0, 1, 2, 3),
                 frozen_stages=-1, norm_cfg=None, norm_eval=True,
                 with_cp=False, style="pytorch", pretrained=None,
                 init_cfg=None, **kwargs):
        super().__init__()
        self.depth = depth
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval
        self.with_cp = with_cp
        self.pretrained = pretrained

        timm_name = _TIMM_MAP.get(depth, f"resnet{depth}")

        # 判断是否使用 timm 内置预训练
        use_timm_pretrained = (pretrained is True)

        self.backbone = timm.create_model(
            timm_name,
            pretrained=use_timm_pretrained,
            features_only=True,
            # timm 的 stage 0 是 stem, mmdet 的 out_indices 0 是 layer1
            # 所以 timm 需要输出 max(out_indices)+2 个 stage (含 stem)
            # 然后从 timm 结果中用 i+1 取对应 layer
            out_indices=list(range(max(out_indices) + 2)),
        )

        # 保存 timm 所有 stage 的 channel 信息, 方便后续按 mmdet 索引取
        all_channels = self.backbone.feature_info.channels()
        # mmdet out_indices 映射: mmdet_i → timm_i+1
        self.num_features = [all_channels[i + 1] for i in out_indices]

        # 加载自定义预训练权重 (字符串路径或 URL)
        if isinstance(pretrained, str):
            self._load_custom_pretrained(pretrained)

        self._freeze_stages()

    def _load_custom_pretrained(self, path: str):
        if path.startswith(("http://", "https://")):
            sd = torch.hub.load_state_dict_from_url(path, map_location="cpu")
        else:
            sd = torch.load(path, map_location="cpu", weights_only=False)
        if hasattr(sd, 'get'):
            sd = sd.get("state_dict", sd.get("model", sd))
        if any(k.startswith("module.") for k in sd):
            sd = {k.removeprefix("module."): v for k, v in sd.items()}
        missing, unexpected = self.backbone.load_state_dict(sd, strict=False)
        from flashocc.core.log import logger
        if missing:
            logger.info(f"[ResNet] Missing keys: {len(missing)}")
        if unexpected:
            logger.info(f"[ResNet] Unexpected keys: {len(unexpected)}")

    def _freeze_stages(self):
        if self.frozen_stages < 0:
            return
        # 冻结 stem
        if hasattr(self.backbone, "conv1"):
            for m in [self.backbone.conv1, self.backbone.bn1]:
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False
        # 冻结各 stage (timm resnet: layer1 ~ layer4)
        for i in range(self.frozen_stages):
            layer_name = f"layer{i + 1}"
            if hasattr(self.backbone, layer_name):
                layer = getattr(self.backbone, layer_name)
                layer.eval()
                for p in layer.parameters():
                    p.requires_grad = False

    def forward(self, x):
        features = self.backbone(x)
        # mmdet out_indices i 对应 timm features[i+1] (timm 0 是 stem)
        outs = tuple(features[i + 1] for i in self.out_indices)
        return outs

    def train(self, mode=True):
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            _bn_names = frozenset({'BatchNorm2d', 'SyncBatchNorm'})
            for m in self.modules():
                if type(m).__name__ in _bn_names:
                    m.eval()


__all__ = ["BasicBlock", "Bottleneck", "ResNet"]
