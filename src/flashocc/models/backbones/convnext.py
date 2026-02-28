"""ConvNeXt backbone via timm."""
from __future__ import annotations

import timm
import torch.nn as nn

from flashocc.models import BACKBONES


@BACKBONES.register
class TimmConvNeXt(nn.Module):
    """ConvNeXt backbone wrapped from timm.create_model."""

    def __init__(
        self,
        model_name: str = "convnext_tiny",
        out_indices=(2, 3),
        pretrained=True,
        with_cp: bool = False,
        init_cfg=None,
        **kwargs,
    ):
        super().__init__()
        self.out_indices = tuple(out_indices)
        self.with_cp = with_cp

        checkpoint_path = pretrained if isinstance(pretrained, str) else None
        use_timm_pretrained = bool(pretrained is True)

        self.backbone = timm.create_model(
            model_name,
            pretrained=use_timm_pretrained,
            checkpoint_path=checkpoint_path,
            features_only=True,
            out_indices=self.out_indices,
            **kwargs,
        )

        if self.with_cp and hasattr(self.backbone, "set_grad_checkpointing"):
            self.backbone.set_grad_checkpointing(enable=True)

        self.num_features = self.backbone.feature_info.channels()

    def forward(self, x):
        feats = self.backbone(x)
        return tuple(feats)

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self.with_cp and hasattr(self.backbone, "set_grad_checkpointing"):
            self.backbone.set_grad_checkpointing(enable=True)


__all__ = ["TimmConvNeXt"]
