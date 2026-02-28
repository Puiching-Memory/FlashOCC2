"""Simplified MVXTwoStageDetector for FlashOCC.

BEVDet inherits directly from MVXTwoStageDetector.
Only the image pipeline is implemented because FlashOCC is camera-only.
"""
import torch
import torch.nn as nn
from flashocc.core import BaseModule
from flashocc.models import DETECTORS


@DETECTORS.register
class MVXTwoStageDetector(BaseModule):
    """Camera-only 3D 检测器基类.

    仅实现图像分支, 用于 FlashOCC / BEVDet 系列模型.
    """

    def __init__(self,
                 img_backbone=None,
                 img_neck=None,
                 pts_bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 **kwargs):
        super().__init__(init_cfg=init_cfg)

        if img_backbone is not None:
            self.img_backbone = img_backbone
        if img_neck is not None:
            self.img_neck = img_neck
        if pts_bbox_head is not None:
            self.pts_bbox_head = pts_bbox_head

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    # ---------- properties ----------
    @property
    def with_img_backbone(self):
        return hasattr(self, 'img_backbone') and self.img_backbone is not None

    @property
    def with_img_neck(self):
        return hasattr(self, 'img_neck') and self.img_neck is not None

    @property
    def with_pts_bbox(self):
        return hasattr(self, 'pts_bbox_head') and self.pts_bbox_head is not None

    # ---------- forward ----------
    def forward(self, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)

    def forward_train(self, **kwargs):
        raise NotImplementedError

    def forward_test(self, **kwargs):
        raise NotImplementedError

    # ---------- pts head helpers ----------
    def forward_pts_train(self, pts_feats, gt_bboxes_3d, gt_labels_3d,
                          img_metas, gt_bboxes_ignore=None):
        """Forward the pts branch during training."""
        if not self.with_pts_bbox:
            return {}
        outs = self.pts_bbox_head(pts_feats)
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs)
        return losses

    def simple_test_pts(self, x, img_metas, rescale=False):
        """Simple test for pts bbox head."""
        outs = self.pts_bbox_head(x)
        bbox_list = self.pts_bbox_head.get_bboxes(outs, img_metas, rescale=rescale)
        bbox_results = [
            dict(boxes_3d=bboxes, scores_3d=scores, labels_3d=labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results
