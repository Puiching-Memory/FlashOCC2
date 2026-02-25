"""CrossEntropyLoss — 简化封装, 基于 torch.nn.

用 ``torch.nn.CrossEntropyLoss`` / ``torch.nn.BCEWithLogitsLoss`` 替换原自定义实现。
保留 ``CrossEntropyLoss`` 注册接口。
"""
from __future__ import annotations

import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F

from flashocc.models import LOSSES
from flashocc.models.losses.utils import weight_reduce_loss


def cross_entropy(pred, label, weight=None, reduction='mean', avg_factor=None,
                  class_weight=None, ignore_index=-100, avg_non_ignore=False):
    """CrossEntropy loss (简化)."""
    ignore_index = -100 if ignore_index is None else ignore_index
    loss = F.cross_entropy(pred, label, weight=class_weight,
                           reduction='none', ignore_index=ignore_index)
    if avg_factor is None and avg_non_ignore and reduction == 'mean':
        avg_factor = label.numel() - (label == ignore_index).sum().item()
    if weight is not None:
        weight = weight.float()
    return weight_reduce_loss(loss, weight=weight, reduction=reduction,
                              avg_factor=avg_factor)


def binary_cross_entropy(pred, label, weight=None, reduction='mean',
                         avg_factor=None, class_weight=None, ignore_index=-100,
                         avg_non_ignore=False):
    """Binary CrossEntropy loss (简化)."""
    ignore_index = -100 if ignore_index is None else ignore_index
    if pred.dim() != label.dim():
        # one-hot expand
        bin_labels = label.new_full((label.size(0), pred.size(-1)), 0)
        valid_mask = (label >= 0) & (label != ignore_index)
        inds = torch.nonzero(valid_mask & (label < pred.size(-1)), as_tuple=False)
        if inds.numel() > 0:
            bin_labels[inds, label[inds]] = 1
        valid_mask = valid_mask.view(-1, 1).expand_as(bin_labels).float()
        if weight is None:
            weight = valid_mask
        else:
            weight = weight.view(-1, 1).repeat(1, pred.size(-1)) * valid_mask
        label = bin_labels
    else:
        valid_mask = ((label >= 0) & (label != ignore_index)).float()
        weight = weight * valid_mask if weight is not None else valid_mask

    if avg_factor is None and avg_non_ignore and reduction == 'mean':
        avg_factor = valid_mask.sum().item()

    loss = F.binary_cross_entropy_with_logits(
        pred, label.float(), pos_weight=class_weight, reduction='none')
    return weight_reduce_loss(loss, weight.float(), reduction=reduction,
                              avg_factor=avg_factor)


def mask_cross_entropy(pred, target, label, reduction='mean', avg_factor=None,
                       class_weight=None, ignore_index=None, **kwargs):
    """Mask CrossEntropy loss."""
    assert ignore_index is None, "BCE loss does not support ignore_index"
    assert reduction == 'mean' and avg_factor is None
    num_rois = pred.size(0)
    inds = torch.arange(0, num_rois, dtype=torch.long, device=pred.device)
    pred_slice = pred[inds, label].squeeze(1)
    return F.binary_cross_entropy_with_logits(
        pred_slice, target, weight=class_weight, reduction='mean')[None]


@LOSSES.register_module(force=True)
class CrossEntropyLoss(nn.Module):
    def __init__(self, use_sigmoid=False, use_mask=False, reduction='mean',
                 class_weight=None, ignore_index=None, loss_weight=1.0,
                 avg_non_ignore=False):
        super().__init__()
        assert not (use_sigmoid and use_mask)
        self.use_sigmoid = use_sigmoid
        self.use_mask = use_mask
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.ignore_index = ignore_index
        self.avg_non_ignore = avg_non_ignore

        if use_sigmoid:
            self.cls_criterion = binary_cross_entropy
        elif use_mask:
            self.cls_criterion = mask_cross_entropy
        else:
            self.cls_criterion = cross_entropy

    def forward(self, cls_score, label, weight=None, avg_factor=None,
                reduction_override=None, ignore_index=None, **kwargs):
        reduction = reduction_override or self.reduction
        if ignore_index is None:
            ignore_index = self.ignore_index
        class_weight = (cls_score.new_tensor(self.class_weight)
                        if self.class_weight is not None else None)
        return self.loss_weight * self.cls_criterion(
            cls_score, label, weight, class_weight=class_weight,
            reduction=reduction, avg_factor=avg_factor,
            ignore_index=ignore_index, avg_non_ignore=self.avg_non_ignore,
            **kwargs)
