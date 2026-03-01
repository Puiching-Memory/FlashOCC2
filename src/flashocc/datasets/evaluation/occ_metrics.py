"""Occupancy 评估指标 — 用 torchmetrics 替换手写 mIoU.

Metric_mIoU 改为基于 ``torchmetrics.JaccardIndex`` 实现。
Metric_FScore 保留原结构 (无现成 pip 替代)。
"""
from __future__ import annotations

import numpy as np
import os
from functools import reduce
from pathlib import Path

import torch
from sklearn.neighbors import KDTree
from termcolor import colored

from torchmetrics.classification import MulticlassJaccardIndex

from flashocc.constants import (
    OCC_CLASS_NAMES, POINT_CLOUD_RANGE, OCCUPANCY_SIZE, VOXEL_SIZE,
)

np.seterr(divide='ignore', invalid='ignore')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def pcolor(string, color, on_color=None, attrs=None):
    return colored(string, color, on_color, attrs)


class Metric_mIoU:
    """mIoU 指标 — 基于 torchmetrics.JaccardIndex."""

    def __init__(self, save_dir='.', num_classes=18,
                 use_lidar_mask=False, use_image_mask=False):
        self.class_names = list(OCC_CLASS_NAMES)
        self.save_dir = save_dir
        self.use_lidar_mask = use_lidar_mask
        self.use_image_mask = use_image_mask
        self.num_classes = num_classes

        self.point_cloud_range = list(POINT_CLOUD_RANGE)
        self.occupancy_size = list(OCCUPANCY_SIZE)
        self.voxel_size = VOXEL_SIZE
        self.occ_xdim = int((self.point_cloud_range[3] - self.point_cloud_range[0]) / self.occupancy_size[0])
        self.occ_ydim = int((self.point_cloud_range[4] - self.point_cloud_range[1]) / self.occupancy_size[1])
        self.occ_zdim = int((self.point_cloud_range[5] - self.point_cloud_range[2]) / self.occupancy_size[2])

        # torchmetrics JaccardIndex
        self.jaccard = MulticlassJaccardIndex(
            num_classes=num_classes,
            average=None,  # per-class IoU
            ignore_index=255,
            sync_on_compute=False,
        )
        self.cnt = 0

    def add_batch(self, semantics_pred, semantics_gt, mask_lidar, mask_camera):
        self.cnt += 1
        if self.use_image_mask:
            masked_gt = semantics_gt[mask_camera]
            masked_pred = semantics_pred[mask_camera]
        elif self.use_lidar_mask:
            masked_gt = semantics_gt[mask_lidar]
            masked_pred = semantics_pred[mask_lidar]
        else:
            masked_gt = semantics_gt
            masked_pred = semantics_pred

        # 转为 torch tensor (torchmetrics 需要)
        pred_t = torch.as_tensor(masked_pred.flatten(), dtype=torch.long)
        gt_t = torch.as_tensor(masked_gt.flatten(), dtype=torch.long)
        self.jaccard.update(pred_t, gt_t)

    def count_miou(self):
        from flashocc.core.log import logger
        # 计算 per-class IoU
        mIoU = self.jaccard.compute().numpy()

        # 获取混淆矩阵 (torchmetrics 内部 confmat)
        confmat = None
        if hasattr(self.jaccard, 'confmat'):
            confmat = self.jaccard.confmat.cpu().numpy()

        logger.info(f'===> per class IoU of {self.cnt} samples:')
        per_class = {}
        for i in range(self.num_classes - 1):
            iou_val = round(float(mIoU[i]) * 100, 2)
            logger.info(f'===> {self.class_names[i]} - IoU = {iou_val}')
            per_class[self.class_names[i]] = iou_val

        mean_iou = float(np.nanmean(mIoU[:self.num_classes - 1]) * 100)
        logger.info(f'===> mIoU of {self.cnt} samples: {round(mean_iou, 2)}')

        result = {
            'mIoU': mIoU,
            'mIoU_mean': mean_iou,
            'per_class_iou': per_class,
            'num_samples': self.cnt,
        }
        if confmat is not None:
            result['confusion_matrix'] = confmat
        return result


class Metric_FScore:
    """F-Score 指标 (保留原实现，无直接 pip 替代)."""

    def __init__(self, leaf_size=10, threshold_acc=0.6, threshold_complete=0.6,
                 voxel_size=None, range=None,
                 void=[17, 255], use_lidar_mask=False, use_image_mask=False):
        if voxel_size is None:
            voxel_size = list(OCCUPANCY_SIZE)
        if range is None:
            range = list(POINT_CLOUD_RANGE)
        self.leaf_size = leaf_size
        self.threshold_acc = threshold_acc
        self.threshold_complete = threshold_complete
        self.voxel_size = voxel_size
        self.range = range
        self.void = void
        self.use_lidar_mask = use_lidar_mask
        self.use_image_mask = use_image_mask
        self.cnt = 0
        self.tot_acc = 0.0
        self.tot_cmpl = 0.0
        self.tot_f1_mean = 0.0
        self.eps = 1e-8

    def voxel2points(self, voxel):
        mask = np.logical_not(
            reduce(np.logical_or, [voxel == v for v in self.void]))
        occIdx = np.where(mask)
        points = np.concatenate([
            occIdx[0][:, None] * self.voxel_size[0] + self.voxel_size[0] / 2 + self.range[0],
            occIdx[1][:, None] * self.voxel_size[1] + self.voxel_size[1] / 2 + self.range[1],
            occIdx[2][:, None] * self.voxel_size[2] + self.voxel_size[2] / 2 + self.range[2],
        ], axis=1)
        return points

    def add_batch(self, semantics_pred, semantics_gt, mask_lidar, mask_camera):
        self.cnt += 1
        if self.use_image_mask:
            semantics_gt[mask_camera == False] = 255
            semantics_pred[mask_camera == False] = 255
        elif self.use_lidar_mask:
            semantics_gt[mask_lidar == False] = 255
            semantics_pred[mask_lidar == False] = 255

        ground_truth = self.voxel2points(semantics_gt)
        prediction = self.voxel2points(semantics_pred)
        if prediction.shape[0] == 0:
            accuracy = completeness = fmean = 0
        else:
            prediction_tree = KDTree(prediction, leaf_size=self.leaf_size)
            ground_truth_tree = KDTree(ground_truth, leaf_size=self.leaf_size)
            complete_distance, _ = prediction_tree.query(ground_truth)
            accuracy_distance, _ = ground_truth_tree.query(prediction)
            completeness = (complete_distance.flatten() < self.threshold_complete).mean()
            accuracy = (accuracy_distance.flatten() < self.threshold_acc).mean()
            fmean = 2.0 / (1 / (accuracy + self.eps) + 1 / (completeness + self.eps))

        self.tot_acc += accuracy
        self.tot_cmpl += completeness
        self.tot_f1_mean += fmean

    def count_fscore(self):
        from flashocc.core.log import logger
        logger.info(f'\n######## F score: {self.tot_f1_mean / self.cnt} #######')


