"""数据管线基础: to_tensor, LoadImageFromFile, LoadAnnotations."""

import numpy as np
import torch
from PIL import Image


def to_tensor(data):
    """将数据转换为 torch.Tensor."""
    if isinstance(data, torch.Tensor):
        return data
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    if isinstance(data, (list, tuple)):
        return torch.tensor(data)
    if isinstance(data, int):
        return torch.LongTensor([data])
    if isinstance(data, float):
        return torch.FloatTensor([data])
    raise TypeError(f"Cannot convert type {type(data)} to tensor")


class LoadImageFromFile:
    """从文件加载图片."""

    def __init__(self, to_float32=False, color_type="color", **kwargs):
        self.to_float32 = to_float32
        self.color_type = color_type

    def __call__(self, results):
        filename = results.get("img_filename", results.get("filename", None))
        if filename is None:
            return results
        img = np.array(Image.open(filename).convert("RGB"))
        if self.to_float32:
            img = img.astype(np.float32)
        results["img"] = img
        results["img_shape"] = img.shape
        results["ori_shape"] = img.shape
        return results


class LoadAnnotations:
    """加载标注信息 (bboxes, labels, masks 等)."""

    def __init__(self, with_bbox=True, with_label=True, with_mask=False,
                 with_seg=False, **kwargs):
        self.with_bbox = with_bbox
        self.with_label = with_label
        self.with_mask = with_mask
        self.with_seg = with_seg

    def __call__(self, results):
        ann = results.get("ann_info", {})
        if self.with_bbox and "gt_bboxes" in ann:
            results["gt_bboxes"] = ann["gt_bboxes"]
        if self.with_label and "gt_labels" in ann:
            results["gt_labels"] = ann["gt_labels"]
        return results


__all__ = ["to_tensor", "LoadImageFromFile", "LoadAnnotations"]
