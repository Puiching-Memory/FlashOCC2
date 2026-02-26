"""mmdet3d.datasets.custom_3d — Custom3DDataset base class."""
import copy
import os.path as osp

import numpy as np
import torch
from torch.utils.data import Dataset

from flashocc.core.io import load
from .builder import DATASETS


class Custom3DDataset(Dataset):
    """Base dataset class for 3D detection.

    Args:
        data_root (str): Root directory of the dataset.
        ann_file (str): Path to the annotation file.
        pipeline (list[dict]): Processing pipeline.
        classes (tuple[str]): Class names.
        modality (dict | None): Input modality.
        box_type_3d (str): Box type ('LiDAR', 'Camera', 'Depth').
        filter_empty_gt (bool): Filter samples with empty GT.
        test_mode (bool): Whether in test mode.
    """

    CLASSES = None
    PALETTE = None

    def __init__(self,
                 data_root,
                 ann_file,
                 pipeline=None,
                 classes=None,
                 modality=None,
                 box_type_3d='LiDAR',
                 filter_empty_gt=True,
                 test_mode=False,
                 **kwargs):
        super().__init__()
        self.data_root = data_root
        self.ann_file = ann_file
        self.test_mode = test_mode
        self.modality = modality
        self.filter_empty_gt = filter_empty_gt
        self.box_type_3d = box_type_3d

        self.CLASSES = self.get_classes(classes)
        self.cat2id = {name: i for i, name in enumerate(self.CLASSES)}

        # Load annotations
        self.data_infos = self.load_annotations(self.ann_file)

        # Build pipeline
        if pipeline is not None:
            from .pipelines import Compose
            self.pipeline = Compose(pipeline)
        else:
            self.pipeline = None

        # Filter empty
        if not test_mode:
            self._set_group_flag()

    def load_annotations(self, ann_file):
        """Load annotations from ann_file.

        Returns a list of data info dicts.
        """
        data = load(ann_file)
        data_infos = list(data['infos'])
        self.metadata = data.get('metadata', {})
        self.version = self.metadata.get('version', 'unknown')
        return data_infos

    @classmethod
    def get_classes(cls, classes=None):
        """Return class names tuple."""
        if classes is None:
            return cls.CLASSES or ()
        if hasattr(classes, 'upper'):
            # File path — one class per line
            with open(classes) as f:
                class_names = tuple(line.strip() for line in f if line.strip())
            return class_names
        if hasattr(classes, '__iter__'):
            return tuple(classes)
        raise ValueError(f'Unsupported type {type(classes)} for classes.')

    def _set_group_flag(self):
        self.flag = np.zeros(len(self), dtype=np.uint8)

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_data(idx)
        while True:
            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def prepare_train_data(self, idx):
        input_dict = self.get_data_info(idx)
        if input_dict is None:
            return None
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict) if self.pipeline else input_dict
        if self.filter_empty_gt and example is not None:
            gt_labels_3d = example.get('gt_labels_3d', None)
            if gt_labels_3d is not None and hasattr(gt_labels_3d, 'data'):
                gt_labels_3d = gt_labels_3d.data
            if gt_labels_3d is not None and len(gt_labels_3d) == 0:
                return None
        return example

    def prepare_test_data(self, idx):
        input_dict = self.get_data_info(idx)
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict) if self.pipeline else input_dict
        return example

    def get_data_info(self, index):
        """Get data info for the sample at *index*.

        Subclasses should override this.
        """
        info = self.data_infos[index]
        input_dict = dict(
            sample_idx=info.get('token', index),
            pts_filename=info.get('lidar_path', ''),
            sweeps=info.get('sweeps', []),
            timestamp=info.get('timestamp', 0) / 1e6,
        )
        if 'ann_infos' in info:
            input_dict['ann_info'] = info['ann_infos']
        return input_dict

    def pre_pipeline(self, results):
        """Pre-processing before pipeline (add paths etc.)."""
        results['img_prefix'] = self.data_root
        results['seg_prefix'] = self.data_root
        results['proposal_file'] = None
        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['seg_fields'] = []
        results['box_type_3d'] = self.box_type_3d
        results['box_mode_3d'] = 0  # LiDAR

    def _rand_another(self, idx):
        return np.random.randint(len(self))

    def get_ann_info(self, idx):
        return self.data_infos[idx].get('ann_infos', None)

    def format_results(self, outputs, **kwargs):
        """Format results for evaluation. Override in subclasses."""
        raise NotImplementedError

    def evaluate(self, results, **kwargs):
        """Evaluate predictions. Override in subclasses."""
        raise NotImplementedError
