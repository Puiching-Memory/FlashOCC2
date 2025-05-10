# Copyright (c) OpenMMLab. All rights reserved.
import os
from mmengine.registry import TRANSFORMS
import numpy as np
import torch

@TRANSFORMS.register_module()
class LoadOccGTFromFile():
    def __call__(self, results):
        occ_labels = np.load(results['occ_gt_path'])
        semantics = occ_labels['semantics']
        mask_lidar = occ_labels['mask_lidar']
        mask_camera = occ_labels['mask_camera']

        semantics = torch.from_numpy(semantics)
        mask_lidar = torch.from_numpy(mask_lidar)
        mask_camera = torch.from_numpy(mask_camera)

        if results.get('flip_dx', False):
            semantics = torch.flip(semantics, [0])
            mask_lidar = torch.flip(mask_lidar, [0])
            mask_camera = torch.flip(mask_camera, [0])

        if results.get('flip_dy', False):
            semantics = torch.flip(semantics, [1])
            mask_lidar = torch.flip(mask_lidar, [1])
            mask_camera = torch.flip(mask_camera, [1])

        results['voxel_semantics'] = semantics
        results['mask_lidar'] = mask_lidar
        results['mask_camera'] = mask_camera

        return results