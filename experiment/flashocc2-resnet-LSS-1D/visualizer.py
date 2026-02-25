import sys
import os

sys.path.append(os.path.abspath("./"))
from lib.cfg.base import VisualizerBase

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import v2
from torchvision.utils import save_image
from tensordict import TensorDict


class visualizer(VisualizerBase):
    def __init__(self):

        self.image_transforms = v2.Compose(
            [
                v2.Normalize(
                    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                    std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
                ),
                v2.ToDtype(torch.uint8, scale=True),
            ]
        )

    def decode_output(self,output:TensorDict):
        image = self.image_transforms(output["images"])

        return 

    def decode_target(self,target:TensorDict):
        #image = self.image_transforms(target["images"])
        save_image(target["images"], os.path.join(self.save_path, "target.png"))

        return


if __name__ == "__main__":
    import torch.utils.data
    from torch.utils.data import RandomSampler, DistributedSampler
    from torchdata.nodes import SamplerWrapper, ParallelMapper, Loader, pin_memory
    from tqdm import tqdm
    from lib.datasets.openocc import datasetOpenOCC

    vis = visualizer()
    dataset = datasetOpenOCC("dataset/nuscenes","val")
    sampler = RandomSampler(dataset)

    node = SamplerWrapper(sampler)
    node = ParallelMapper(node, map_fn=dataset.__getitem__, num_workers=1, method="process")
    loader = Loader(node)
    
    for (
        images,
        (
            sensor2ego_translation,
            sensor2ego_rotation,
            cam_ego2global_translation,
            cam_ego2global_rotation,
            sensor2lidar_rotation,
            sensor2lidar_translation,
            cam_intrinsic,
        ),
        (
            lidar2ego_translation,
            lidar2ego_rotation,
            lidar_ego2global_translation,
            lodar_ego2global_rotation
        ),
        (
            occ_instances,
            occ_semantics,
            occ_flow
        )
        ) in tqdm(loader,total=len(dataset)):

        vis.decode_target(TensorDict({
            "images": images,
            }))
        break

