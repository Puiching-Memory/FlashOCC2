import sys
import os

sys.path.append(os.path.abspath("./"))

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import tensordict
from lib.utils.logger import logger

class model(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model('mobilenetv4_conv_medium', pretrained=True, num_classes=0, global_pool='', cache_dir="./checkpoint/")
        self.layer1 = nn.Conv2d(1280 * 6, 272, kernel_size=1, stride=1, padding=0)
        self.upsample = nn.Upsample(scale_factor=7, mode='bilinear', align_corners=True)
        self.pool = nn.AdaptiveAvgPool2d((200, 200))
    def forward(self, x: torch.Tensor)->tensordict.TensorDict:
        # input image: torch.Size([B, 6, 3, 900, 1600])
        
        # reshape input: B, 6, 3, 900, 1600 -> B*6, 3, 900, 1600
        x = x.view(x.shape[0] * x.shape[1], x.shape[2], x.shape[3], x.shape[4])

        # backbone: B*6, 3, 900, 1600 -> B*6, 1280, 29, 50
        x = self.backbone(x)

        # B, 1280 * 6, 29, 50
        x = x.view(x.shape[0] // 6, x.shape[1]*6, x.shape[2], x.shape[3])
        
        # B, 1280 * 6, 203, 350
        x = self.upsample(x)

        # B, 272, 203, 350
        x = self.layer1(x)

        # B, 272, 200, 200
        x = self.pool(x)

        # B, 200, 200, 16, 17
        x = x.view(x.shape[0], x.shape[2], x.shape[3], x.shape[1] // 17, 17)
 
        # output: torch.Size([B, 200, 200, 16, 17])
        return tensordict.TensorDict({'output': x}, batch_size=x.shape[0])


if __name__ == "__main__":
    from lib.datasets.openocc import datasetOpenOCC
    from torch.utils.data import RandomSampler, DistributedSampler, BatchSampler
    from torchdata.nodes import SamplerWrapper, ParallelMapper, Loader, Batcher, Prefetcher, PinMemory
    from pytorch3d.transforms import quaternion_to_matrix

    model0 = model()
    model0.train()
    model0.cuda()

    dataset = datasetOpenOCC("dataset/nuscenes","val")
    sampler = RandomSampler(dataset)

    node = SamplerWrapper(sampler)
    node = ParallelMapper(node, map_fn=dataset.__getitem__, num_workers=1, method="process")
    # node = Batcher(node, batch_size=2) # TODO:多批次处理会打包为List,这与我们期望的每个参数打包不同
    # node = PinMemory(node)
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
        ) in loader:

        sensor2ego_rotation = quaternion_to_matrix(sensor2ego_rotation)
        cam_ego2global_rotation = quaternion_to_matrix(cam_ego2global_rotation)

        sensor2ego_translation = sensor2ego_translation.unsqueeze(1)
        images = images.unsqueeze(0)
        images = images.cuda()
        logger.debug(images.shape)

        x = model0(images)
        logger.debug(x) 
        break

