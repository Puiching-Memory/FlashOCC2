import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmdet3d.registry import MODELS
from mmengine.runner.amp import autocast


@MODELS.register_module()
class FlashOcc2Head(BaseModule):
    def __init__(self, channels, num_classes):
        super(FlashOcc2Head, self).__init__()

        self.head = nn.Sequential(
            nn.Conv2d(768, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 272, kernel_size=1),
            nn.ReLU(),
        )

    def forward(self, x):
        # input: bev_feat: (B*6, C, Dy, Dx)
        # torch.Size([6, 128, 200, 200])
        BN, C, W, H = x.shape
        #print("1", x.shape)

        x = x.reshape(1, -1, W, H)  # (B, C*6, W, H)
        #print("2", x.shape)

        x = self.head(x)
        #print("3", x.shape)

        # (B,200,200,16,17)
        x = x.reshape(BN // 6, 200, 200, 16, 17)
        #print("4", x.shape)

        # output:torch.Size([1, 200, 200, 16, 17]) B, x, y, z, cls
        return x
