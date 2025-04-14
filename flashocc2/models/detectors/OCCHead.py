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
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2, padding=0),
            nn.Conv2d(128, 272, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((200, 200)),
        )


    def forward(self, x):
        # input: torch.Size([B*N, 512, 29, 50])
        BN, C, W, H = x.shape

        x = self.head(x)  # torch.Size([B*N, 272, 200, 200])

        # (B*6,200,200,272)
        x = x.permute(0, 2, 3, 1)

        # (B,6,40000,272)
        x = x.view(-1, 6, 40000, 272)

        # (B,1,40000,272)
        #x = self.c_fusion(x)
        x = x.mean(dim=1)
        #print('1',x.shape)

        # (B,200,200,16,17)
        x = x.view(-1,200,200,16,17)
        #print('2',x.shape)
        # output:torch.Size([1, 200, 200, 16, 17]) B, x, y, z, cls
        return x
