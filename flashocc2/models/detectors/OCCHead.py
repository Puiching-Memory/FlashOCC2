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
            # 第一层转置卷积：将空间尺寸从 29x50 → 58x100，通道数减半
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # 第二层转置卷积：空间尺寸 → 116x200，通道数继续减半
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # 第三层转置卷积：空间尺寸 → 232x400，通道数调整为 16×17=272
            nn.ConvTranspose2d(128, 272, kernel_size=2, stride=2, padding=0),
            nn.ReLU(),
            # 插值调整到精确的 200x200 空间尺寸
            # nn.Upsample(size=(200, 200), mode='bilinear', align_corners=False),
            # 自适应池化层调整空间尺寸到 200x200
            nn.AdaptiveAvgPool2d((200, 200)),
        )

        self.c_fusion = nn.Sequential(nn.AdaptiveAvgPool3d((1,40000,272)))

    @autocast("cuda", torch.float32)
    def forward(self, x):
        # input: torch.Size([B*N, 512, 29, 50])
        BN, C, W, H = x.shape

        x = self.head(x)  # torch.Size([B*6, 272, 200, 200])

        # (B*6,200,200,272)
        x = x.permute(0, 2, 3, 1)

        # (B,6,40000,272)
        x = x.view(-1, 6, 40000, 272)

        # (B,1,40000,272)
        x = self.c_fusion(x)
        #print('1',x.shape)

        # (B,200,200,16,17)
        x = x.view(-1,200,200,16,17)
        #print('2',x.shape)
        # output:torch.Size([1, 200, 200, 16, 17]) B, x, y, z, cls
        return x
