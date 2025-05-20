import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmdet3d.registry import MODELS
from mmengine.runner.amp import autocast


@MODELS.register_module()
class FlashOcc2Head(BaseModule):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.in_dim = in_dim
        self.num_classes = num_classes

        self.head = nn.Sequential(
            nn.Conv2d(in_dim*6, 512, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(128, 16*num_classes, kernel_size=3, stride=1, padding=1),
        )
        self.predicter = nn.Sequential(
            nn.Linear(16*num_classes, 16*num_classes*2),
            nn.SiLU(),
            nn.Linear(16*num_classes*2, 16*num_classes),
        )

    def forward(self, x):
        B, N, C, H, W = x.shape
        x = x.view(B, N*C, H, W)  # 合并时间步和通道
        
        x = self.head(x)
        x = x[:, :, :200, :200]  # 裁剪到 (B, 16*17, 200, 200)
        
        x = x.permute(0, 3, 2, 1).contiguous()  # (B, 200, 200, 16*17)
        x = self.predicter(x)
        x = x.reshape(B, 200, 200, 16, self.num_classes)
        
        return x