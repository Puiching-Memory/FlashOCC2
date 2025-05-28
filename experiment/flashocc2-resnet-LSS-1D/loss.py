import sys
import os

sys.path.append(os.path.abspath("./"))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from lib.cfg.base import LossBase
from lib.utils.assigner import notassigner
from lib.utils.metrics import bbox_iou
from lib.models.loss import varifocal_loss
from lib.utils.logger import logger


class loss(LossBase):
    def __init__(self):
        self.heatmap_loss = nn.BCEWithLogitsLoss()

    def __call__(
        self, output: tuple[torch.Tensor, ...], target: dict[torch.Tensor]
    ) -> tuple[torch.Tensor, dict[torch.Tensor]]:
        """
        计算损失函数
        ---
        output:dict[torch.Tensor]
        return:torch.Tensor
            损失函数值
        """

        # print(output[5].shape, target["heatmap3D"].shape)
        loss_heatmap = self.heatmap_loss(
            output[5].reshape(-1),
            target["heatmap3D"].reshape(-1),
        )

        loss = loss_heatmap
        loss_info = {"loss_heatmap3D": loss_heatmap}

        return loss, loss_info

    def __str__(self):
        return f"{self.__class__.__name__}\n{vars(self)}"


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    B, num_classes, H, W = 16, 2, 48, 160

    logits = torch.rand(
        [B, num_classes, H, W], dtype=torch.float16, requires_grad=True, device=device
    )
    # logits = logits.reshape(B,H,W,num_classes)
    labels = torch.rand([B, num_classes, H, W], dtype=torch.float64, device=device)
    # labels = torch.randint(0,num_classes,[B, H, W])
    print(logits, labels)
    print(logits.shape, labels.shape)
    print(logits.dtype, labels.dtype)

    print(varifocal_loss(logits, labels))
