import sys
import os

sys.path.append(os.path.abspath("./"))

import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.models import block


class model(nn.Module):
    def __init__(self):
        super().__init__()

        self.backboneP3 = nn.Sequential(
            block.Conv(3, 64, 3, 2),  # 0-P1/2
            block.Conv(64, 128, 3, 2),  # 1-P2/4
            block.C3k2(128, 256, 2, False, 0.25),  # 2
            block.Conv(256, 256, 3, 2),  # 3-P3/8
        )
        self.backboneP4 = nn.Sequential(
            block.C3k2(256, 512, 2, False, 0.25),  # 4
            block.Conv(512, 512, 3, 2),  # 5-P4/16
        )
        self.backboneP5 = nn.Sequential(
            block.C3k2(512, 512, 2, True, 0.5),  # 6
            block.Conv(512, 1024, 3, 2),  # 7-P5/32
        )
        self.backboneTop = nn.Sequential(
            block.C3k2(1024, 1024, 2, True, 0.5),  # 8
            block.SPPF(1024, 1024, 5),  # 9
        )
        self.heatmap3D = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 80-1, 1, 1, 0),
        )
        self.upsample1 = block.DySample(1024, 2, "lp", 4, False)
        self.upsample2 = block.DySample(512, 2, "lp", 4, False)
        self.neckConv1 = block.C3k2(1024 + 512, 512, 2, False, 0.5)
        self.neckConv2 = block.C3k2(512 + 256, 256, 2, False, 0.5)

    def forward(self, x: torch.Tensor) -> tuple:
        # input shape (B, 3, 384, 1280)
        x_p3 = self.backboneP3(x)
        x_p4 = self.backboneP4(x_p3)
        x_p5 = self.backboneP5(x_p4)
        x_backbone = self.backboneTop(x_p5)

        x = self.upsample1(x_backbone)  # 11
        x = torch.cat([x_p4, x], dim=1)  # 12-cat backbone P4
        x = self.neckConv1(x)  # 13
        x = self.upsample2(x)  # 14
        x = torch.cat([x_p3, x], dim=1)  # 15-# cat backbone P3
        x_neck = self.neckConv2(x)  # 16 (P3/8-small)

        return (
            x_p3,  # 0
            x_p4,  # 1
            x_p5,  # 2
            x_backbone,  # 3-(B, 1024, 12, 40)
            x_neck,  # 4-(B, 256, 48, 160)
            self.heatmap3D(x_neck),  # 5-(B, N, 48, 160)
        )


if __name__ == "__main__":
    import math
    from matplotlib import pyplot as plt
    from dataset import data_set
    from lib.cam.cnn_cam import module_feature_saver

    test_model = model()
    test_model.eval()

    print(test_model)

    dataset = data_set()

    modle_layers = list(test_model.children())

    # 注册hook获取模型中间层输出
    savers = []
    for layer in modle_layers:
        savers.append(module_feature_saver(layer))

    print(f"number of layers: {len(savers)}")

    # 执行一次推理
    for inputs, targets, data_info in dataset.get_test_loader():
        outputs = test_model(inputs)
        for index, o in enumerate(outputs):
            print(f"output-{index}: {o.shape}")
        break

    # 绘图
    fig, axes = plt.subplots(math.ceil((len(savers) + 1) / 5), 5)
    axes = axes.flatten()

    # 绘制输入图像
    axes[0].imshow(inputs[0].permute(1, 2, 0))
    axes[0].title.set_text("input_image")

    # 绘制中间层输出
    for index, saver in enumerate(savers):
        axes[index + 1].imshow(saver.temp_feature)
        axes[index + 1].title.set_text(f"{saver.layer_name}-{index}")

    # # 绘制检测头输出
    # for index, output in enumerate(outputs):
    #     axes[index+len(savers)+1].imshow(outputs[index][0][0].detach().numpy())
    #     axes[index+len(savers)+1].title.set_text(f"output-{index}")

    plt.show()
