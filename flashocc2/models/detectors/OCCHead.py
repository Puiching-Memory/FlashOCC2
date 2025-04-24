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
            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.Conv2d(
                2048,
                512,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.Conv2d(
                512,
                16 * 17,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )

        self.predicter = nn.Sequential(
            nn.Linear(16 * 17, 16 * 17 * 2),
            nn.Softplus(),
            nn.Linear(16 * 17 * 2, 16 * 17),
        )

    def forward(self, x):
        # input: bev_feat: (B, C, Dy, Dx)
        
        B, C, W, H = x.shape
        print("A", x.shape)

        x = self.head(x).permute(0, 3, 2, 1).contiguous()
        print("B", x.shape) # B, Dz*cls, 200, 200 -> B, 200, 200, Dz*cls

        x = self.predicter(x)
        print("C", x.shape)

        x = x.reshape(B, W*4, H*4, 16, 17)

        # output:torch.Size([1, 200, 200, 16, 17]) B, x, y, z, cls
        return x


@MODELS.register_module()
class BEVOCCHead2D(BaseModule):
    def __init__(
        self,
        in_dim=256,
        out_dim=256,
        Dz=16,
        use_mask=True,
        num_classes=18,
        use_predicter=True,
        class_balance=False,
        loss_occ=None,
    ):
        super(BEVOCCHead2D, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.Dz = Dz
        out_channels = out_dim if use_predicter else num_classes * Dz
        self.final_conv = nn.Conv2d(
            self.in_dim,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.use_predicter = use_predicter
        if use_predicter:
            self.predicter = nn.Sequential(
                nn.Linear(self.out_dim, self.out_dim * 2),
                nn.Softplus(),
                nn.Linear(self.out_dim * 2, num_classes * Dz),
            )

        self.use_mask = use_mask
        self.num_classes = num_classes

        self.class_balance = class_balance
        if self.class_balance:
            class_weights = torch.from_numpy(
                1 / np.log(nusc_class_frequencies[:num_classes] + 0.001)
            )
            self.cls_weights = class_weights
            loss_occ["class_weight"] = class_weights  # ce loss
        self.loss_occ = build_loss(loss_occ)

    def forward(self, img_feats):
        """
        Args:
            img_feats: (B, C, Dy, Dx)

        Returns:

        """
        # (B, C, Dy, Dx) --> (B, C, Dy, Dx) --> (B, Dx, Dy, C)
        occ_pred = self.final_conv(img_feats).permute(0, 3, 2, 1)
        bs, Dx, Dy = occ_pred.shape[:3]
        if self.use_predicter:
            # (B, Dx, Dy, C) --> (B, Dx, Dy, 2*C) --> (B, Dx, Dy, Dz*n_cls)
            occ_pred = self.predicter(occ_pred)
            occ_pred = occ_pred.view(bs, Dx, Dy, self.Dz, self.num_classes)

        return occ_pred
