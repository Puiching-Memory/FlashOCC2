from typing import Optional, Union, Dict, List
from mmdet3d.structures.det3d_data_sample import (
    ForwardResults,
    OptSampleList,
    SampleList,
    PointData
)

from focal_loss.focal_loss import FocalLoss
from mmdet3d.models.losses import LovaszLoss
import torch
import numpy as np
from mmdet3d.models import Base3DSegmentor
from mmengine.logging import logger

# from mmengine.model import BaseModel
from mmdet3d.registry import MODELS


@MODELS.register_module()
class Flashocc2Orchestrator(Base3DSegmentor):

    def __init__(
        self,
        backbone,
        neck,
        view_transformer,
        mixter,
        head,
    ):
        super().__init__()

        self.backbone = MODELS.build(backbone)
        self.neck = MODELS.build(neck)
        # self.view_transformer = MODELS.build(view_transformer)
        # self.mixter = MODELS.build(mixter)
        self.head = MODELS.build(head)
        self.loss_cls = LovaszLoss(loss_type='multi_class',
                                      per_sample=False,
                                      reduction='none')

    def _forward(
        self, batch_inputs: dict, batch_data_samples: OptSampleList = None
    ) -> torch.Tensor:

        # batch_inputs: dict_keys(['img', 'occ_200'])
        # img: list[torch.Size([6, 3, 900, 1600]),...]
        # occ_200: list[torch.Size([N, 4]),...]
        img_feats_dict = self.extract_feat(
            torch.stack(batch_inputs["img"], dim=1)
        )  # torch.Size([B*N, 512, 29, 50])
        x = self.head(
            img_feats_dict["img_feats"]
        )  # TODO: torch.Size([1, 200, 200, 16, 17]) B, x, y, z, cls

        return x

    def loss(
        self, batch_inputs: dict, batch_data_samples: SampleList
    ) -> Dict[str, torch.Tensor]:
        """Calculate losses from a batch of inputs and data samples."""

        # batch_inputs: dict_keys(['img', 'occ_200'])
        # img: list[torch.Size([6, 3, 900, 1600]),...]
        # occ_200: list[torch.Size([N, 4]),...]

        batch_outputs = self._forward(
            batch_inputs, batch_data_samples
        )  # B, X, Y, Z, Cls

        voxels_gt = self.multiscale_supervision(
            batch_inputs["occ_200"],
            [1, 1, 1],
            [len(batch_data_samples), 200, 200, 16],
        )  # torch.Size([B, 200, 200, 16])

        voxels_gt = torch.flatten(voxels_gt, start_dim=0) # -1
        batch_outputs = batch_outputs.view(-1,batch_outputs.shape[4]) # -1,Cls
        batch_outputs = batch_outputs.softmax(dim=-1)

        loss1 = torch.nan_to_num(self.loss_cls(batch_outputs, voxels_gt))

        return {"loss": loss1}

    def predict(self, batch_inputs: dict, batch_data_samples: SampleList) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing."""
        batch_outputs = self._forward(
            batch_inputs, batch_data_samples
        )  # B, X, Y, Z, Cls

        voxel_logit = batch_outputs.reshape(len(batch_data_samples), 200, 200, 16, 17)
        voxel_pred = torch.argmax(voxel_logit, dim=-1)

        voxels_gt = self.multiscale_supervision(
            batch_inputs["occ_200"],
            [1, 1, 1],
            [len(batch_data_samples), 200, 200, 16],
        )  # torch.Size([B, 200, 200, 16])
        
        #voxels_gt = voxels_gt.reshape(len(batch_data_samples),-1)

        for i in range(len(batch_data_samples)):
            batch_data_samples[i].set_data({
                'pts_seg_logits': PointData(pts_seg_logits=voxel_logit[i]),
                'pred_pts_seg': PointData(pts_semantic_mask=voxel_pred[i])
            })
            # 设置评估用的 ground truth
            batch_data_samples[i].eval_ann_info['pts_semantic_mask'] = \
                voxels_gt[i].cpu().numpy().astype(np.uint8)
            
        return batch_data_samples

    def encode_decode(
        self, batch_inputs: torch.Tensor, batch_data_samples: SampleList
    ) -> torch.Tensor:
        """Placeholder for encode images with backbone and decode into a
        semantic segmentation map of the same size as input."""
        return

    def multiscale_supervision(
        self, gt_occ: list[torch.Tensor], ratio: list, gt_shape: list
    ) -> torch.Tensor:
        # gt_occ: list[torch.tensor(N,4)] x,y,z,cls
        # ratio: downsample_X, downsample_Y, downsample_Z
        # gt_shape: B, X, Y, Z
        gt = (
            torch.zeros([gt_shape[0], gt_shape[1], gt_shape[2], gt_shape[3]])
            .to(gt_occ[0].device)
            .type(torch.long)
        )
        for i in range(gt.shape[0]):
            coords_x = gt_occ[i][:, 0].to(torch.float) // ratio[0]
            coords_y = gt_occ[i][:, 1].to(torch.float) // ratio[1]
            coords_z = gt_occ[i][:, 2].to(torch.float) // ratio[2]
            coords_x = coords_x.to(torch.long)
            coords_y = coords_y.to(torch.long)
            coords_z = coords_z.to(torch.long)
            coords = torch.stack([coords_x, coords_y, coords_z], dim=1)
            gt[i, coords[:, 0], coords[:, 1], coords[:, 2]] = gt_occ[i][:, 3]

        return gt

    def extract_feat(self, batch_inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract features of images."""
        # N = 输入视图数量,例如:输入6视图
        x = batch_inputs
        B, N, C, H, W = x.size()
        x = x.reshape(B * N, C, H, W)

        x = self.backbone(x)
        x = self.neck(x)[0]

        return {
            "img_feats": x,  # torch.Size([B*N, 512, 29, 50])
        }
 