from typing import Optional, Union, Dict, List
from mmdet3d.structures.det3d_data_sample import (
    ForwardResults,
    OptSampleList,
    SampleList,
    PointData,
)

from .losses import FocalLoss
from mmdet3d.models.losses import LovaszLoss
from torch.nn import CrossEntropyLoss
import torch
import numpy as np
import cv2
from mmdet3d.models import Base3DSegmentor
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
        self.view_transformer = MODELS.build(view_transformer)
        # self.mixter = MODELS.build(mixter)
        self.head = MODELS.build(head)
        self.loss_cls = LovaszLoss(
            loss_type="multi_class", per_sample=False, reduction="none"
        )
        # self.loss_cls = FocalLoss(gamma=2,task_type='multi-class',num_classes=17)
        # self.loss_depth = torch.nn.CrossEntropyLoss()
        self.deg = [0, 0, 0]

    def _forward(
        self, batch_inputs: dict, batch_data_samples: OptSampleList = None
    ) -> torch.Tensor:

        # batch_inputs: dict_keys(['img', 'occ_200'])
        # img: list[torch.Size([6, 3, 900, 1600]),...]
        # occ_200: list[torch.Size([N, 4]),...]

        device = batch_inputs["img"][0].device

        img_feats_dict = self.extract_feat(torch.stack(batch_inputs["img"], dim=0))

        # print("img_feats_dict",img_feats_dict["img_feats"].shape) # torch.Size([1, 6, 512, 29, 50])

        ego2img = torch.tensor(
            np.array([i.ego2img for i in batch_data_samples]),
            dtype=torch.float32,
            device=device,
        )  # (B, N, 4, 4)
        cam2img = torch.tensor(
            np.array([i.cam2img for i in batch_data_samples]),
            dtype=torch.float32,
            device=device,
        )  # (B, N, 4, 4)
        ego2global = torch.tensor(
            np.array([i.ego2global for i in batch_data_samples]),
            dtype=torch.float32,
            device=device,
        )  # (B, 4, 4)
        cam2ego = torch.tensor(
            np.array([i.cam2ego for i in batch_data_samples]),
            dtype=torch.float32,
            device=device,
        )  # (B, N, 4, 4)

        B = ego2img.shape[0]
        N = ego2img.shape[1]

        # cam -> ego: (B, N, 4, 4) @ (B, N, 4, 4)
        # cam2ego = torch.linalg.inv(ego2img) @ torch.linalg.inv(cam2img)

        # 截取前 3×3 部分 → (B, N, 3, 3)
        intrins = cam2img[:, :, :3, :3]

        post_rots = (
            torch.eye(3, device=device).unsqueeze(0).unsqueeze(0).repeat(B, N, 1, 1)
        )  # (B, N, 3, 3)
        post_trans = torch.zeros((B, N, 3), device=device)  # (B, N, 3)
        bda_rot = torch.eye(3, device=device).unsqueeze(0).repeat(B, 1, 1)  # (B, 3, 3)
        ego2global = ego2global.unsqueeze(1).repeat(1, N, 1, 1)

        # print("cam2ego",cam2ego,cam2ego.shape)
        # print("intrins",intrins,intrins.shape)

        bev_feat, depth_feat = self.view_transformer(
            [
                img_feats_dict["img_feats"],
                cam2ego,
                ego2global,
                intrins,
                post_rots,
                post_trans,
                bda_rot,
            ]
        )  # tuple(bev_feat: (B, C, Dy, Dx), depth: (B*N, D, fH, fW))

        for i in range(bev_feat[0].shape[0]):
            channel = bev_feat[0][i]
            img_data = channel.detach().cpu().numpy()
            img_max = np.max(img_data)
            img_min = np.min(img_data)
            img_data = (img_data - img_min) / (img_max - img_min)
            img_data = (img_data * 255).astype(np.uint8)
            cv2.imwrite(f"./temp/bev_{i}.jpg", img_data)

        for i in range(depth_feat[0].shape[0]):
            channel = depth_feat[0][i]
            img_data = channel.detach().cpu().numpy()
            img_max = np.max(img_data)
            img_min = np.min(img_data)
            img_data = (img_data - img_min) / (img_max - img_min)
            img_data = (img_data * 255).astype(np.uint8)
            cv2.imwrite(f"./temp/depth_{i}.jpg", img_data)

        x = self.head(bev_feat)

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
        )  # B, X, Y, Z, Cls (B,200,200,16,17)

        voxels_gt = self.multiscale_supervision(
            batch_inputs["occ_200"],
            [1, 1, 1],
            [len(batch_data_samples), 200, 200, 16],
        )  # torch.Size([B, 200, 200, 16])

        voxels_gt = torch.flatten(voxels_gt, start_dim=0)  # -1
        batch_outputs = batch_outputs.view(-1, batch_outputs.shape[4])  # -1,Cls

        # print(torch.unique(voxels_gt))

        # 过滤掉类别为255的行
        ignore_index = voxels_gt != 255
        voxels_gt = voxels_gt[ignore_index]
        batch_outputs = batch_outputs[ignore_index]

        # batch_outputs = batch_outputs.softmax(dim=-1)
        # print("compare",batch_outputs,voxels_gt)

        lossA = torch.nan_to_num(self.loss_cls(batch_outputs, voxels_gt))

        return {"loss": lossA * 100}

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

        # voxels_gt = torch.flatten(voxels_gt, start_dim=0)  # -1
        # batch_outputs = batch_outputs.view(-1, batch_outputs.shape[4])  # -1,Cls

        for i in range(len(batch_data_samples)):
            batch_data_samples[i].set_data(
                {
                    "pts_seg_logits": PointData(pts_seg_logits=voxel_logit[i]),
                    "pred_pts_seg": PointData(pts_semantic_mask=voxel_pred[i]),
                    # "pred_pts_seg": PointData(pts_semantic_mask=voxels_gt[i]), # 输入真值以检查EVAL计算
                }
            )
            # 设置评估用的 ground truth
            batch_data_samples[i].eval_ann_info["pts_semantic_mask"] = (
                voxels_gt[i].cpu().numpy().astype(np.uint8)
            )

        mask = voxels_gt != 0
        indices = torch.argmax(mask.long(), dim=-1, keepdim=True)
        selected_voxels_gt = torch.gather(voxels_gt, dim=-1, index=indices).squeeze(-1)
        selected_voxels_gt = selected_voxels_gt.cpu().detach().numpy()
        selected_voxels_gt = (selected_voxels_gt / np.max(selected_voxels_gt) * 255).astype(np.uint8)
        print(selected_voxels_gt.shape)

        cv2.imwrite(
            f"./temp/vis.jpg",
            selected_voxels_gt[0]
        )

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

        # print(gt_occ[0].shape, gt_occ[0].dtype, gt_occ[0].device)
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
        B, N, C, H, W = batch_inputs.size()  # 1 6 3 900 1600
        #print("raw_input", B, N, C, H, W)

        x = x.reshape(B * N, C, H, W)
        #print("input", x.shape)  # torch.Size([6, 3, 900, 1600])

        x = self.backbone(x)
        #print("backbone", len(x), x[0].shape)  # 1 torch.Size([6, 2048, 29, 50])

        x = self.neck(x)
        #print("neck", len(x), x[0].shape)  # 6 torch.Size([512, 29, 50])

        x = torch.stack(x, dim=0)

        return {
            "img_feats": x,
        }
