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

palette = [
    [0, 0, 0],  # noise                                         0
    [255, 120, 50],  # barrier              orange              1
    [255, 192, 203],  # bicycle              pink               2
    [255, 255, 0],  # bus                  yellow               3
    [0, 150, 245],  # car                  blue                 4
    [0, 255, 255],  # construction_vehicle cyan                 5
    [255, 127, 0],  # motorcycle           dark orange          6
    [255, 0, 0],  # pedestrian           red                    7
    [255, 240, 150],  # traffic_cone         light yellow       8
    [135, 60, 0],  # trailer              brown                 9
    [160, 32, 240],  # truck                purple              10
    [255, 0, 255],  # driveable_surface    dark pink            11
    [139, 137, 137],  # other_flat           dark red           12
    [75, 0, 75],  # sidewalk             dard purple            13
    [150, 240, 80],  # terrain              light green         14
    [230, 230, 250],  # manmade              white              15
    [0, 175, 0],  # vegetation           green                  16
    [200,200,200],# 未被任何事物占据的体素                        17
]


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
        self.mixter = MODELS.build(mixter)
        self.head = MODELS.build(head)
        self.loss_cls = LovaszLoss(
            loss_type="multi_class", per_sample=False, reduction="none"
        )
        # self.loss_cls = FocalLoss(gamma=2,task_type='multi-class',num_classes=17)
        # self.loss_depth = torch.nn.CrossEntropyLoss()

    def _forward(
        self, batch_inputs: dict, batch_data_samples: OptSampleList = None
    ) -> torch.Tensor:
        # batch_inputs: dict_keys(['img', 'voxel_semantics', 'intrinsics', 'extrinsic', 'cam2ego', 'ego2global'])
        #   img: list[torch.Size([6, 3, 900, 1600]),...]
        # batch_data_samples: dict_keys(['gt_pts_seg', 'gt_instances_3d', 'gt_instances', 'eval_ann_info'])

        device = batch_inputs["img"][0].device

        img_feats_dict = self.extract_feat(torch.stack(batch_inputs["img"], dim=0))

        # print("img_feats_dict",img_feats_dict["img_feats"].shape) # torch.Size([1, 6, 512, 29, 50])

        x = self.head(img_feats_dict["img_feats"])

        return x

    def loss(
        self, batch_inputs: dict, batch_data_samples: SampleList
    ) -> Dict[str, torch.Tensor]:
        """Calculate losses from a batch of inputs and data samples."""
        # batch_inputs: dict_keys(['img', 'voxel_semantics', 'intrinsics', 'extrinsic', 'cam2ego', 'ego2global'])
        #   img: list[torch.Size([6, 3, 900, 1600]),...]
        # batch_data_samples: dict_keys(['gt_pts_seg', 'gt_instances_3d', 'gt_instances', 'eval_ann_info'])

        batch_outputs = self._forward(
            batch_inputs, batch_data_samples
        )  # B, X, Y, Z, Cls (B,200,200,16,17)

        # voxels_gt = self.multiscale_supervision(
        #     batch_inputs["occ_200"],
        #     [1, 1, 1],
        #     [len(batch_data_samples), 200, 200, 16],
        # )  # torch.Size([B, 200, 200, 16])

        voxels_gt = torch.stack(batch_inputs["voxel_semantics"],dim=0)

        # 真值鸟瞰图可视化
        mask = voxels_gt != 17
        indices = torch.argmax(mask.long(), dim=-1, keepdim=True)
        selected_voxels_gt = torch.gather(voxels_gt, dim=-1, index=indices).squeeze(-1)
        selected_voxels_gt = selected_voxels_gt.cpu().detach().numpy()
        #selected_voxels_gt[selected_voxels_gt == 17] = 0 
        
        palette_np = np.array(palette, dtype=np.uint8)
        selected_voxels_gt = palette_np[selected_voxels_gt]

        cv2.imwrite(f"./temp/vis_gt.jpg", selected_voxels_gt[0])

        # 预测鸟瞰图可视化
        voxel_pred = batch_outputs.argmax(dim=-1)
        mask = voxel_pred != 17
        indices = torch.argmax(mask.long(), dim=-1, keepdim=True)
        selected_voxels_pred = torch.gather(voxel_pred, dim=-1, index=indices).squeeze(-1)
        selected_voxels_pred = selected_voxels_pred.cpu().detach().numpy()
        #selected_voxels_pred[selected_voxels_pred == 17] = 0 
        
        palette_np = np.array(palette, dtype=np.uint8)
        selected_voxels_pred = palette_np[selected_voxels_pred]

        cv2.imwrite(f"./temp/vis_pred.jpg", selected_voxels_pred[0])

        voxels_gt = torch.flatten(voxels_gt, start_dim=0)  # -1
        batch_outputs = batch_outputs.view(-1, batch_outputs.shape[4])  # -1,Cls

        # 过滤掉类别为17的行
        #ignore_index = voxels_gt != 17
        #voxels_gt = voxels_gt[ignore_index]
        #batch_outputs = batch_outputs[ignore_index]

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

        # voxels_gt = self.multiscale_supervision(
        #     batch_inputs["occ_200"],
        #     [1, 1, 1],
        #     [len(batch_data_samples), 200, 200, 16],
        # )  # torch.Size([B, 200, 200, 16])
        voxels_gt = torch.stack(batch_inputs["voxel_semantics"],dim=0)

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

        # 真值鸟瞰图可视化
        mask = voxels_gt != 17
        indices = torch.argmax(mask.long(), dim=-1, keepdim=True)
        selected_voxels_gt = torch.gather(voxels_gt, dim=-1, index=indices).squeeze(-1)
        selected_voxels_gt = selected_voxels_gt.cpu().detach().numpy()
        #selected_voxels_gt[selected_voxels_gt == 17] = 0 
        
        palette_np = np.array(palette, dtype=np.uint8)
        selected_voxels_gt = palette_np[selected_voxels_gt]

        cv2.imwrite(f"./temp/vis_gt.jpg", selected_voxels_gt[0])

        # 预测鸟瞰图可视化
        mask = voxel_pred != 17
        indices = torch.argmax(mask.long(), dim=-1, keepdim=True)
        selected_voxels_pred = torch.gather(voxel_pred, dim=-1, index=indices).squeeze(-1)
        selected_voxels_pred = selected_voxels_pred.cpu().detach().numpy()
        #selected_voxels_pred[selected_voxels_pred == 17] = 0 
        
        palette_np = np.array(palette, dtype=np.uint8)
        selected_voxels_pred = palette_np[selected_voxels_pred]

        cv2.imwrite(f"./temp/vis_pred.jpg", selected_voxels_pred[0])

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
        print("raw_input", B, N, C, H, W)

        x = x.reshape(B * N, C, H, W)
        print("input", x.shape)  # torch.Size([6, 3, 900, 1600])

        x = self.backbone(x)
        print("backbone", len(x), x[0].shape)  # 1 torch.Size([6, 2048, 29, 50])

        x = self.neck(x)
        print("neck", len(x), x[0].shape)  # 6 torch.Size([512, 29, 50])

        x = torch.stack(x, dim=0)

        return {
            "img_feats": x,
        }

def prepare_inputs(inputs):
    # split the inputs into each frame
    assert len(inputs) == 7
    B, N, C, H, W = inputs[0].shape
    imgs, sensor2egos, ego2globals, intrins, post_rots, post_trans, bda = \
        inputs

    sensor2egos = sensor2egos.view(B, N, 4, 4)
    ego2globals = ego2globals.view(B, N, 4, 4)

    # calculate the transformation from adj sensor to key ego
    keyego2global = ego2globals[:, 0,  ...].unsqueeze(1)    # (B, 1, 4, 4)
    global2keyego = torch.inverse(keyego2global.double())   # (B, 1, 4, 4)
    sensor2keyegos = \
        global2keyego @ ego2globals.double() @ sensor2egos.double()     # (B, N_views, 4, 4)
    sensor2keyegos = sensor2keyegos.float()

    return [imgs, sensor2keyegos, ego2globals, intrins,
            post_rots, post_trans, bda]