"""FlashOCC-R50 — 组合式 Python 配置.

BEVDetOCC + ResNet50 占用预测模型完整实验描述。
 import + Lazy 引用, 拼写错误即 ImportError, 参数错误即 TypeError。
 所有场景参数用 dataclass 表达, 零 dict, IDE 全程补全。
"""
from __future__ import annotations

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR

from flashocc.config import Lazy, Experiment, DataConfig, GridConfig, BDAAugConfig

# ------ 模型组件 (import 即安全) ------
from flashocc.models.backbones._resnet_base import ResNet
from flashocc.models.backbones.resnet import CustomResNet
from flashocc.models.necks.fpn import CustomFPN
from flashocc.models.necks.lss_fpn import FPN_LSS
from flashocc.models.necks.view_transformer import LSSViewTransformer
from flashocc.models.heads.occ_head import BEVOCCHead2D
from flashocc.models.losses.cross_entropy import CrossEntropyLoss
from flashocc.models.detectors.bevdet_occ import BEVDetOCC

# ------ 数据组件 ------
from flashocc.datasets.nuscenes_occ import NuScenesDatasetOccpancy
from flashocc.datasets.pipelines.loading import (
    PrepareImageInputs,
    LoadAnnotationsBEVDepth,
    LoadOccGTFromFile,
    PointToMultiViewDepth,
)
from flashocc.datasets.pipelines._compat_pipelines import LoadPointsFromFile
from flashocc.datasets.pipelines.formatting import DefaultFormatBundle3D, Collect3D
from flashocc.datasets.pipelines._compat_pipelines import MultiScaleFlipAug3D

# =====================================================================
#  场景参数 — dataclass, IDE 全程补全, 拼写错误即 TypeError
# =====================================================================

class_names = [
    "car", "truck", "construction_vehicle", "bus", "trailer",
    "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone",
]

numC_Trans = 64

data_config = DataConfig(
    cams=["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
          "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"],
    Ncams=6,
    input_size=(256, 704),
    src_size=(900, 1600),
    resize=(-0.06, 0.11),
    rot=(-5.4, 5.4),
    flip=True,
    crop_h=(0.0, 0.0),
    resize_test=0.00,
)

grid_config = GridConfig(
    x=(-40, 40, 0.4),
    y=(-40, 40, 0.4),
    z=(-1, 5.4, 6.4),
    depth=(1.0, 45.0, 0.5),
)

bda_aug_conf = BDAAugConfig(
    rot_lim=(0.0, 0.0),
    scale_lim=(1.0, 1.0),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5,
)

# =====================================================================
#  模型 — 组合式构建
# =====================================================================

model = Lazy(BEVDetOCC,
    img_backbone=Lazy(ResNet,
        depth=50,
        num_stages=4,
        out_indices=[2, 3],
        frozen_stages=-1,
        norm_eval=False,
        with_cp=True,
        style="pytorch",
        pretrained="torchvision://resnet50",
    ),
    img_neck=Lazy(CustomFPN,
        in_channels=[1024, 2048],
        out_channels=256,
        num_outs=1,
        start_level=0,
        out_ids=[0],
    ),
    img_view_transformer=Lazy(LSSViewTransformer,
        grid_config=grid_config,
        input_size=[256, 704],
        in_channels=256,
        out_channels=numC_Trans,
        sid=False,
        collapse_z=True,
        downsample=16,
    ),
    img_bev_encoder_backbone=Lazy(CustomResNet,
        numC_input=numC_Trans,
        num_channels=[numC_Trans * 2, numC_Trans * 4, numC_Trans * 8],
    ),
    img_bev_encoder_neck=Lazy(FPN_LSS,
        in_channels=numC_Trans * 8 + numC_Trans * 2,
        out_channels=256,
    ),
    occ_head=Lazy(BEVOCCHead2D,
        in_dim=256,
        out_dim=256,
        Dz=16,
        use_mask=True,
        num_classes=18,
        use_predicter=True,
        class_balance=False,
        loss_occ=Lazy(CrossEntropyLoss,
            use_sigmoid=False,
            ignore_index=255,
            loss_weight=1.0,
        ),
    ),
)

# =====================================================================
#  数据管线 — Lazy 包装每个 transform
# =====================================================================

train_pipeline = [
    Lazy(PrepareImageInputs,
         is_train=True, data_config=data_config, sequential=False),
    Lazy(LoadAnnotationsBEVDepth,
         bda_aug_conf=bda_aug_conf, classes=class_names, is_train=True),
    Lazy(LoadOccGTFromFile),
    Lazy(LoadPointsFromFile,
         coord_type="LIDAR", load_dim=5, use_dim=5),
    Lazy(PointToMultiViewDepth,
         downsample=1, grid_config=grid_config),
    Lazy(DefaultFormatBundle3D, class_names=class_names),
    Lazy(Collect3D,
         keys=["img_inputs", "gt_depth", "voxel_semantics",
               "mask_lidar", "mask_camera"]),
]

test_pipeline = [
    Lazy(PrepareImageInputs,
         data_config=data_config, sequential=False),
    Lazy(LoadAnnotationsBEVDepth,
         bda_aug_conf=bda_aug_conf, classes=class_names, is_train=False),
    Lazy(LoadPointsFromFile,
         coord_type="LIDAR", load_dim=5, use_dim=5),
    Lazy(MultiScaleFlipAug3D,
         img_scale=[1333, 800], pts_scale_ratio=1, flip=False,
         transforms=[
             Lazy(DefaultFormatBundle3D,
                  class_names=class_names, with_label=False),
             Lazy(Collect3D, keys=["points", "img_inputs"]),
         ]),
]

# =====================================================================
#  数据集
# =====================================================================

_DATA_ROOT = "data/nuscenes/"

train_data = Lazy(NuScenesDatasetOccpancy,
    data_root=_DATA_ROOT,
    ann_file=_DATA_ROOT + "bevdetv2-nuscenes_infos_train.pkl",
    classes=class_names,
    modality=dict(use_camera=True, use_lidar=True, use_radar=False, use_map=False, use_external=False),
    pipeline=train_pipeline,
    test_mode=False,
    use_valid_flag=True,
    box_type_3d="LiDAR",
    stereo=False,
    filter_empty_gt=False,
    img_info_prototype="bevdet",
)

val_data = Lazy(NuScenesDatasetOccpancy,
    data_root=_DATA_ROOT,
    ann_file=_DATA_ROOT + "bevdetv2-nuscenes_infos_val.pkl",
    classes=class_names,
    modality=dict(use_camera=True, use_lidar=True, use_radar=False, use_map=False, use_external=False),
    pipeline=test_pipeline,
    stereo=False,
    filter_empty_gt=False,
    img_info_prototype="bevdet",
)

# =====================================================================
#  实验 — 单一对象, 完整描述
# =====================================================================

experiment = Experiment(
    model=model,
    train_data=train_data,
    val_data=val_data,

    samples_per_gpu=4,
    workers_per_gpu=4,

    optimizer=Lazy(AdamW, lr=1e-4, weight_decay=1e-2),
    lr_scheduler=Lazy(MultiStepLR, milestones=[24], gamma=0.1),
    warmup_iters=200,
    warmup_ratio=0.001,
    grad_max_norm=5.0,

    max_epochs=24,
    load_from="ckpts/bevdet-r50-cbgs.pth",

    checkpoint_interval=1,
    max_keep_ckpts=5,
    eval_interval=1,
    eval_start=20,

    log_interval=50,
    seed=0,
)
