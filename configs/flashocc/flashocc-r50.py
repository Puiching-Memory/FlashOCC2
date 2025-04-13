_base_ = ["../_base_/datasets/nus-3d.py", "../_base_/default_runtime.py"]

custom_imports = dict(
    imports=[
        "flashocc2",
    ],
    allow_failed_imports=False,
)

# https://mmdetection3d.readthedocs.io/zh-cn/latest/advanced_guides/datasets/nuscenes.html

point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
class_names = [
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
]

# data_config = {
#     "cams": [
#         "CAM_FRONT_LEFT",
#         "CAM_FRONT",
#         "CAM_FRONT_RIGHT",
#         "CAM_BACK_LEFT",
#         "CAM_BACK",
#         "CAM_BACK_RIGHT",
#     ],
#     "Ncams": 6,
#     "input_size": (256, 704),
#     "src_size": (900, 1600),
#     # Augmentation
#     "resize": (-0.06, 0.11),
#     "rot": (-5.4, 5.4),
#     "flip": True,
#     "crop_h": (0.0, 0.0),
#     "resize_test": 0.00,
# }

grid_config = {
    "x": [-40, 40, 0.4],
    "y": [-40, 40, 0.4],
    "z": [-1, 5.4, 6.4],
    "depth": [1.0, 45.0, 0.5],
}

numC_Trans = 64

model = dict(
    type="BEVDetOCC",
    img_backbone=dict(
        type="mmdet.ResNet",
        depth=50,
        num_stages=4,
        out_indices=(2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type="BN", requires_grad=True),
        norm_eval=False,
        with_cp=True,
        style="pytorch",
        pretrained="torchvision://resnet50",
    ),
    img_neck=dict(
        type="CustomFPN",
        in_channels=[1024, 2048],
        out_channels=256,
        num_outs=1,
        start_level=0,
        out_ids=[0],
    ),
    img_view_transformer=dict(
        type="LSSViewTransformer",
        grid_config=grid_config,
        input_size=(256, 704),
        in_channels=256,
        out_channels=numC_Trans,
        sid=False,
        collapse_z=True,
        downsample=16,
    ),
    img_bev_encoder_backbone=dict(
        type="CustomResNet",
        numC_input=numC_Trans,
        num_channels=[numC_Trans * 2, numC_Trans * 4, numC_Trans * 8],
    ),
    img_bev_encoder_neck=dict(
        type="FPN_LSS", in_channels=numC_Trans * 8 + numC_Trans * 2, out_channels=256
    ),
    occ_head=dict(
        type="BEVOCCHead2D",
        in_dim=256,
        out_dim=256,  # out_dim=128 for M0!!!
        Dz=16,
        use_mask=True,
        num_classes=18,
        use_predicter=True,
        class_balance=False,
        loss_occ=dict(
            type="CrossEntropyLoss",
            use_sigmoid=False,
            ignore_index=255,
            loss_weight=1.0,
        ),
    ),
)

train_pipeline = [
    dict(
        type="LoadMultiViewImageFromFiles",
        to_float32=True,
        num_views=6,
    ),
    dict(
        type="LoadAnnotations3D",
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False,
    ),
    dict(type="ObjectNameFilter", classes=class_names),
]

data_prefix = dict(
    CAM_FRONT="samples/CAM_FRONT",
    CAM_FRONT_LEFT="samples/CAM_FRONT_LEFT",
    CAM_FRONT_RIGHT="samples/CAM_FRONT_RIGHT",
    CAM_BACK="samples/CAM_BACK",
    CAM_BACK_RIGHT="samples/CAM_BACK_RIGHT",
    CAM_BACK_LEFT="samples/CAM_BACK_LEFT",
)
train_dataloader = dict(
    batch_size=2,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type="NuScenesDataset",
        data_root="./data/nuScenes",
        ann_file="nuscenes_infos_train.pkl",
        data_prefix=data_prefix,
        modality=dict(
            use_camera=True,
            use_lidar=False,
        ),
        pipeline=train_pipeline,
        test_mode=False,
    ),
)


# with det pretrain; use_mask=True;
# ===> per class IoU of 6019 samples:
# ===> others - IoU = 6.74
# ===> barrier - IoU = 37.65
# ===> bicycle - IoU = 10.26
# ===> bus - IoU = 39.55
# ===> car - IoU = 44.36
# ===> construction_vehicle - IoU = 14.88
# ===> motorcycle - IoU = 13.4
# ===> pedestrian - IoU = 15.79
# ===> traffic_cone - IoU = 15.38
# ===> trailer - IoU = 27.44
# ===> truck - IoU = 31.73
# ===> driveable_surface - IoU = 78.82
# ===> other_flat - IoU = 37.98
# ===> sidewalk - IoU = 48.7
# ===> terrain - IoU = 52.5
# ===> manmade - IoU = 37.89
# ===> vegetation - IoU = 32.24
# ===> mIoU of 6019 samples: 32.08

# with det pretrain; use_mask=False; class_balance=True
# ===> per class IoU of 6019 samples:
# ===> others - IoU = 4.49
# ===> barrier - IoU = 29.59
# ===> bicycle - IoU = 7.38
# ===> bus - IoU = 30.32
# ===> car - IoU = 32.22
# ===> construction_vehicle - IoU = 13.04
# ===> motorcycle - IoU = 11.91
# ===> pedestrian - IoU = 8.61
# ===> traffic_cone - IoU = 8.11
# ===> trailer - IoU = 7.66
# ===> truck - IoU = 20.84
# ===> driveable_surface - IoU = 48.59
# ===> other_flat - IoU = 26.62
# ===> sidewalk - IoU = 26.08
# ===> terrain - IoU = 20.86
# ===> manmade - IoU = 7.62
# ===> vegetation - IoU = 7.14
# ===> mIoU of 6019 samples: 18.3
