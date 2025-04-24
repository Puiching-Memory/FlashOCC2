_base_ = ["../_base_/default_runtime.py"]  # "../_base_/datasets/nus-3d.py",

custom_imports = dict(
    imports=[
        "flashocc2",
        "swanlab.integration.mmengine",
    ],
    allow_failed_imports=False,
)

# load_from = 'ckpt/r101_dcn_fcos3d_pretrain.pth'

dataset_type = "NuScenesSegDataset"
data_root = "data/nuscenes"
data_prefix = dict(
    pts="samples/LIDAR_TOP",
    pts_semantic_mask="lidarseg/v1.0-trainval",
    CAM_FRONT="samples/CAM_FRONT",
    CAM_FRONT_LEFT="samples/CAM_FRONT_LEFT",
    CAM_FRONT_RIGHT="samples/CAM_FRONT_RIGHT",
    CAM_BACK="samples/CAM_BACK",
    CAM_BACK_RIGHT="samples/CAM_BACK_RIGHT",
    CAM_BACK_LEFT="samples/CAM_BACK_LEFT",
)

input_modality = dict(use_lidar=False, use_camera=True)
backend_args = None

point_cloud_range = [-50.0, -50.0, -5.0, 50.0, 50.0, 3.0]
grid_size_vt = [100, 100, 8]
num_points_per_voxel = 35
nbr_class = 17
use_lidar = False
use_radar = False
use_occ3d = False

model = dict(
    type="Flashocc2Orchestrator",
    backbone=dict(
        type="mmdet.ResNet",
        depth=50,
        num_stages=4,
        out_indices=(3,),
        frozen_stages=-1,
        norm_cfg=dict(type="BN", requires_grad=True),
        norm_eval=False,
        with_cp=False,
        style="pytorch",
        init_cfg=dict(type="Pretrained", checkpoint="torchvision://resnet50"),
    ),
    neck=dict(
        type="mmdet.DilatedEncoder",
        in_channels=2048,
        out_channels=512,
        block_mid_channels=128,
        num_residual_blocks=4,
        block_dilations=[1, 2, 4, 6],
    ),
    view_transformer=dict(
        type="LSSViewTransformer",
        grid_config={
            "x": [-50, 50, 0.5],
            "y": [-50, 50, 0.5],
            "z": [-1, 5.4, 6.4],
            "depth": [1.0, 45.0, 0.5],
        },
        input_size=(900, 1600),
        downsample=32,
        in_channels=512,
        out_channels=64,
    ),
    mixter=dict(
        type="mmdet.ResNet",
        depth=50,
        strides=[1,1,1,1],
        out_indices=(3,),
        in_channels=64,
        style="pytorch",
        deep_stem=False,
        dilations=(1, 1, 2, 4),
        #init_cfg=dict(type="Pretrained", checkpoint="torchvision://resnet50"),
    ),
    head=dict(
        type="FlashOcc2Head",
        channels=0,
        num_classes=nbr_class,
    ),
)

train_transforms = [
    dict(type="PhotoMetricDistortion3D"),
    # dict(
    #     type="AffineResize",
    #     img_scale=(256, 704),
    #     down_ratio=32,
    # ),
]

train_pipeline = [
    dict(
        type="BEVLoadMultiViewImageFromFiles",
        to_float32=False,
        color_type="unchanged",
        num_views=6,
        backend_args=backend_args,
    ),
    dict(type="LoadOccupancy"),
    dict(
        type="LoadAnnotations3D",
        with_bbox_3d=False,
        with_label_3d=False,
        with_seg_3d=True,
        with_attr_label=False,
        seg_3d_dtype="np.uint8",
    ),
    dict(type="MultiViewWrapper", transforms=train_transforms),
    dict(type="SegLabelMapping"),
    dict(
        type="Custom3DPack",
        keys=["img", "occ_200"],
        meta_keys=[
            "ego2img",
            "ego2global",
            # "cam2global",  # 不存在
            "cam2img",
            # "img_shape",  # 不存在
            # "ori_shape",  # 不存在
            "ori_cam2img",
            # "resize_img_shape",  # 不存在
            "lidar2cam",  # 不存在
            "lidar2img",
            "cam2ego",
        ],
    ),
]

val_pipeline = [
    dict(
        type="BEVLoadMultiViewImageFromFiles",
        to_float32=False,
        color_type="unchanged",
        num_views=6,
        backend_args=backend_args,
    ),
    dict(type="LoadOccupancy"),
    dict(
        type="LoadAnnotations3D",
        with_bbox_3d=False,
        with_label_3d=False,
        with_seg_3d=True,
        with_attr_label=False,
        seg_3d_dtype="np.uint8",
    ),
    dict(type="MultiViewWrapper", transforms=train_transforms),
    dict(type="SegLabelMapping"),
    dict(
        type="Custom3DPack",
        keys=["img", "occ_200"],
        meta_keys=[
            "ego2img",
            "ego2global",
            # "cam2global",  # 不存在
            "cam2img",
            # "img_shape",  # 不存在
            # "ori_shape",  # 不存在
            "ori_cam2img",
            # "resize_img_shape",  # 不存在
            "lidar2cam",  # 不存在
            "lidar2img",
            "cam2ego",
        ],
    ),
]

test_pipeline = val_pipeline

train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=data_prefix,
        #indices=100,  # 测试用,mini
        ann_file="nuscenes_infos_occfusion_train.pkl",
        pipeline=train_pipeline,
        test_mode=False,
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=data_prefix,
        #indices=100,  # 测试用,mini
        ann_file="nuscenes_infos_occfusion_val.pkl",
        pipeline=val_pipeline,
        test_mode=True,
    ),
)

test_dataloader = val_dataloader

val_evaluator = dict(
    type="EvalMetric",
    # collect_device="gpu", # 在大规模eval时容易爆显存
    collect_device="cpu",
)

test_evaluator = val_evaluator

vis_backends = [
    dict(
        type="SwanlabVisBackend",
        init_kwargs={  # swanlab.init 参数
            "project": "flashocc2",
            "experiment_name": "flashocc2-r50-LSS-B",  # 实验名称
            "description": "some NOTE here",  # 实验的描述信息
        },
    ),
]

visualizer = dict(
    type="Visualizer",
    vis_backends=vis_backends,
    name="visualizer",
)

optim_wrapper = dict(
    # type='OptimWrapper',
    type="AmpOptimWrapper",  # 启用AMP
    optimizer=dict(type="AdamW", lr=2e-4, weight_decay=0.01),
    # paramwise_cfg=dict(
    #     custom_keys={
    #         "backbone": dict(lr_mult=0.1),
    #     }
    # ),
    # clip_grad=dict(max_norm=35, norm_type=2),
)

param_scheduler = [
    dict(type="LinearLR", start_factor=1e-5, by_epoch=False, begin=0, end=500),
    dict(
        type="CosineAnnealingLR",
        begin=0,
        T_max=24,
        by_epoch=True,
        eta_min=1e-6,
        convert_to_iter_based=True,
    ),
]

train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=24, val_begin=1, val_interval=1)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

default_hooks = dict(
    checkpoint=dict(
        type="CheckpointHook", interval=1, max_keep_ckpts=5, save_optimizer=True
    ),
    logger=dict(type="LoggerHook", interval=50),
)
custom_hooks = [dict(type="EMAHook")]

compile = False
find_unused_parameters = False
