_base_ = ["../_base_/default_runtime.py"]

custom_imports = dict(
    imports=[
        "flashocc2",
        "swanlab.integration.mmengine",
    ],
    allow_failed_imports=False,
)

# load_from = 'ckpt/r101_dcn_fcos3d_pretrain.pth'

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
            "x": [-40, 40, 0.4],
            "y": [-40, 40, 0.4],
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
        strides=[1, 1, 1, 1],
        out_indices=(3,),
        in_channels=64,
        style="pytorch",
        deep_stem=False,
        dilations=(1, 1, 2, 4),
        # init_cfg=dict(type="Pretrained", checkpoint="torchvision://resnet50"),
    ),
    head=dict(
        type="FlashOcc2Head",
        channels=0,
        num_classes=17,
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
    dict(  # 载入多视角图像
        type="LoadMultiViewImageFromFiles",
        to_float32=True,
        num_views=6,
    ),
    dict(  # 载入3D标注
        type="LoadAnnotations3D",
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False,
    ),
    dict(type="LoadOccGTFromFile"),  # 载入OCC真值
    dict(
        type="CustomPack3DDetInputs",  # 打包keys,基本是keys的重新分配与过滤
        keys=[
            "img",
            "gt_bboxes_3d",  # must include
            "gt_labels_3d",  # # must include
            "voxel_semantics",
            "mask_camera",
            "intrinsics",
            "extrinsic",
            "cam2ego",
            "ego2global",
        ],
    ),
]

val_pipeline = [
    dict(  # 载入多视角图像
        type="LoadMultiViewImageFromFiles",
        to_float32=True,
        num_views=6,
    ),
    dict(  # 载入3D标注
        type="LoadAnnotations3D",
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False,
    ),
    dict(type="LoadOccGTFromFile"),  # 载入OCC真值
    dict(
        type="CustomPack3DDetInputs",  # 打包keys,基本是keys的重新分配与过滤
        keys=[
            "img",
            "gt_bboxes_3d",  # must include
            "gt_labels_3d",  # # must include
            "voxel_semantics",
            "mask_camera",
            "intrinsics",
            "extrinsic",
            "cam2ego",
            "ego2global",
        ],
    ),
]

test_pipeline = val_pipeline

dataset_type = "NuScenesDatasetOccupancy"
data_root = "data/nuscenes"
data_prefix = dict(
    CAM_FRONT="samples/CAM_FRONT",
    CAM_FRONT_LEFT="samples/CAM_FRONT_LEFT",
    CAM_FRONT_RIGHT="samples/CAM_FRONT_RIGHT",
    CAM_BACK="samples/CAM_BACK",
    CAM_BACK_RIGHT="samples/CAM_BACK_RIGHT",
    CAM_BACK_LEFT="samples/CAM_BACK_LEFT",
)
modality = dict(
    use_camera=True,
    use_lidar=False,
)

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
        modality=modality,
        # indices=100,  # 测试用,mini
        ann_file="bevdetv2-nuscenes_infos_train.pkl",
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
        modality=modality,
        # indices=100,  # 测试用,mini
        ann_file="bevdetv2-nuscenes_infos_val.pkl",
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
