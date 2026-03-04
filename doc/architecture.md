# FlashOCC 项目架构

## 概述

FlashOCC 是一个基于 BEV (鸟瞰图) 的 3D 占用预测模型，
使用 Channel-to-Height 策略实现快速高效的占用预测。

项目已从 mmdetection3d 框架完全独立，不依赖任何 mmlab 包。

## 包结构

```
src/flashocc/
├── __init__.py           # 包入口, 版本号
├── constants.py          # 全局常量 (类别名、网格参数、ImageNet统计量等)
├── config/               # 纯 Python 配置系统
│   ├── lazy.py           # Lazy 描述符、Experiment 数据类
│   └── types.py          # DataConfig, GridConfig, BDAAugConfig
├── core/                 # 核心基础设施
│   ├── base_module.py    # BaseModule (支持 init_cfg 的 nn.Module 基类)
│   ├── checkpoint.py     # load_checkpoint / save_checkpoint
│   ├── dist.py           # init_dist / get_dist_info / master_only
│   ├── env.py            # collect_env / setup_multi_processes
│   ├── fp16.py           # force_fp32 / wrap_fp16_model
│   ├── functional.py     # multi_apply / reduce_mean
│   ├── hooks.py          # HOOKS 注册表 + Hook 基类
│   ├── io.py             # load / dump (json/pickle/yaml)
│   ├── nn.py             # ConvModule / build_conv_layer / build_norm_layer / 初始化
│   ├── registry.py       # Registry 类
│   ├── bbox/             # 3D 边界框 & 点类
│   │   ├── bbox.py       # BaseInstance3DBoxes, LiDARInstance3DBoxes
│   │   └── points.py     # BasePoints, LiDARPoints, get_points_type
│   └── ops/              # 自定义算子
│       ├── _ext.py       # JIT CUDA 扩展加载
│       ├── bev_pool_v2.py# BEV 池化 v2 CUDA 内核
│       ├── bev_pool_v3.py# BEV 池化 v3 CUDA 内核 (高性能)
│       ├── bev_pool_v3_triton.py  # BEV 池化 v3 Triton 后端
│       └── voxel_pooling_prepare_v3.py  # 融合 voxel prepare
├── datasets/             # 数据加载与评估
│   ├── base_dataset.py   # Custom3DDataset 抽象基类
│   ├── builder.py        # DATASETS / PIPELINES 注册表, build_dataset
│   ├── nuscenes_bevdet.py# NuScenesDatasetBEVDet (检测评估格式)
│   ├── nuscenes_occ.py   # NuScenesOccDataset (占用评估, mIoU/RayIoU)
│   ├── dali_decode.py    # DALI GPU 图像解码
│   ├── pipelines/        # 数据变换管线
│   │   ├── base.py       # LoadMultiViewImageFromFiles, GlobalRotScaleTrans 等
│   │   ├── loading.py    # BEVDet 专用加载 (PrepareImageInputs, LoadOccGTFromFile)
│   │   ├── formatting.py # DefaultFormatBundle3D, Collect3D
│   │   ├── compose.py    # Compose (管线组合器)
│   │   └── _compat_pipelines.py  # LoadPointsFromFile, MultiScaleFlipAug3D
│   └── evaluation/       # 评估指标
│       ├── occ_metrics.py# Metric_mIoU, Metric_FScore
│       ├── ray_metrics.py# Metric_RayIoU
│       └── ray_pq.py     # RayPanopticMetric
├── engine/               # 训练 / 测试引擎
│   ├── trainer.py        # train_model, build_dataloader
│   ├── tester.py         # single_gpu_test
│   ├── inference.py      # init_model
│   ├── seed.py           # init_random_seed, set_random_seed
│   ├── parallel.py       # DataContainer / DataParallel
│   └── hooks/            # 训练钩子
│       ├── ema.py        # MEGVIIEMAHook
│       ├── sequential_control.py  # SequentialControlHook
│       └── syncbn_control.py      # SyncBNControlHook
└── models/               # 模型定义
    ├── __init__.py       # MODELS/DETECTORS/BACKBONES/NECKS/HEADS/LOSSES 注册表
    ├── backbones/        # ResNet (torchvision) + CustomResNet
    ├── necks/            # FPN, LSSFPN, LSSViewTransformer (含 DepthNet)
    ├── heads/            # OccHead (Channel-to-Height + 语义分类)
    ├── detectors/        # MVXTwoStageDetector → BEVDet → BEVDetOCC
    └── losses/           # CrossEntropyLoss, lovász loss 等
```

## 配置系统

采用纯 Python 配置，核心组件：

- **`Lazy(cls, **kwargs)`**: 延迟构造描述符。记录类和参数，调用 `.build()` 时才实例化。
- **`Experiment`**: 数据类，包含 model、data、optimizer 等全部实验设置。
- **`DataConfig` / `GridConfig` / `BDAAugConfig`**: 类型化的配置数据类。

```python
# configs/flashocc_r50.py
experiment = Experiment(
    model=Lazy(BEVDetOCC,
        img_backbone=Lazy(ResNet, depth=50, ...),
        img_neck=Lazy(CustomFPN, ...),
        ...
    ),
    train_data=Lazy(NuScenesDatasetOccpancy, ...),
    val_data=Lazy(NuScenesDatasetOccpancy, ...),
    samples_per_gpu=8,
    workers_per_gpu=16,
    ...
)
```

优势：
- 所有组件通过 Python `import` 直接引用 — IDE 可跳转/重构/类型检查
- 无 YAML 解析、无字符串 `type` 查找
- 配置即代码，可自由使用 Python 表达式

## 注册表系统

注册表仍用于运行时按名称查找组件（数据管线等场景），但配置层面推荐直接 import：

```python
from flashocc.models import DETECTORS

@DETECTORS.register_module()
class BEVDetOCC(BEVDet):
    ...
```

当前注册表：`MODELS`, `DETECTORS`, `BACKBONES`, `NECKS`, `HEADS`, `LOSSES`, `DATASETS`, `PIPELINES`, `HOOKS`

## 模型继承链

```
BaseModule (nn.Module + init_cfg)
  └── MVXTwoStageDetector
        └── BEVDet (image → BEV → features)
              └── BEVDetOCC (+ OccHead → 3D occupancy grid)
```

## 数据流

```
6 路相机图像
    │
    ▼
img_backbone (ResNet-50)  →  多尺度特征 [C3, C4, C5]
    │
    ▼
img_neck (FPN)  →  融合特征
    │
    ▼
LSSViewTransformer (含 DepthNet)  →  BEV 特征 (200×200)  ← 显式深度估计
    │
    ▼
img_bev_encoder (ResNet blocks)  →  增强 BEV 特征
    │
    ▼
OccHead (Channel-to-Height)  →  3D 占用网格 (200×200×16, 18 类)
```

### Sky Freespace 辅助监督 (可选)

利用 SAM3 离线生成的天空掩码, 在 3D 体素空间施加 "已知空闲" 监督:

```
SAM3 天空掩码 (2D, per camera)
    │
    ▼
沿视锥射线投影到 3D 体素 ── get_ego_coor() 将 (fH, fW) × D 个采样点映射到 ego 坐标
    │
    ▼
天空射线穿过的 3D 体素 → 目标类别 = Free/Empty (class 17)
    │
    ▼
辅助 Cross-Entropy Loss (loss_sky_freespace) ← 仅训练时, 不影响推理
```

关键设计: **不修改任何 2D 特征流** — 避免硬门控 (Hard Gating) 导致的:
- 射线状伪影 (特征截断产生高频边界)
- 梯度阻断 (天空区域 OCC 梯度消失)
- 假阳性误杀 (SAM3 误判直接消灭目标特征)

详见 [`doc/sky_freespace.md`](sky_freespace.md)。

## 常量定义

全局常量集中在 `constants.py`：

| 常量 | 值 | 说明 |
|------|----|------|
| `OCC_CLASS_NAMES` | 18 个类别名 | NuScenes 占用语义类别 |
| `NUM_OCC_CLASSES` | 18 | 类别总数 |
| `POINT_CLOUD_RANGE` | [-40, -40, -1, 40, 40, 5.4] | 点云范围 (米) |
| `VOXEL_SIZE` | 0.4 | 体素边长 (米) |
| `OCC_GRID_SHAPE` | (200, 200, 16) | 占用网格形状 |
| `IMAGENET_MEAN/STD` | 标准值 | 图像归一化参数 |
