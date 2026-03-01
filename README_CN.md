# FlashOCC v2.0

基于 Channel-to-Height 插件的快速、高效 3D 占用预测。

基于 [FlashOCC](https://arxiv.org/abs/2311.12058)、[Panoptic-FlashOCC](https://arxiv.org/pdf/2406.10527) 和 [UltimateDO](https://arxiv.org/abs/2409.11160)。

## 特色

- 修复了BGR颜色通道问题 https://github.com/HuangJunJie2017/BEVDet/issues/274
- 新的BEV pool V3 支持 torch.compiled CUDA，显著提升性能
- DAIL数据管道

## 快速开始

```bash
# 安装
uv venv .venv --python 3.12 && source .venv/bin/activate
uv sync

# 准备数据（见 doc/nuscenes_data.md）
python tools/create_data_flashocc2.py

# 训练（单 GPU）
python tools/train.py configs/flashocc_r50.py

# 训练（多 GPU）
torchrun --nproc_per_node=4 tools/train.py configs/flashocc_r50.py

# 测试
python tools/test.py configs/flashocc_r50.py work_dirs/flashocc_r50/epoch_24.pth --eval occ

# 统计数据集类别分布
python tools/analyze_class_distribution.py data/flashocc2-nuscenes_infos_train.pkl --no-show
```

## 项目结构

```
FlashOCC/
├── configs/                  # Python 配置文件
│   └── flashocc_r50.py       # R50 单帧配置
├── src/flashocc/             # 源代码包
│   ├── constants.py          # 全局常量（类别名、网格参数等）
│   ├── config/               # 配置系统（Lazy 描述符、数据类）
│   ├── core/                 # 核心基础设施
│   │   ├── base_module.py    # BaseModule（支持 init_cfg）
│   │   ├── checkpoint.py     # 权重加载/保存
│   │   ├── dist.py           # 分布式工具
│   │   ├── fp16.py           # 混合精度（force_fp32, wrap_fp16_model）
│   │   ├── functional.py     # multi_apply, reduce_mean
│   │   ├── nn.py             # ConvModule, build_conv/norm_layer, 初始化
│   │   ├── registry.py       # 注册表模式
│   │   ├── bbox/             # 3D 边界框 & 点类
│   │   └── ops/              # CUDA 算子（bev_pool_v2, bev_pool_v3）
│   ├── datasets/             # 数据加载与评估
│   │   ├── base_dataset.py   # Custom3DDataset 基类
│   │   ├── nuscenes_occ.py   # NuScenesOccDataset（占用评估）
│   │   ├── nuscenes_bevdet.py# NuScenesDatasetBEVDet（检测评估）
│   │   ├── pipelines/        # 数据变换（加载、增强、格式化）
│   │   └── evaluation/       # 评估指标（mIoU、RayIoU、RayPQ）
│   ├── engine/               # 训练与推理
│   │   ├── trainer.py        # 训练循环
│   │   ├── tester.py         # 测试循环
│   │   ├── inference.py      # single_gpu_test
│   │   ├── seed.py           # 随机种子工具
│   │   ├── parallel.py       # DataParallel 封装
│   │   └── hooks/            # 训练钩子（EMA、SyncBN 等）
│   └── models/               # 模型定义
│       ├── backbones/        # ResNet, CustomResNet
│       ├── necks/            # FPN, LSSFPN, ViewTransformer
│       ├── heads/            # OccHead
│       ├── detectors/        # BEVDet → BEVDetOCC 流水线
│       └── losses/           # 交叉熵损失工具
├── tools/                    # 命令行脚本
│   ├── train.py              # 训练入口
│   ├── test.py               # 测试入口
│   ├── analyze_class_distribution.py  # 数据集类别分布分析
│   ├── dist_train.sh         # 多 GPU 训练脚本
│   ├── dist_test.sh          # 多 GPU 测试脚本
│   └── create_data_flashocc2.py # 数据准备
├── data/nuscenes/            # 数据集（不被 git 跟踪）
├── ckpts/                    # 预训练权重
└── pyproject.toml            # 依赖与构建配置
```

## 配置系统

配置文件为纯 Python 文件，使用 `Lazy` 描述符和数据类——无 YAML，无基于字典的注册表：

```python
# configs/flashocc_r50.py
from flashocc.config import Experiment, Lazy, DataConfig, GridConfig
from flashocc.models.backbones.resnet import ResNet
from flashocc.models.necks.fpn import FPN

experiment = Experiment(
    model=Lazy(BEVDetOCC,
        img_backbone=Lazy(ResNet, depth=50, ...),
        img_neck=Lazy(FPN, ...),
        ...
    ),
    data=DataConfig(...),
    ...
)
```

所有模型组件通过 Python `import` 直接引用——IDE 跳转、重构和类型检查开箱即用。

## 性能指标

| 模型                    | 骨干网络 | 输入尺寸 | mIoU  | 参数量  |
| ----------------------- | -------- | -------- | ----- | ------- |
| FlashOCC M1 (1f)        | R50      | 256×704  | 32.08 | 44.74M  |
| FlashOCC-4D-Stereo (2f) | R50      | 256×704  | 37.84 | -       |
| FlashOCC-4D-Stereo (2f) | Swin-B   | 512×1408 | 43.52 | 144.99M |

## 文档

- [安装指南](doc/install.md)
- [项目架构](doc/architecture.md)
- [NuScenes 数据准备](doc/nuscenes_data.md)
- [训练指南](doc/training.md)
- [测试指南](doc/testing.md)

## 引用

```bibtex
@article{yu2024ultimatedo,
  title={UltimateDO: An Efficient Framework to Marry Occupancy Prediction with 3D Object Detection via Channel2height},
  author={Yu, Zichen and Shu, Changyong},
  journal={arXiv preprint arXiv:2409.11160},
  year={2024}
}

@article{yu2024panoptic,
  title={Panoptic-FlashOcc: An Efficient Baseline to Marry Semantic Occupancy with Panoptic via Instance Center},
  author={Yu, Zichen and Shu, Changyong and Sun, Qianpu and Linghu, Junjie and Wei, Xiaobao and Yu, Jiangyong and Liu, Zongdai and Yang, Dawei and Li, Hui and Chen, Yan},
  journal={arXiv preprint arXiv:2406.10527},
  year={2024}
}

@article{yu2023flashocc,
  title={FlashOcc: Fast and Memory-Efficient Occupancy Prediction via Channel-to-Height Plugin},
  author={Zichen Yu and Changyong Shu and Jiajun Deng and Kangjie Lu and Zongdai Liu and Jiangyong Yu and Dawei Yang and Hui Li and Yan Chen},
  year={2023},
  eprint={2311.12058},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```

## 致谢

基于 [BEVDet](https://github.com/HuangJunJie2017/BEVDet)、[FB-BEV](https://github.com/NVlabs/FB-BEV.git)、[RenderOcc](https://github.com/pmj110119/RenderOcc.git) 和 [SparseBEV](https://github.com/MCG-NJU/SparseBEV.git)。
