# FlashOCC v2.0

Fast and memory-efficient 3D occupancy prediction via Channel-to-Height plugin.

Based on [FlashOCC](https://arxiv.org/abs/2311.12058), [Panoptic-FlashOCC](https://arxiv.org/pdf/2406.10527), and [UltimateDO](https://arxiv.org/abs/2409.11160).

## Quick Start

```bash
# Install
uv venv .venv --python 3.12 && source .venv/bin/activate
uv sync

# Prepare data (see doc/nuscenes_data.md)
python tools/create_data_flashocc2.py

# Train (single GPU)
python tools/train.py configs/flashocc_r50.py

# Train (multi-GPU)
torchrun --nproc_per_node=4 tools/train.py configs/flashocc_r50.py

# Test
python tools/test.py configs/flashocc_r50.py work_dirs/flashocc_r50/epoch_24.pth --eval occ

# Analyze dataset class distribution
python tools/analyze_class_distribution.py data/flashocc2-nuscenes_infos_train.pkl --no-show
```

## Project Structure

```
FlashOCC/
├── configs/                  # Python config files
│   └── flashocc_r50.py       # R50 single-frame config
├── src/flashocc/             # Package source
│   ├── constants.py          # Project-wide constants (classes, grid, etc.)
│   ├── config/               # Config system (Lazy descriptors, dataclasses)
│   ├── core/                 # Base infrastructure
│   │   ├── base_module.py    # BaseModule with init_cfg support
│   │   ├── checkpoint.py     # Checkpoint load/save
│   │   ├── dist.py           # Distributed utilities
│   │   ├── fp16.py           # Mixed precision (force_fp32, wrap_fp16_model)
│   │   ├── functional.py     # multi_apply, reduce_mean
│   │   ├── nn.py             # ConvModule, build_conv/norm_layer, init
│   │   ├── registry.py       # Registry pattern
│   │   ├── bbox/             # 3D bounding box & point classes
│   │   └── ops/              # CUDA ops (bev_pool_v2, bev_pool_v3)
│   ├── datasets/             # Data loading & evaluation
│   │   ├── base_dataset.py   # Custom3DDataset base class
│   │   ├── nuscenes_occ.py   # NuScenesOccDataset (OCC evaluation)
│   │   ├── nuscenes_bevdet.py# NuScenesDatasetBEVDet (detection eval)
│   │   ├── pipelines/        # Data transforms (loading, augmentation, formatting)
│   │   └── evaluation/       # Metrics (mIoU, RayIoU, RayPQ)
│   ├── engine/               # Training & inference
│   │   ├── trainer.py        # Training loop
│   │   ├── tester.py         # Test loop
│   │   ├── inference.py      # single_gpu_test
│   │   ├── seed.py           # Random seed utilities
│   │   ├── parallel.py       # DataParallel wrappers
│   │   └── hooks/            # Training hooks (EMA, SyncBN, etc.)
│   └── models/               # Model definitions
│       ├── backbones/        # ResNet, CustomResNet
│       ├── necks/            # FPN, LSSFPN, ViewTransformer
│       ├── heads/            # OccHead
│       ├── detectors/        # BEVDet → BEVDetOCC pipeline
│       └── losses/           # CrossEntropyLoss utilities
├── tools/                    # CLI scripts
│   ├── train.py              # Training entry point
│   ├── test.py               # Testing entry point
│   ├── analyze_class_distribution.py  # Dataset class distribution analysis
│   ├── dist_train.sh         # Multi-GPU training wrapper
│   ├── dist_test.sh          # Multi-GPU testing wrapper
│   └── create_data_flashocc2.py # Data preparation
├── data/nuscenes/            # Dataset (not tracked by git)
├── ckpts/                    # Pretrained weights
└── pyproject.toml            # Dependencies & build config
```

## Config System

Configs are pure Python files using `Lazy` descriptors and dataclasses — no YAML, no dict-based registries:

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

All model components are imported directly — IDE navigation, refactoring, and type-checking work out of the box.

## Metrics

| Model                   | Backbone | Input    | mIoU  | Params  |
| ----------------------- | -------- | -------- | ----- | ------- |
| FlashOCC M1 (1f)        | R50      | 256×704  | 32.08 | 44.74M  |
| FlashOCC-4D-Stereo (2f) | R50      | 256×704  | 37.84 | -       |
| FlashOCC-4D-Stereo (2f) | Swin-B   | 512×1408 | 43.52 | 144.99M |

## Documentation

- [Installation](doc/install.md)
- [Architecture](doc/architecture.md)
- [NuScenes Data Preparation](doc/nuscenes_data.md)
- [Training](doc/training.md)
- [Testing](doc/testing.md)

## Citation

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

## Acknowledgement

Based on [BEVDet](https://github.com/HuangJunJie2017/BEVDet), [FB-BEV](https://github.com/NVlabs/FB-BEV.git), [RenderOcc](https://github.com/pmj110119/RenderOcc.git), and [SparseBEV](https://github.com/MCG-NJU/SparseBEV.git).
