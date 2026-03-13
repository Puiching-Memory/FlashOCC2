# FlashOCC v2.0

An improved engineering version of FlashOCC, with mm-series dependencies removed, and support for the latest Torch and CUDA devices.

## Highlights

- Fixed the BGR color channel issue: https://github.com/HuangJunJie2017/BEVDet/issues/274
- New BEV Pool V3 supports torch.compile CUDA and significantly improves performance
- DAIL data pipeline

## Request For Help

We are currently unable to reproduce the FlashOCC_ResNet50 result (mIoU 32.08).
We do not have enough development bandwidth to investigate this for now.
If you discover anything, we would greatly appreciate your analysis in advance.

## Quick Start

```bash
# Install
uv venv .venv --python 3.12 && source .venv/bin/activate
uv sync

# Prepare data (see doc/nuscenes_data.md)
python tools/create_data_flashocc2.py

# Train (single GPU)
python tools/train.py configs/flashocc_r50.py

# Train (multi GPU)
torchrun --nproc_per_node=4 tools/train.py configs/flashocc_r50.py

# Test
python tools/test.py configs/flashocc_r50.py work_dirs/flashocc_r50/epoch_24.pth --eval occ

# Analyze class distribution in the dataset
python tools/analyze_class_distribution.py data/flashocc2-nuscenes_infos_train.pkl --no-show
```

## Project Structure

```text
FlashOCC/
├── configs/                  # Python config files
│   └── flashocc_r50.py       # R50 single-frame config
├── src/flashocc/             # Source package
│   ├── constants.py          # Global constants (class names, grid params, etc.)
│   ├── config/               # Config system (Lazy descriptor, dataclasses)
│   ├── core/                 # Core infrastructure
│   │   ├── base_module.py    # BaseModule (supports init_cfg)
│   │   ├── checkpoint.py     # Weight loading/saving
│   │   ├── dist.py           # Distributed utilities
│   │   ├── fp16.py           # Mixed precision (force_fp32, wrap_fp16_model)
│   │   ├── functional.py     # multi_apply, reduce_mean
│   │   ├── nn.py             # ConvModule, build_conv/norm_layer, initialization
│   │   ├── registry.py       # Registry pattern
│   │   ├── bbox/             # 3D bounding boxes and point classes
│   │   └── ops/              # CUDA ops (bev_pool_v2, bev_pool_v3)
│   ├── datasets/             # Data loading and evaluation
│   │   ├── base_dataset.py   # Custom3DDataset base class
│   │   ├── nuscenes_occ.py   # NuScenesOccDataset (occupancy evaluation)
│   │   ├── nuscenes_bevdet.py# NuScenesDatasetBEVDet (detection evaluation)
│   │   ├── pipelines/        # Data transforms (loading, augmentation, formatting)
│   │   └── evaluation/       # Metrics (mIoU, RayIoU, RayPQ)
│   ├── engine/               # Training and inference
│   │   ├── trainer.py        # Training loop
│   │   ├── tester.py         # Testing loop
│   │   ├── inference.py      # single_gpu_test
│   │   ├── seed.py           # Random seed utilities
│   │   ├── parallel.py       # DataParallel wrapper
│   │   └── hooks/            # Training hooks (EMA, SyncBN, etc.)
│   └── models/               # Model definitions
│       ├── backbones/        # ResNet, CustomResNet
│       ├── necks/            # FPN, LSSFPN, ViewTransformer
│       ├── heads/            # OccHead
│       ├── detectors/        # BEVDet -> BEVDetOCC pipeline
│       └── losses/           # Cross-entropy loss utilities
├── tools/                    # CLI scripts
│   ├── train.py              # Training entry
│   ├── test.py               # Testing entry
│   ├── analyze_class_distribution.py  # Dataset class distribution analysis
│   ├── dist_train.sh         # Multi-GPU training script
│   ├── dist_test.sh          # Multi-GPU testing script
│   └── create_data_flashocc2.py # Data preparation
├── data/nuscenes/            # Dataset (not tracked by git)
├── ckpts/                    # Pretrained checkpoints
└── pyproject.toml            # Dependencies and build config
```

## Configuration System

Configuration files are pure Python files using Lazy descriptors and dataclasses, with no YAML and no dict-based registry:

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

All model components are referenced directly through Python imports, so IDE navigation, refactoring, and type checking work out of the box.

## Performance

| Model                    | Backbone | Input Size | mIoU  | Params  |
| ------------------------ | -------- | ---------- | ----- | ------- |
| FlashOCC M1 (1f)         | R50      | 256×704    | 32.08 | 44.74M  |
| FlashOCC-4D-Stereo (2f)  | R50      | 256×704    | 37.84 | -       |
| FlashOCC-4D-Stereo (2f)  | Swin-B   | 512×1408   | 43.52 | 144.99M |

## Documentation

- [Installation Guide](doc/install.md)
- [Project Architecture](doc/architecture.md)
- [NuScenes Data Preparation](doc/nuscenes_data.md)
- [Training Guide](doc/training.md)
- [Testing Guide](doc/testing.md)

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

## Acknowledgements

Based on [BEVDet](https://github.com/HuangJunJie2017/BEVDet), [FB-BEV](https://github.com/NVlabs/FB-BEV.git), [RenderOcc](https://github.com/pmj110119/RenderOcc.git), and [SparseBEV](https://github.com/MCG-NJU/SparseBEV.git).
