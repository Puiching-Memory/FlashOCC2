# NuScenes 数据准备

## 1. 下载数据集

从 [NuScenes 官网](https://www.nuscenes.org/nuscenes) 下载：

- Full dataset (v1.0-trainval): 约 300GB
- Mini dataset (v1.0-mini): 约 4GB（用于快速验证）

## 2. 数据目录结构

将数据放置在 `data/nuscenes/` 目录下：

```
data/nuscenes/
├── maps/
├── samples/
│   ├── CAM_BACK/
│   ├── CAM_BACK_LEFT/
│   ├── CAM_BACK_RIGHT/
│   ├── CAM_FRONT/
│   ├── CAM_FRONT_LEFT/
│   ├── CAM_FRONT_RIGHT/
│   ├── LIDAR_TOP/
│   └── ...
├── sweeps/
│   └── ...
├── v1.0-trainval/
│   ├── attribute.json
│   ├── calibrated_sensor.json
│   ├── category.json
│   ├── ego_pose.json
│   ├── instance.json
│   ├── log.json
│   ├── map.json
│   ├── sample_annotation.json
│   ├── sample_data.json
│   ├── sample.json
│   ├── scene.json
│   ├── sensor.json
│   └── visibility.json
└── lidarseg/
    └── v1.0-trainval/
```

> **注意**: `data/` 目录存放数据，不被 git 跟踪。处理数据的代码在 `src/flashocc/datasets/` 下。

## 3. 生成 BEVDet 格式数据信息

```bash
python tools/create_data_bevdet.py
```

> **注意**: `create_data_bevdet.py` 依赖 `nuscenes_converter.py`，后者需要 `mmcv` 和 `mmdet3d`。
> 如果已有预生成的 pkl 文件，可跳过此步骤。

生成的文件：
- `data/nuscenes/bevdetv2-nuscenes_infos_train.pkl`
- `data/nuscenes/bevdetv2-nuscenes_infos_val.pkl`

## 4. Occupancy GT 数据

FlashOCC 训练需要体素化的占用真值：

```
data/nuscenes/
└── gts/
    ├── scene-0001/
    │   ├── <sample_token>/   # 每帧的占用网格
    │   └── ...
    └── ...
```

每个目录下的 numpy 文件包含：
- `voxel_semantics`: (200, 200, 16) uint8, 体素语义标签
- `mask_lidar`: (200, 200, 16) bool, LiDAR 可见区域
- `mask_camera`: (200, 200, 16) bool, 相机可见区域

占用网格参数（定义在 `src/flashocc/constants.py`）：
- 点云范围：[-40, -40, -1, 40, 40, 5.4] 米
- 体素大小：0.4 米
- 网格形状：(200, 200, 16)
- 语义类别：18 类

## 5. 验证数据完整性

```bash
python -c "
import pickle
with open('data/nuscenes/bevdetv2-nuscenes_infos_train.pkl', 'rb') as f:
    data = pickle.load(f)
print(f'训练集样本数: {len(data[\"infos\"])}')
"
```

预期输出：`训练集样本数: 28130`
