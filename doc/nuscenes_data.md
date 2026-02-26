# NuScenes 数据准备

## 1. 下载数据集

从 [NuScenes 官网](https://www.nuscenes.org/nuscenes) 下载

## 2. 数据目录结构

将数据放置在 `data/nuScenes/` 目录下：

```
data/nuScenes/
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

## 3. 生成 BEVDet 格式数据信息

```bash
python tools/create_data_flashocc2.py \
    --root-path data/nuScenes \
    --out-dir data/ \
    --extra-tag flashocc2-nuscenes \
    --max-sweeps 0 \
    --parallel-backend process \
    --num-workers 8
```

生成的文件：
- `data/nuScenes/flashocc2-nuscenes_infos_train.pkl`
- `data/nuScenes/flashocc2-nuscenes_infos_val.pkl`

> 可选参数：
> - `--path-mode relative|absolute`：控制 pkl 中 `data_path` / `occ_path` 存储相对路径或绝对路径。
> - `--occ-root <path>`：指定 OCC GT 根目录，默认 `<root-path>/gts`。
> - `--num-workers <int>`：样本级并行 worker 数，设为 `1` 表示关闭并行。
> - `--parallel-backend thread|process`：并行后端；`thread`（默认）内存占用更低，`process` 速度可能更高但占用更大。
> - 建议先从 `thread + 4~8 workers` 起步，再按机器资源逐步调高。

## 4. Occupancy GT 数据

FlashOCC 训练需要体素化的占用真值：

```
data/nuScenes/
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
with open('data/nuScenes/flashocc2-nuscenes_infos_train.pkl', 'rb') as f:
    data = pickle.load(f)
print(f'训练集样本数: {len(data[\"infos\"])}')
"
```

预期输出：`训练集样本数: 28130`
