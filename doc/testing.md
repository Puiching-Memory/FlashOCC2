# 模型测试与评估

## 单 GPU 测试

```bash
python tools/test.py configs/flashocc_r50.py work_dirs/flashocc_r50/epoch_24.pth --eval occ
```

## 多 GPU 测试

```bash
torchrun --nproc_per_node=8 tools/test.py \
    configs/flashocc_r50.py work_dirs/flashocc_r50/epoch_24.pth \
    --launcher pytorch --eval occ

# 使用便捷脚本
bash tools/dist_test.sh configs/flashocc_r50.py work_dirs/flashocc_r50/epoch_24.pth 8 --eval occ
```

## 命令行参数

```
python tools/test.py <config> <checkpoint> [options]

必选：
  config                 Python 配置文件路径（.py）
  checkpoint             checkpoint 文件路径

可选：
  --out PATH             保存结果到 pkl 文件
  --eval METRICS         评估指标（如 occ）
  --gpu-id ID            GPU ID（单卡模式，默认: 0）
  --launcher {none,pytorch,slurm}  分布式启动器（默认: none）
```

## 保存预测结果

```bash
python tools/test.py configs/flashocc_r50.py epoch_24.pth --out results/flashocc_r50.pkl
```

## 评估指标

### 占用预测（Occupancy）

- **mIoU**: 平均交并比（mean Intersection over Union）
- **RayIoU**: 基于射线的 IoU（更贴近实际感知场景）
- **RayPQ**: 基于射线的全景质量（Panoptic Quality）

### NuScenes 占用语义类别（18 类）

| ID  | 类别                 | ID  | 类别              |
| --- | -------------------- | --- | ----------------- |
| 0   | others               | 9   | trailer           |
| 1   | barrier              | 10  | truck             |
| 2   | bicycle              | 11  | driveable_surface |
| 3   | bus                  | 12  | other_flat        |
| 4   | car                  | 13  | sidewalk          |
| 5   | construction_vehicle | 14  | terrain           |
| 6   | motorcycle           | 15  | manmade           |
| 7   | pedestrian           | 16  | vegetation        |
| 8   | traffic_cone         | 17  | free              |

完整类别列表定义在 `src/flashocc/constants.py` 的 `OCC_CLASS_NAMES` 中。

### 参考性能

| 模型                    | Backbone | mIoU  |
| ----------------------- | -------- | ----- |
| FlashOCC M1 (1f)        | R50      | 32.08 |
| FlashOCC-4D-Stereo (2f) | R50      | 37.84 |
| FlashOCC-4D-Stereo (2f) | Swin-B   | 43.52 |
