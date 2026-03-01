# 模型测试与评估

## 单 GPU 测试

```bash
python tools/test.py configs/flashocc_r50.py work_dirs/flashocc_r50/epoch_24.pth --eval occ

# 指定 CSV 输出路径
python tools/test.py configs/flashocc_r50.py work_dirs/flashocc_r50/epoch_24.pth \
  --eval occ --csv-out results/flashocc_r50_eval.csv
```

## 目录批量评估（按顺序）

```bash
# 当 checkpoint 传入目录时，会处理该目录下一级所有权重文件并按自然顺序（如 epoch_1, epoch_2, ..., epoch_24）评估
python tools/test.py configs/flashocc_r50.py work_dirs/flashocc_r50 --eval occ

# 可选：手动指定汇总 CSV 文件路径
python tools/test.py configs/flashocc_r50.py work_dirs/flashocc_r50 \
  --eval occ --csv-out results/flashocc_r50_all_epochs.csv
```

## 多 GPU 测试

```bash
torchrun --nproc_per_node=8 tools/test.py \
  configs/flashocc_r50.py work_dirs/flashocc_r50 \
    --launcher pytorch --eval occ

# 使用便捷脚本
bash tools/dist_test.sh configs/flashocc_r50.py work_dirs/flashocc_r50 8 --eval occ
```

## 命令行参数

```
python tools/test.py <config> <checkpoint_or_dir> [options]

必选：
  config                 Python 配置文件路径（.py）
  checkpoint_or_dir      checkpoint 文件路径，或目录路径（目录下一级权重将按顺序批量评估）

可选：
  --eval METRICS         评估指标（如 occ）
  --csv-out PATH         保存评估汇总结果到 CSV 文件
  --gpu-id ID            GPU ID（单卡模式，默认: 0）
  --launcher {none,pytorch,slurm}  分布式启动器（默认: none）
```

## 保存评估结果（CSV）

```bash
python tools/test.py configs/flashocc_r50.py work_dirs/flashocc_r50/epoch_24.pth \
    --eval occ --csv-out results/flashocc_r50_eval.csv
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
