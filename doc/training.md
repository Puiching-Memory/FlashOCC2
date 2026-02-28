# 模型训练指南

## 快速开始

### 单 GPU 训练

```bash
python tools/train.py configs/flashocc_r50.py
```

### 多 GPU 训练（DDP）

```bash
# 4 卡训练
torchrun --nproc_per_node=4 tools/train.py configs/flashocc_r50.py --launcher pytorch

# 使用便捷脚本
bash tools/dist_train.sh configs/flashocc_r50.py 4
```

### SLURM 集群训练

```bash
srun -p partition --gres=gpu:4 --ntasks=4 --ntasks-per-node=4 \
    python tools/train.py configs/flashocc_r50.py --launcher slurm
```

## 命令行参数

```
python tools/train.py <config> [options]

必选：
  config                 Python 配置文件路径（.py）

可选：
  --work-dir DIR         工作目录（默认: work_dirs/<config_name>）
  --resume-from PATH     从 checkpoint 恢复训练
  --validate             训练时进行验证
  --seed SEED            随机种子
  --deterministic        启用确定性模式
  --launcher {none,pytorch,slurm}  分布式启动器（默认: none）
  --gpu-id ID            GPU ID（单卡模式，默认: 0）
```

## 预训练权重

训练前需下载 BEVDet 预训练权重到 `ckpts/` 目录：

| 模型            | 文件                        |
| --------------- | --------------------------- |
| BEVDet-R50-CBGS | `ckpts/bevdet-r50-cbgs.pth` |

权重来源：[BEVDet](https://github.com/HuangJunJie2017/BEVDet)

配置文件中通过 `load_from` 指定预训练路径：

```python
# configs/flashocc_r50.py
experiment = Experiment(
    load_from="ckpts/bevdet-r50-cbgs.pth",
    ...
)
```

## 训练输出

训练产物保存在 `work_dirs/<config_name>/`：

```
work_dirs/flashocc_r50/
├── 20240101_120000.log     # 训练日志
├── epoch_1.pth             # 各 epoch checkpoint
├── epoch_2.pth
├── ...
├── epoch_24.pth
└── epoch_24.pth
```

## 训练参数参考（FlashOCC-R50）

| 参数       | 值                        |
| ---------- | ------------------------- |
| batch_size | 8 × GPU 数                |
| 学习率     | 1e-4                      |
| 优化器     | AdamW (weight_decay=1e-2) |
| 学习率策略 | Step (step=[24])          |
| 梯度裁剪   | max_norm=5                |
| EMA        | init_updates=10560        |
| Epoch      | 24                        |
| 训练时间   | ~10h (4×A100)             |
