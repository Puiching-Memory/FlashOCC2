# FlashOCC 安装指南

## 系统要求

- **Python**: >= 3.10
- **PyTorch**: >= 2.5 (CUDA 12.8)
- **GPU**: NVIDIA GPU (推荐 A100 / 4090)
- **操作系统**: Ubuntu 22.04+

## 快速安装

### 1. 创建虚拟环境

```bash
# 使用 uv（推荐）
uv venv .venv --python 3.12
source .venv/bin/activate

# 或使用 conda
conda create -n flashocc python=3.12 -y
conda activate flashocc
```

### 2. 安装依赖

```bash
# 使用 uv（推荐，自动处理 PyTorch CUDA 版本）
uv sync

# 或手动安装
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -e .
```

### 3. 验证安装

```bash
python -c "
from flashocc.config import load_experiment
exp = load_experiment('configs/flashocc_r50.py')
model = exp.build_model()
print(f'FlashOCC OK — {sum(p.numel() for p in model.parameters()):,} params')
"
```

预期输出：`FlashOCC OK — 44,744,312 params`

## CUDA 扩展

BEV Pooling CUDA 算子在首次使用时通过 PyTorch JIT 自动编译（需要 `ninja`，已在 `pyproject.toml` 中声明）。

编译产物缓存在 `~/.cache/torch_extensions/`。如果遇到编译问题：

```bash
# 清除构建缓存
rm -rf ~/.cache/torch_extensions/

# 确认 CUDA 工具链可用
nvcc --version
python -c "import torch; print(torch.cuda.is_available())"
```

## 目录结构

```
FlashOCC/
├── configs/              # Python 配置文件
│   └── flashocc_r50.py
├── src/flashocc/         # 源代码包
│   ├── constants.py      # 全局常量
│   ├── config/           # 纯 Python 配置系统
│   ├── core/             # 核心基础设施
│   ├── datasets/         # 数据集 & 评估
│   ├── engine/           # 训练 / 测试引擎
│   └── models/           # 模型定义
├── tools/                # 训练 / 测试 / 分析脚本
├── data/nuscenes/        # 数据目录（不被 git 跟踪）
├── ckpts/                # 预训练权重
└── pyproject.toml        # 依赖声明
```

## 依赖列表

核心依赖（完整列表见 `pyproject.toml`）：

| 包              | 版本要求 | 用途                    |
| --------------- | -------- | ----------------------- |
| torch           | >= 2.5   | 深度学习框架            |
| torchvision     | >= 0.20  | 图像处理、ResNet 预训练 |
| nuscenes-devkit | >= 1.2.0 | NuScenes 数据集 API     |
| numpy           | >= 1.26  | 数值计算                |
| timm            | >= 1.0   | 模型组件                |
| ninja           | >= 1.11  | CUDA 扩展 JIT 编译      |
| matplotlib      | >= 3.8   | 类别分布可视化工具    |
