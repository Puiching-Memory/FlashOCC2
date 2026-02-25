"""组合式配置 — Lazy 延迟构建 + Experiment 实验描述.

's|/desay120T/ct/dev/uid01954/FlashOCC/||''s|/desay120T/ct/dev/uid01954//desay120T/ct/dev/uid01954/FlashOCC/src/flashocc
--------
1. **所有组件都是真实 Python 类** — import 即检查, IDE 可跳转/补全
2. **Lazy(cls, **kw)** — 延迟构建描述符, "我要用这个类 + 这些参数"
3. **Experiment** — 一次实验的完整描述, 训练引擎直接消费
4. **无字典配置, 无字符串类型名, 无 YAML, 无运行时覆写**


----
>>> from flashocc.config import Lazy, Experiment
>>> from flashocc.models.backbones._resnet_base import ResNet
>>> backbone = Lazy(ResNet, depth=50)
>>> backbone.build()   # → ResNet(depth=50)
"""
from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass, field
from typing import Any, List, Optional, Type

import torch


# =====================================================================
#  Lazy — 延迟构建描述符
# =====================================================================

class Lazy:
    """延迟构建描述符 — 记录类 + 参数, 调用 build() 时才实例化.

    Parameters
    ----------
    cls : type
        要构建的目标类.
    **kwargs
        传给 ``cls.__init__`` 的参数, 支持嵌套 ``Lazy``.
    """

    __slots__ = ("_cls", "_kwargs")

    def __init__(self, cls: type, /, **kwargs: Any) -> None:
        if not callable(cls):
            raise TypeError(f"Lazy 第一个参数必须是可调用对象, 收到 {type(cls)}")
        self._cls = cls
        self._kwargs = kwargs

    @property
    def cls(self) -> type:
        return self._cls

    @property
    def kwargs(self) -> dict:
        return dict(self._kwargs)

    def build(self) -> Any:
        """递归解析嵌套 Lazy, 调用 cls(**resolved_kwargs)."""
        resolved = {k: _resolve(v) for k, v in self._kwargs.items()}
        return self._cls(**resolved)

    def override(self, **kwargs: Any) -> "Lazy":
        """返回参数覆盖后的新 Lazy (不可变)."""
        merged = {**self._kwargs, **kwargs}
        return Lazy(self._cls, **merged)

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self._kwargs.items())
        return f"Lazy({self._cls.__name__}, {args})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Lazy):
            return NotImplemented
        return self._cls is other._cls and self._kwargs == other._kwargs


# =====================================================================
#  Experiment — 实验描述
# =====================================================================

@dataclass
class Experiment:
    """一次实验的完整描述 — 训练引擎直接消费此对象.

    所有字段类型明确, IDE 可自动补全、静态检查。
    """

    # ---- 模型 ----
    model: Lazy

    # ---- 数据 ----
    train_data: Lazy
    val_data: Lazy
    test_data: Optional[Lazy] = None
    samples_per_gpu: int = 4
    workers_per_gpu: int = 4

    # ---- 优化器 (Lazy 包装 torch.optim 类) ----
    optimizer: Lazy = field(
        default_factory=lambda: Lazy(torch.optim.AdamW, lr=1e-4, weight_decay=1e-2))

    # ---- 学习率调度器 (Lazy 包装 torch.optim.lr_scheduler 类) ----
    lr_scheduler: Optional[Lazy] = None

    # ---- Warmup ----
    warmup_iters: int = 0
    warmup_ratio: float = 0.001

    # ---- 梯度裁剪 ----
    grad_max_norm: Optional[float] = None

    # ---- 训练 ----
    max_epochs: int = 24

    # ---- 预训练 / 恢复 ----
    load_from: Optional[str] = None
    resume_from: Optional[str] = None

    # ---- 检查点 ----
    checkpoint_interval: int = 1
    max_keep_ckpts: int = 5

    # ---- 评估 ----
    eval_interval: int = 1
    eval_start: int = 20

    # ---- 运行时 ----
    work_dir: Optional[str] = None
    seed: int = 0
    log_interval: int = 50
    find_unused_parameters: bool = False

    # ================================================================
    #  构建方法
    # ================================================================

    def build_model(self) -> torch.nn.Module:
        """构建模型."""
        return self.model.build()

    def build_train_dataset(self) -> Any:
        """构建训练数据集."""
        return self.train_data.build()

    def build_val_dataset(self) -> Any:
        """构建验证数据集."""
        return self.val_data.build()

    def build_test_dataset(self) -> Any:
        """构建测试数据集."""
        return (self.test_data or self.val_data).build()

    def build_optimizer(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        """构建优化器 — Lazy.cls(model.parameters(), **Lazy.kwargs)."""
        return self.optimizer.cls(model.parameters(), **self.optimizer.kwargs)

    def build_lr_scheduler(
        self, optimizer: torch.optim.Optimizer
    ) -> Optional[Any]:
        """构建学习率调度器."""
        if self.lr_scheduler is None:
            return None
        return self.lr_scheduler.cls(optimizer, **self.lr_scheduler.kwargs)


# =====================================================================
#  加载函数
# =====================================================================

def load_experiment(filepath: str) -> Experiment:
    """从 Python 配置文件加载 Experiment.

    配置文件须定义模块级变量 ``experiment: Experiment``.
    """
    filepath = os.path.abspath(filepath)
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"配置文件不存在: {filepath}")

    spec = importlib.util.spec_from_file_location("_pyconfig", filepath)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载配置: {filepath}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]

    if not hasattr(mod, "experiment"):
        raise AttributeError(
            f"配置文件 {filepath} 中未定义 'experiment'。"
            f"\n须定义: experiment = Experiment(...)")
    exp = mod.experiment
    if not isinstance(exp, Experiment):
        raise TypeError(f"'experiment' 须为 Experiment, 得到 {type(exp)}")
    return exp


# =====================================================================
#  内部
# =====================================================================

def _resolve(v: Any) -> Any:
    """递归解析 Lazy → 实例."""
    if isinstance(v, Lazy):
        return v.build()
    if isinstance(v, dict):
        return {dk: _resolve(dv) for dk, dv in v.items()}
    if isinstance(v, (list, tuple)):
        return type(v)(_resolve(item) for item in v)
    return v
