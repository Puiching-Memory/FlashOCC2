"""模型初始化和推理."""
import torch

from flashocc.config import load_experiment, Experiment
from flashocc.core import load_checkpoint


def init_model(config, checkpoint=None, device="cuda:0"):
    """从 Python 配置文件初始化模型.

    Args:
        config: 配置文件路径 (str) 或 Experiment 对象。
        checkpoint: checkpoint 文件路径。
        device: 设备字符串。

    Returns:
        nn.Module: 初始化好的模型。
    """
    if isinstance(config, str):
        exp = load_experiment(config)
    elif isinstance(config, Experiment):
        exp = config
    else:
        raise TypeError(f"config 须为 str 或 Experiment, 得到 {type(config)}")

    model = exp.build_model()
    if checkpoint is not None:
        load_checkpoint(model, checkpoint, map_location="cpu")
    model.to(device)
    model.eval()
    return model


__all__ = ["init_model"]
