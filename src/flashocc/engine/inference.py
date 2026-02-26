"""模型初始化和推理."""
import torch
from pydantic import BaseModel, ConfigDict

from flashocc.config import load_experiment, Experiment
from flashocc.core import load_checkpoint


class _InitModelInput(BaseModel):
    config: str | Experiment
    checkpoint: str | None = None
    device: str = "cuda:0"

    model_config = ConfigDict(arbitrary_types_allowed=True)


def init_model(config, checkpoint=None, device="cuda:0"):
    """从 Python 配置文件初始化模型.

    Args:
        config: 配置文件路径 (str) 或 Experiment 对象。
        checkpoint: checkpoint 文件路径。
        device: 设备字符串。

    Returns:
        nn.Module: 初始化好的模型。
    """
    validated = _InitModelInput.model_validate(
        {"config": config, "checkpoint": checkpoint, "device": device}
    )

    match validated.config:
        case str() as cfg_path:
            exp = load_experiment(cfg_path)
        case Experiment() as exp_obj:
            exp = exp_obj
        case _:
            raise TypeError(f"config 须为 str 或 Experiment, 得到 {type(validated.config)}")

    model = exp.build_model()
    if validated.checkpoint is not None:
        load_checkpoint(model, validated.checkpoint, map_location="cpu")
    model.to(validated.device)
    model.eval()
    return model


__all__ = ["init_model"]
