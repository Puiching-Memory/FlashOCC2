"""检查点 (Checkpoint) 加载与保存."""

from __future__ import annotations

import os
import os.path as osp
from collections import OrderedDict

import torch
import torch.nn as nn

from flashocc.core.log import logger

def _load_checkpoint_raw(filename: str, map_location=None):
    """从文件 / URL 加载 checkpoint."""
    if filename.startswith(("http://", "https://")):
        return torch.hub.load_state_dict_from_url(filename, map_location=map_location)
    return torch.load(filename, map_location=map_location, weights_only=False)


def load_state_dict(module: nn.Module, state_dict: dict, strict: bool = False,
                    logger=None):
    """加载 state_dict, 处理形状不匹配."""
    from flashocc.core.log import logger as _default_logger
    log = logger or _default_logger
    unexpected, missing = [], []
    own = module.state_dict()

    for name, param in state_dict.items():
        if name in own:
            if own[name].shape == param.shape:
                own[name].copy_(param)
            else:
                log.warning(f"形状不匹配 {name}: {own[name].shape} vs {param.shape}")
                missing.append(name)
        else:
            unexpected.append(name)

    for name in own:
        if name not in state_dict:
            missing.append(name)

    if unexpected:
        log.warning(f"意外的 key: {unexpected[:10]}...")
    if missing:
        log.warning(f"缺失的 key: {missing[:10]}...")


def load_checkpoint(model: nn.Module, filename: str, map_location=None,
                    strict: bool = False, logger=None, **kwargs):
    """加载检查点到模型."""
    ckpt = _load_checkpoint_raw(filename, map_location=map_location)

    if hasattr(ckpt, 'keys'):
        state_dict = ckpt.get("state_dict", ckpt.get("model", ckpt))
    else:
        raise RuntimeError(f"不支持的 checkpoint 类型: {type(ckpt)}")

    # 去除 'module.' 前缀
    keys = list(state_dict.keys())
    if keys and keys[0].startswith("module."):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    load_state_dict(model, state_dict, strict=strict, logger=logger)
    return ckpt


def save_checkpoint(model: nn.Module, filename: str, optimizer=None, meta=None):
    """保存检查点."""
    os.makedirs(osp.dirname(osp.abspath(filename)), exist_ok=True)
    sd = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    ckpt = {"state_dict": sd, "meta": meta or {}}
    if optimizer is not None:
        ckpt["optimizer"] = optimizer.state_dict()
    torch.save(ckpt, filename)
