"""并行 / 数据容器工具."""

from __future__ import annotations

import torch
import torch.nn as nn


# =====================================================================
#  DataContainer
# =====================================================================


class DataContainer:
    """数据容器, 为 DataParallel 提供 stack/cpu_only 等元信息."""

    def __init__(self, data, stack: bool = False, padding_value: int = 0,
                 cpu_only: bool = False, pad_dims: int = 2):
        self._data = data
        self._stack = stack
        self._padding_value = padding_value
        self._cpu_only = cpu_only
        self._pad_dims = pad_dims

    @property
    def data(self):
        return self._data

    @property
    def datatype(self):
        return self._data.type() if isinstance(self._data, torch.Tensor) else type(self._data)

    @property
    def stack(self):
        return self._stack

    @property
    def padding_value(self):
        return self._padding_value

    @property
    def cpu_only(self):
        return self._cpu_only

    @property
    def pad_dims(self):
        return self._pad_dims

    def __repr__(self):
        return f"DataContainer({self._data})"

    def __len__(self):
        return len(self._data)


# =====================================================================
#  DataParallel
# =====================================================================


def _unwrap(data):
    """递归解包 DataContainer."""
    if isinstance(data, DataContainer):
        return data.data
    if isinstance(data, dict):
        return {k: _unwrap(v) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return type(data)(_unwrap(d) for d in data)
    return data


class FlashDataParallel(nn.DataParallel):
    """DataParallel, 自动解包 DataContainer."""

    def train_step(self, *inputs, **kwargs):
        return self.forward(*inputs, **kwargs)

    def val_step(self, *inputs, **kwargs):
        return self.forward(*inputs, **kwargs)

    def scatter(self, inputs, kwargs, device_ids):
        return super().scatter(_unwrap(inputs), _unwrap(kwargs), device_ids)


class FlashDistributedDataParallel(nn.parallel.DistributedDataParallel):
    """DistributedDataParallel."""

    def train_step(self, *inputs, **kwargs):
        return self.forward(*inputs, **kwargs)

    def val_step(self, *inputs, **kwargs):
        return self.forward(*inputs, **kwargs)


__all__ = [
    "DataContainer",
    "FlashDataParallel", "FlashDistributedDataParallel",
]
