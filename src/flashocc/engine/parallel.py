"""数据容器."""

from __future__ import annotations


class DataContainer:
    """数据容器, 为 collate/scatter 提供 stack/cpu_only 等元信息."""

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
        return self._data.type() if hasattr(self._data, 'type') and callable(self._data.type) else type(self._data)

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


__all__ = ["DataContainer"]
