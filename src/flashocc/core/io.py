"""文件 I/O 工具."""

from __future__ import annotations

import json
import os
import os.path as osp
import pickle

import yaml


def mkdir_or_exist(dir_name: str, mode: int = 0o777):
    """创建目录 (若不存在)."""
    if dir_name:
        os.makedirs(dir_name, mode=mode, exist_ok=True)


def load(file: str, file_format: str | None = None, **kwargs):
    """加载 json / yaml / pickle 文件."""
    ext = osp.splitext(file)[1].lower()
    fmt = file_format or {
        ".json": "json", ".yaml": "yaml", ".yml": "yaml",
        ".pkl": "pickle", ".pickle": "pickle",
    }.get(ext)
    # 标准化别名
    if fmt in ("pkl", "pickle"):
        fmt = "pickle"
    if fmt is None:
        raise ValueError(f"不支持的文件格式: {ext}")

    mode = "rb" if fmt == "pickle" else "r"
    with open(file, mode) as f:
        if fmt == "json":
            return json.load(f, **kwargs)
        if fmt == "yaml":
            return yaml.safe_load(f)
        return pickle.load(f, **kwargs)


def dump(obj, file: str | None = None, file_format: str | None = None, **kwargs):
    """导出 json / yaml / pickle."""
    if file is not None:
        ext = osp.splitext(file)[1].lower()
        fmt = file_format or {
            ".json": "json", ".yaml": "yaml", ".yml": "yaml",
            ".pkl": "pickle", ".pickle": "pickle",
        }.get(ext)
        mode = "wb" if fmt == "pickle" else "w"
        with open(file, mode) as f:
            if fmt == "json":
                json.dump(obj, f, **kwargs)
            elif fmt == "yaml":
                yaml.dump(obj, f, **kwargs)
            elif fmt == "pickle":
                pickle.dump(obj, f, **kwargs)
    else:
        fmt = file_format
        if fmt == "json":
            return json.dumps(obj, **kwargs)
        if fmt == "yaml":
            return yaml.dump(obj, **kwargs)
        if fmt == "pickle":
            return pickle.dumps(obj, **kwargs)



