"""通用类注册器.

使用方式:
    MODELS = Registry("models")

    @MODELS.register
    class MyModel(nn.Module):
        ...

    model = MODELS.build({"type": "MyModel", "hidden": 64})
"""

from __future__ import annotations

import copy
import inspect
from typing import Any


class Registry:
    """简洁的类注册表, 支持 ``register`` 装饰器与 ``build`` 工厂方法."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._module_dict: dict[str, type] = {}

    # ------------------------------------------------------------------
    # 注册
    # ------------------------------------------------------------------
    def register(self, cls_or_name=None, *, force: bool = False):
        """用作装饰器或函数调用来注册一个类.

        用法::

            @REGISTRY.register
            class Foo: ...

            @REGISTRY.register("Bar")
            class Baz: ...

            REGISTRY.register(SomeClass)
        """
        # 直接作为装饰器 @REGISTRY.register
        if cls_or_name is None or (hasattr(cls_or_name, 'upper') and callable(cls_or_name.upper)):
            name = cls_or_name

            def wrapper(cls):
                self._do_register(cls, name=name, force=force)
                return cls

            return wrapper

        # @REGISTRY.register  (无括号)
        if inspect.isclass(cls_or_name):
            self._do_register(cls_or_name, force=force)
            return cls_or_name

        raise TypeError(f"register 期望类或字符串, 收到 {type(cls_or_name)}")

    def _do_register(self, cls, *, name: str | None = None, force: bool = False):
        reg_name = name or cls.__name__
        if reg_name in self._module_dict and not force:
            raise KeyError(f"'{reg_name}' 已在 {self.name} 中注册")
        self._module_dict[reg_name] = cls

    # ------------------------------------------------------------------
    # 构建
    # ------------------------------------------------------------------
    def build(self, cfg: dict, default_args: dict | None = None) -> Any:
        """根据 cfg['type'] 查找注册类并实例化.

        Args:
            cfg: 至少包含 ``type`` 键.
            default_args: 默认参数, 优先级低于 cfg.
        """
        # 将 Config / DictConfig / ListConfig 转为纯 Python 容器
        if hasattr(cfg, 'to_dict'):
            cfg = cfg.to_dict()
        elif hasattr(cfg, '_metadata'):
            # OmegaConf DictConfig: has _metadata attribute
            from omegaconf import OmegaConf
            cfg = OmegaConf.to_container(cfg, resolve=True)
        else:
            cfg = copy.deepcopy(cfg)
        if default_args:
            for k, v in default_args.items():
                cfg.setdefault(k, v)
        type_name = cfg.pop("type")
        cls = self.get(type_name)
        return cls(**cfg)

    def get(self, key: str) -> type:
        if key not in self._module_dict:
            raise KeyError(f"'{key}' 未在 {self.name} 注册表中找到. "
                           f"可用: {list(self._module_dict.keys())}")
        return self._module_dict[key]

    def __contains__(self, key: str) -> bool:
        return key in self._module_dict

    def __repr__(self) -> str:
        return (f"Registry(name={self.name!r}, "
                f"items={list(self._module_dict.keys())})")
