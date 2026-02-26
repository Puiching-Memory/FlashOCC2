"""训练钩子注册表 — pydantic 配置 + Registry."""
from pydantic import BaseModel, ConfigDict

from .registry import Registry

HOOKS = Registry("hooks")


class Hook:
    """钩子基类 — 所有钩子生命周期方法."""
    def before_run(self, runner):
        pass
    def after_run(self, runner):
        pass
    def before_epoch(self, runner):
        pass
    def after_epoch(self, runner):
        pass
    def before_iter(self, runner):
        pass
    def after_iter(self, runner):
        pass
    def before_train_epoch(self, runner):
        pass
    def after_train_epoch(self, runner):
        pass
    def before_train_iter(self, runner):
        pass
    def after_train_iter(self, runner):
        pass


class HookConfig(BaseModel):
    """钩子配置基类 — 子类可扩展字段."""
    model_config = ConfigDict(extra="allow")
    type: str
    priority: int = 50


__all__ = ["HOOKS", "Hook", "HookConfig"]
