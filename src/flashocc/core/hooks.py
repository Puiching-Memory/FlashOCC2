"""训练钩子注册表."""
from .registry import Registry

HOOKS = Registry("hooks")


class Hook:
    """钩子基类."""
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


__all__ = ["HOOKS", "Hook"]
