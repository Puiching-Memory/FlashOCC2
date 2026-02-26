# Copyright (c) OpenMMLab. All rights reserved.
from pydantic import BaseModel, field_validator

from flashocc.core.hooks import HOOKS, Hook
from .utils import is_parallel

__all__ = ['SequentialControlHook']


class SequentialControlConfig(BaseModel):
    """顺序控制钩子配置."""
    temporal_start_epoch: int = 1

    @field_validator("temporal_start_epoch")
    @classmethod
    def _epoch_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"temporal_start_epoch 不能为负, 收到 {v}")
        return v


@HOOKS.register_module()
class SequentialControlHook(Hook):
    """ """

    def __init__(self, temporal_start_epoch=1):
        super().__init__()
        cfg = SequentialControlConfig(temporal_start_epoch=temporal_start_epoch)
        self.temporal_start_epoch = cfg.temporal_start_epoch

    def set_temporal_flag(self, runner, flag):
        if is_parallel(runner.model.module):
            runner.model.module.module.with_prev=flag
        else:
            runner.model.module.with_prev = flag

    def before_run(self, runner):
        self.set_temporal_flag(runner, False)

    def before_train_epoch(self, runner):
        if runner.epoch > self.temporal_start_epoch:
            self.set_temporal_flag(runner, True)