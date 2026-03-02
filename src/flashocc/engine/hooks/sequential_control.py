# Copyright (c) OpenMMLab. All rights reserved.
from flashocc.core.hooks import HOOKS, Hook
from flashocc.engine.hooks.utils import is_parallel as _is_parallel

__all__ = ['SequentialControlHook']


@HOOKS.register
class SequentialControlHook(Hook):
    """在指定 epoch 后启用 temporal 模式."""

    def __init__(self, temporal_start_epoch=1):
        super().__init__()
        assert temporal_start_epoch >= 0, f"temporal_start_epoch 不能为负, 收到 {temporal_start_epoch}"
        self.temporal_start_epoch = int(temporal_start_epoch)

    def set_temporal_flag(self, runner, flag):
        if _is_parallel(runner.model.module):
            runner.model.module.module.with_prev = flag
        else:
            runner.model.module.with_prev = flag

    def before_run(self, runner):
        self.set_temporal_flag(runner, False)

    def before_train_epoch(self, runner):
        if runner.epoch > self.temporal_start_epoch:
            self.set_temporal_flag(runner, True)