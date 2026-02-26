# Copyright (c) OpenMMLab. All rights reserved.
from pydantic import BaseModel, field_validator

from flashocc.core.hooks import HOOKS, Hook
from .utils import is_parallel
from torch.nn import SyncBatchNorm

__all__ = ['SyncbnControlHook']


class SyncbnControlConfig(BaseModel):
    """SyncBN 控制钩子配置."""
    syncbn_start_epoch: int = 1

    @field_validator("syncbn_start_epoch")
    @classmethod
    def _epoch_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"syncbn_start_epoch 不能为负, 收到 {v}")
        return v


@HOOKS.register_module()
class SyncbnControlHook(Hook):
    """ """

    def __init__(self, syncbn_start_epoch=1):
        super().__init__()
        cfg = SyncbnControlConfig(syncbn_start_epoch=syncbn_start_epoch)
        self.is_syncbn = False
        self.syncbn_start_epoch = cfg.syncbn_start_epoch

    def cvt_syncbn(self, runner):
        if is_parallel(runner.model.module):
            runner.model.module.module=\
                SyncBatchNorm.convert_sync_batchnorm(runner.model.module.module,
                                                     process_group=None)
        else:
            runner.model.module=\
                SyncBatchNorm.convert_sync_batchnorm(runner.model.module,
                                                     process_group=None)

    def before_train_epoch(self, runner):
        if runner.epoch>= self.syncbn_start_epoch and not self.is_syncbn:
            from flashocc.core.log import logger
            logger.info('start use syncbn')
            self.cvt_syncbn(runner)
            self.is_syncbn=True

