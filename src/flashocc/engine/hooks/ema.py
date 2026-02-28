# Copyright (c) OpenMMLab. All rights reserved.
# modified from megvii-bevdepth.
import math
import os
from copy import deepcopy
from typing import Optional

import torch
from pydantic import BaseModel, ConfigDict, field_validator

from flashocc.core import load_state_dict
from flashocc.core.dist import master_only
from flashocc.core.log import logger
from flashocc.core.hooks import HOOKS, Hook

from .utils import is_parallel

__all__ = ['ModelEMA']


class EMAConfig(BaseModel):
    """EMA 钩子配置."""
    model_config = ConfigDict(extra="allow")
    init_updates: int = 0
    decay: float = 0.9990
    resume: Optional[str] = None

    @field_validator("decay")
    @classmethod
    def _decay_in_range(cls, v: float) -> float:
        if not 0.0 < v < 1.0:
            raise ValueError(f"decay 应在 (0, 1), 收到 {v}")
        return v


class ModelEMA:
    """Model Exponential Moving Average from https://github.com/rwightman/
    pytorch-image-models Keep a moving average of everything in the model
    state_dict (parameters and buffers).

    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/
    ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training
    schemes to perform well.
    This class is sensitive where it is initialized in the sequence
    of model init, GPU assignment and distributed training wrappers.
    """

    def __init__(self, model, decay=0.9999, updates=0):
        """
        Args:
            model (nn.Module): model to apply EMA.
            decay (float): ema decay reate.
            updates (int): counter of EMA updates.
        """
        # Create EMA(FP32)
        self.ema_model = deepcopy(model).eval()
        self.ema = self.ema_model.module.module if is_parallel(
            self.ema_model.module) else self.ema_model.module
        self.updates = updates
        # decay exponential ramp (to help early epochs)
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, trainer, model):
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = model.module.state_dict() if is_parallel(
                model) else model.state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1.0 - d) * msd[k].detach()


@HOOKS.register
class MEGVIIEMAHook(Hook):
    """EMAHook used in BEVDepth — pydantic 验证配置参数.

    Modified from https://github.com/Megvii-Base
    Detection/BEVDepth/blob/main/callbacks/ema.py.
    """

    def __init__(self, init_updates=0, decay=0.9990, resume=None):
        super().__init__()
        cfg = EMAConfig(init_updates=init_updates, decay=decay, resume=resume)
        self.init_updates = cfg.init_updates
        self.resume = cfg.resume
        self.decay = cfg.decay

    def before_run(self, runner):
        bn_model_list = list()
        bn_model_dist_group_list = list()
        for model_ref in runner.model.modules():
            if type(model_ref).__name__ == 'SyncBatchNorm':
                bn_model_list.append(model_ref)
                bn_model_dist_group_list.append(model_ref.process_group)
                model_ref.process_group = None
        runner.ema_model = ModelEMA(runner.model, self.decay)

        for bn_model, dist_group in zip(bn_model_list,
                                        bn_model_dist_group_list):
            bn_model.process_group = dist_group
        runner.ema_model.updates = self.init_updates

        if self.resume is not None:
            logger.info(f'resume ema checkpoint from {self.resume}')
            cpt = torch.load(self.resume, map_location='cpu')
            load_state_dict(runner.ema_model.ema, cpt['state_dict'])
            runner.ema_model.updates = cpt['updates']

    def after_train_iter(self, runner):
        runner.ema_model.update(runner, runner.model.module)

    def after_train_epoch(self, runner):
        # if self.is_last_epoch(runner):   # 只保存最后一个epoch的ema权重.
        self.save_checkpoint(runner)

    @master_only
    def save_checkpoint(self, runner):
        state_dict = runner.ema_model.ema.state_dict()
        ema_checkpoint = {
            'epoch': runner.epoch,
            'state_dict': state_dict,
            'updates': runner.ema_model.updates
        }
        save_path = f'epoch_{runner.epoch+1}_ema.pth'
        save_path = os.path.join(runner.work_dir, save_path)
        torch.save(ema_checkpoint, save_path)
        logger.info(f'Saving ema checkpoint at {save_path}')
