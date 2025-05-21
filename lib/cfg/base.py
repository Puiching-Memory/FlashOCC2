import abc
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
import numpy as np
from typing import Optional, Tuple, Union, List, Dict, Any


class LossBase(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def __call__(
        self, output: dict[torch.Tensor], target: dict[torch.Tensor]
    ) -> torch.Tensor: ...


class OptimizerBase(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self, model: torch.nn.Module): ...

    @abc.abstractmethod
    def get_optimizer(self) -> torch.optim.Optimizer: ...


class SchedulerBase(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self, optimizer: torch.optim.Optimizer): ...

    @abc.abstractmethod
    def get_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler: ...


class DataSetBase(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def get_train_loader(self) -> DataLoader: ...

    @abc.abstractmethod
    def get_val_loader(self) -> DataLoader: ...

    @abc.abstractmethod
    def get_test_loader(self) -> DataLoader: ...

    @abc.abstractmethod
    def get_bath_size(self) -> int: ...

    @abc.abstractmethod
    def get_num_workers(self) -> int: ...


class TrainerBase(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def get_end_epoch(self) -> int: ...

    @abc.abstractmethod
    def get_save_path(self) -> str: ...

    @abc.abstractmethod
    def get_log_interval(self) -> int: ...

    @abc.abstractmethod
    def get_seed(self) -> int: ...
    
    @abc.abstractmethod
    def get_resume_checkpoint(self) -> Optional[str]: ...
    
    @abc.abstractmethod
    def get_start_epoch(self) -> int: ...
    
    @abc.abstractmethod
    def set_start_epoch(self, epoch: int) -> None: ...

    @abc.abstractmethod
    def is_cudnn(self) -> bool: ...

    @abc.abstractmethod
    def is_amp(self) -> bool: ...


class VisualizerBase(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def decode_output(self) -> dict[np.ndarray]: ...

    @abc.abstractmethod
    def decode_target(self) -> dict[np.ndarray]: ...
