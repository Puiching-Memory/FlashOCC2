import sys
import os
sys.path.append(os.path.abspath("./"))
from lib.cfg.base import SchedulerBase

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler

class scheduler(SchedulerBase):
    def __init__(self,optimizer):
        self.optimizer = optimizer
    
    def get_scheduler(self):
        return lr_scheduler.CosineAnnealingLR(self.optimizer,T_max=50,eta_min=0,last_epoch=-1)