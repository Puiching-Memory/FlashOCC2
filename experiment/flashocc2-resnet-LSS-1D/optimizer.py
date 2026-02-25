import sys
import os
sys.path.append(os.path.abspath("./"))
from lib.cfg.base import OptimizerBase

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class optimizer(OptimizerBase):
    def __init__(self, model):
        self.model = model
    
    def get_optimizer(self):
        #return optim.SGD(self.model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)
        return optim.AdamW(self.model.parameters(), lr=3e-4, betas=(0.9,0.999),weight_decay=1e-4)