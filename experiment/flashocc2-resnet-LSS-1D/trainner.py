import sys
import os

sys.path.append(os.path.abspath("./"))

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.cfg.base import TrainerBase


class trainner(TrainerBase):
    def __init__(self):
        self.start_epoch = 0
        self.end_epoch = 1
        self.save_path = r"C:\workspace\github\monolite\experiment\monolite_YOLO11_centernet\checkpoint"
        self.resume_checkpoint = None
        #self.resume_checkpoint = "C:\workspace\github\monolite\experiment\monolite_YOLO11_centernet\checkpoint\model.pth"
        self.log_interval = 1
        self.seed = 114514
        self.amp = True
        self.cudnn = True
        
    def get_end_epoch(self):
        return self.end_epoch

    def get_save_path(self):
        return self.save_path

    def get_log_interval(self):
        return self.log_interval
    
    def get_seed(self):
        return self.seed

    def is_amp(self):
        return self.amp

    def is_cudnn(self):
        return self.cudnn
    
    def get_resume_checkpoint(self):
        return self.resume_checkpoint
    
    def get_start_epoch(self):
        return self.start_epoch
    
    def set_start_epoch(self, start_epoch:int)->None:
        self.start_epoch = start_epoch

if __name__ == "__main__":
    trainer = trainner()
