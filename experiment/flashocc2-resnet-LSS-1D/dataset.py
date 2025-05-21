import sys
import os

sys.path.append(os.path.abspath("./"))
from lib.datasets.openocc import KITTI
from lib.cfg.base import DataSetBase

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class data_set(DataSetBase):
    def __init__(self):
        # data config
        self.root_dir = r"C:\Users\11386\Downloads\kitti3d"
        self.drop_last_val = True
        self.split = "trainval"
        self.batch_size = 8  # We used the BN layer, so a value of >=2 is recommended
        self.num_workers = 1

        self.class_map = {"Car": 0, "Pedestrian": 1, "Cyclist": 2}
        self.random_flip = 0.5
        self.random_crop = 0.5
        self.scale = 0.4
        self.shift = 0.1
        self.image_size = [384, 1280]

        # build up dataset
        self.train_set = KITTI(self.root_dir, self.split, self.class_map)
        self.val_set = KITTI(self.root_dir, self.split, self.class_map)
        self.test_set = KITTI(self.root_dir, self.split, self.class_map)

    def get_bath_size(self):
        return self.batch_size

    def get_num_workers(self):
        return self.num_workers

    def get_train_loader(self):
        return DataLoader(
            dataset=self.train_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True, # https://github.com/pytorch/pytorch/issues/13246
            prefetch_factor=2,
        )

    def get_val_loader(self):
        return DataLoader(
            dataset=self.val_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=self.drop_last_val,
        )

    def get_test_loader(self):
        return DataLoader(
            dataset=self.test_set,
            batch_size=1,
            num_workers=0,
            shuffle=True,
            pin_memory=False,
            drop_last=False,
        )


if __name__ == "__main__":
    data = data_set()
    for i, (inputs, targets, info) in enumerate(data.get_train_loader()):
        print(inputs.shape)
        print(targets)
        print(info)
        break
