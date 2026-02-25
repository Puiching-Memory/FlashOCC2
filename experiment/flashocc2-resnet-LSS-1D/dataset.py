import sys
import os

sys.path.append(os.path.abspath("./"))
from lib.datasets.openocc import datasetOpenOCC
from lib.cfg.base import DataSetBase

from torch.utils.data import RandomSampler, DistributedSampler
from torchdata.nodes import SamplerWrapper, ParallelMapper, Loader, pin_memory

class datasetPlugin(DataSetBase):
    def __init__(self):
        # data config
        self.root_dir = r"dataset/nuscenes"
        self.num_workers = 1

        # build up dataset
        self.dataset_train = datasetOpenOCC(self.root_dir, "train")
        self.dataset_val = datasetOpenOCC(self.root_dir, "val")
        self.dataset_test = datasetOpenOCC(self.root_dir, "test")
    def get_num_workers(self):
        return self.num_workers

    def get_train_loader(self):
        sampler = RandomSampler(self.dataset_train)
        node = SamplerWrapper(sampler)
        node = ParallelMapper(node, map_fn=self.dataset_train.__getitem__, num_workers=self.num_workers, method="process")
        loader = Loader(node)
        return loader
    
    def get_val_loader(self):
        sampler = RandomSampler(self.dataset_val)
        node = SamplerWrapper(sampler)
        node = ParallelMapper(node, map_fn=self.dataset_val.__getitem__, num_workers=self.num_workers, method="process")
        loader = Loader(node)
        return loader

    def get_test_loader(self):
        sampler = RandomSampler(self.dataset_test)
        node = SamplerWrapper(sampler)
        node = ParallelMapper(node, map_fn=self.dataset_test.__getitem__, num_workers=self.num_workers, method="process")
        loader = Loader(node)
        return loader

if __name__ == "__main__":
    testDataset = datasetPlugin()
    for input in testDataset.get_train_loader():
        print(input)
        break
