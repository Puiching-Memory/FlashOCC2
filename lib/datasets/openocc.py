import os
import numpy as np
import torch
import torch.utils.data
from torch.utils.data import RandomSampler, DistributedSampler
from torchdata.nodes import SamplerWrapper, ParallelMapper, Loader, pin_memory
import torchvision
from torchvision.transforms import v2
import time
import sys
import pickle
from pathlib import Path
import cv2

sys.path.append(str(Path().resolve()))

class datasetOpenOCC(torch.utils.data.Dataset):
    """
    doc: https://github.com/OpenDriveLab/OccNet?tab=readme-ov-file#openocc-dataset

    """
    def __init__(self, root_dir:str, split:str)->None:
        self.root_dir = root_dir
        self.split = split

        # 载入数据集
        assert split in ["train", "val"]

        with open(Path(root_dir) / f"nuscenes_infos_{split}_occ.pkl","rb") as f:
            self.data = pickle.load(f)
        
        # dict_keys(['infos', 'metadata'])
        #   metadata: {'version': 'v1.0-trainval'}
        #   infos: list len=6019
        #       dict_keys(['lidar_path', 'token', 'sweeps', 'cams', 'lidar2ego_translation', 'lidar2ego_rotation', 'ego2global_translation', 'ego2global_rotation', 'timestamp', 'gt_boxes', 'gt_names', 'gt_velocity', 'num_lidar_pts', 'num_radar_pts', 'valid_flag', 'occ_path'])
        # import pprint
        # pprint.pp(data['infos'][0])

        # split_dir = os.path.join(root_dir, f"nuscenes_infos_{split}_occ.pkl")
        # self.idx_list = [x.strip() for x in open(split_dir).readlines()]

        # 图像转换
        self.image_transforms = v2.Compose(
            [
                v2.Resize(size=(384, 1280)),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def getImageTensor(self, image_path: str) -> torch.Tensor:
        assert os.path.exists(image_path)
        image = torchvision.io.decode_image(image_path)  # (C,H,W)
        image = self.image_transforms(image)
        return image
    def __len__(self)->int:
        return len(self.data['infos'])
    def __getitem__(self, index:int):
        dataload_time = time.perf_counter_ns()
        clip = self.data['infos'][index]

        images = torch.zeros((6,3,384,1280),dtype=torch.float32)
        for index, (cam_k, cam_v) in enumerate(clip['cams'].items()):
            #print(cam_k,cam_v)
            p = Path(cam_v['data_path'])
            p = p.relative_to(str(Path(*p.parts[:p.parts.index("nuscenes")+1])))
            images[index] = self.getImageTensor(str(self.root_dir / p))
        
        return images


if __name__ == "__main__":
    from pyinstrument import Profiler
    from tqdm import tqdm

    profiler = Profiler()
    profiler.start()

    dataset = datasetOpenOCC("dataset/nuscenes","val")
    sampler = RandomSampler(dataset)

    node = SamplerWrapper(sampler)
    node = ParallelMapper(node, map_fn=dataset.__getitem__, num_workers=16, method="process")
    loader = Loader(node)
    
    for images in tqdm(loader,total=len(dataset)):
        images = images.to("cuda:0")
        print(images.shape,images.dtype,images.device)

    profiler.stop()
    profiler.print()

    with open("profiler.html", "w") as f:
        f.write(profiler.output_html())
