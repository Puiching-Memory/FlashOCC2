import sys
import os
sys.path.append(os.path.abspath("./"))

import numpy as np
import torch
import torch.utils.data
from torch.utils.data import RandomSampler, DistributedSampler
from torchdata.nodes import SamplerWrapper, ParallelMapper, Loader, pin_memory
import torchvision
from torchvision.transforms import v2
import pickle
from pathlib import Path

from lib.utils.logger import logger

class datasetOpenOCC(torch.utils.data.Dataset):
    """
    ### documentation
    https://github.com/OpenDriveLab/OccNet?tab=readme-ov-file#openocc-dataset

    ### path structure
    ```
    nuscenes
    ├── maps
    ├── nuscenes_infos_train_occ.pkl
    ├── nuscenes_infos_val_occ.pkl
    ├── nuscenes_infos_test_occ.pkl
    ├── openocc_v2
    ├── samples
    ├── v1.0-test
    └── v1.0-trainval
    ```
    """
    def __init__(self, root_dir:str, split:str)->None:
        self.root_dir = root_dir
        self.split = split

        # 载入数据集
        assert split in ["train", "val", "test"]

        logger.debug(f"Loading {Path(root_dir) / f'nuscenes_infos_{split}_occ.pkl'} ...")
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
    
    def getOCCTensor(self, occ_path: str) -> tuple[torch.Tensor,...]:
        assert os.path.exists(occ_path)
        with np.load(occ_path) as f:
            # NpzFile 'labels.npz' with keys: instances, semantics, flow
            occ_instances = torch.tensor(f['instances'])
            occ_semantics = torch.tensor(f['semantics'])
            occ_flow = torch.tensor(f['flow'])
        return occ_instances, occ_semantics, occ_flow

    def __len__(self)->int:
        return len(self.data['infos'])
    def __getitem__(self, index:int):
        clip = self.data['infos'][index]

        # init data space
        images = torch.zeros((6,3,384,1280),dtype=torch.float64)
        sensor2ego_translation = torch.zeros((6,3),dtype=torch.float64)
        sensor2ego_rotation = torch.zeros((6,4),dtype=torch.float64)
        cam_ego2global_translation = torch.zeros((6,3),dtype=torch.float64)
        cam_ego2global_rotation = torch.zeros((6,4),dtype=torch.float64)
        sensor2lidar_rotation = torch.zeros((6,3,3),dtype=torch.float64)
        sensor2lidar_translation = torch.zeros((6,3),dtype=torch.float64)
        cam_intrinsic = torch.zeros((6,3,3),dtype=torch.float64)
        timestamp = torch.zeros((6),dtype=torch.uint64)

        # CAM_FRONT -> CAM_FRONT_RIGHT -> CAM_FRONT_LEFT -> CAM_BACK -> CAM_BACK_LEFT -> CAM_BACK_RIGHT
        for index, (cam_k, cam_v) in enumerate(clip['cams'].items()):
            # load image from data_path
            p = Path(cam_v['data_path'])
            p = p.relative_to(str(Path(*p.parts[:p.parts.index("nuscenes")+1])))
            images[index] = self.getImageTensor(str(self.root_dir / p))

            # load camera intrinsics and extrinsics
            sensor2ego_translation[index] = torch.tensor(cam_v['sensor2ego_translation'],dtype=torch.float64)
            sensor2ego_rotation[index] = torch.tensor(cam_v['sensor2ego_rotation'],dtype=torch.float64)
            cam_ego2global_translation[index] = torch.tensor(cam_v['ego2global_translation'],dtype=torch.float64)
            cam_ego2global_rotation[index] = torch.tensor(cam_v['ego2global_rotation'],dtype=torch.float64)
            sensor2lidar_rotation[index] = torch.tensor(cam_v['sensor2lidar_rotation'],dtype=torch.float64)
            sensor2lidar_translation[index] = torch.tensor(cam_v['sensor2lidar_translation'],dtype=torch.float64)
            cam_intrinsic[index] = torch.tensor(cam_v['cam_intrinsic'],dtype=torch.float64)

            # load timestamp
            timestamp[index] = torch.tensor(cam_v['timestamp'],dtype=torch.uint64)
        
        # load vehicle intrinsics and extrinsics
        lidar2ego_translation = torch.tensor(clip['lidar2ego_translation'])
        lidar2ego_rotation = torch.tensor(clip['lidar2ego_rotation'])
        lidar_ego2global_translation = torch.tensor(clip['ego2global_translation'])
        lodar_ego2global_rotation = torch.tensor(clip['ego2global_rotation'])

        # load OCC GT from occ_path
        p = Path(clip['occ_path'])
        p = p.relative_to(str(Path(*p.parts[:p.parts.index("nuscenes")+1])))
        occ_instances, occ_semantics, occ_flow = self.getOCCTensor(str(self.root_dir / p))

        return (
            images,
            (
                sensor2ego_translation,
                sensor2ego_rotation,
                cam_ego2global_translation,
                cam_ego2global_rotation,
                sensor2lidar_rotation,
                sensor2lidar_translation,
                cam_intrinsic,
            ),
            (
                lidar2ego_translation,
                lidar2ego_rotation,
                lidar_ego2global_translation,
                lodar_ego2global_rotation
            ),
            (
                occ_instances,
                occ_semantics,
                occ_flow
            )
            )


if __name__ == "__main__":
    from pyinstrument import Profiler
    from tqdm import tqdm

    profiler = Profiler()
    profiler.start()

    dataset = datasetOpenOCC("dataset/nuscenes","val")
    sampler = RandomSampler(dataset)

    node = SamplerWrapper(sampler)
    node = ParallelMapper(node, map_fn=dataset.__getitem__, num_workers=1, method="process")
    loader = Loader(node)
    
    for (
        images,
        (
            sensor2ego_translation,
            sensor2ego_rotation,
            cam_ego2global_translation,
            cam_ego2global_rotation,
            sensor2lidar_rotation,
            sensor2lidar_translation,
            cam_intrinsic,
        ),
        (
            lidar2ego_translation,
            lidar2ego_rotation,
            lidar_ego2global_translation,
            lodar_ego2global_rotation
        ),
        (
            occ_instances,
            occ_semantics,
            occ_flow
        )
        ) in tqdm(loader,total=len(dataset)):
        pass

    profiler.stop()
    profiler.print()

    with open("profiler.html", "w") as f:
        f.write(profiler.output_html())
