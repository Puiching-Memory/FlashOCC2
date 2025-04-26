from mmengine.registry import DATASETS
from mmdet3d.datasets import NuScenesDataset
from typing import Union, List
import json
import os
import numpy as np
from scipy.spatial.transform import Rotation as R 

@DATASETS.register_module()
class NuScenesDatasetOccupancy(NuScenesDataset):
    def __init__(self, *args, **kwargs):
        print("NuScenesDatasetOccupancy_args", args, kwargs)

        with open(os.path.join(kwargs["data_root"], "annotations.json")) as f:
            self.annotations = load_annotations(json.load(f))

        self.data_root = kwargs["data_root"]

        super().__init__(*args, **kwargs)

    def parse_data_info(self, info: dict) -> Union[List[dict], dict]:
        # info: dict_keys(['sample_idx', 'token', 'timestamp', 'ego2global', 'images', 'lidar_points', 'instances', 'cam_instances'])
        print(info.keys())
        parsed_info = super().parse_data_info(info)
        # parsed_info: dict_keys(['sample_idx', 'token', 'timestamp', 'ego2global', 'images', 'lidar_points', 'instances', 'cam_instances', 'ann_info', 'eval_ann_info'])
        print(parsed_info.keys())
        print(parsed_info["sample_idx"], parsed_info["token"])
        parsed_info["occ_gt_path"] = os.path.join(
            self.data_root, self.annotations[parsed_info["token"]]["gt_path"]
        )

        intrinsics = np.empty((6, 3, 3))
        extrinsic = np.empty((6, 4, 4))
        cam2ego = np.empty((6, 4, 4))
        ego2global = np.empty((4, 4))
        for index, v in enumerate(
            self.annotations[parsed_info["token"]]["camera_sensor"].values()
        ):
            intrinsics[index] = v["intrinsics"]
            extrinsic[index][:3,3] = v["extrinsic"]["translation"]
            extrinsic[index][:3,:3] = R.from_quat(v["extrinsic"]["rotation"]).as_matrix()
            extrinsic[index][3] = np.array([0,0,0,1])
            cam2ego[index][:3,3] = v["extrinsic"]["translation"]
            cam2ego[index][:3,:3] = R.from_quat(v["ego_pose"]["rotation"]).as_matrix()
            cam2ego[index][3] = np.array([0,0,0,1])

        ego2global[:3,3] = self.annotations[parsed_info["token"]]["ego_pose"]["translation"]
        ego2global[:3,:3] = R.from_quat(self.annotations[parsed_info["token"]]["ego_pose"]["rotation"]).as_matrix()
        ego2global[3] = np.array([0,0,0,1])

        parsed_info["intrinsics"] = intrinsics
        parsed_info["extrinsic"] = extrinsic
        parsed_info["cam2ego"] = cam2ego
        parsed_info["ego2global"] = ego2global

        print("-" * 20)
        return parsed_info


def load_annotations(anno: dict):
    # dict_keys(['train_split', 'val_split', 'scene_infos'])
    #   train_split: ['scene-0001', 'scene-0002', ...]
    #   val_split: ['scene-0001', 'scene-0002', ...]
    #   scene_infos: dict_keys(['scene-0001', 'scene-0002', ...])
    #       dict_keys(['e93e98b63d3b40209056d129dc53ceee', ...])
    #           dict_keys(['timestamp', 'ego_pose', 'camera_sensor', 'gt_path', 'prev', 'next'])
    #               camera_sensor: dict_keys(['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT'])
    #                   dict_keys(['intrinsics', 'extrinsic', 'ego_pose', 'img_path'])
    #                       extrinsic: dict_keys(['translation','rotation'])
    #                       ego_pose: dict_keys(['translation','rotation'])

    output_anno = {}

    # 将scene展平,方便查找
    for scene_id, scene_data in anno["scene_infos"].items():
        for token, token_data in scene_data.items():
            output_anno[token] = token_data

    return output_anno
