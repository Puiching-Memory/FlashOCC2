# Copyright (c) OpenMMLab. All rights reserved.
import os
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import flashocc
import torch
import cv2
import numpy as np
from pyquaternion import Quaternion

from flashocc.core.log import logger, progress_bar
from flashocc.datasets import DATASETS
from flashocc.constants import OCC_CLASS_NAMES
from .nuscenes_bevdet import NuScenesDatasetBEVDet as NuScenesDataset
from .evaluation.occ_metrics import Metric_mIoU, Metric_FScore
from .evaluation.ray_metrics import main as calc_rayiou
from torch.utils.data import DataLoader, Dataset
from .evaluation.ray_metrics import main_raypq
import glob


# ---- 模块级函数: 供多进程池调用 (避免 pickle self) ----
def _load_gt_worker(occ_path: str):
    """在子进程中解压 GT npz 文件 (bypass GIL)."""
    occ_gt = np.load(os.path.join(occ_path, 'labels.npz'))
    return (
        occ_gt['semantics'],
        occ_gt['mask_lidar'].astype(bool),
        occ_gt['mask_camera'].astype(bool),
    )


# ---------------------------------------------------------------------------
# EgoPoseDataset (原 ego_pose_dataset.py, 仅在 ray-iou 评估中使用)
# ---------------------------------------------------------------------------

def _trans_matrix(T, R):
    """构造 4x4 变换矩阵."""
    tm = np.eye(4)
    tm[:3, :3] = R
    tm[:3, 3] = T
    return tm


class EgoPoseDataset(Dataset):
    """提供 ego pose 信息用于 ray-iou 评估."""

    def __init__(self, data_infos):
        self.data_infos = data_infos
        np.set_printoptions(precision=3, suppress=True)

    @staticmethod
    def get_scene_token(info):
        return info.get('scene_token', '')

    @staticmethod
    def get_ego_from_lidar(info):
        lidar2ego_r = info['lidar2ego_rotation']
        lidar2ego_t = info['lidar2ego_translation']
        return _trans_matrix(
            np.array(lidar2ego_t),
            Quaternion(lidar2ego_r).rotation_matrix)

    @staticmethod
    def get_global_pose(info, inverse=False):
        ego2global_r = info['ego2global_rotation']
        ego2global_t = info['ego2global_translation']
        mat = _trans_matrix(
            np.array(ego2global_t),
            Quaternion(ego2global_r).rotation_matrix)
        if inverse:
            mat = np.linalg.inv(mat)
        return mat

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, idx):
        info = self.data_infos[idx]
        token = info['token']

        ego_from_lidar = self.get_ego_from_lidar(info)
        global_from_ego = self.get_global_pose(info)
        lidar_origin = (global_from_ego @ ego_from_lidar @ np.array([0, 0, 0, 1]))[:3]
        lidar_origin = torch.tensor(lidar_origin, dtype=torch.float32)

        return token, lidar_origin


colors_map = np.array(
    [
        [0,   0,   0, 255],  # 0 undefined
        [255, 158, 0, 255],  # 1 car  orange
        [0, 0, 230, 255],    # 2 pedestrian  Blue
        [47, 79, 79, 255],   # 3 sign  Darkslategrey
        [220, 20, 60, 255],  # 4 CYCLIST  Crimson
        [255, 69, 0, 255],   # 5 traiffic_light  Orangered
        [255, 140, 0, 255],  # 6 pole  Darkorange
        [233, 150, 70, 255], # 7 construction_cone  Darksalmon
        [255, 61, 99, 255],  # 8 bycycle  Red
        [112, 128, 144, 255],# 9 motorcycle  Slategrey
        [222, 184, 135, 255],# 10 building Burlywood
        [0, 175, 0, 255],    # 11 vegetation  Green
        [165, 42, 42, 255],  # 12 trunk  nuTonomy green
        [0, 207, 191, 255],  # 13 curb, road, lane_marker, other_ground
        [75, 0, 75, 255], # 14 walkable, sidewalk
        [255, 0, 0, 255], # 15 unobsrvd
        [0, 0, 0, 0],  # 16 undefined
        [0, 0, 0, 0],  # 16 undefined
    ])


@DATASETS.register
class NuScenesDatasetOccpancy(NuScenesDataset):
    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        input_dict = super(NuScenesDatasetOccpancy, self).get_data_info(index)
        # standard protocol modified from SECOND.Pytorch
        # input_dict['occ_gt_path'] = os.path.join(self.data_root, self.data_infos[index]['occ_path'])
        input_dict['occ_gt_path'] = self.data_infos[index]['occ_path']
        return input_dict

    def evaluate(self, occ_results, runner=None, show_dir=None, **eval_kwargs):
        metric = eval_kwargs['metric'][0]
        logger.info(f"metric = {metric}")
        if metric == 'ray-iou':
            occ_gts = []
            occ_preds = []
            lidar_origins = []
            inst_gts = []
            inst_preds = []
            
            logger.info('Starting Evaluation...')

            data_loader = DataLoader(
                EgoPoseDataset(self.data_infos),
                batch_size=1,
                shuffle=False,
                num_workers=8
            )

            sample_tokens = [info['token'] for info in self.data_infos]

            for i, batch in enumerate(data_loader):
                # if i > 5:
                #     break
                token = batch[0][0]
                output_origin = batch[1]

                data_id = sample_tokens.index(token)
                info = self.data_infos[data_id]
                # occ_gt = np.load(os.path.join(self.data_root, info['occ_path'], 'labels.npz'))
                # occ_gt = np.load(os.path.join(info['occ_path'], 'labels.npz'))
                occ_gt = np.load(os.path.join(info['occ_path'].replace('data/nuscenes/gts/', 'data/nuscenes/occ3d_panoptic/'), 'labels.npz'))
                gt_semantics = occ_gt['semantics']      # (Dx, Dy, Dz)
                mask_lidar = occ_gt['mask_lidar'].astype(bool)      # (Dx, Dy, Dz)
                mask_camera = occ_gt['mask_camera'].astype(bool)    # (Dx, Dy, Dz)
                occ_pred = occ_results[data_id]['pred_occ'].cpu().numpy()     # (Dx, Dy, Dz)
                # occ_pred = occ_results[data_id]['pred_occ']     # (Dx, Dy, Dz)

                lidar_origins.append(output_origin)
                occ_gts.append(gt_semantics)
                occ_preds.append(occ_pred)

                if 'pano_inst' in occ_results[data_id].keys():
                    pano_inst = occ_results[data_id]['pano_inst'].cpu()
                    # pano_inst = torch.from_numpy(occ_results[data_id]['pano_inst'])
                    pano_inst = pano_inst.squeeze(0).numpy()
                    gt_instances = occ_gt['instances']
                    inst_gts.append(gt_instances)
                    inst_preds.append(pano_inst)
                    
            eval_results = calc_rayiou(occ_preds, occ_gts, lidar_origins)
            if len(inst_preds) > 0:
                eval_results.update(main_raypq(occ_preds, occ_gts, inst_preds, inst_gts, lidar_origins))
            # eval_results = main_raypq(occ_preds, occ_gts, inst_preds, inst_gts, lidar_origins)
        else:
            num_classes = 18
            use_lidar_mask = False
            use_image_mask = True

            logger.info('Starting Evaluation...')
            num_samples = len(occ_results)

            # ---- 直接累积混淆矩阵 (比 torchmetrics 快, 省去 tensor 转换) ----
            confmat = np.zeros((num_classes, num_classes), dtype=np.int64)

            # 收集 GT 路径供多进程加载
            occ_paths = [self.data_infos[i]['occ_path'] for i in range(num_samples)]

            # 多进程池: npz 解压是 CPU 密集, ThreadPool 受 GIL 限制;
            # 用 forkserver 避免 fork CUDA 上下文.
            n_workers = min(os.cpu_count() or 8, 32)
            chunksize = max(1, num_samples // (n_workers * 4))
            logger.info(f'GT loading: {n_workers} workers, chunksize={chunksize}')

            mp_ctx = mp.get_context('forkserver')
            with mp_ctx.Pool(processes=n_workers) as pool:
                gt_iter = pool.imap(_load_gt_worker, occ_paths, chunksize=chunksize)

                for index in progress_bar(range(num_samples), desc="Eval"):
                    gt_semantics, mask_lidar, mask_camera = next(gt_iter)

                    occ_pred = occ_results[index]
                    pred = occ_pred['pred_occ'] if (hasattr(occ_pred, 'keys') and 'pred_occ' in occ_pred) else occ_pred

                    # torch tensor -> numpy
                    if hasattr(pred, 'cpu'):
                        pred = pred.cpu().numpy()

                    # 应用 mask
                    if use_image_mask:
                        masked_gt = gt_semantics[mask_camera]
                        masked_pred = pred[mask_camera]
                    elif use_lidar_mask:
                        masked_gt = gt_semantics[mask_lidar]
                        masked_pred = pred[mask_lidar]
                    else:
                        masked_gt = gt_semantics.ravel()
                        masked_pred = pred.ravel()

                    g = masked_gt.astype(np.int64).ravel()
                    p = masked_pred.astype(np.int64).ravel()

                    # 过滤无效值 (GT 中可能有 255 = unlabeled)
                    valid = (g >= 0) & (g < num_classes) & (p >= 0) & (p < num_classes)
                    if not valid.all():
                        g = g[valid]
                        p = p[valid]

                    # bincount 直接累加混淆矩阵 (单次向量化操作, 极快)
                    confmat += np.bincount(
                        g * num_classes + p,
                        minlength=num_classes * num_classes,
                    ).reshape(num_classes, num_classes)

                    if show_dir is not None:
                        info = self.data_infos[index]
                        flashocc.mkdir_or_exist(show_dir)
                        scene_name = [tem for tem in info['occ_path'].split('/') if 'scene-' in tem][0]
                        sample_token = info['token']
                        flashocc.mkdir_or_exist(os.path.join(show_dir, scene_name, sample_token))
                        save_path = os.path.join(show_dir, scene_name, sample_token, 'pred.npz')
                        np.savez_compressed(save_path, pred=pred, gt=gt_semantics, sample_token=sample_token)

            # ---- 从混淆矩阵计算 per-class IoU ----
            tp = np.diag(confmat).astype(np.float64)
            fp = confmat.sum(axis=0).astype(np.float64) - tp
            fn = confmat.sum(axis=1).astype(np.float64) - tp
            iou = tp / (tp + fp + fn + 1e-10)

            class_names = list(OCC_CLASS_NAMES)
            logger.info(f'===> per class IoU of {num_samples} samples:')
            per_class = {}
            for i in range(num_classes - 1):
                iou_val = round(float(iou[i]) * 100, 2)
                logger.info(f'===> {class_names[i]} - IoU = {iou_val}')
                per_class[class_names[i]] = iou_val

            mean_iou = float(np.nanmean(iou[:num_classes - 1]) * 100)
            logger.info(f'===> mIoU of {num_samples} samples: {round(mean_iou, 2)}')

            eval_results = {
                'mIoU': iou,
                'mIoU_mean': mean_iou,
                'per_class_iou': per_class,
                'num_samples': num_samples,
                'confusion_matrix': confmat,
            }

        return eval_results


    def vis_occ(self, semantics):
        # simple visualization of result in BEV
        semantics_valid = np.logical_not(semantics == 17)
        d = np.arange(16).reshape(1, 1, 16)
        d = np.repeat(d, 200, axis=0)
        d = np.repeat(d, 200, axis=1).astype(np.float32)
        d = d * semantics_valid
        selected = np.argmax(d, axis=2)

        selected_torch = torch.from_numpy(selected)
        semantics_torch = torch.from_numpy(semantics)

        occ_bev_torch = torch.gather(semantics_torch, dim=2,
                                     index=selected_torch.unsqueeze(-1))
        occ_bev = occ_bev_torch.numpy()

        occ_bev = occ_bev.flatten().astype(np.int32)
        occ_bev_vis = colors_map[occ_bev].astype(np.uint8)
        occ_bev_vis = occ_bev_vis.reshape(200, 200, 4)[::-1, ::-1, :3]
        occ_bev_vis = cv2.resize(occ_bev_vis,(400,400))
        return occ_bev_vis
