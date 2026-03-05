# Copyright (c) OpenMMLab. All rights reserved.
import os

import numpy as np
import torch
from pyquaternion import Quaternion

from flashocc.core.bbox.points import BasePoints, get_points_type
from flashocc.datasets.pipelines.base import LoadAnnotations, LoadImageFromFile
from flashocc.core.bbox.bbox import LiDARInstance3DBoxes
from flashocc.datasets.builder import PIPELINES
from flashocc.constants import IMAGENET_MEAN, IMAGENET_STD
from flashocc.engine.parallel import DataContainer as DC


@PIPELINES.register
class PrepareImageInputs(object):
    """准备多视角图像输入 — DALI GPU 解码版.

    Workers 只读取 JPEG 原始字节 + 计算增广参数/矩阵 (CPU),
    实际图像解码和变换延迟到 trainer 主线程的 GPU 上批量执行.
    """

    def __init__(
            self,
            data_config,
            is_train=False,
            sequential=False,
    ):
        self.is_train = is_train
        self.data_config = data_config
        self.sequential = sequential

    def choose_cams(self):
        """
        Returns:
            cam_names: List[CAM_Name0, CAM_Name1, ...]
        """
        if self.is_train and self.data_config.Ncams < len(
                self.data_config.cams):
            cam_names = np.random.choice(
                self.data_config.cams,
                self.data_config.Ncams,
                replace=False)
        else:
            cam_names = self.data_config.cams
        return cam_names

    def sample_augmentation(self, H, W, flip=None, scale=None):
        """
        Args:
            H:
            W:
            flip:
            scale:
        Returns:
            resize: resize比例float.
            resize_dims: (resize_W, resize_H)
            crop: (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip: 0 / 1
            rotate: 随机旋转角度float
        """
        fH, fW = self.data_config.input_size
        if self.is_train:
            resize = float(fW) / float(W)
            resize += np.random.uniform(*self.data_config.resize)    # resize的比例, 位于[fW/W − 0.06, fW/W + 0.11]之间.
            resize_dims = (int(W * resize), int(H * resize))            # resize后的size
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.data_config.crop_h)) *
                         newH) - fH     # s * H - H_in
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))       # max(0, s * W - fW)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = self.data_config.flip and np.random.choice([0, 1])
            rotate = np.random.uniform(*self.data_config.rot)
        else:
            resize = float(fW) / float(W)
            if scale is not None:
                resize += scale
            else:
                resize += self.data_config.resize_test
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_config.crop_h)) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False if flip is None else flip
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def img_transform_core(self, img, resize_dims, crop, flip, rotate):
        # DALI 版: 不做实际变换, 仅返回原图 (变换延迟到 GPU)
        return img

    def get_rot(self, h):
        return torch.Tensor([
            [np.cos(h), np.sin(h)],
            [-np.sin(h), np.cos(h)],
        ])

    def img_transform(self, post_rot, post_tran, resize, resize_dims,
                      crop, flip, rotate):
        """计算图像增广对应的 post_rot / post_tran 矩阵 (纯数学, 不操作图像).

        Args:
            post_rot: torch.eye(2)
            post_tran: torch.eye(2)
            resize: float, resize的比例.
            resize_dims: Tuple(W, H), resize后的图像尺寸
            crop: (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip: bool
            rotate: float 旋转角度
        Returns:
            post_rot: Tensor (2, 2)
            post_tran: Tensor (2, )
        """
        # post-homography transformation
        # 将上述变换以矩阵表示.
        post_rot *= resize
        post_tran -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            post_rot = A.matmul(post_rot)
            post_tran = A.matmul(post_tran) + b
        A = self.get_rot(rotate / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        post_rot = A.matmul(post_rot)
        post_tran = A.matmul(post_tran) + b

        return post_rot, post_tran

    def get_sensor_transforms(self, info, cam_name):
        """
        Args:
            info:
            cam_name: 当前要读取的CAM.
        Returns:
            sensor2ego: (4, 4)
            ego2global: (4, 4)
        """
        w, x, y, z = info['cams'][cam_name]['sensor2ego_rotation']      # 四元数格式
        # sensor to ego
        sensor2ego_rot = torch.Tensor(
            Quaternion(w, x, y, z).rotation_matrix)     # (3, 3)
        sensor2ego_tran = torch.Tensor(
            info['cams'][cam_name]['sensor2ego_translation'])   # (3, )
        sensor2ego = sensor2ego_rot.new_zeros((4, 4))
        sensor2ego[3, 3] = 1
        sensor2ego[:3, :3] = sensor2ego_rot
        sensor2ego[:3, -1] = sensor2ego_tran

        # ego to global
        w, x, y, z = info['cams'][cam_name]['ego2global_rotation']      # 四元数格式
        ego2global_rot = torch.Tensor(
            Quaternion(w, x, y, z).rotation_matrix)     # (3, 3)
        ego2global_tran = torch.Tensor(
            info['cams'][cam_name]['ego2global_translation'])   # (3, )
        ego2global = ego2global_rot.new_zeros((4, 4))
        ego2global[3, 3] = 1
        ego2global[:3, :3] = ego2global_rot
        ego2global[:3, -1] = ego2global_tran
        return sensor2ego, ego2global

    def get_inputs(self, results, flip=None, scale=None):
        """读取 JPEG 字节 + 计算增广矩阵 (CPU 端, 不做图像解码).

        Returns (存入 results dict):
            img_inputs: (placeholder, sensor2egos, ego2globals, intrins, post_rots, post_trans)
            jpeg_bytes: list[bytes]  — N_cam 个 JPEG 原始字节
            img_aug_params: list[tuple] — N_cam 个 (resize_dims, crop, flip, rotate)
        """
        jpeg_bytes = []
        img_aug_params = []
        sensor2egos = []
        ego2globals = []
        intrins = []
        post_rots = []
        post_trans = []
        cam_names = self.choose_cams()
        results['cam_names'] = cam_names

        src_H, src_W = self.data_config.src_size   # (900, 1600)

        for cam_name in cam_names:
            cam_data = results['curr']['cams'][cam_name]
            filename = cam_data['data_path']

            # ---- 读取 JPEG 原始字节 (快速 I/O, 不解码) ----
            with open(filename, 'rb') as f:
                jpeg_data = f.read()
            jpeg_bytes.append(jpeg_data)

            # 初始化图像增广的旋转和平移矩阵
            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)
            # 当前相机内参
            intrin = torch.Tensor(cam_data['cam_intrinsic'])

            # 获取当前相机的sensor2ego(4x4), ego2global(4x4)矩阵.
            sensor2ego, ego2global = \
                self.get_sensor_transforms(results['curr'], cam_name)

            # image view augmentation (resize, crop, horizontal flip, rotate)
            img_augs = self.sample_augmentation(
                H=src_H, W=src_W, flip=flip, scale=scale)
            resize, resize_dims, crop, flip, rotate = img_augs

            # 保存增广参数 (GPU 解码后应用)
            img_aug_params.append((resize_dims, crop, flip, rotate))

            # 计算 post_rot, post_tran 矩阵 (纯数学, 与原逻辑完全一致)
            post_rot2, post_tran2 = \
                self.img_transform(post_rot, post_tran,
                                   resize=resize,
                                   resize_dims=resize_dims,
                                   crop=crop,
                                   flip=flip,
                                   rotate=rotate)

            # for convenience, make augmentation matrices 3x3
            post_tran = torch.zeros(3)
            post_rot = torch.eye(3)
            post_tran[:2] = post_tran2
            post_rot[:2, :2] = post_rot2

            if self.sequential:
                assert 'adjacent' in results
                for adj_info in results['adjacent']:
                    filename_adj = adj_info['cams'][cam_name]['data_path']
                    with open(filename_adj, 'rb') as f:
                        jpeg_bytes.append(f.read())
                    img_aug_params.append((resize_dims, crop, flip, rotate))

            intrins.append(intrin)
            sensor2egos.append(sensor2ego)
            ego2globals.append(ego2global)
            post_rots.append(post_rot)
            post_trans.append(post_tran)

        if self.sequential:
            for adj_info in results['adjacent']:
                post_trans.extend(post_trans[:len(cam_names)])
                post_rots.extend(post_rots[:len(cam_names)])
                intrins.extend(intrins[:len(cam_names)])

                for cam_name in cam_names:
                    sensor2ego, ego2global = \
                        self.get_sensor_transforms(adj_info, cam_name)
                    sensor2egos.append(sensor2ego)
                    ego2globals.append(ego2global)

        # Placeholder for imgs — 实际数据将在 GPU 端由 DALI 填充
        fH, fW = self.data_config.input_size   # (256, 704)
        n_views = len(sensor2egos)
        imgs_placeholder = torch.zeros(n_views, 3, fH, fW)

        sensor2egos = torch.stack(sensor2egos)
        ego2globals = torch.stack(ego2globals)
        intrins = torch.stack(intrins)
        post_rots = torch.stack(post_rots)
        post_trans = torch.stack(post_trans)
        results['canvas'] = []

        return (imgs_placeholder, sensor2egos, ego2globals, intrins,
                post_rots, post_trans), jpeg_bytes, img_aug_params

    def __call__(self, results):
        img_inputs, jpeg_bytes, img_aug_params = self.get_inputs(results)
        results['img_inputs'] = img_inputs
        # DC(cpu_only=True) — collate 时收集为 list, scatter 时保留在 CPU
        results['jpeg_bytes'] = DC(jpeg_bytes, stack=False, cpu_only=True)
        results['img_aug_params'] = DC(img_aug_params, stack=False, cpu_only=True)
        return results


@PIPELINES.register
class LoadAnnotationsBEVDepth(object):
    def __init__(self, bda_aug_conf, classes, is_train=True):
        self.bda_aug_conf = bda_aug_conf
        self.is_train = is_train
        self.classes = classes

    def sample_bda_augmentation(self):
        """Generate bda augmentation values based on bda_config."""
        if self.is_train:
            rotate_bda = np.random.uniform(*self.bda_aug_conf.rot_lim)
            scale_bda = np.random.uniform(*self.bda_aug_conf.scale_lim)
            flip_dx = np.random.uniform() < self.bda_aug_conf.flip_dx_ratio
            flip_dy = np.random.uniform() < self.bda_aug_conf.flip_dy_ratio
        else:
            rotate_bda = 0
            scale_bda = 1.0
            flip_dx = False
            flip_dy = False
        return rotate_bda, scale_bda, flip_dx, flip_dy

    def bev_transform(self, gt_boxes, rotate_angle, scale_ratio, flip_dx,
                      flip_dy):
        """
        Args:
            gt_boxes: (N, 9)
            rotate_angle:
            scale_ratio:
            flip_dx: bool
            flip_dy: bool

        Returns:
            gt_boxes: (N, 9)
            rot_mat: (3, 3）
        """
        rotate_angle = torch.tensor(rotate_angle / 180 * np.pi)
        rot_sin = torch.sin(rotate_angle)
        rot_cos = torch.cos(rotate_angle)
        rot_mat = torch.Tensor([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0],
                                [0, 0, 1]])
        scale_mat = torch.Tensor([[scale_ratio, 0, 0], [0, scale_ratio, 0],
                                  [0, 0, scale_ratio]])
        flip_mat = torch.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        if flip_dx:     # 沿着y轴翻转
            flip_mat = flip_mat @ torch.Tensor([[-1, 0, 0], [0, 1, 0],
                                                [0, 0, 1]])
        if flip_dy:     # 沿着x轴翻转
            flip_mat = flip_mat @ torch.Tensor([[1, 0, 0], [0, -1, 0],
                                                [0, 0, 1]])
        rot_mat = flip_mat @ (scale_mat @ rot_mat)    # 变换矩阵(3, 3)
        if gt_boxes.shape[0] > 0:
            gt_boxes[:, :3] = (
                rot_mat @ gt_boxes[:, :3].unsqueeze(-1)).squeeze(-1)     # 变换后的3D框中心坐标
            gt_boxes[:, 3:6] *= scale_ratio    # 变换后的3D框尺寸
            gt_boxes[:, 6] += rotate_angle     # 旋转后的3D框的方位角
            # 翻转也会进一步改变方位角
            if flip_dx:
                gt_boxes[:, 6] = 2 * torch.asin(torch.tensor(1.0)) - gt_boxes[:, 6]
            if flip_dy:
                gt_boxes[:, 6] = -gt_boxes[:, 6]
            gt_boxes[:, 7:] = (
                rot_mat[:2, :2] @ gt_boxes[:, 7:].unsqueeze(-1)).squeeze(-1)
        return gt_boxes, rot_mat

    def __call__(self, results):
        gt_boxes, gt_labels = results['ann_infos']      # (N_gt, 9),  (N_gt, )
        gt_boxes = torch.Tensor(np.array(gt_boxes))
        gt_labels = torch.tensor(np.array(gt_labels))
        rotate_bda, scale_bda, flip_dx, flip_dy = self.sample_bda_augmentation()

        bda_mat = torch.zeros(4, 4)
        bda_mat[3, 3] = 1
        # gt_boxes: (N, 9)  BEV增广变换后的3D框
        # bda_rot: (3, 3)   BEV增广矩阵, 包括旋转、缩放和翻转.
        gt_boxes, bda_rot = self.bev_transform(gt_boxes, rotate_bda, scale_bda,
                                               flip_dx, flip_dy)
        bda_mat[:3, :3] = bda_rot

        if len(gt_boxes) == 0:
            gt_boxes = torch.zeros(0, 9)
        results['gt_bboxes_3d'] = \
            LiDARInstance3DBoxes(gt_boxes, box_dim=gt_boxes.shape[-1],
                                 origin=(0.5, 0.5, 0.5))
        results['gt_labels_3d'] = gt_labels

        imgs, sensor2egos, ego2globals, intrins = results['img_inputs'][:4]
        post_rots, post_trans = results['img_inputs'][4:]
        results['img_inputs'] = (imgs, sensor2egos, ego2globals, intrins, post_rots,
                                 post_trans, bda_rot)

        results['flip_dx'] = flip_dx
        results['flip_dy'] = flip_dy
        results['rotate_bda'] = rotate_bda
        results['scale_bda'] = scale_bda

        # if 'voxel_semantics' in results:
        #     if flip_dx:
        #         results['voxel_semantics'] = results['voxel_semantics'][::-1, ...].copy()
        #         results['mask_lidar'] = results['mask_lidar'][::-1, ...].copy()
        #         results['mask_camera'] = results['mask_camera'][::-1, ...].copy()
        #     if flip_dy:
        #         results['voxel_semantics'] = results['voxel_semantics'][:, ::-1, ...].copy()
        #         results['mask_lidar'] = results['mask_lidar'][:, ::-1, ...].copy()
        #         results['mask_camera'] = results['mask_camera'][:, ::-1, ...].copy()

        return results


@PIPELINES.register
class PointToMultiViewDepth(object):
    def __init__(self, grid_config, downsample=1):
        self.downsample = downsample
        self.grid_config = grid_config

    def points2depthmap(self, points, height, width):
        """
        Args:
            points: (N_points, 3):  3: (u, v, d)
            height: int
            width: int

        Returns:
            depth_map：(H, W)
        """
        height, width = height // self.downsample, width // self.downsample
        depth_map = torch.zeros((height, width), dtype=torch.float32)
        coor = torch.round(points[:, :2] / self.downsample)     # (N_points, 2)  2: (u, v)
        depth = points[:, 2]    # (N_points, )哦
        kept1 = (coor[:, 0] >= 0) & (coor[:, 0] < width) & (
            coor[:, 1] >= 0) & (coor[:, 1] < height) & (
                depth < self.grid_config.depth[1]) & (
                    depth >= self.grid_config.depth[0])
        # 获取有效投影点.
        coor, depth = coor[kept1], depth[kept1]    # (N, 2), (N, )
        ranks = coor[:, 0] + coor[:, 1] * width
        sort = (ranks + depth / 100.).argsort()
        coor, depth, ranks = coor[sort], depth[sort], ranks[sort]
        kept2 = torch.ones(coor.shape[0], device=coor.device, dtype=torch.bool)
        kept2[1:] = (ranks[1:] != ranks[:-1])
        coor, depth = coor[kept2], depth[kept2]
        coor = coor.to(torch.long)
        depth_map[coor[:, 1], coor[:, 0]] = depth
        return depth_map

    def __call__(self, results):
        points_lidar = results['points']
        imgs, sensor2egos, ego2globals, intrins = results['img_inputs'][:4]
        post_rots, post_trans, bda = results['img_inputs'][4:]
        # 使用 placeholder tensor 的 shape 获取目标图像尺寸
        img_H, img_W = imgs.shape[2], imgs.shape[3]
        depth_map_list = []
        for cid in range(len(results['cam_names'])):
            cam_name = results['cam_names'][cid]    # CAM_TYPE
            # 猜测liadr和cam不是严格同步的，因此lidar_ego和cam_ego可能会不一致.
            # 因此lidar-->cam的路径不采用:   lidar --> ego --> cam
            # 而是： lidar --> lidar_ego --> global --> cam_ego --> cam
            lidar2lidarego = np.eye(4, dtype=np.float32)
            lidar2lidarego[:3, :3] = Quaternion(
                results['curr']['lidar2ego_rotation']).rotation_matrix
            lidar2lidarego[:3, 3] = results['curr']['lidar2ego_translation']
            lidar2lidarego = torch.from_numpy(lidar2lidarego)

            lidarego2global = np.eye(4, dtype=np.float32)
            lidarego2global[:3, :3] = Quaternion(
                results['curr']['ego2global_rotation']).rotation_matrix
            lidarego2global[:3, 3] = results['curr']['ego2global_translation']
            lidarego2global = torch.from_numpy(lidarego2global)

            cam2camego = np.eye(4, dtype=np.float32)
            cam2camego[:3, :3] = Quaternion(
                results['curr']['cams'][cam_name]
                ['sensor2ego_rotation']).rotation_matrix
            cam2camego[:3, 3] = results['curr']['cams'][cam_name][
                'sensor2ego_translation']
            cam2camego = torch.from_numpy(cam2camego)

            camego2global = np.eye(4, dtype=np.float32)
            camego2global[:3, :3] = Quaternion(
                results['curr']['cams'][cam_name]
                ['ego2global_rotation']).rotation_matrix
            camego2global[:3, 3] = results['curr']['cams'][cam_name][
                'ego2global_translation']
            camego2global = torch.from_numpy(camego2global)

            cam2img = np.eye(4, dtype=np.float32)
            cam2img = torch.from_numpy(cam2img)
            cam2img[:3, :3] = intrins[cid]

            # lidar --> lidar_ego --> global --> cam_ego --> cam
            lidar2cam = torch.inverse(camego2global.matmul(cam2camego)).matmul(
                lidarego2global.matmul(lidar2lidarego))
            lidar2img = cam2img.matmul(lidar2cam)
            points_img = points_lidar.tensor[:, :3].matmul(
                lidar2img[:3, :3].T) + lidar2img[:3, 3].unsqueeze(0)     # (N_points, 3)  3: (ud, vd, d)
            points_img = torch.cat(
                [points_img[:, :2] / points_img[:, 2:3], points_img[:, 2:3]],
                1)      # (N_points, 3):  3: (u, v, d)

            # 再考虑图像增广
            points_img = points_img.matmul(
                post_rots[cid].T) + post_trans[cid:cid + 1, :]      # (N_points, 3):  3: (u, v, d)
            depth_map = self.points2depthmap(points_img,
                                             img_H,     # H
                                             img_W      # W
                                             )
            depth_map_list.append(depth_map)
        depth_map = torch.stack(depth_map_list)
        results['gt_depth'] = depth_map
        return results


@PIPELINES.register
class LoadOccGTFromFile(object):
    def __call__(self, results):
        occ_gt_path = results['occ_gt_path']
        occ_gt_path = os.path.join(occ_gt_path, "labels.npz")

        occ_labels = np.load(occ_gt_path)
        semantics = occ_labels['semantics']
        mask_lidar = occ_labels['mask_lidar']
        mask_camera = occ_labels['mask_camera']

        semantics = torch.from_numpy(semantics)
        mask_lidar = torch.from_numpy(mask_lidar)
        mask_camera = torch.from_numpy(mask_camera)

        if results.get('flip_dx', False):
            semantics = torch.flip(semantics, [0])
            mask_lidar = torch.flip(mask_lidar, [0])
            mask_camera = torch.flip(mask_camera, [0])

        if results.get('flip_dy', False):
            semantics = torch.flip(semantics, [1])
            mask_lidar = torch.flip(mask_lidar, [1])
            mask_camera = torch.flip(mask_camera, [1])

        results['voxel_semantics'] = semantics
        results['mask_lidar'] = mask_lidar
        results['mask_camera'] = mask_camera

        return results


@PIPELINES.register
class LoadSkyMask(object):
    """加载 SAM3 离线生成的天空二值掩码.

    SAM3 掩码目录结构与 nuScenes 的 samples/ 相同:
        sky_mask_root/samples/CAM_XXX/<same_filename_stem>.png
    掩码为 uint8 灰度图, 255=天空, 0=前景.

    Args:
        sky_mask_root (str): SAM3 掩码根目录, 例如 "data/SAM3".
        downsample (int): 下采样倍率, 与 LSSViewTransformer 的 downsample 保持一致.
            直接使用 nearest-neighbor resize 降低 mask 分辨率到特征图尺寸.
    """

    def __init__(self, sky_mask_root='data/SAM3', downsample=16,
                 input_size=(256, 704)):
        self.sky_mask_root = sky_mask_root
        self.downsample = downsample
        # 目标特征图尺寸 (固定, 与 LSSViewTransformer 一致)
        self.feat_H = input_size[0] // downsample
        self.feat_W = input_size[1] // downsample

    def __call__(self, results):
        from PIL import Image

        cam_names = results.get('cam_names', [])
        masks = []
        for cam_name in cam_names:
            cam_data = results['curr']['cams'][cam_name]
            # 原始图像路径形如: data/nuScenes/samples/CAM_FRONT/xxx.jpg
            orig_path = cam_data['data_path']
            # 从路径中提取 samples/CAM_XXX/stem
            parts = orig_path.replace('\\', '/').split('/')
            # 找到 "samples" 在路径中的位置
            try:
                idx = parts.index('samples')
            except ValueError:
                idx = -3  # fallback
            rel = '/'.join(parts[idx:])
            # 替换后缀 .jpg -> .png
            rel_png = os.path.splitext(rel)[0] + '.png'
            mask_path = os.path.join(self.sky_mask_root, rel_png)

            if os.path.exists(mask_path):
                mask = Image.open(mask_path).convert('L')
                mask_np = np.array(mask, dtype=np.float32) / 255.0   # 0.0 或 1.0
            else:
                # 如果没有对应的 mask，默认全部为前景 (不遮挡)
                src_H, src_W = 900, 1600
                mask_np = np.zeros((src_H, src_W), dtype=np.float32)

            # 下采样到特征图分辨率
            # 先应用与图像相同的增广参数 (resize + crop + flip)
            # 这里我们对 mask 也做同样的空间变换
            mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

            # 先 resize 到增广后 size, 再 crop
            aug_params = results.get('img_aug_params', None)
            if aug_params is not None and hasattr(aug_params, 'data'):
                aug_params = aug_params.data
            if aug_params is not None:
                cam_idx = list(cam_names).index(cam_name)
                resize_dims, crop, flip, rotate = aug_params[cam_idx]
                resize_W, resize_H = resize_dims
                # resize
                mask_tensor = torch.nn.functional.interpolate(
                    mask_tensor, size=(resize_H, resize_W), mode='bilinear',
                    align_corners=False)
                # crop (crop_w, crop_h, crop_w + fW, crop_h + fH)
                cw, ch, cw2, ch2 = int(crop[0]), int(crop[1]), int(crop[2]), int(crop[3])
                # clamp to valid range
                ch = max(0, ch)
                cw = max(0, cw)
                ch2 = min(mask_tensor.shape[2], ch2)
                cw2 = min(mask_tensor.shape[3], cw2)
                mask_tensor = mask_tensor[:, :, ch:ch2, cw:cw2]
                # flip
                if flip:
                    mask_tensor = torch.flip(mask_tensor, [-1])
                # NOTE: rotation 对 binary mask 影响很小, 忽略以保持简洁

            # 下采样到固定的特征图分辨率 (fH, fW)
            mask_tensor = torch.nn.functional.interpolate(
                mask_tensor, size=(self.feat_H, self.feat_W), mode='bilinear',
                align_corners=False)
            mask_tensor = (mask_tensor.squeeze(0).squeeze(0) > 0.5).float()  # (fH, fW)
            masks.append(mask_tensor)

        if masks:
            sky_mask = torch.stack(masks, dim=0)  # (N_cams, fH, fW)
        else:
            sky_mask = torch.zeros(1, 1, 1)

        results['sky_mask'] = sky_mask
        return results

