# Copyright (c) Phigent Robotics. All rights reserved.
import torch
import torch.nn.functional as F

from .bevdet import BEVDet
from flashocc.models import DETECTORS


@DETECTORS.register
class BEVDetOCC(BEVDet):
    def __init__(self,
                 occ_head=None,
                 upsample=False,
                 use_depth_loss=False,
                 sky_freespace_loss_weight=0.0,
                 **kwargs):
        super(BEVDetOCC, self).__init__(**kwargs)
        self.occ_head = occ_head
        self.pts_bbox_head = None
        self.upsample = upsample
        self.use_depth_loss = use_depth_loss
        self.sky_freespace_loss_weight = sky_freespace_loss_weight

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
        """Forward training function."""
        img_feats, pts_feats, depth = self.extract_feat(
            points, img_inputs=img_inputs, img_metas=img_metas, **kwargs)

        losses = dict()

        # ---- Depth Loss: LiDAR 投影深度监督 LSS 深度预测 ----
        # 注意: 官方 BEVDetOCC 没有 depth loss, 只有 BEVDepthOCC 才有.
        # use_depth_loss 默认 False, 与官方行为一致.
        if self.use_depth_loss and 'gt_depth' in kwargs and kwargs['gt_depth'] is not None:
            loss_depth = self.img_view_transformer.get_depth_loss(
                kwargs['gt_depth'], depth)
            losses['loss_depth'] = loss_depth

        voxel_semantics = kwargs['voxel_semantics']     # (B, Dx, Dy, Dz)
        mask_camera = kwargs['mask_camera']     # (B, Dx, Dy, Dz)

        occ_bev_feature = img_feats[0]
        if self.upsample:
            occ_bev_feature = F.interpolate(occ_bev_feature, scale_factor=2,
                                            mode='bilinear', align_corners=True)

        loss_occ, occ_pred = self.forward_occ_train(occ_bev_feature, voxel_semantics, mask_camera)
        losses.update(loss_occ)

        # ---- Sky Freespace 射线级 Loss ----
        if self.sky_freespace_loss_weight > 0 and 'sky_mask' in kwargs:
            sky_fs_loss = self._compute_sky_freespace_loss(
                kwargs['sky_mask'], occ_pred)
            losses.update(sky_fs_loss)

        return losses

    def forward_occ_train(self, img_feats, voxel_semantics, mask_camera):
        """
        Args:
            img_feats: (B, C, Dz, Dy, Dx) / (B, C, Dy, Dx)
            voxel_semantics: (B, Dx, Dy, Dz)
            mask_camera: (B, Dx, Dy, Dz)
        Returns:
            loss_occ: dict
            occ_pred: (B, Dx, Dy, Dz, n_cls)
        """
        outs = self.occ_head(img_feats)
        loss_occ = self.occ_head.loss(
            outs,  # (B, Dx, Dy, Dz, n_cls)
            voxel_semantics,  # (B, Dx, Dy, Dz)
            mask_camera,  # (B, Dx, Dy, Dz)
        )
        return loss_occ, outs

    # ------------------------------------------------------------------
    #  Sky Freespace Loss — 射线级 3D 体素空间 "已知空闲" 监督
    # ------------------------------------------------------------------

    @torch.amp.autocast('cuda', enabled=False)
    def _compute_sky_freespace_loss(self, sky_mask, occ_pred):
        """将 SAM3 天空掩码沿相机射线投影到 3D 体素空间.

        对于 2D 图像中被标注为 "天空" 的像素, 其沿视锥的所有深度
        采样点映射到的 3D 体素都应当是 Free/Empty (class n_cls-1).
        以此为监督信号施加辅助 Cross-Entropy Loss.

        与旧版 Sky-Gate 硬门控 (在 2D 阶段乘零阻断特征) 不同:
        - 不修改任何 2D 特征流, 保持 LSS 梯度链路完整
        - 在 3D 占据空间直接提供 "负样本" 监督
        - 天空射线产生的体素作为高质量 Free-Space 先验

        Args:
            sky_mask: (B, N, fH, fW) float — 1=天空, 0=前景
                      (已由 LoadSkyMask pipeline 下采样到特征图分辨率)
            occ_pred: (B, Dx, Dy, Dz, n_cls) — OCC head 输出 logits
        Returns:
            dict with ``loss_sky_freespace`` tensor, or empty dict
        """
        sensor_inputs = getattr(self, '_cached_sensor_inputs', None)
        if sensor_inputs is None:
            return {}

        vt = self.img_view_transformer
        n_cls = occ_pred.shape[-1]
        FREE_CLASS = n_cls - 1          # class 17 = free/empty

        Dx = int(vt.grid_size[0])       # 200
        Dy = int(vt.grid_size[1])       # 200
        occ_Dz = self.occ_head.Dz       # 16

        # ---- 1. 计算视锥在 ego 坐标系下的坐标 (无需梯度) ----
        with torch.no_grad():
            # sensor_inputs: [imgs, sensor2keyegos, ego2globals, intrins,
            #                 post_rots, post_trans, bda]
            ego_coor = vt.get_ego_coor(
                *sensor_inputs[1:7])            # (B, N, D, fH, fW, 3)
            B, N, D, fH, fW, _ = ego_coor.shape

            # ---- 2. 转换为体素索引 ----
            x_lower = vt.grid_lower_bound[0].item()
            y_lower = vt.grid_lower_bound[1].item()
            x_interval = vt.grid_interval[0].item()
            y_interval = vt.grid_interval[1].item()
            z_lower = vt.grid_config.z[0]
            z_upper = vt.grid_config.z[1]
            z_interval = (z_upper - z_lower) / occ_Dz

            vx = ((ego_coor[..., 0] - x_lower) / x_interval).long()
            vy = ((ego_coor[..., 1] - y_lower) / y_interval).long()
            vz = ((ego_coor[..., 2] - z_lower) / z_interval).long()

            # ---- 3. 有效范围 & 天空掩码联合筛选 ----
            valid = ((vx >= 0) & (vx < Dx) &
                     (vy >= 0) & (vy < Dy) &
                     (vz >= 0) & (vz < occ_Dz))

            # sky_mask: (B, N, fH, fW) -> (B, N, D, fH, fW)
            sky_expanded = (sky_mask > 0.5).unsqueeze(2).expand(
                B, N, D, fH, fW)
            sky_valid = sky_expanded & valid

            if not sky_valid.any():
                return {}

            # ---- 4. 散射到体素线性索引 (去重) ----
            batch_idx = torch.arange(
                B, device=vx.device
            ).view(B, 1, 1, 1, 1).expand(B, N, D, fH, fW)

            sv = sky_valid.reshape(-1)
            linear_idx = (
                batch_idx.reshape(-1)[sv] * (Dx * Dy * occ_Dz) +
                vx.reshape(-1)[sv] * (Dy * occ_Dz) +
                vy.reshape(-1)[sv] * occ_Dz +
                vz.reshape(-1)[sv]
            )
            linear_idx = linear_idx.unique()

        if linear_idx.numel() == 0:
            return {}

        # ---- 5. Cross-Entropy: 天空射线体素 -> Free 类别 ----
        occ_flat = occ_pred.reshape(-1, n_cls).float()  # (B*Dx*Dy*Dz, n_cls)
        preds_sky = occ_flat[linear_idx]                 # (K, n_cls)

        target = torch.full(
            (preds_sky.shape[0],), FREE_CLASS,
            device=preds_sky.device, dtype=torch.long)

        loss = F.cross_entropy(preds_sky, target, reduction='mean')
        return {'loss_sky_freespace': loss * self.sky_freespace_loss_weight}

    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    rescale=False,
                    **kwargs):
        img_feats, _, _ = self.extract_feat(
            points, img_inputs=img, img_metas=img_metas, **kwargs)

        occ_bev_feature = img_feats[0]
        if self.upsample:
            occ_bev_feature = F.interpolate(occ_bev_feature, scale_factor=2,
                                            mode='bilinear', align_corners=True)

        occ_list = self.simple_test_occ(occ_bev_feature, img_metas)
        return occ_list

    def simple_test_occ(self, img_feats, img_metas=None):
        """
        Args:
            img_feats: (B, C, Dz, Dy, Dx) / (B, C, Dy, Dx)
        Returns:
            occ_preds: List[(Dx, Dy, Dz), ...]
        """
        outs = self.occ_head(img_feats)
        if not hasattr(self.occ_head, "get_occ_gpu"):
            occ_preds = self.occ_head.get_occ(outs, img_metas)
        else:
            occ_preds = self.occ_head.get_occ_gpu(outs, img_metas)
        return occ_preds

    def forward_dummy(self,
                      points=None,
                      img_metas=None,
                      img_inputs=None,
                      **kwargs):
        img_feats, pts_feats, depth = self.extract_feat(
            points, img_inputs=img_inputs, img_metas=img_metas, **kwargs)
        occ_bev_feature = img_feats[0]
        if self.upsample:
            occ_bev_feature = F.interpolate(occ_bev_feature, scale_factor=2,
                                            mode='bilinear', align_corners=True)
        outs = self.occ_head(occ_bev_feature)
        return outs
