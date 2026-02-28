# Copyright (c) Phigent Robotics. All rights reserved.
import torch.nn.functional as F
from .bevdet import BEVDet
from flashocc.models import DETECTORS


@DETECTORS.register
class BEVDetOCC(BEVDet):
    def __init__(self,
                 occ_head=None,
                 upsample=False,
                 **kwargs):
        super(BEVDetOCC, self).__init__(**kwargs)
        self.occ_head = occ_head
        self.pts_bbox_head = None
        self.upsample = upsample

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
        voxel_semantics = kwargs['voxel_semantics']     # (B, Dx, Dy, Dz)
        mask_camera = kwargs['mask_camera']     # (B, Dx, Dy, Dz)

        occ_bev_feature = img_feats[0]
        if self.upsample:
            occ_bev_feature = F.interpolate(occ_bev_feature, scale_factor=2,
                                            mode='bilinear', align_corners=True)

        loss_occ = self.forward_occ_train(occ_bev_feature, voxel_semantics, mask_camera)
        losses.update(loss_occ)
        return losses

    def forward_occ_train(self, img_feats, voxel_semantics, mask_camera):
        """
        Args:
            img_feats: (B, C, Dz, Dy, Dx) / (B, C, Dy, Dx)
            voxel_semantics: (B, Dx, Dy, Dz)
            mask_camera: (B, Dx, Dy, Dz)
        """
        outs = self.occ_head(img_feats)
        loss_occ = self.occ_head.loss(
            outs,  # (B, Dx, Dy, Dz, n_cls)
            voxel_semantics,  # (B, Dx, Dy, Dz)
            mask_camera,  # (B, Dx, Dy, Dz)
        )
        return loss_occ

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
