import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mmengine.model import BaseModule
from mmengine.runner.amp import autocast
from mmdet3d.registry import MODELS
from mmdet.models.utils.misc import multi_apply
from .bottleneckaspp import BottleNeckASPP
from .efficientvitblock import EfficientViTBlock
from .fusion import DynamicFusion2D, DynamicFusion3D

class SingleScaleInverseMatrixVT(BaseModule):
    def __init__(self,
                 feature_strides,
                 in_index=-1,
                 in_channel=512,
                 grid_size=[100, 100, 8],
                 x_bound=[-50, 50],
                 y_bound=[-50, 50],
                 z_bound=[-5., 3.],
                 sampling_rate=4,
                 num_cams=None,
                 enable_fix=False,
                 use_lidar=False,
                 use_radar=False):
        super().__init__()
        self.grid_size = torch.tensor(grid_size)
        self.x_bound = x_bound
        self.y_bound = y_bound
        self.z_bound = z_bound
        self.sampling_rate = sampling_rate
        self.in_index = in_index
        self.ds_rate = feature_strides
        self.coord = self._create_gridmap_anchor()
        if enable_fix:
            self.fix_param = torch.load(f'./fix_param_small/{self.in_index}.pth.tar')
        self.enable_fix = enable_fix
        self.use_lidar = use_lidar
        self.use_radar = use_radar
        self.num_cams = num_cams
        self.down_conv3d = nn.Sequential(nn.Conv3d(512,in_channel,1),
                                        nn.BatchNorm3d(in_channel),
                                        nn.ReLU(),
                                        nn.Conv3d(in_channel,in_channel,3,padding=1),
                                        nn.BatchNorm3d(in_channel),
                                        nn.ReLU(),
                                        nn.Conv3d(in_channel,in_channel,3,padding=1),
                                        nn.BatchNorm3d(in_channel),
                                        nn.ReLU())
        self.xy_conv = nn.Sequential(nn.Conv2d(512,in_channel,1),
                                    nn.BatchNorm2d(in_channel),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channel,in_channel,3,padding=1),
                                    nn.BatchNorm2d(in_channel),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channel,in_channel,3,padding=1),
                                    nn.BatchNorm2d(in_channel),
                                    nn.ReLU())
        self.combine_coeff = nn.Conv3d(in_channel, 1, kernel_size=1)
        self.aspp_xy = BottleNeckASPP(in_channel,in_channel,[1, 6, 12, 18])
             
        if in_index == 0: 
            self.bev_attn_layer = EfficientViTBlock(type='s',
                                                ed=in_channel,
                                                kd=8,
                                                nh=8,
                                                ar=1,
                                                resolution=self.grid_size[0], # Feature Map Size
                                                kernels=[5 for _ in range(8)]
                                                )
        elif in_index == 1:
            self.bev_attn_layer = EfficientViTBlock(type='s',
                                                ed=in_channel,
                                                kd=16,
                                                nh=8,
                                                ar=1,
                                                resolution=self.grid_size[0],
                                                kernels=[5 for _ in range(8)]
                                                )
        else:
            self.bev_attn_layer = EfficientViTBlock(type='s',
                                                ed=in_channel,
                                                kd=32,
                                                nh=8,
                                                ar=1,
                                                resolution=self.grid_size[0],
                                                kernels=[5 for _ in range(8)]
                                                )
        
    def _create_gridmap_anchor(self):
        # create a gridmap anchor with shape of (X, Y, Z, sampling_rate**3, 3)
        grid_size = self.sampling_rate * self.grid_size
        coord = torch.zeros(grid_size[0], grid_size[1], grid_size[2], 3)
        x_coord = torch.linspace(self.x_bound[0], self.x_bound[1], grid_size[0])
        y_coord = torch.linspace(self.y_bound[0], self.y_bound[1], grid_size[1])
        z_coord = torch.linspace(self.z_bound[0], self.z_bound[1], grid_size[2])
        ones = torch.ones(grid_size[0], grid_size[1], grid_size[2], 1)
        coord[:, :, :, 0] = x_coord.reshape(-1, 1, 1)
        coord[:, :, :, 1] = y_coord.reshape(1, -1, 1)
        coord[:, :, :, 2] = z_coord.reshape(1, 1, -1)
        coord = torch.cat([coord, ones], dim=-1)
        # taking multi sampling points into a single grid
        new_coord = coord.reshape(self.grid_size[0], self.sampling_rate,
                                  self.grid_size[1], self.sampling_rate,
                                  self.grid_size[2], self.sampling_rate, 4). \
            permute(0, 2, 4, 1, 3, 5, 6).reshape(self.grid_size[0], self.grid_size[1],
                                                 self.grid_size[2], -1, 4)
        return new_coord

    @torch.no_grad()
    def get_vt_matrix(self, img_feats, img_metas):
        batch_vt = multi_apply(self._get_vt_matrix_single,img_feats,img_metas)
        res = tuple(torch.stack(vt) for vt in batch_vt)
        return res
    
    @autocast('cuda',torch.float32)
    def _get_vt_matrix_single(self, img_feat, img_meta):
        Nc, C, H, W = img_feat.shape
        # lidar2img: (Nc, 4, 4)
        lidar2img = img_meta['lidar2img']
        lidar2img = np.asarray(lidar2img)
        lidar2img = torch.tensor(lidar2img,device=img_feat.device,dtype=torch.float32)
        img_shape = img_meta['img_shape']
        # global_coord: (X * Y * Z, Nc, S, 4, 1)
        global_coord = self.coord.clone().to(lidar2img.device)
        X, Y, Z, S, _ = global_coord.shape
        global_coord = global_coord.view(X * Y * Z, 1, S, 4, 1).repeat(1, Nc, 1, 1, 1)
        # lidar2img: (X * Y * Z, Nc, S, 4, 4)
        lidar2img = lidar2img.unsqueeze(0).unsqueeze(2).repeat(X * Y * Z, 1, S, 1, 1)
        # ref_points: (X * Y * Z, Nc, S, 4), 4: (λW, λH, λ, 1) or (λU, λV, λ, 1)
        ref_points = torch.matmul(lidar2img.to(torch.float32), global_coord.to(torch.float32)).squeeze(-1)
        ref_points[..., 0] = ref_points[..., 0] / ref_points[..., 2]
        ref_points[..., 1] = ref_points[..., 1] / ref_points[..., 2]
        # remove invalid sampling points
        invalid_w = torch.logical_or(ref_points[..., 0] < 0.,ref_points[..., 0] > (img_shape[1] - 1))
        invalid_h = torch.logical_or(ref_points[..., 1] < 0.,ref_points[..., 1] > (img_shape[0] - 1))
        invalid_d = ref_points[..., 2] < 0.

        ref_points = torch.div(ref_points[..., :2], self.ds_rate, rounding_mode='floor').to(torch.long)
        # select valid cams
        if self.num_cams is not None:
            assert type(self.num_cams) == int
            valid_cams = torch.logical_not(invalid_w | invalid_h | invalid_d)
            valid_cams = valid_cams.permute(1, 0, 2).reshape(Nc, -1).sum(dim=-1)
            _, valid_cams_idx = torch.topk(valid_cams, self.num_cams)
            ref_points = ref_points[:, valid_cams_idx, :, :]
            Nc = self.num_cams
        else:
            valid_cams_idx = torch.arange(Nc, device=lidar2img.device)
        # still need (0, 1, 2...) encoding
        cam_index = torch.arange(Nc, device=lidar2img.device).unsqueeze(0).unsqueeze(2).repeat(X * Y * Z, 1, S).unsqueeze(-1)
        # ref_points: (X * Y * Z, Nc * S, 3), 3: (W, H, Nc)
        ref_points = torch.cat([ref_points, cam_index], dim=-1)
        ref_points[(invalid_w[:, valid_cams_idx] |
                    invalid_h[:, valid_cams_idx] |
                    invalid_d[:, valid_cams_idx])] = -1
        ref_points = ref_points.view(X * Y * Z, -1, 3)
        # ref_points_flatten: (X * Y * Z, Nc * S), 1: H * W * nc + W * h + w
        ref_points_flatten = ref_points[..., 2] * H * W + ref_points[..., 1] * W + ref_points[..., 0]
        # factorize 3D
        ref_points_flatten = ref_points_flatten.reshape(X, Y, Z, -1)
        ref_points_xyz = ref_points_flatten.reshape(X * Y * Z, -1)
        ref_points_z = ref_points_flatten.permute(0, 1, 3, 2).reshape(X * Y, -1)

        # create vt matrix with sparse matrix
        valid_idx_xyz = torch.nonzero(ref_points_xyz > 0)
        valid_idx_z = torch.nonzero(ref_points_z > 0)
        
        idx_xyz = torch.stack([ref_points_xyz[valid_idx_xyz[:, 0],valid_idx_xyz[:, 1]],valid_idx_xyz[:, 0]],dim=0).unique(dim=1)
        v_xyz = torch.ones(idx_xyz.shape[1]).to(img_feat.device)
        vt_xyz = torch.sparse_coo_tensor(indices=idx_xyz, values=v_xyz, size=[Nc * H * W, X * Y * Z])
        div_xyz = vt_xyz.sum(0).to_dense().clip(min=1)
        
        idx_xy = torch.stack([ref_points_z[valid_idx_z[:, 0],valid_idx_z[:, 1]],valid_idx_z[:, 0]],dim=0).unique(dim=1)
        v_xy = torch.ones(idx_xy.shape[1]).to(img_feat.device)
        vt_xy = torch.sparse_coo_tensor(indices=idx_xy, values=v_xy, size=[Nc * H * W, X * Y])
        div_xy = vt_xy.sum(0).to_dense().clip(min=1)
        
        return vt_xyz, vt_xy, div_xyz, div_xy, valid_cams_idx

    @autocast('cuda',torch.float32)
    def forward(self, 
                img_feats, 
                img_metas):
        X, Y, Z = self.grid_size
        B, _, C, H, W = img_feats.shape

        if self.enable_fix:
            vt_xyz = self.fix_param['vt_xyz'].to(img_feats.device)
            vt_xy = self.fix_param['vt_xy'].to(img_feats.device)
            div_xyz = self.fix_param['div_xyz'].to(img_feats.device)
            div_xy = self.fix_param['div_xy'].to(img_feats.device)
            valid_nc = self.fix_param['valid_nc'].to(img_feats.device)
            valid_nc = valid_nc.unsqueeze(0).repeat(B, 1)
        else:
            vt_xyzs, vt_xys, div_xyzs, div_xys, valid_nc = self.get_vt_matrix(img_feats, img_metas)
        
        valid_nc = valid_nc.unsqueeze(2).unsqueeze(3).unsqueeze(4).expand(-1, -1, C, H, W)
        img_feats = torch.gather(img_feats, 1, valid_nc)
        img_feats = img_feats.permute(0, 2, 1, 3, 4).reshape(B, C, -1)
        
        cam_xyz_feats, cam_xy_feats = [], []
        for idx in range(img_feats.shape[0]):
            img_feat = img_feats[idx]
            if not self.enable_fix:
                vt_xyz = vt_xyzs[idx]
                vt_xy = vt_xys[idx]
                div_xyz = div_xyzs[idx]
                div_xy = div_xys[idx]
            vt_xyz = vt_xyz.to_sparse_csr()
            vt_xy = vt_xy.to_sparse_csr()
            cam_xyz = torch.sparse.mm(img_feat,vt_xyz) / div_xyz
            cam_xyz_feat = cam_xyz.view(C, X, Y, Z)
            cam_xy = torch.sparse.mm(img_feat,vt_xy) / div_xy
            cam_xy_feat = cam_xy.view(C, X, Y)
            cam_xyz_feats.append(cam_xyz_feat)
            cam_xy_feats.append(cam_xy_feat)
        
        cam_xyz_feats = torch.stack(cam_xyz_feats)
        cam_xy_feats = torch.stack(cam_xy_feats)
        cam_xyz_feats = self.down_conv3d(cam_xyz_feats)
        cam_xy_feats = self.xy_conv(cam_xy_feats)
                
        # Apply ASPP on final 3D volume BEV slice
        cam_bevs = self.bev_attn_layer(cam_xy_feats)
        cam_bevs = self.aspp_xy(cam_bevs)
        coeff = self.combine_coeff(cam_xyz_feats).sigmoid()
        cam_xyz_feats = cam_xyz_feats + coeff * cam_bevs.unsqueeze(-1)
        
        return cam_xyz_feats
    
    @autocast('cuda',torch.float32)
    def forward_two(self, 
                    img_feats, 
                    img_metas, 
                    lidar_xyz_feat, 
                    lidar_xy_feat):
        X, Y, Z = self.grid_size
        B, _, C, H, W = img_feats.shape

        if self.enable_fix:
            vt_xyz = self.fix_param['vt_xyz'].to(img_feats.device)
            vt_xy = self.fix_param['vt_xy'].to(img_feats.device)
            div_xyz = self.fix_param['div_xyz'].to(img_feats.device)
            div_xy = self.fix_param['div_xy'].to(img_feats.device)
            valid_nc = self.fix_param['valid_nc'].to(img_feats.device)
            valid_nc = valid_nc.unsqueeze(0).repeat(B, 1)
        else:
            vt_xyzs, vt_xys, div_xyzs, div_xys, valid_nc = self.get_vt_matrix(img_feats, img_metas)
        
        valid_nc = valid_nc.unsqueeze(2).unsqueeze(3).unsqueeze(4).expand(-1, -1, C, H, W)
        img_feats = torch.gather(img_feats, 1, valid_nc)
        img_feats = img_feats.permute(0, 2, 1, 3, 4).reshape(B, C, -1)
        
        cam_xyz_feats, cam_xy_feats = [], []
        for idx in range(img_feats.shape[0]):
            img_feat = img_feats[idx]
            if not self.enable_fix:
                vt_xyz = vt_xyzs[idx]
                vt_xy = vt_xys[idx]
                div_xyz = div_xyzs[idx]
                div_xy = div_xys[idx]
            vt_xyz = vt_xyz.to_sparse_csr()
            vt_xy = vt_xy.to_sparse_csr()
            cam_xyz = torch.sparse.mm(img_feat,vt_xyz) / div_xyz
            cam_xyz_feat = cam_xyz.view(C, X, Y, Z)
            cam_xy = torch.sparse.mm(img_feat,vt_xy) / div_xy
            cam_xy_feat = cam_xy.view(C, X, Y)
            cam_xyz_feats.append(cam_xyz_feat)
            cam_xy_feats.append(cam_xy_feat)
        
        cam_xyz_feats = torch.stack(cam_xyz_feats)
        cam_xy_feats = torch.stack(cam_xy_feats)
        cam_xyz_feats = self.down_conv3d(cam_xyz_feats)
        cam_xy_feats = self.xy_conv(cam_xy_feats)
        
        if self.use_lidar:
            cam_atten_3d = self.cam_atten_3D(cam_xyz_feats)
            cam_atten_2d = self.cam_atten_2D(cam_xy_feats)
            lidar_atten_3d = self.lidar_atten_3D(lidar_xyz_feat)
            lidar_atten_2d = self.lidar_atten_2D(lidar_xy_feat)
            
            cam_xyz_feats = lidar_atten_3d * cam_xyz_feats
            cam_xy_feats = lidar_atten_2d * cam_xy_feats
            lidar_xyz_feat = cam_atten_3d * lidar_xyz_feat
            lidar_xy_feat = cam_atten_2d * lidar_xy_feat
        
        merged_xyz_feat = torch.cat([cam_xyz_feats,lidar_xyz_feat],dim=1)
        merged_xyz_feat = self.xyz_fusion(merged_xyz_feat)
        
        merged_xy_feat = torch.cat([cam_xy_feats,lidar_xy_feat],dim=1)
        merged_xy_feat = self.xy_fusion(merged_xy_feat)
        
        # Apply ASPP on final 3D volume BEV slice
        merged_bev = self.bev_attn_layer(merged_xy_feat)
        merged_bev = self.aspp_xy(merged_bev)
        coeff = self.combine_coeff(merged_xyz_feat).sigmoid()
        merged_xyz_feat = merged_xyz_feat + coeff * merged_bev.unsqueeze(-1)
        
        return merged_xyz_feat
    
    @autocast('cuda',torch.float32)
    def forward_three(self, 
                      img_feats, 
                      img_metas, 
                      lidar_xyz_feat, 
                      lidar_xy_feat,
                      radar_xyz_feat,
                      radar_xy_feat):
        X, Y, Z = self.grid_size
        B, _, C, H, W = img_feats.shape

        if self.enable_fix:
            vt_xyz = self.fix_param['vt_xyz'].to(img_feats.device)
            vt_xy = self.fix_param['vt_xy'].to(img_feats.device)
            div_xyz = self.fix_param['div_xyz'].to(img_feats.device)
            div_xy = self.fix_param['div_xy'].to(img_feats.device)
            valid_nc = self.fix_param['valid_nc'].to(img_feats.device)
            valid_nc = valid_nc.unsqueeze(0).repeat(B, 1)
        else:
            vt_xyzs, vt_xys, div_xyzs, div_xys, valid_nc = self.get_vt_matrix(img_feats, img_metas)

        
        valid_nc = valid_nc.unsqueeze(2).unsqueeze(3).unsqueeze(4).expand(-1, -1, C, H, W)
        img_feats = torch.gather(img_feats, 1, valid_nc)
        img_feats = img_feats.permute(0, 2, 1, 3, 4).reshape(B, C, -1)
        
        cam_xyz_feats, cam_xy_feats = [], []
        for idx in range(img_feats.shape[0]):
            img_feat = img_feats[idx]
            if not self.enable_fix:
                vt_xyz = vt_xyzs[idx]
                vt_xy = vt_xys[idx]
                div_xyz = div_xyzs[idx]
                div_xy = div_xys[idx]
            vt_xyz = vt_xyz.to_sparse_csr()
            vt_xy = vt_xy.to_sparse_csr()
            cam_xyz = torch.sparse.mm(img_feat,vt_xyz) / div_xyz
            cam_xyz_feat = cam_xyz.view(C, X, Y, Z)
            cam_xy = torch.sparse.mm(img_feat,vt_xy) / div_xy
            cam_xy_feat = cam_xy.view(C, X, Y)
            cam_xyz_feats.append(cam_xyz_feat)
            cam_xy_feats.append(cam_xy_feat)
        
        cam_xyz_feats = torch.stack(cam_xyz_feats)
        cam_xy_feats = torch.stack(cam_xy_feats)
        cam_xyz_feats = self.down_conv3d(cam_xyz_feats)
        cam_xy_feats = self.xy_conv(cam_xy_feats)
        
        cam_atten_3d = self.cam_atten_3D(cam_xyz_feats)
        cam_atten_2d = self.cam_atten_2D(cam_xy_feats)
        lidar_atten_3d = self.lidar_atten_3D(lidar_xyz_feat)
        lidar_atten_2d = self.lidar_atten_2D(lidar_xy_feat)
        
        cam_xyz_feats = lidar_atten_3d * cam_xyz_feats
        cam_xy_feats = lidar_atten_2d * cam_xy_feats
        lidar_xyz_feat = cam_atten_3d * lidar_xyz_feat
        lidar_xy_feat = cam_atten_2d * lidar_xy_feat
        
        merged_xyz_feat = torch.cat([cam_xyz_feats,lidar_xyz_feat,radar_xyz_feat],dim=1)
        merged_xyz_feat = self.xyz_fusion(merged_xyz_feat)
        
        merged_xy_feat = torch.cat([cam_xy_feats,lidar_xy_feat,radar_xy_feat],dim=1)
        merged_xy_feat = self.xy_fusion(merged_xy_feat)
        
        # Apply ASPP on final 3D volume BEV slice
        merged_bev = self.bev_attn_layer(merged_xy_feat)
        merged_bev = self.aspp_xy(merged_bev)
        coeff = self.combine_coeff(merged_xyz_feat).sigmoid()
        merged_xyz_feat = merged_xyz_feat + coeff * merged_bev.unsqueeze(-1)
        
        return merged_xyz_feat
        