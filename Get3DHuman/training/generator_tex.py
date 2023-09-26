# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import torch
from torch_utils import persistence
from training.networks_stylegan2 import Generator as StyleGAN2Backbone
from training.networks_stylegan2 import Discriminator as DisBackbone

from libs.HGFilters_ import HGFilter
# from libs.HGFilters_pifu import HGFilter
from libs.MLP import MLP
import torch.nn as nn


from libs.geometry import orthogonal, index 

import numpy as np
from surface_tracing import SDFRenderer

# D_cache = DisBackbone(0, 125, 256, block_kwargs=opt_stylegan.D_kwargs["block_kwargs"]
#                                  , mapping_kwargs=opt_stylegan.D_kwargs["mapping_kwargs"]
#                                  , epilogue_kwargs=opt_stylegan.D_kwargs["epilogue_kwargs"])


# D_cache = DisBackbone(0, 128, 256)

# feature = torch.rand([1,256,128,128]).float()

# d_out = D_cache(feature, 0)
# # 

# @persistence.persistent_class
class I3DGAN_Generator(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality.
        c_dim,                      # Conditioning label (C) dimensionality.
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output resolution.
        img_channels,               # Number of output color channels.
        
        render_size,
        use_gpu,
        cuda_,
        
        mapping_kwargs      = {},   # Arguments for MappingNetwork.
        **synthesis_kwargs,         # Arguments for SynthesisNetwork.
    ):
        super().__init__()
        self.z_dim=z_dim
        self.c_dim=c_dim
        self.w_dim=w_dim
        self.img_resolution=img_resolution
        self.img_channels=img_channels
        # self.renderer = ImportanceRenderer()
        # self.ray_sampler = RaySampler()
        # self.backbone = StyleGAN2Backbone(z_dim, c_dim, w_dim, img_resolution=img_resolution, img_channels=img_channels, mapping_kwargs=mapping_kwargs, **synthesis_kwargs)
        self.backbone_shape = StyleGAN2Backbone(self.z_dim, self.c_dim, self.w_dim, img_resolution=512, img_channels=16, mapping_kwargs={'num_layers':2}, **synthesis_kwargs)
        self.backbone_texture = StyleGAN2Backbone(self.z_dim, self.c_dim, self.w_dim, img_resolution=512, img_channels=16, mapping_kwargs={'num_layers':2}, **synthesis_kwargs)
    
        # self.superresolution = dnnlib.util.construct_class_by_name(class_name=rendering_kwargs['superresolution_module'], channels=32, img_resolution=img_resolution, sr_num_fp16_res=sr_num_fp16_res, sr_antialias=rendering_kwargs['sr_antialias'], **sr_kwargs)
        # self.decoder = OSGDecoder(32, {'decoder_lr_mul': rendering_kwargs.get('decoder_lr_mul', 1), 'decoder_output_dim': 32})
        # self.neural_rendering_resolution = 64
        # self.rendering_kwargs = rendering_kwargs
        self.cuda_ = cuda_
        # self._last_planes = Nonefilter_texturefilter_texture
        self.filter_shape = HGFilter(1, 2, 16, 256,  'group', 'ave_pool', False)
        # self.filter_shape = HGFilter(1, 2, 16, 256,  'group', 'ave_pool', False)
        self.filter_texture = HGFilter(1, 2, 16+16, 256,  'group', 'ave_pool', False)
        
        self.mlp_shape = MLP([257, 512, 256, 128, 1], 2, [1, 2], norm=None, last_op=nn.Sigmoid())
        self.mlp_texture = MLP([513, 1024, 512, 256, 128, 3], 2, [2, 3], norm=None, last_op=nn.Sigmoid())
        
        # DR 
        '''DR'''

        self.render_size = render_size
        cam = ((self.render_size-1)/2)
        self.cam_intrinsic = np.array([[cam, 0., cam], [0., cam, cam], [0., 0., 1.]])
        self.img_size = self.render_size
        buffer_size, threshold, ratio, use_depth2normal = 3, 1e-5, 1.0, False
        self.radius = 2.1
        self.use_gpu = use_gpu
        
        self.transform_matrix_ortho = np.array([[1., 0., 0.], [0., 1, 0.], [0., 0., -1.]])
        self.sdf_renderer = SDFRenderer(cuda_, self.mlp_shape, self.cam_intrinsic, img_hw=(self.img_size,self.img_size),
                                          transform_matrix=self.transform_matrix_ortho, radius=self.radius,
                                        march_step=55, buffer_size=buffer_size, 
                                        threshold=threshold, ray_marching_ratio=ratio, 
                                        use_depth2normal=use_depth2normal,use_gpu= cuda_.type!='cpu')
        
        
                
        self.Dis_shape = DisBackbone(self.c_dim, img_resolution=128, img_channels=256)
        self.Dis_texture = DisBackbone(self.c_dim, img_resolution=128, img_channels=512)
        # self.Dis_normal = DisBackbone(self.c_dim, img_resolution=512, img_channels=3)
        # self.Dis_depth = DisBackbone(self.c_dim, img_resolution=512, img_channels=1)
    
    
    def query_texture(self, infer_texture_cache, points, calibs, spatial_enc, labels=None):
        '''
        infer_texture_cache, points, calibs, spatial_enc= infer_texture_cache, verts_train_query_i, calib_tensor_real, spatial_enc
        '''
        xyz_rgb = orthogonal(points, calibs, None)
        xy_rgb = xyz_rgb[:, :2, :]
        # self.xyz_rgb = xyz_rgb
        sp_feat_rgb = spatial_enc(xyz_rgb)
        texture_feat_list = [index(infer_texture_cache[0], xy_rgb), sp_feat_rgb]    
        point_local_feat_texture = torch.cat(texture_feat_list, 1)
        
        # point_local_feat = torch.cat(point_local_feat_list, 1)
        pred_rgb = infer_texture_cache[1](point_local_feat_texture)[0]
        return pred_rgb
    
    
    def mapping_shape(self, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False):
        # if self.rendering_kwargs['c_gen_conditioning_zero']:
                # c = torch.zeros_like(c)
        return self.backbone_shape.mapping(z, c , truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)

    def synthesis_shape(self, ws, c, neural_rendering_resolution=None, update_emas=False, cache_backbone=False, use_cached_backbone=False, **synthesis_kwargs):
        # cam2world_matrix = c[:, :16].view(-1, 4, 4)
        # intrinsics = c[:, 16:25].view(-1, 3, 3)
        planes = self.backbone_shape.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
        return planes
    
    def forward_shape_(self, z, c, truncation_psi=1, truncation_cutoff=None, neural_rendering_resolution=None, update_emas=False, cache_backbone=False, use_cached_backbone=False, **synthesis_kwargs):
        # Render a batch of generated images.
        ws_shape = self.mapping_shape(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        feature_shape = self.synthesis_shape(ws_shape, c, update_emas=update_emas, neural_rendering_resolution=neural_rendering_resolution, cache_backbone=cache_backbone, use_cached_backbone=use_cached_backbone, **synthesis_kwargs)

        shape_field, _ = self.filter_shape(feature_shape)
        
        return feature_shape, shape_field[0]

    
    def forward_shape_f(self, z, c, truncation_psi=1, truncation_cutoff=None, neural_rendering_resolution=None, update_emas=False, cache_backbone=False, use_cached_backbone=False, **synthesis_kwargs):
        # Render a batch of generated images.
        ws_shape = self.mapping_shape(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        feature_shape = self.synthesis_shape(ws_shape, c, update_emas=update_emas, neural_rendering_resolution=neural_rendering_resolution, cache_backbone=cache_backbone, use_cached_backbone=use_cached_backbone, **synthesis_kwargs)
        self.feature_shape = feature_shape
        
        # feature_volume_shape, _ = self.filter_shape(feature_shape)
        # self.feature_volume_shape = feature_volume_shape[0]
        return feature_shape
    
    def forward_shape_mix(self, z, c, truncation_psi=1, truncation_cutoff=None, neural_rendering_resolution=None, update_emas=False, cache_backbone=False, use_cached_backbone=False, **synthesis_kwargs):
        # Render a batch of generated images.
        ws_shape = self.mapping_shape(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        feature_shape = self.synthesis_shape(ws_shape, c, update_emas=update_emas, neural_rendering_resolution=neural_rendering_resolution, cache_backbone=cache_backbone, use_cached_backbone=use_cached_backbone, **synthesis_kwargs)
        self.feature_shape = feature_shape
        
        feature_volume_shape, _ = self.filter_shape(feature_shape)
        self.feature_volume_shape = feature_volume_shape[0]
        return feature_volume_shape[0], feature_shape, ws_shape
    
    def forward_shape_mix_infer(self, z, c, truncation_psi=1, truncation_cutoff=None, neural_rendering_resolution=None, update_emas=False, cache_backbone=False, use_cached_backbone=False, **synthesis_kwargs):
        # Render a batch of generated images.
        with torch.no_grad():
            ws_shape = self.mapping_shape(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
            feature_shape = self.synthesis_shape(ws_shape, c, update_emas=update_emas, neural_rendering_resolution=neural_rendering_resolution, cache_backbone=cache_backbone, use_cached_backbone=use_cached_backbone, **synthesis_kwargs)
            self.feature_shape = feature_shape
            
            feature_volume_shape, _ = self.filter_shape(feature_shape)
            self.feature_volume_shape = feature_volume_shape[0]
        return feature_volume_shape[0], feature_shape, ws_shape
    
    def forward_shape_2(self, z, c, truncation_psi=1, truncation_cutoff=None, neural_rendering_resolution=None, update_emas=False, cache_backbone=False, use_cached_backbone=False, **synthesis_kwargs):
        # Render a batch of generated images.
        ws_shape = self.mapping_shape(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        feature_shape = self.synthesis_shape(ws_shape, c, update_emas=update_emas, neural_rendering_resolution=neural_rendering_resolution, cache_backbone=cache_backbone, use_cached_backbone=use_cached_backbone, **synthesis_kwargs)
        self.feature_shape = feature_shape
        
        feature_volume_shape, _ = self.filter_shape(feature_shape)
        self.feature_volume_shape = feature_volume_shape[0]
        return feature_volume_shape[0], ws_shape
    
    def mapping_texture(self, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False):
        # if self.rendering_kwargs['c_gen_conditioning_zero']:
                # c = torch.zeros_like(c)
        return self.backbone_texture.mapping(z, c , truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)

    def synthesis_texture(self, ws, c, neural_rendering_resolution=None, update_emas=False, cache_backbone=False, use_cached_backbone=False, **synthesis_kwargs):
        # cam2world_matrix = c[:, :16].view(-1, 4, 4)
        # intrinsics = c[:, 16:25].view(-1, 3, 3)
        planes = self.backbone_texture.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
        return planes
  
      
    def forward_DR(self, z_shape, c, R, T, calib, cam_x, cam_y, cuda, spatial_enc, not_train_DR, truncation_psi=1, truncation_cutoff=None,  update_emas=False, cache_backbone=False, use_cached_backbone=False, **synthesis_kwargs):
        
        # Render shape depth, and find the points.
        # with torch.no_grad():
        ws_shape = self.mapping_shape(z_shape, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        feature_shape_dr = self.synthesis_shape(ws_shape, c, update_emas=update_emas, neural_rendering_resolution=None, cache_backbone=cache_backbone, use_cached_backbone=use_cached_backbone, **synthesis_kwargs)
        feature_volume_shape_dr, _ = self.filter_shape(feature_shape_dr)
            
        self.cache_image = torch.ones([z_shape.size(0), self.render_size*self.render_size]).to(device=cuda) >2
        render_output = self.sdf_renderer.render_ortho_points(cam_x, cam_y, [feature_volume_shape_dr[0], calib, orthogonal, spatial_enc, index, self.cache_image],
                                                   R, T, no_grad=not_train_DR,  sample_index_type='min_abs', no_grad_depth=not_train_DR, no_grad_normal=False)
        
        # render_output = self.sdf_renderer.render_ortho([latent_pifu, calib, self.projection,self.spatial_enc,self.index],
        #                                          R, T, profile=False, sample_index_type='min_abs', ray_marching_type=self.ray_marching_type, no_grad_depth=(not grad_settings["depth"]), no_grad_normal=(not grad_settings["normal"]))
        depth_rendered, min_points, normal_rendered = render_output
        # depth, min_points = self.DR_depnorm(R, T, calib, self.feature_volume_shape)
        self.feature_shape_dr = feature_shape_dr
        self.feature_volume_shape_dr = feature_volume_shape_dr
        
        return depth_rendered, min_points, normal_rendered, feature_shape_dr, feature_volume_shape_dr[0]
       
    def forward_DR_infer(self, z_shape, c, R, T, calib, cam_x, cam_y, cuda, spatial_enc, not_train_DR, truncation_psi=1, truncation_cutoff=None,  update_emas=False, cache_backbone=False, use_cached_backbone=False, **synthesis_kwargs):
        
        # Render shape depth, and find the points.
        with torch.no_grad():
            ws_shape = self.mapping_shape(z_shape, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
            self.feature_shape_dr = self.synthesis_shape(ws_shape, c, update_emas=update_emas, neural_rendering_resolution=None, cache_backbone=cache_backbone, use_cached_backbone=use_cached_backbone, **synthesis_kwargs).repeat([1,3,1,1])
            self.feature_volume_shape_dr, _ = self.filter_shape(self.feature_shape_dr)
             
        # self.cam_x, self.cam_y = torch.tensor(cam_x).to(device=cuda), torch.tensor(cam_y).to(device=cuda)
        self.cache_image = torch.ones([z_shape.size(0), self.render_size*self.render_size]).to(device=self.cuda) >2
        render_output = self.sdf_renderer.render_ortho_points(cam_x, cam_y, [self.feature_volume_shape_dr[0], calib, orthogonal, spatial_enc, index, self.cache_image],
                                                   R, T, no_grad=not_train_DR,  sample_index_type='min_abs', no_grad_depth=not_train_DR, no_grad_normal=True)
        
        # render_output = self.sdf_renderer.render_ortho([latent_pifu, calib, self.projection,self.spatial_enc,self.index],
        #                                          R, T, profile=False, sample_index_type='min_abs', ray_marching_type=self.ray_marching_type, no_grad_depth=(not grad_settings["depth"]), no_grad_normal=(not grad_settings["normal"]))
        depth_rendered, min_points = render_output
        # depth, min_points = self.DR_depnorm(R, T, calib, self.feature_volume_shape)
         
        return depth_rendered, min_points, self.feature_shape_dr, self.feature_volume_shape_dr[0]
       
        
    def forward_texture(self, shape_feature, z_texture, c, truncation_psi=1, truncation_cutoff=None,  update_emas=False, cache_backbone=False, use_cached_backbone=False, **synthesis_kwargs):
        
        # depth_rendered, min_points = render_output

        ws_shape = self.mapping_texture(z_texture, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        feature_texture = self.synthesis_texture(ws_shape, c, update_emas=update_emas, neural_rendering_resolution=None, cache_backbone=cache_backbone, use_cached_backbone=use_cached_backbone, **synthesis_kwargs)
        # feature_volume_texture, _ = self.filter_texture(torch.cat([feature_texture, shape_feature],1))
        # feature_volume_texture, _ = self.filter_texture(torch.cat([shape_feature, feature_texture],1))
        feature_volume_texture, _ = self.filter_texture(torch.cat([feature_texture, shape_feature],1))

        
        self.feature_texture = feature_texture
        self.feature_volume_texture = feature_volume_texture[0]
        return feature_volume_texture[0]
    
    
    def forward_avatar_t(self, z_shape, c_s, z_texture, t_c, truncation_psi=1, truncation_cutoff=None, neural_rendering_resolution=None, update_emas=False, cache_backbone=False, use_cached_backbone=False, **synthesis_kwargs):
        # Render a batch of generated images.
        with torch.no_grad():
            ws_shape = self.mapping_shape(z_shape, c_s, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
            feature_shape = self.synthesis_shape(ws_shape, c_s, update_emas=update_emas, neural_rendering_resolution=neural_rendering_resolution, cache_backbone=cache_backbone, use_cached_backbone=use_cached_backbone, **synthesis_kwargs)
            shape_field, _ = self.filter_shape(feature_shape)
            
        ws_texture = self.mapping_texture(z_texture, t_c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        feature_texture = self.synthesis_texture(ws_texture, t_c, update_emas=update_emas, neural_rendering_resolution=None, cache_backbone=cache_backbone, use_cached_backbone=use_cached_backbone, **synthesis_kwargs)
        texture_field, _ = self.filter_texture(torch.cat([feature_texture, feature_shape],1))
        # texture_field, _ = self.filter_texture(torch.cat([feature_shape, feature_texture],1))
        # texture_field, _ = self.filter_texture(torch.cat([feature_shape, feature_texture],1))
        

        return  shape_field[0], texture_field[0], feature_shape
    
    def forward_image(self, z_texture, c, shape_feature, min_points, calib, spatial_enc, truncation_psi=1, truncation_cutoff=None,  update_emas=False, cache_backbone=False, use_cached_backbone=False, **synthesis_kwargs):
        
        # depth_rendered, min_points = render_output

        ws_shape = self.mapping_texture(z_texture, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        feature_texture = self.synthesis_texture(ws_shape, c, update_emas=update_emas, neural_rendering_resolution=None, cache_backbone=cache_backbone, use_cached_backbone=use_cached_backbone, **synthesis_kwargs)
        # feature_volume_texture, _ = self.filter_texture(torch.cat([shape_feature[0], self.feature_texture],1))
        feature_volume_texture, _ = self.filter_texture(torch.cat([shape_feature[0], self.feature_texture],1))
        
        feature_volume_cat = torch.cat([feature_volume_texture[0], shape_feature[1]],1)
        # feature_volume_cat = torch.cat([shape_feature[1], feature_volume_texture[0]],1)
        
        pred_texture = self.decode_texture(self.mlp_texture, [feature_volume_cat, calib, orthogonal, spatial_enc, index], min_points).squeeze(-1)
        # pred_texture = self.decode_texture(self.mlp_texture, [self.feature_texture[0], calib, orthogonal, spatial_enc, index, self.cache_image], min_points).squeeze(-1)
        pred_texture = pred_texture.reshape(-1, 3, self.render_size, self.render_size)
        
        self.feature_texture = feature_texture
        self.feature_volume_texture = feature_volume_texture
        return pred_texture
    

    def decode_texture(self, decoder, latent_list, points, clamp_dist=0.1, MAX_POINTS=300000, no_grad=False):
        
        # decoder, latent_list, points = sdf_renderer.decoder, latent, points_now
        
        points = torch.clamp(points, -0.99, 0.99)
    
        start, num_all = 0, points.shape[-1]
        output_list = []
        while True:
            end = min(start + MAX_POINTS, num_all)
            latent_pifu = latent_list[0]
            calibs = latent_list[1].repeat(points.shape[0],1,1)
            
            points_list = points
            # points_list = points_list.transpose(2,1)

            xyz = latent_list[2](points_list, calibs, transforms=None)
            xy = xyz[:, :2, :]
            sp_feat = latent_list[3](xyz, calibs=calibs)
            point_local_feat_list = [latent_list[4](latent_pifu, xy), sp_feat]       
            latent_vector = torch.cat(point_local_feat_list, 1)
            
            sdf_batch = decoder(latent_vector)[0].squeeze(1)

            start = end
            # if no_grad:
            #     sdf_batch = sdf_batch.detach()
            output_list.append(sdf_batch)
            if end == num_all:
                break
        rgb = torch.cat(output_list, 0)

        return rgb
    
    
    def query_mlp(self, samples, calibs, spatial_enc):
        xyz = orthogonal(samples, calibs, None)
        xy = xyz[:, :2, :]

        sp_feat = spatial_enc(xyz, calibs=calibs)
        # intermediate_preds_list = []
        # phi = None
        point_local_feat_list = [index(self.feature_texture, xy), sp_feat]       
        point_local_feat = torch.cat(point_local_feat_list, 1)
        pred, _ = self.mlp_texture(point_local_feat)
        return pred
    

    def forward_dis_shape(self, feature, c):
        our_d = self.Dis_shape(feature, c) 
        return our_d 
    
    def forward_dis_texture(self, feature, c):
        our_d_t = self.Dis_texture(feature, c) 
        return our_d_t 
    
    # def forward_dis_normal(self, feature, c):
    #     our_n = self.Dis_normal(feature, c) 
    #     return our_n 
    
    # def forward_dis_depth(self, feature, c):
    #     our_n = self.Dis_depth(feature, c) 
    #     return our_n 
        
# z_dim = 512           
# c_dim = 512                         
# w_dim = 512                  
# img_resolution = 512                
# img_channels = 512               
# StyleGAN2Backbone(z_dim, c_dim, w_dim, img_resolution=img_resolution, img_channels=img_channels, mapping_kwargs=mapping_kwargs, **synthesis_kwargs)

# backbone = StyleGAN2Backbone(512, 0, 512, img_resolution=512, img_channels=16, mapping_kwargs={'num_layers':2})
# 
# z_shape = torch.rand(512)
# feature_shape = backbone.forward_shape(z_shape)



# z_shape = torch.rand([1,512])

# stylepifu = PlaneGenerator(512, 0, 512, img_resolution=512, img_channels=16, use_gpu = False, mapping_kwargs={'num_layers':2})
    

# feature_shape = stylepifu.forward_shape(z_shape,0)


# z_texture = torch.rand([1,512])

# feature_shape = stylepifu.forward_texture(z_texture,0)


# feature_volume_shape = stylepifu.feature_volume_shape

