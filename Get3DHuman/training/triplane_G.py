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
from libs.HGFilters_hd import HGFilter
from libs.MLP import MLP
import torch.nn as nn
# from training.volumetric_rendering.renderer import ImportanceRenderer
# from training.volumetric_rendering.ray_sampler import RaySampler
# import dnnlib

import numpy as np
# from diff_render.renderer import SDFRenderer
# from diff_render.decoder_utils import load_decoder
from surface_tracing import SDFRenderer

@persistence.persistent_class
class TriPlaneGenerator(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality.
        c_dim,                      # Conditioning label (C) dimensionality.
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output resolution.
        img_channels,               # Number of output color channels.
        sr_num_fp16_res     = 0,
        mapping_kwargs      = {},   # Arguments for MappingNetwork.
        rendering_kwargs    = {},
        sr_kwargs = {},
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
        self.backbone = StyleGAN2Backbone(z_dim, c_dim, w_dim, img_resolution=512, img_channels=16, mapping_kwargs=mapping_kwargs, **synthesis_kwargs)
        # self.superresolution = dnnlib.util.construct_class_by_name(class_name=rendering_kwargs['superresolution_module'], channels=32, img_resolution=img_resolution, sr_num_fp16_res=sr_num_fp16_res, sr_antialias=rendering_kwargs['sr_antialias'], **sr_kwargs)
        # self.decoder = OSGDecoder(32, {'decoder_lr_mul': rendering_kwargs.get('decoder_lr_mul', 1), 'decoder_output_dim': 32})
        self.neural_rendering_resolution = 64
        self.rendering_kwargs = rendering_kwargs
    
        self._last_planes = None
        self.image_filter = HGFilter(1, 2, 16, 256,  'group', 'ave_pool', False)
        
        self.mlp = MLP([257, 512, 256, 128, 1], 2, [1, 2], norm=None,last_op=nn.Sigmoid())
        
        # DR 
        '''DR'''
#        self.ray_marching_type = 'pyramid_recursive'
        self.ray_marching_type = 'recursive'
        # self.cam_intrinsic = np.array([[128, 0., 128], [0., 128, 128], [0., 0., 1.]])
        render_size = 128
        cam = ((render_size-1)/2)
        self.cam_intrinsic = np.array([[cam, 0., cam], [0., cam, cam], [0., 0., 1.]])
        self.img_size = render_size
        buffer_size, threshold, ratio, use_depth2normal = 3, 1e-5, 0.95, False
        self.radius = 2.1
        self.use_gpu = False
        
        self.transform_matrix_ortho = np.array([[1., 0., 0.], [0., 1, 0.], [0., 0., -1.]])
        self.sdf_renderer = SDFRenderer(self.cuda, self.mlp, self.cam_intrinsic, img_hw=(self.img_size,self.img_size),
                                         transform_matrix=self.transform_matrix_ortho, radius=self.radius,
                                        march_step=100, buffer_size=buffer_size, 
                                        threshold=threshold, ray_marching_ratio=ratio, 
                                        use_depth2normal=use_depth2normal,use_gpu=self.use_gpu)
        
    
    def DR_depnorm(self, R, T, calib):
        
        grad_settings={"depth": True, "normal": True, "silhouette": True}
        
        '''In pifuhd the latent_pifu = [bs, 256,256,256], 
           in DR, latent_dist =  [bs, 256],
           So we modify the code in latent_dist to [latent_pifu, calib,]'''
        
#        geo_trans = [self.projection,self.spatial_enc,self.index]
        latent_pifu = self.im_feat_list[0]
        render_output = self.sdf_renderer.render_ortho([latent_pifu, calib, self.projection,self.spatial_enc,self.index],
                                                 R, T, profile=False, sample_index_type='min_abs', ray_marching_type=self.ray_marching_type, no_grad_depth=(not grad_settings["depth"]), no_grad_normal=(not grad_settings["normal"]))
        depth_rendered, normal_rendered, valid_mask_rendered, min_sdf_sample = render_output
    
        return depth_rendered, normal_rendered, valid_mask_rendered, min_sdf_sample

    
    def mapping(self, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False):
        # if self.rendering_kwargs['c_gen_conditioning_zero']:
                # c = torch.zeros_like(c)
        return self.backbone.mapping(z, c , truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)

    def synthesis(self, ws, c, neural_rendering_resolution=None, update_emas=False, cache_backbone=False, use_cached_backbone=False, **synthesis_kwargs):
        # cam2world_matrix = c[:, :16].view(-1, 4, 4)
        # intrinsics = c[:, 16:25].view(-1, 3, 3)

        if neural_rendering_resolution is None:
            neural_rendering_resolution = self.neural_rendering_resolution
        else:
            self.neural_rendering_resolution = neural_rendering_resolution

        # Create a batch of rays for volume rendering
        # ray_origins, ray_directions = self.ray_sampler(cam2world_matrix, intrinsics, neural_rendering_resolution)

        # Create triplanes by running StyleGAN backbone
        # N, M, _ = ray_origins.shape
        if use_cached_backbone and self._last_planes is not None:
            planes = self._last_planes
        else:
            planes = self.backbone.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
        if cache_backbone:
            self._last_planes = planes

        # Reshape output into three 32-channel planes
        # planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])

        # Perform volume rendering
        # feature_samples, depth_samples, weights_samples = self.renderer(planes, self.decoder, ray_origins, ray_directions, self.rendering_kwargs) # channels last

        # Reshape into 'raw' neural-rendered image
        # H = W = self.neural_rendering_resolution
        # feature_image = feature_samples.permute(0, 2, 1).reshape(N, feature_samples.shape[-1], H, W).contiguous()
        # depth_image = depth_samples.permute(0, 2, 1).reshape(N, 1, H, W)

        # Run superresolution to get final image
        # rgb_image = feature_image[:, :3]
        # sr_image = self.superresolution(rgb_image, feature_image, ws, noise_mode=self.rendering_kwargs['superresolution_noise_mode'], **{k:synthesis_kwargs[k] for k in synthesis_kwargs.keys() if k != 'noise_mode'})
        
        return planes
    
    # def sample(self, coordinates, directions, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
    #     # Compute RGB features, density for arbitrary 3D coordinates. Mostly used for extracting shapes. 
    #     ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
    #     planes = self.backbone.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
    #     planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])
    #     return self.renderer.run_model(planes, self.decoder, coordinates, directions, self.rendering_kwargs)

    # def sample_mixed(self, coordinates, directions, ws, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
    #     # Same as sample, but expects latent vectors 'ws' instead of Gaussian noise 'z'
    #     planes = self.backbone.synthesis(ws, update_emas = update_emas, **synthesis_kwargs)
    #     planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])
    #     return self.renderer.run_model(planes, self.decoder, coordinates, directions, self.rendering_kwargs)

    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, neural_rendering_resolution=None, update_emas=False, cache_backbone=False, use_cached_backbone=False, **synthesis_kwargs):
        # Render a batch of generated images.
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        self.feature_volume = self.synthesis(ws, c, update_emas=update_emas, neural_rendering_resolution=neural_rendering_resolution, cache_backbone=cache_backbone, use_cached_backbone=use_cached_backbone, **synthesis_kwargs)
        
        self.im_feat_list, self.normx = self.image_filter(self.feature_volume)
        
        return self.im_feat_list
    
    
    
# from training.networks_stylegan2 import FullyConnectedLayer

# class OSGDecoder(torch.nn.Module):
#     def __init__(self, n_features, options):
#         super().__init__()
#         self.hidden_dim = 64

#         self.net = torch.nn.Sequential(
#             FullyConnectedLayer(n_features, self.hidden_dim, lr_multiplier=options['decoder_lr_mul']),
#             torch.nn.Softplus(),
#             FullyConnectedLayer(self.hidden_dim, 1 + options['decoder_output_dim'], lr_multiplier=options['decoder_lr_mul'])
#         )
        
#     def forward(self, sampled_features, ray_directions):
#         # Aggregate features
#         sampled_features = sampled_features.mean(1)
#         x = sampled_features

#         N, M, C = x.shape
#         x = x.view(N*M, C)

#         x = self.net(x)
#         x = x.view(N, M, -1)
#         rgb = torch.sigmoid(x[..., 1:])*(1 + 2*0.001) - 0.001 # Uses sigmoid clamping from MipNeRF
#         sigma = x[..., 0:1]
#         return {'rgb': rgb, 'sigma': sigma}
