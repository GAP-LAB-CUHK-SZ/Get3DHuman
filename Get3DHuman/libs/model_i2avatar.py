import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 
from libs.BasePIFuNet import BasePIFuNet
from libs.MLP import MLP

from DepthNormalizer_selfpifu import DepthNormalizer
from libs.HGFilters_hd import HGFilter
from libs.net_util import init_net
from libs.networks import define_G
import cv2
# import model_u
# from diff_render.renderer import SDFRenderer
# from diff_render.decoder_utils import load_decoder
from libs.models_social_media import Generate
from libs.geometry import orthogonal, index 

class i2avatar(BasePIFuNet):
    '''
     uses stacked hourglass as an image encoder.
    '''

    def __init__(self):
        super(i2avatar, self).__init__()
        self.name = 'i2avatar'

        self.netG_n = define_G(3, 3, 64, "global", 4, 9, 1, 3, "instance")
#       
        self.netG_d = Generate(6,1)
        self.i2m = Generate(3,1)
        
     
        self.image_filter = HGFilter(1, 2, 3, 256,  'group', 'ave_pool', False)

        self.mlp = MLP([257, 512, 256, 128, 1], 2, [1, 2], norm=None, last_op=nn.Sigmoid())

        self.spatial_enc = DepthNormalizer()
# 
        init_net(self)
        print('img2avatar initialized')
        
        
        self.image_filter_texture = HGFilter(1, 2, 3, 256,  'group', 'ave_pool', False)
        self.mlp_texture = MLP([513, 1024, 512, 256, 128, 3], 2, [2, 3], norm=None, last_op=nn.Sigmoid())
        
        
        

    def filter_texture(self, images):
        pred_m = self.i2m.forward(images)
        pred_m_s = pred_m > 0.5
        pred_m_s_3 = pred_m_s.repeat(1,3,1,1)*1
        
        pred_norm = self.netG_n.forward(images*pred_m_s_3)
        pred_depth = self.netG_d.forward(torch.cat([images*pred_m_s_3, pred_norm],1))
        pred_depth = pred_depth.repeat([1,3,1,1])
        
        shape_feature = self.filter_dep(pred_depth)
        
        tex_feature = self.filter_image(images*pred_m_s_3)
        
        # self.im_feat_all = torch.cat([im_feat, self.im_feat_texture[0][0]], 1)
        return shape_feature, tex_feature, pred_m_s_3, pred_norm, pred_depth
        
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
    
        
    def query_texture_(self, im_feat_all, points, calibs, spatial_enc, labels=None):
        '''
        im_feat_all, points, calibs = model_RGB.im_feat_all, sample_local, calib_tensor
        '''
        if labels is not None:
            self.labels_rgb = labels

        xyz_rgb = self.projection(points, calibs, None)
        xy_rgb = xyz_rgb[:, :2, :]
        self.xyz_rgb = xyz_rgb
        sp_feat_rgb = spatial_enc(xyz_rgb)

        texture_feat_list = [self.index(im_feat_all, xy_rgb), sp_feat_rgb]    
        point_local_feat_texture = torch.cat(texture_feat_list, 1)
        
        # point_local_feat = torch.cat(point_local_feat_list, 1)
        self.pred_rgb = self.mlp_texture(point_local_feat_texture)[0]
        
    
    def img2dep(self,images):
        pred_m = self.i2m.forward(images)
        pred_m_s = pred_m > 0.5
        pred_m_s_3 = pred_m_s.repeat(1,3,1,1)*1
        
        pred_norm = self.netG_n.forward(images*pred_m_s_3)
        pred_depth = self.netG_d.forward(torch.cat([images*pred_m_s_3, pred_norm],1))
        pred_depth = pred_depth.repeat([1,3,1,1])
        
        return pred_m_s_3, pred_norm, pred_depth
    
    def img2norm(self,images):
#        input_img = torch.cat([images, denspose],1)
#        pred_norm = self.netG_n.forward(torch.cat([images, denspose],1))
        pred_norm = self.netG_n.forward(images)
        return pred_norm
    
    
    def img2depth(self,images, denspose, normal):
        pred_depth = self.netG_d.forward(torch.cat([images, normal],1))
        pred_depth = pred_depth.repeat([1,3,1,1])
        return pred_depth
    
    def filter_image(self, image):
        '''
        apply a fully convolutional network to images.
        the resulting feature will be stored.
        args:
            images: [B, C, H, W]
        '''
        im_feat_list, _ = self.image_filter_texture(image)

        return im_feat_list[0]
    
    
    def filter_dep(self, depth):
        '''
        apply a fully convolutional network to images.
        the resulting feature will be stored.
        args:
            images: [B, C, H, W]
        '''
        im_feat_list, _ = self.image_filter(depth)
        return im_feat_list[0]
    
    def filter(self, depth):
        '''
        apply a fully convolutional network to images.
        the resulting feature will be stored.
        args:
            images: [B, C, H, W]
        '''

#        self.im_feat_list, self.normx = self.image_filter(torch.cat([depth, images, norm],1))
        self.im_feat_list, self.normx = self.image_filter(depth)
        # If it is not in training, only produce the last im_feat
        if not self.training:
            self.im_feat_list = [self.im_feat_list[-1]]
        
    def query(self, points, calibs, transforms=None, labels=None, update_pred=True, update_phi=True):
        '''
        given 3d points, we obtain 2d projection of these given the camera matrices.
        filter needs to be called beforehand.
        the prediction is stored to self.preds
        args:
            points: [B, 3, N] 3d points in world space
            calibs: [B, 3, 4] calibration matrices for each image
            transforms: [B, 2, 3] image space coordinate transforms
            labels: [B, C, N] ground truth labels (for supervision only)
        return:
            [B, C, N] prediction
        '''
        if labels is not None:
            self.labels = labels

        xyz = self.projection(points, calibs, transforms)
        xy = xyz[:, :2, :]
        
        self.xyz = xyz

        in_img = (xy[:, 0] >= -1.0) & (xy[:, 0] <= 1.0) & (xy[:, 1] >= -1.0) & (xy[:, 1] <= 1.0)
        in_bb = in_img[:, None, :].detach().float()
        
        self.in_bb = in_bb

        if labels is not None:
#            self.labels = labels
            self.labels = (1-in_bb)*0.9 + in_bb * labels

        sp_feat = self.spatial_enc(xyz, calibs=calibs)

        intermediate_preds_list = []

        phi = None
#        for i, im_feat in enumerate(a):
#            point_local_feat_list = [index(im_feat, xy), sp_feat]       
#            point_local_feat = torch.cat(point_local_feat_list, 1)
            
        for i, im_feat in enumerate(self.im_feat_list):
            point_local_feat_list = [self.index(im_feat, xy), sp_feat]       
            point_local_feat = torch.cat(point_local_feat_list, 1)
            pred, phi = self.mlp(point_local_feat)
            pred = in_bb * pred
            pred =  (1-in_bb)*0.9 +pred

            intermediate_preds_list.append(pred)
        
        self.point_local_feat =point_local_feat
        if update_phi:
            self.phi = phi

        if update_pred:
            self.intermediate_preds_list = intermediate_preds_list
            self.preds = self.intermediate_preds_list[-1]
            
            
