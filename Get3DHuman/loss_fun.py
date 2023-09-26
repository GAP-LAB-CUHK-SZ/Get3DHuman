

import numpy as np
from libs.DepthNormalizer import DepthNormalizer
from libs.geometry import orthogonal, index 
import torch.nn as nn
import torch.nn.functional as F 
import torch

from torch_utils.ops import conv2d_gradfix

error_term={'mse': nn.MSELoss(), 'l1': nn.L1Loss()}

style_mixing_prob, r1_gamma, pl_batch_shrink, pl_decay, pl_weight = 0.9, 10, 2, 0.01, 2


def surface_rgb_loss(mlp, pc, calibs, feature_gt, feature_pred, spatial_enc,weight):
    sdf_gt = query_rgb_mlp(pc, calibs, spatial_enc, feature_gt, mlp)
    sdf_pred = query_rgb_mlp(pc, calibs, spatial_enc, feature_pred, mlp)
    
    return l1_loss(sdf_gt, sdf_pred)*weight, l2_loss(sdf_gt, sdf_pred)*weight, [sdf_gt, sdf_pred]

def query_rgb_mlp(samples, calibs, spatial_enc,feature, mlp):
    xyz = orthogonal(samples, calibs, None)
    xy = xyz[:, :2, :]

    sp_feat = spatial_enc(xyz, calibs=calibs)
    # intermediate_preds_list = []
    # phi = None
    point_local_feat_list = [index(feature, xy), sp_feat]       
    point_local_feat = torch.cat(point_local_feat_list, 1)
    pred, _ = mlp(point_local_feat)
    return pred



def path_reg(gen_img, gen_ws, pl_mean):
    # pl_mean = torch.zeros([], device=cuda)
    # batch_size_g = gen_img.shape[0] // pl_batch_shrink
    pl_noise = torch.randn_like(gen_img) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])
    with conv2d_gradfix.no_weight_gradients():
        pl_grads = torch.autograd.grad(outputs=[(gen_img * pl_noise).sum()], inputs=[gen_ws], create_graph=True, only_inputs=True)[0]
    pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
    pl_mean = pl_mean.lerp(pl_lengths.mean(), pl_decay)
    pl_mean.copy_(pl_mean.detach())
    pl_penalty = (pl_lengths - pl_mean).square()
    loss_Gpl = pl_penalty * pl_weight
    return loss_Gpl

def d_r1_loss(real_pred, real_img):
    with conv2d_gradfix.no_weight_gradients():
        r1_grads = torch.autograd.grad(outputs=[real_pred.sum()], inputs=[real_img], create_graph=True, only_inputs=True)[0]
    r1_penalty = r1_grads.square().sum([1,2,3])
    grad_penalty = r1_penalty * (r1_gamma / 2)
    return grad_penalty


def norm_loss(fake_n, real_n, mask = None) :
        #   fake_n,real_n =  pred_n_f_, normal_f_tensor
        '''
        fake_map_arr = fake_map.detach().cpu().numpy()
        fake_map_tf = tf.constant(fake_map_arr)
        output_mask_tf = tf.abs(fake_map_arr) < 1e-5
        output_no0_tf = tf.where(output_mask_tf, 1e-5*torch.ones([1,3,512,512]), fake_map_tf)
        output_mag_tf  = tf.expand_dims(tf.sqrt(tf.reduce_sum(tf.square(output_no0_tf ),3)),-1)
        output_unit_tf  = tf.divide(output_no0_tf ,output_mag_tf )

        '''        
        fake_map_0 = fake_n.permute([0,2,3,1])
        real_n_0 = real_n.permute([0,2,3,1])
#        fake_map_0 = fake_n
#        real_n_0 = real_n
        output_mask_t = torch.abs(fake_map_0)< 1e-5
        
        output_no0_t = torch.where(output_mask_t, 1e-5*torch.ones_like(fake_map_0), fake_map_0)
        output_mag_t = torch.unsqueeze(torch.sqrt(torch.sum(torch.square(output_no0_t),3)),-1)
        output_unit_t = torch.div(output_no0_t,output_mag_t)
#        real_mask_t = real_n_0 > -0.98
        real_mask_t = real_n_0 != 0
        
        z_mask_t = real_mask_t[...,0]
        a11_t = torch.masked_select(torch.sum(torch.square(output_unit_t),3),z_mask_t)
        a22_t = torch.masked_select(torch.sum(torch.square(real_n_0),3),z_mask_t)
        a12_t = torch.masked_select(torch.sum(torch.mul(output_unit_t,real_n_0),3),z_mask_t)
    
        cos_angle_t = a12_t/torch.sqrt(torch.mul(a11_t,a22_t))
        cos_angle_clipped_t = torch.clamp(torch.where(torch.isnan(cos_angle_t),-1*torch.ones_like(cos_angle_t),cos_angle_t),-1,1)
        loss_geonorm = torch.mean(3.1415926/2-cos_angle_clipped_t-torch.pow(cos_angle_clipped_t,3)/6-torch.pow(cos_angle_clipped_t,5)*3/40-torch.pow(cos_angle_clipped_t,7)*5/112-torch.pow(cos_angle_clipped_t,9)*35/1152)
 
        return loss_geonorm 


def l1_loss(gt, pred):
    return error_term["l1"](gt, pred)


def l2_loss(gt, pred):
    return error_term["mse"](gt, pred)



def surface_loss(mlp, mlp_gan, pc, calibs, feature_gt, feature_pred, spatial_enc,weight):
    sdf_gt = query_mlp(pc, calibs, spatial_enc, feature_gt, mlp)
    sdf_pred = query_mlp(pc, calibs, spatial_enc, feature_pred, mlp_gan)
    
    return l1_loss(sdf_gt, sdf_pred)*weight, l2_loss(sdf_gt, sdf_pred)*weight, [sdf_gt, sdf_pred]


def query_mlp( samples, calibs, spatial_enc, feature, mlp):
    xyz = orthogonal(samples, calibs, None)
    xy = xyz[:, :2, :]
    in_img = (xy[:, 0] >= -1.0) & (xy[:, 0] <= 1.0) & (xy[:, 1] >= -1.0) & (xy[:, 1] <= 1.0)
    in_bb = in_img[:, None, :].detach().float()
    sp_feat = spatial_enc(xyz, calibs=calibs)
    # intermediate_preds_list = []
    phi = None
    point_local_feat_list = [index(feature, xy), sp_feat]       
    point_local_feat = torch.cat(point_local_feat_list, 1)
    pred, phi = mlp(point_local_feat)
    pred = in_bb * pred
    pred =  (1-in_bb)*0.9 +pred
        
    return pred
    
# def reconstruction_sdf_mlp(mlp, feature, spatial_enc, cuda, calib_tensor,
#                    coords,mat, use_octree=False, num_samples=10000):

#     def eval_func(points):
#         points = np.expand_dims(points, axis=0)
#         points = np.repeat(points,1, axis=0)
#         samples = torch.from_numpy(points).to(device=cuda).float()
#         # net.query(samples, calib_tensor)
#         # pred = net.get_preds()[0][0]
#         pred = query_mlp( samples, calib_tensor, spatial_enc, feature, mlp)
#         return pred.detach().cpu().numpy()
# #    print(use_octree)
#     # Then we evaluate the grid
#     if use_octree:
#         sdf = eval_grid_octree(coords, eval_func, num_samples=num_samples)
# #        sdf = np.where(sdf>0,0,1)
#     else:
#         sdf = eval_grid(coords, eval_func, num_samples=num_samples)
#     return sdf,mat







