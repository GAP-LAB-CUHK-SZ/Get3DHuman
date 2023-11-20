import cv2
import numpy as np
import os
from PIL import Image, ImageOps
from PIL.ImageFilter import GaussianBlur
import random
import torchvision.transforms as transforms
import torch

from PIL import Image
import numpy as np
import torch
import os
import cv2
from PIL import Image, ImageOps
import torchvision.transforms as transforms
import random
import platform

from libs.DepthNormalizer import DepthNormalizer
from libs.geometry import orthogonal, index 

from skimage import measure
import math


to_tensor_512 = transforms.Compose([transforms.Resize(512),transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) ])

def inference_avatar(Get3DHuman, model_i2i, shape_field, texture_feature, args, calib_define, calib, kernel, spatial_enc, R, T, cuda):
    point_rgb_all = np.zeros([1,6])
    
    save_imgs = []
    for cam_x_i in [180, 0]:
        cam_x, cam_y = torch.tensor(cam_x_i*1.0).to(device=cuda), torch.tensor(0.0).to(device=cuda)
        
        with torch.no_grad():
            empty_image = torch.ones([shape_field.size(0), args.render_size*args.render_size]).to(device=cuda) >2
            render_output = Get3DHuman.sdf_renderer.render_ortho_points(cam_x, cam_y, [shape_field, calib, orthogonal, spatial_enc, index, empty_image],
                                                           R, T, no_grad=True,  sample_index_type='min_abs', no_grad_depth=True, no_grad_normal=True)
            depth_rendered, min_points = render_output
            mask = (depth_rendered<2)*(depth_rendered>0.4 ) 
            mask_3_pred = (mask*1.0).detach().cpu().numpy()[0]
            mask_3_pred = cv2.dilate(mask_3_pred,kernel,iterations = 1)
            mask_3_pred = torch.tensor((mask_3_pred[None,None,:,:]).repeat(3,1)*1).cuda()

            depth_pred = ((depth_rendered/2) * mask)[0].detach().cpu().numpy()
            
            # 
            # mask_3_pred_show = mask_3_pred.detach().cpu().numpy()[0,0]
            # mask_show = mask.detach().cpu().numpy()[0]
            
            
            mask_image = (mask*1.0).detach().cpu().numpy()[0]
            mask_image = np.repeat(mask_image[:,:,np.newaxis], 3,2)
    
        with torch.no_grad():
            pred_texture = Get3DHuman.decode_texture(Get3DHuman.mlp_texture, [texture_feature, calib, orthogonal, spatial_enc, index], min_points).squeeze(-1)
            pred_texture = pred_texture.reshape(-1, 3,  args.render_size,  args.render_size)
        
        
        img_pred = ((pred_texture*mask_3_pred) + 1-mask_3_pred)
        img_pred_show = img_pred.permute(0,2,3,1).detach().cpu().numpy()[0]
        
        with torch.no_grad():
            img_refine = model_i2i.img2img(img_pred)
            img_refine_show = ((img_refine+1)/2)[0].permute(1,2,0).detach().cpu().numpy()
        
        img_pred_crop_ = img_pred_show*mask_image + (1-mask_image)
        img_pred_crop = crop_render(img_pred_crop_, mask_image)
        
        img_refine_crop_ = img_refine_show*mask_image + (1-mask_image)
        img_refine_crop = crop_render(img_refine_crop_, mask_image)

        

        save_imgs.append(img_refine_crop)
       
        img_r = np.flip(img_refine_show*255, 0)
        img_r = img_r[...,[2,1,0]]
        
        data_uint16 = np.flip(depth_pred*255*255, 0) # x
        mask_r = np.where(data_uint16 < 10000,0,1)
        dep_map = data_uint16.copy()
        h, w = dep_map.shape
        vid, uid = np.where(mask_r > 0.01)

        uv_mat = np.ones((len(vid), 4))
        z_buffer_data_numpy  = dep_map[vid, uid] / 255./ 255.
        uv_mat[:,2] = z_buffer_data_numpy
        uv_mat[:, 0] = uid/(args.render_size)
        uv_mat[:, 1] = vid/(args.render_size)
        uv_mat= uv_mat*2-1
        uv_mat_4 = uv_mat
        uv_mat_test_2 = np.matmul(uv_mat_4, calib_define)
        uv_mat_test_2_r = uv_mat_test_2.copy()
        cam_x, cam_y = 0, math.radians(cam_x_i)
        rotate = make_rotate(cam_x, cam_y, 0)
        uv_mat_test_2_r[:,:3] = np.matmul(uv_mat_test_2[:,:3], rotate)
        vert = uv_mat_test_2_r
        rgb_buffer_numpy  = img_r[vid, uid]
        point_rgb = np.concatenate([vert[:,:3], rgb_buffer_numpy],1)
        point_rgb_all = np.concatenate((point_rgb_all, point_rgb),0)
        
    return point_rgb_all, save_imgs
    



def create_grid(resX, resY, resZ, b_min=np.array([0, 0, 0]), b_max=np.array([1, 1, 1]), transform=None):
    '''
    Create a dense grid of given resolution and bounding box
    :param resX: resolution along X axis
    :param resY: resolution along Y axis
    :param resZ: resolution along Z axis
    :param b_min: vec3 (x_min, y_min, z_min) bounding box corner
    :param b_max: vec3 (x_max, y_max, z_max) bounding box corner
    :return: [3, resX, resY, resZ] coordinates of the grid, and transform matrix from mesh index
    '''
    coords = np.mgrid[:resX, :resY, :resZ]
    coords = coords.reshape(3, -1)
    coords_matrix = np.eye(4)
    length = b_max - b_min
    coords_matrix[0, 0] = length[0] / resX
    coords_matrix[1, 1] = length[1] / resY
    coords_matrix[2, 2] = length[2] / resZ
    coords_matrix[0:3, 3] = b_min
    coords = np.matmul(coords_matrix[:3, :3], coords) + coords_matrix[:3, 3:4]
    if transform is not None:
        coords = np.matmul(transform[:3, :3], coords) + transform[:3, 3:4]
        coords_matrix = np.matmul(transform, coords_matrix)
    coords = coords.reshape(3, resX, resY, resZ)
    return coords, coords_matrix


res = 512
b_min = np.array([-128,  -28, -128])
b_max = np.array([128, 228, 128])
coords, mat = create_grid(res, res, res, b_min, b_max, transform=None)

def crop_render(img_array, mask, zone=6):
    # img_array = np.array(image)
    w,h,_ = img_array.shape
    img_bg_index = np.where(mask> 0.1) 
    img_h_max, img_h_min = min(img_bg_index[0].max() + zone, w-1), max(img_bg_index[0].min()-zone, 0)
    img_w_max, img_w_min = min(img_bg_index[1].max() + zone, w-1), max(img_bg_index[1].min()-zone, 0)
    
    crop_array = img_array[img_h_min:img_h_max,img_w_min:img_w_max,:]
    
    return crop_array


def test_avatar(save_root_1, spatial_enc, random_S, random_C, epoch, save_index, mat, save_root, model, model_sdf, calib_tensor_real, cuda_rank):
    # model = I3DGAN_model
    # random_S, random_C = z_shape_r, z_shape_r
    # calib_tensor_real
    shape_field, texture_field = model.module.infer_avatar(random_S, 0, random_C, 0)
    sdf_pred, mat  = reconstruction_sdf_mlp(model_sdf.module.mlp, shape_field, spatial_enc, cuda_rank, calib_tensor_real, coords, mat)
    sdf_pred_ = 1 - sdf_pred
    verts_pred, faces_pred, normals_pred, values_pred = measure.marching_cubes(sdf_pred_, 0.5)
     # transform verts into world coordinate system
    verts_pred = np.matmul(mat[:3, :3], verts_pred.T) + mat[:3, 3:4]
    verts_pred = verts_pred.T
    
    verts_train_query = torch.tensor(verts_pred.T.copy()).float().to(device=cuda_rank)[None]
    # pre_rgb_train = []
    num_sample_color = 10000
    
    # infer_texture_mlp = model.mlp_texture
    infer_texture_cache = [torch.cat([shape_field,texture_field],1), model.module.mlp_texture]
    # infer_texture_cache = [model_sdf.im_feat_all, model_sdf.mlp_texture]
    # model_sdf
    pre_rgb_train = []
    for sample_i in range(int(verts_train_query.shape[2]/num_sample_color)):
        verts_train_query_i = verts_train_query[:,:,sample_i*num_sample_color:(sample_i+1)*num_sample_color]
        # model_sdf.query_texture(model_sdf.im_feat_all, verts_train_query_i, calib_tensor_real)
        rgb_cache = model.module.query_texture(infer_texture_cache, verts_train_query_i, calib_tensor_real, spatial_enc)
        pre_rgb_train.append(rgb_cache.detach().cpu().numpy())
    sample_i  = sample_i + 1
    verts_train_query_i = verts_train_query[:,:,sample_i*num_sample_color:]
    # model_sdf.query_texture(model_sdf.im_feat_all, verts_train_query_i, calib_tensor_real)
    rgb_cache = model.module.query_texture(infer_texture_cache, verts_train_query_i, calib_tensor_real, spatial_enc)
    pre_rgb_train.append(rgb_cache.detach().cpu().numpy())
    pre_rgb_train_ = np.concatenate(pre_rgb_train,-1)*255
    
    save_obj_mesh_rgb('%s/color_%02d_%04d.obj'% (save_root_1, save_index, epoch),verts_pred,faces_pred, pre_rgb_train_[0]/255)

def test_avatar_infer(save_root_1, spatial_enc, random_S, random_C, epoch, save_index, mat, save_root, model, model_sdf, calib_tensor_real, cuda_rank):
    # model = I3DGAN_model
    # random_S, random_C = z_shape_r, z_shape_r
    # calib_tensor_real
    shape_field, texture_field = model.infer_avatar(random_S, 0, random_C, 0)
    sdf_pred, mat  = reconstruction_sdf_mlp(model_sdf.mlp, shape_field, spatial_enc, cuda_rank, calib_tensor_real, coords, mat)
    sdf_pred_ = 1 - sdf_pred
    verts_pred, faces_pred, normals_pred, values_pred = measure.marching_cubes(sdf_pred_, 0.5)
     # transform verts into world coordinate system
    verts_pred = np.matmul(mat[:3, :3], verts_pred.T) + mat[:3, 3:4]
    verts_pred = verts_pred.T
    
    verts_train_query = torch.tensor(verts_pred.T.copy()).float().to(device=cuda_rank)[None]
    # pre_rgb_train = []
    num_sample_color = 10000
    
    # infer_texture_mlp = model.mlp_texture
    infer_texture_cache = [torch.cat([shape_field,texture_field],1), model.mlp_texture]
    # infer_texture_cache = [model_sdf.im_feat_all, model_sdf.mlp_texture]
    # model_sdf
    pre_rgb_train = []
    for sample_i in range(int(verts_train_query.shape[2]/num_sample_color)):
        verts_train_query_i = verts_train_query[:,:,sample_i*num_sample_color:(sample_i+1)*num_sample_color]
        # model_sdf.query_texture(model_sdf.im_feat_all, verts_train_query_i, calib_tensor_real)
        rgb_cache = model.query_texture(infer_texture_cache, verts_train_query_i, calib_tensor_real, spatial_enc)
        pre_rgb_train.append(rgb_cache.detach().cpu().numpy())
    sample_i  = sample_i + 1
    verts_train_query_i = verts_train_query[:,:,sample_i*num_sample_color:]
    # model_sdf.query_texture(model_sdf.im_feat_all, verts_train_query_i, calib_tensor_real)
    rgb_cache = model.query_texture(infer_texture_cache, verts_train_query_i, calib_tensor_real, spatial_enc)
    pre_rgb_train.append(rgb_cache.detach().cpu().numpy())
    pre_rgb_train_ = np.concatenate(pre_rgb_train,-1)*255
    
    save_obj_mesh_rgb('%s/color_%02d_%04d.obj'% (save_root_1, save_index, epoch),verts_pred,faces_pred, pre_rgb_train_[0]/255)
   
def avatar_infer_shape(spatial_enc, cuda, calib_r, mat, shape_field, save_root, mlp, coords):
    # model = I3DGAN_model
    # random_S, random_C = z_shape_r, z_shape_r
    # shape_field, texture_field = model.infer_avatar(random_S, 0, random_C, 0)
    sdf_pred, mat  = reconstruction_sdf_mlp(mlp, shape_field, spatial_enc, cuda, calib_r, coords, mat)
    sdf_pred_ = 1 - sdf_pred
    verts_pred, faces_pred, normals_pred, values_pred = measure.marching_cubes(sdf_pred_, 0.5)
     # transform verts into world coordinate system
    verts_pred = np.matmul(mat[:3, :3], verts_pred.T) + mat[:3, 3:4]
    verts_pred = verts_pred.T
    
    # verts_train_query = torch.tensor(verts_pred.T.copy()).float().to(device=cuda)[None]
    return verts_pred, faces_pred


def make_rotate(rx, ry, rz):
    sinX = np.sin(rx)
    sinY = np.sin(ry)
    sinZ = np.sin(rz)

    cosX = np.cos(rx)
    cosY = np.cos(ry)
    cosZ = np.cos(rz)

    Rx = np.zeros((3,3))
    Rx[0, 0] = 1.0
    Rx[1, 1] = cosX
    Rx[1, 2] = -sinX
    Rx[2, 1] = sinX
    Rx[2, 2] = cosX

    Ry = np.zeros((3,3))
    Ry[0, 0] = cosY
    Ry[0, 2] = sinY
    Ry[1, 1] = 1.0
    Ry[2, 0] = -sinY
    Ry[2, 2] = cosY

    Rz = np.zeros((3,3))
    Rz[0, 0] = cosZ
    Rz[0, 1] = -sinZ
    Rz[1, 0] = sinZ
    Rz[1, 1] = cosZ
    Rz[2, 2] = 1.0
    R = np.matmul(np.matmul(Rz,Ry),Rx)
    return R

def avatar_infer_texture(spatial_enc, cuda, calib_r, shape_field, texture_field, name, verts_pred, faces_pred, save_root, model):
    verts_train_query = torch.tensor(verts_pred.T.copy()).float().to(device=cuda)[None]
    # pre_rgb_train = []
    num_sample_color = 10000
    
    # infer_texture_mlp = model.mlp_texture
    infer_texture_cache = [torch.cat([shape_field,texture_field],1), model.mlp_texture]
    # infer_texture_cache = [model_sdf.im_feat_all, model_sdf.mlp_texture]
    # model_sdf
    pre_rgb_train = []
    for sample_i in range(int(verts_train_query.shape[2]/num_sample_color)):
        verts_train_query_i = verts_train_query[:,:,sample_i*num_sample_color:(sample_i+1)*num_sample_color]
        # model_sdf.query_texture(model_sdf.im_feat_all, verts_train_query_i, calib_tensor_real)
        rgb_cache = model.query_texture(infer_texture_cache, verts_train_query_i, calib_r, spatial_enc)
        pre_rgb_train.append(rgb_cache.detach().cpu().numpy())
    sample_i  = sample_i + 1
    verts_train_query_i = verts_train_query[:,:,sample_i*num_sample_color:]
    # model_sdf.query_texture(model_sdf.im_feat_all, verts_train_query_i, calib_tensor_real)
    rgb_cache = model.query_texture(infer_texture_cache, verts_train_query_i, calib_r, spatial_enc)
    pre_rgb_train.append(rgb_cache.detach().cpu().numpy())
    pre_rgb_train_ = np.concatenate(pre_rgb_train,-1)
    
    save_obj_mesh_rgb('%s/%s.obj'% (save_root, name),verts_pred,faces_pred, pre_rgb_train_[0])
    



def save_obj_mesh_rgb(mesh_path, verts, faces, rgb):
    file = open(mesh_path, 'w')
    # RGB_ = rgb.astype("uint8")
    RGB_ = rgb
    for v_index, v in enumerate(verts):
        file.write('v %.4f %.4f %.4f %f %f %f\n' % (v[0], v[1], v[2], RGB_[:,v_index][0], RGB_[:,v_index][1], RGB_[:,v_index][2],))
    for f in faces:
        f_plus = f + 1
        file.write('f %d %d %d\n' % (f_plus[0], f_plus[2], f_plus[1]))
    file.close()
    
def recon_sample_feat(random_z, epoch, save_index, mat, save_root, feature_shape_sample, mlp, spatial_enc, cuda, coords):
    # random_z = np.load(r'G:\E\PhD\Program_stylepifu\data\style_pifu\imgs_step1\img00000_z.npy')
    calib_tensor_real =  torch.Tensor(np.array([[[ 0.00927734,  0.0000,  0.0000,  0.0000],
                                           [ 0.0000, -0.00927734,  0.0000,  0.88134766],
                                           [ 0.0000,  0.0000,  0.00927734,  0.0000],
                                           [ 0.0000,  0.0000,  0.0000,  1.0000]]])).to(device=cuda)
    
    # feature_shape_sample,_ = model_infer(torch.tensor(random_z), 0)
   
    sdf_pred, mat  = reconstruction_sdf_mlp(mlp, feature_shape_sample, spatial_enc, cuda, calib_tensor_real, coords, mat)
    sdf_pred_ = 1 - sdf_pred
    verts_pred, faces_pred, normals_pred, values_pred = measure.marching_cubes(sdf_pred_, 0.5)
   
     # transform verts into world coordinate system
    verts_pred = np.matmul(mat[:3, :3], verts_pred.T) + mat[:3, 3:4]
    verts_pred = verts_pred.T
    save_obj_mesh(os.path.join(save_root,'test_%06d_%02d.obj'%(epoch, save_index)) ,verts_pred,faces_pred)
 



def recon_sample_n(random_z, epoch, save_index, mat, save_root, model_infer, model_sdf, spatial_enc, cuda, coords):
    # random_z = np.load(r'G:\E\PhD\Program_stylepifu\data\style_pifu\imgs_step1\img00000_z.npy')
    calib_tensor_real =  torch.Tensor(np.array([[[ 0.00927734,  0.0000,  0.0000,  0.0000],
                                           [ 0.0000, -0.00927734,  0.0000,  0.88134766],
                                           [ 0.0000,  0.0000,  0.00927734,  0.0000],
                                           [ 0.0000,  0.0000,  0.0000,  1.0000]]])).to(device=cuda)
    
    feature_shape_sample, feature_n = model_infer(torch.tensor(random_z), 0)
   
    sdf_pred, mat  = reconstruction_sdf_mlp(model_sdf.mlp, feature_shape_sample, spatial_enc, cuda, calib_tensor_real, coords, mat)
    sdf_pred_ = 1 - sdf_pred
    verts_pred, faces_pred, normals_pred, values_pred = measure.marching_cubes(sdf_pred_, 0.5)
   
     # transform verts into world coordinate system
    verts_pred = np.matmul(mat[:3, :3], verts_pred.T) + mat[:3, 3:4]
    verts_pred = verts_pred.T
    save_obj_mesh(os.path.join(save_root,'test_%06d_%02d.obj'%(epoch, save_index)) ,verts_pred,faces_pred)
    
    return feature_n

def recon_sample(random_z, epoch, save_index, mat, save_root, model_infer, model_sdf, spatial_enc, cuda, coords):
    # random_z = np.load(r'G:\E\PhD\Program_stylepifu\data\style_pifu\imgs_step1\img00000_z.npy')
    calib_tensor_real =  torch.Tensor(np.array([[[ 0.00927734,  0.0000,  0.0000,  0.0000],
                                           [ 0.0000, -0.00927734,  0.0000,  0.88134766],
                                           [ 0.0000,  0.0000,  0.00927734,  0.0000],
                                           [ 0.0000,  0.0000,  0.0000,  1.0000]]])).to(device=cuda)
    
    feature_shape_sample,_ = model_infer(torch.tensor(random_z), 0)
   
    sdf_pred, mat  = reconstruction_sdf_mlp(model_sdf.mlp, feature_shape_sample, spatial_enc, cuda, calib_tensor_real, coords, mat)
    sdf_pred_ = 1 - sdf_pred
    verts_pred, faces_pred, normals_pred, values_pred = measure.marching_cubes(sdf_pred_, 0.5)
   
     # transform verts into world coordinate system
    verts_pred = np.matmul(mat[:3, :3], verts_pred.T) + mat[:3, 3:4]
    verts_pred = verts_pred.T
    save_obj_mesh(os.path.join(save_root,'test_%06d_%02d.obj'%(epoch, save_index)) ,verts_pred,faces_pred)
 


def query_mlp( points, calibs, spatial_enc, feature, mlp):
    xyz = orthogonal(points, calibs, None)
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
    
def reconstruction_sdf_mlp(mlp, feature, spatial_enc, cuda, calib_tensor,
                   coords,mat, use_octree=False, num_samples=10000):

    def eval_func(points):
        points = np.expand_dims(points, axis=0)
        points = np.repeat(points,1, axis=0)
        samples = torch.from_numpy(points).to(device=cuda).float()
        # net.query(samples, calib_tensor)
        # pred = net.get_preds()[0][0]
        pred = query_mlp( samples, calib_tensor, spatial_enc, feature, mlp)
        return pred.detach().cpu().numpy()
#    print(use_octree)
    # Then we evaluate the grid
    if use_octree:
        sdf = eval_grid_octree(coords, eval_func, num_samples=num_samples)
#        sdf = np.where(sdf>0,0,1)
    else:
        sdf = eval_grid(coords, eval_func, num_samples=num_samples)
    return sdf,mat




def reconstruction_sdf(net, cuda, calib_tensor,
                   coords,mat, use_octree=False, num_samples=10000):
    '''
    Reconstruct meshes from sdf predicted by the network.
    :param net: a BasePixImpNet object. call image filter beforehead.
    :param cuda: cuda device
    :param calib_tensor: calibration tensor
    :param resolution: resolution of the grid cell
    :param b_min: bounding box corner [x_min, y_min, z_min]
    :param b_max: bounding box corner [x_max, y_max, z_max]
    :param use_octree: whether to use octree acceleration
    :param num_samples: how many points to query each gpu iteration
    :return: marching cubes results.
    '''
    # First we create a grid by resolution
    # and transforming matrix for grid coordinates to real world xyz
#    coords, mat = create_grid(resolution, resolution, resolution,
#                              b_min, b_max, transform=transform)
#    
    # net, cuda, calib_tensor, resolution, b_min, b_max = net, cuda, calib_tensor, opt.resolution, b_min, b_max
    
    # Then we define the lambda function for cell evaluation
    
    def eval_func(points):
        points = np.expand_dims(points, axis=0)
        points = np.repeat(points,1, axis=0)
        samples = torch.from_numpy(points).to(device=cuda).float()
        net.query(samples, calib_tensor)
        pred = net.get_preds()[0][0]
        return pred.detach().cpu().numpy()
#    print(use_octree)
    # Then we evaluate the grid
    if use_octree:
        sdf = eval_grid_octree(coords, eval_func, num_samples=num_samples)
#        sdf = np.where(sdf>0,0,1)
    else:
        sdf = eval_grid(coords, eval_func, num_samples=num_samples)
    return sdf,mat


def load_real(real_path):
    # real_path =  opt_data.dataroot_real
    if platform.system().lower() == 'windows':
        dep_path = real_path + "_dep"
        emask_path =  real_path + "_emask"
    if platform.system().lower() != 'windows':
        dep_path = real_path[:-1] + "_dep"
        emask_path =  real_path[:-1] + "_emask"
    list_all = []
    img_list = os.listdir(real_path)
    img_list.sort()
    for img_list_i in range(int(len(img_list)/2)):
        list_all.append([os.path.join(real_path, img_list[img_list_i*2]),
                         os.path.join(real_path, img_list[img_list_i*2+1]),
                         os.path.join(dep_path, img_list[img_list_i*2][:-7]+"depth.png"),
                         os.path.join(emask_path, img_list[img_list_i*2][:-7]+"emask.jpg")])
    return list_all



def read_real_erode(mask_path, img_path, dep_path, emask_path, load_size=512 ,aug=True):
    depth_f_img = Image.open(dep_path)
    mask = Image.open(mask_path).convert('L')
    emask = Image.open(emask_path).convert('L')
    render = Image.open(img_path).convert('RGB')
#        dp_img = Image.open(dp_path)
    
#        name_save = os.path.split(render_path)[-1][:3]
    
    if aug == True:
        pad_size = int(np.random.rand(1)/4 * load_size)
        render = ImageOps.expand(render, pad_size, fill=0)
        mask = ImageOps.expand(mask, pad_size, fill=0)
        emask = ImageOps.expand(emask, pad_size, fill=0)
        
        depth_f_img = ImageOps.expand(depth_f_img, pad_size, fill=255*255)
#                dp_img = ImageOps.expand(dp_img, pad_size, fill=0)
        w, h = render.size
#        th, tw = load_size, load_size
        x1, y1, x2, y2 = np.random.randint(0,50,4)
        render = render.crop((x1, y1, w - x2, h - y2))
        mask = mask.crop((x1, y1, w - x2, h - y2))
        emask = emask.crop((x1, y1, w - x2, h - y2))
        depth_f_img = depth_f_img.crop((x1, y1, w - x2, h - y2))
        
        
    depth_f_img_512 = depth_f_img.resize((512,512), Image.BILINEAR)
    depth_f_img_512 = np.array(depth_f_img_512)/65025
    depth_f_img_512 = np.repeat(depth_f_img_512[np.newaxis,:],3,0)*2-1
    depth_f_img_512 = torch.tensor(depth_f_img_512).float()

    render_512 = render.resize((512, 512), Image.BILINEAR)
    render_512 = to_tensor_512(render_512)
    
#    render_512 = render.resize((512, 512), Image.BILINEAR)
    
    mask = mask.resize((512, 512), Image.NEAREST)
    mask_512 = transforms.ToTensor()(mask).float()
    
    emask = emask.resize((512, 512), Image.NEAREST)
    emask_512 = transforms.ToTensor()(emask).float()
    
#        dp_img = to_tensor_512(dp_img)
    

    render_512 = ( mask_512.expand_as(render_512) * render_512 )[None]
    
    return render_512, mask_512, depth_f_img_512, emask_512

def read_real(mask_path, img_path, dep_path, load_size=512 ,aug=True):
    depth_f_img = Image.open(dep_path)
    mask = Image.open(mask_path).convert('L')
    render = Image.open(img_path).convert('RGB')
#        dp_img = Image.open(dp_path)
    
#        name_save = os.path.split(render_path)[-1][:3]
    
    if aug == True:
        pad_size = int(np.random.rand(1)/4 * load_size)
        render = ImageOps.expand(render, pad_size, fill=0)
        mask = ImageOps.expand(mask, pad_size, fill=0)

        depth_f_img = ImageOps.expand(depth_f_img, pad_size, fill=255*255)
#                dp_img = ImageOps.expand(dp_img, pad_size, fill=0)
        w, h = render.size
#        th, tw = load_size, load_size
        x1, y1, x2, y2 = np.random.randint(0,50,4)
        render = render.crop((x1, y1, w - x2, h - y2))
        mask = mask.crop((x1, y1, w - x2, h - y2))
        depth_f_img = depth_f_img.crop((x1, y1, w - x2, h - y2))
        
        
    depth_f_img_512 = depth_f_img.resize((512,512), Image.BILINEAR)
    depth_f_img_512 = np.array(depth_f_img_512)/65025
    depth_f_img_512 = np.repeat(depth_f_img_512[np.newaxis,:],3,0)*2-1
    depth_f_img_512 = torch.tensor(depth_f_img_512).float()

    render_512 = render.resize((512, 512), Image.BILINEAR)
    render_512 = to_tensor_512(render_512)
    
#    render_512 = render.resize((512, 512), Image.BILINEAR)
    
    mask = mask.resize((512, 512), Image.NEAREST)
#    mask_512 = transforms.Resize(512)(mask)
    mask_512 = transforms.ToTensor()(mask).float()
#        dp_img = to_tensor_512(dp_img)
    

    render_512 = ( mask_512.expand_as(render_512) * render_512 )[None]
    
    return render_512, mask_512, depth_f_img_512
    
    


def depth2mesh(data_uint16, save_path, name_uint16, image_size=512):

#    image_size = 256
    data_uint16 = cv2.resize(data_uint16, (image_size,image_size))
    calib_1 = np.array([[ 107.78947529,    0.  ,    0.        ,    0. ],[   0. ,  107.78947529,    0.,    0.],
                        [   0.        ,    0.  , -107.78947529,    0. ],[   0. ,   95.        ,    0.,    1.]])
    
    
    
    data_uint16 = data_uint16
    
    data_uint16 = np.flip(data_uint16, 0) # x
    #data_uint16 = np.flip(data_uint16, 1)  # y
    #data_uint16 = cv2.flip(data_uint16,1)
    mask = np.where(data_uint16 >= 255*255-10000,0,1)
    dep_map = data_uint16.copy()
    #trans_depth2mesh_uint16(data_uint16,"test_1.obj")
    h, w = dep_map.shape
    
    out_name = os.path.join(save_path,"%s_d2m"%name_uint16)
    
    if mask is None:
        thres = np.mean(dep_map)
        thres = np.clip(thres, a_min=np.min(dep_map), a_max=255*255-10100)
        vid, uid = np.where(dep_map < thres)
    else:
        vid, uid = np.where(mask > 0.01)
    ### calculate the inverse point cloud
    uv_mat = np.ones((len(vid), 4), dtype=np.float)
    #uv_mat[:, 0] = (uid - h / 2.) / (h / 2.)
    z_buffer_data_numpy  = dep_map[vid, uid] / 255./ 255.
    
    uv_mat[:,2] = z_buffer_data_numpy
    uv_mat[:, 0] = uid/image_size
    uv_mat[:, 1] = vid/image_size
    #viewZ =  (z_buffer_data_numpy ) * (cam.near- cam.far ) - cam.near
    
    uv_mat= uv_mat*2-1
    uv_mat_4 = uv_mat
    uv_mat_test_2 = np.matmul(uv_mat_4, calib_1)
    
    vert = uv_mat_test_2
    
    # save vertex and faces:
    f = open(out_name + '.obj', 'w')
    nv = 0
    vidx_map = np.full_like(dep_map, fill_value=-1, dtype=np.int)
    for i in range(0, len(vid)):
        f.write('v %f %f %f\n' % (vert[i][0], vert[i][1], vert[i][2]))
        vidx_map[vid[i], uid[i]] = nv
        nv += 1
    
    for i in range(0, h-2):
        for j in range(0, w-2):
            if vidx_map[i, j] >= 0 and vidx_map[i, j+1] >= 0 and vidx_map[i+1, j] >= 0 and vidx_map[i+1, j+1] >= 0:
    #            f.write('f %d %d %d\n' % (vidx_map[i + 1, j] + 1, vidx_map[i, j + 1] + 1, vidx_map[i, j] + 1))
    #            f.write('f %d %d %d\n' % (vidx_map[i, j + 1] + 1, vidx_map[i + 1, j] + 1, vidx_map[i + 1, j + 1] + 1))
                f.write('f %d %d %d\n' % (vidx_map[i , j] + 1, vidx_map[i, j + 1] + 1, vidx_map[i + 1, j] + 1))
                f.write('f %d %d %d\n' % (vidx_map[i + 1, j + 1] + 1, vidx_map[i + 1, j] + 1, vidx_map[i , j + 1] + 1))
    #            f.write('f %d %d %d\n' % (vidx_map[i + 1, j] +1, vidx_map[i, j + 1] -1, vidx_map[i, j]  -1))
    #            f.write('f %d %d %d\n' % (vidx_map[i, j + 1] +1, vidx_map[i + 1, j]  -1, vidx_map[i + 1, j + 1]  -1))
    
    f.close()
    return vid, uid, vert



def save_mesh(opt, net, cuda, data, save_path, resolution,points_num, use_octree=True):
    #  opt, net, cuda, data, save_path=opt, netG_new_pifu, cuda, test_data, save_path_0
#    image_tensor_512 = data['img_512'].to(device=cuda)
#    image_tensor = data['img'].to(device=cuda)
#    calib_tensor = data['calib'].to(device=cuda)
#    normal_f_tensor = data['normal_f_img_list'].to(device=cuda)
#    normal_b_tensor = data['normal_b_img_list'].to(device=cuda)
#    label_tensor = data['labels']
#    net.filter(image_tensor_512,normal_f_tensor,normal_b_tensor)
    
    image_tensor_512 = data['img_512'].to(device=cuda)
    image_tensor_1024 = data['img_1024'].to(device=cuda)
    image_tensor = data['img'].to(device=cuda)
    calib_tensor = data['calib'].to(device=cuda)
    sample_tensor = data['samples'].to(device=cuda)
    label_tensor = data['labels'].to(device=cuda)
    normal_f_tensor = data['normal_f_img_list'].to(device=cuda)
    normal_b_tensor = data['normal_b_img_list'].to(device=cuda)
    
    depth_f_tensor = data['depth_f_img'].to(device=cuda)
    
    dp_tensor = data['dp_512'].to(device=cuda)
    
    b_min = data['b_min']
    b_max = data['b_max']
    
#    net.forward(image_tensor_1024.unsqueeze(1), normal_f_tensor.unsqueeze(1), normal_b_tensor.unsqueeze(1),
#                                             image_tensor_512.unsqueeze(1),sample_tensor[None,None,:], 
#                                             calib_tensor.unsqueeze(1),calib_tensor,label_tensor[:,None,None,:])
#    
    net.forward(image_tensor_512, normal_f_tensor, depth_f_tensor[:,:,:,:], dp_tensor,
                sample_tensor.unsqueeze(0), calib_tensor,label_tensor)

    iou = 1-torch.sum(torch.logical_xor(net.labels_occ.squeeze(1),net.preds_occ[-1])*1).cpu().numpy()/points_num
                
    try:
        save_img_path = save_path
        save_img_list = []
        for v in range(image_tensor.shape[0]):
            save_img = (np.transpose(image_tensor[v].detach().cpu().numpy(), (1, 2, 0)) * 0.5 + 0.5)[:, :, ::-1] * 255.0
            save_img_list.append(save_img)
        save_img = np.concatenate(save_img_list, axis=1)
        Image.fromarray(np.uint8(save_img[:,:,::-1])).save(save_img_path)
        sdf,mat  = reconstruction(
            net, cuda, calib_tensor, resolution, b_min, b_max, use_octree=use_octree)
        
           
#        sdf = np.where(sdf<0,1,0)
        return sdf, mat, iou, net.pred_norm, net.pred_depth
        

    except Exception as e:
        print(e)
        print('Can not create marching cubes at this time.')
        
def save_hd_mesh(opt, net, cuda, data, save_path, resolution,points_num, use_octree=True):
    #  opt, net, cuda, data, save_path=opt, netG_fine_pifu, cuda, test_data, save_path
#    image_tensor_512 = data['img_512'].to(device=cuda)
#    image_tensor_1024 = data['img_1024'].to(device=cuda)
#    image_tensor = data['img'].to(device=cuda)
#    calib_tensor = data['calib'].to(device=cuda)
#    normal_f_tensor = data['normal_f_img_list'].to(device=cuda)
#    normal_b_tensor = data['normal_b_img_list'].to(device=cuda)
#
#    net.filter_global(image_tensor_512.squeeze(1), normal_f_tensor.squeeze(1), normal_b_tensor.squeeze(1))
#    net.filter_local(image_tensor_1024.unsqueeze(0), normal_f_tensor.squeeze(1), normal_b_tensor.squeeze(1),  rect=None)

    image_tensor_512 = data['img_512'].to(device=cuda)
    image_tensor_1024 = data['img_1024'].to(device=cuda)
    image_tensor = data['img'].to(device=cuda)
    calib_tensor = data['calib'].to(device=cuda)
    sample_tensor = data['samples'].to(device=cuda)
    label_tensor = data['labels'].to(device=cuda)
    normal_f_tensor = data['normal_f_img_list'].to(device=cuda)
    normal_b_tensor = data['normal_b_img_list'].to(device=cuda)
   
    net.forward(image_tensor_1024.unsqueeze(1), normal_f_tensor.unsqueeze(1), normal_b_tensor.unsqueeze(1),
                                             image_tensor_512.unsqueeze(1),sample_tensor[None,None,:], 
                                             calib_tensor.unsqueeze(1),calib_tensor,label_tensor[:,None,None,:])

    b_min = data['b_min']
    b_max = data['b_max']
    
    
    iou_fine = 1-torch.sum(torch.logical_xor(net.labels_occ[0,0,:],net.preds_interm_occ[-1][0,0,:])*1).cpu().numpy()/points_num
            
    try:
        save_img_path = save_path
        save_img_list = []
        for v in range(image_tensor.shape[0]):
            save_img = (np.transpose(image_tensor[v].detach().cpu().numpy(), (1, 2, 0)) * 0.5 + 0.5)[:, :, ::-1] * 255.0
            save_img_list.append(save_img)
        save_img = np.concatenate(save_img_list, axis=1)
        Image.fromarray(np.uint8(save_img[:,:,::-1])).save(save_img_path)
        sdf,mat  = reconstruction(
            net, cuda, calib_tensor, resolution, b_min, b_max, use_octree=use_octree)
         
#        sdf = np.where(sdf<0,1,0)
        return  sdf, mat, iou_fine
        
#        sdf = sdf.detach().cpu().numpy()
#        np.save(save_path, sdf.astype("float16"))
#        verts, faces, _, _ = reconstruction(
#            net, cuda, calib_tensor, opt.resolution, b_min, b_max, use_octree=use_octree)
#        verts_tensor = torch.from_numpy(verts.T).unsqueeze(0).to(device=cuda).float()
#        xyz_tensor = net.projection(verts_tensor, calib_tensor[:1])
#        uv = xyz_tensor[:, :2, :]
#        color = index(image_tensor[:1], uv).detach().cpu().numpy()[0].T
#        color = color * 0.5 + 0.5
#        save_obj_mesh_with_color(save_path, verts, faces, color)
    except Exception as e:
        print(e)
        print('Can not create marching cubes at this time.')

def reconstruction(net, cuda, calib_tensor,
                   resolution, b_min, b_max,
                   use_octree=False, num_samples=10000, transform=None):
    '''
    Reconstruct meshes from sdf predicted by the network.
    :param net: a BasePixImpNet object. call image filter beforehead.
    :param cuda: cuda device
    :param calib_tensor: calibration tensor
    :param resolution: resolution of the grid cell
    :param b_min: bounding box corner [x_min, y_min, z_min]
    :param b_max: bounding box corner [x_max, y_max, z_max]
    :param use_octree: whether to use octree acceleration
    :param num_samples: how many points to query each gpu iteration
    :return: marching cubes results.
    '''
    # First we create a grid by resolution
    # and transforming matrix for grid coordinates to real world xyz
    coords, mat = create_grid(resolution, resolution, resolution,
                              b_min, b_max, transform=transform)
    
    # net, cuda, calib_tensor, resolution, b_min, b_max = net, cuda, calib_tensor, opt.resolution, b_min, b_max
    
    # Then we define the lambda function for cell evaluation
    
    def eval_func(points):
        points = np.expand_dims(points, axis=0)
        points = np.repeat(points,1, axis=0)
        samples = torch.from_numpy(points).to(device=cuda).float()
        net.query(samples, calib_tensor)
        pred = net.get_preds()[0][0]
        return pred.detach().cpu().numpy()
#    print(use_octree)
    # Then we evaluate the grid
    if use_octree:
        sdf = eval_grid_octree(coords, eval_func, num_samples=num_samples)
#        sdf = np.where(sdf>0,0,1)
    else:
        sdf = eval_grid(coords, eval_func, num_samples=num_samples)
    return sdf,mat
    # Finally we do marching cubes
#    try:
#        verts, faces, normals, values = measure.marching_cubes_lewiner(sdf, 0.01)
#        # transform verts into world coordinate system
#        verts = np.matmul(mat[:3, :3], verts.T) + mat[:3, 3:4]
#        verts = verts.T
#        return verts, faces, normals, values
#    except:
#        print('error cannot marching cubes')
#        return -1



def save_obj_mesh(mesh_path, verts, faces):
    file = open(mesh_path, 'w')

    for v in verts:
        file.write('v %.4f %.4f %.4f\n' % (v[0], v[1], v[2]))
    for f in faces:
        f_plus = f + 1
        file.write('f %d %d %d\n' % (f_plus[0], f_plus[2], f_plus[1]))
    file.close()

def batch_eval(points, eval_func, num_samples=512 * 512 * 512):
    num_pts = points.shape[1]
    sdf = np.zeros(num_pts)

    num_batches = num_pts // num_samples
    for i in range(num_batches):
        sdf[i * num_samples:i * num_samples + num_samples] = eval_func(
            points[:, i * num_samples:i * num_samples + num_samples])
    if num_pts % num_samples:
        sdf[num_batches * num_samples:] = eval_func(points[:, num_batches * num_samples:])

    return sdf


def eval_grid(coords, eval_func, num_samples=512 * 512 * 512):
    resolution = coords.shape[1:4]
    coords = coords.reshape([3, -1])
    sdf = batch_eval(coords, eval_func, num_samples=num_samples)
    return sdf.reshape(resolution)


def eval_grid_octree(coords, eval_func,
                     init_resolution=64, threshold=0.01,
                     num_samples=512 * 512 * 512):
    resolution = coords.shape[1:4]

    sdf = np.zeros(resolution)

    dirty = np.ones(resolution, dtype=np.bool)
    grid_mask = np.zeros(resolution, dtype=np.bool)

    reso = resolution[0] // init_resolution

    while reso > 0:
        # subdivide the grid
        grid_mask[0:resolution[0]:reso, 0:resolution[1]:reso, 0:resolution[2]:reso] = True
        # test samples in this iteration
        test_mask = np.logical_and(grid_mask, dirty)
        #print('step size:', reso, 'test sample size:', test_mask.sum())
        points = coords[:, test_mask]

        sdf[test_mask] = batch_eval(points, eval_func, num_samples=num_samples)
        dirty[test_mask] = False

        # do interpolation
        if reso <= 1:
            break
        for x in range(0, resolution[0] - reso, reso):
            for y in range(0, resolution[1] - reso, reso):
                for z in range(0, resolution[2] - reso, reso):
                    # if center marked, return
                    if not dirty[x + reso // 2, y + reso // 2, z + reso // 2]:
                        continue
                    v0 = sdf[x, y, z]
                    v1 = sdf[x, y, z + reso]
                    v2 = sdf[x, y + reso, z]
                    v3 = sdf[x, y + reso, z + reso]
                    v4 = sdf[x + reso, y, z]
                    v5 = sdf[x + reso, y, z + reso]
                    v6 = sdf[x + reso, y + reso, z]
                    v7 = sdf[x + reso, y + reso, z + reso]
                    v = np.array([v0, v1, v2, v3, v4, v5, v6, v7])
                    v_min = v.min()
                    v_max = v.max()
                    # this cell is all the same
                    if (v_max - v_min) < threshold:
                        sdf[x:x + reso, y:y + reso, z:z + reso] = (v_max + v_min) / 2
                        dirty[x:x + reso, y:y + reso, z:z + reso] = False
        reso //= 2

    return sdf.reshape(resolution)


def adjust_learning_rate(optimizer, epoch, lr, schedule, gamma):
    """Sets the learning rate to the initial LR decayed by schedule"""
#    if epoch in schedule:
#        lr *= gamma
#        for param_group in optimizer.param_groups:
#            param_group['lr'] = lr
    
    # schedule decay in each schedule
    # gamma decay ratio
    
    if epoch%schedule==0:
        lr *= gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    return lr


def cal_iou(pred_lable,gt_lable,thresh=0.5):
#    gt_lable ,pred_lable =  netG_fine_pifu.labels-0.5,netG_fine_pifu.preds-0.5
    cache = (gt_lable-thresh)*(pred_lable-thresh)
#    a = torch.where(cache>0)
    iou = len(torch.where(cache>0)[0])/len(cache[0,0,:])
    
    return iou







# def depth2mesh(data_uint16, save_path, name_uint16, image_size=512):

# #    image_size = 256
#     data_uint16 = cv2.resize(data_uint16, (image_size,image_size),interpolation=cv2.INTER_NEAREST)
#     calib_1 = np.array([[ 107.78947529,    0.  ,    0.        ,    0. ],[   0. ,  107.78947529,    0.,    0.],
#                         [   0.        ,    0.  , -105.26315789,    0. ],[   0. ,   95.        ,    0.,    1.]])
    
    
#     data_uint16 = data_uint16
    
#     data_uint16 = np.flip(data_uint16, 0) # x
#     #data_uint16 = np.flip(data_uint16, 1)  # y
#     #data_uint16 = cv2.flip(data_uint16,1)
#     mask = np.where(data_uint16 >= 255*255-10000,0,1)
#     dep_map = data_uint16.copy()
#     #trans_depth2mesh_uint16(data_uint16,"test_1.obj")
#     h, w = dep_map.shape
    
#     out_name = os.path.join(save_path,"%s_d2m"%name_uint16)
    
#     if mask is None:
#         thres = np.mean(dep_map)
#         thres = np.clip(thres, a_min=np.min(dep_map), a_max=255*255-10100)
#         vid, uid = np.where(dep_map < thres)
#     else:
#         vid, uid = np.where(mask > 0.01)
#     ### calculate the inverse point cloud
#     uv_mat = np.ones((len(vid), 4), dtype=np.float)
#     #uv_mat[:, 0] = (uid - h / 2.) / (h / 2.)
#     z_buffer_data_numpy  = dep_map[vid, uid] / 255./ 255.
    
#     uv_mat[:,2] = z_buffer_data_numpy
#     uv_mat[:, 0] = uid/image_size
#     uv_mat[:, 1] = vid/image_size
#     #viewZ =  (z_buffer_data_numpy ) * (cam.near- cam.far ) - cam.near
    
#     uv_mat= uv_mat*2-1
#     uv_mat_4 = uv_mat
#     uv_mat_test_2 = np.matmul(uv_mat_4, calib_1)
    
#     vert = uv_mat_test_2
    
#     # save vertex and faces:
#     f = open(out_name + '.obj', 'w')
#     nv = 0
#     vidx_map = np.full_like(dep_map, fill_value=-1, dtype=np.int)
#     for i in range(0, len(vid)):
#         f.write('v %f %f %f\n' % (vert[i][0], vert[i][1], vert[i][2]))
#         vidx_map[vid[i], uid[i]] = nv
#         nv += 1
    
#     for i in range(0, h-2):
#         for j in range(0, w-2):
#             if vidx_map[i, j] >= 0 and vidx_map[i, j+1] >= 0 and vidx_map[i+1, j] >= 0 and vidx_map[i+1, j+1] >= 0:
#                 f.write('f %d %d %d\n' % (vidx_map[i , j] + 1, vidx_map[i, j + 1] + 1, vidx_map[i + 1, j] + 1))
#                 f.write('f %d %d %d\n' % (vidx_map[i + 1, j + 1] + 1, vidx_map[i + 1, j] + 1, vidx_map[i , j + 1] + 1))
#     f.close()
#     return vid, uid, vert
