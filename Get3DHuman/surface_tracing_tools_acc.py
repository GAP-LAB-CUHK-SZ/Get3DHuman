import os, sys
import torch
import torch.nn as nn
import numpy as np
# sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import copy
import time
import cv2
import math

def depth2normal_bs(depth, bg_flag):
    '''
    To compute a normal map from the depth map
    Input:
    - depth:		torch.Tensor (BS, 1, H, W)   
    Return:
    - normal:		torch.Tensor (H, W, 3)
    depth = gep_gt_torch+0  
    '''
    depth_bs = depth*(-215.57895058)  #  2*calib_d2pc[2,2]
    # f_pix_y, f_pix_x =  0.4211, 0.4211  #  1/512*2*calib_d2pc[0,0]) 
    bs, h, w = depth_bs.shape
    f_pix_y, f_pix_x = 215.57895058/(w/2), 215.57895058/(w/2) , #  1/512*2*calib_d2pc[0,0]) 
    # eps = 1e-12

    # bg_flag = (depth > 60000) | (depth == 0)
    depth_bs[bg_flag] = 0.0

    depth_left, depth_right, depth_up, depth_down = torch.zeros(bs, h, w), torch.zeros(bs, h, w),torch.zeros(bs, h, w),torch.zeros(bs, h, w)
    if depth_bs.get_device() != -1:
        device_id = depth_bs.get_device()
        depth_left, depth_right, depth_up, depth_down = depth_left.to(device_id), depth_right.to(device_id), depth_up.to(device_id), depth_down.to(device_id)
    depth_left[:, :, 1:w-1] = depth_bs[:, :, :w-2].clone()
    depth_right[:, :, 1:w-1] = depth_bs[:, :, 2:].clone()
    depth_up[:, 1:h-1, :] = depth_bs[:, :h-2, :].clone()
    depth_down[:, 1:h-1, :] = depth_bs[:, 2:, :].clone()

    dzdx = (depth_right - depth_left)
    dzdy = (depth_down - depth_up)
    
    normal = torch.stack([dzdx, dzdy, -torch.ones_like(dzdx)*f_pix_x]).permute(1, 0, 2, 3)
    normal_length = torch.norm(normal, p=2, dim=1)
    normal = normal / (normal_length + 1e-12)[:,None]
    normal[bg_flag[:,None].repeat(1,3,1,1)] = 0
    
    # normal = (normal+1)/2
    normal[:,0,:,:] = normal[:,0,:,:]*-1 
    normal[:,2,:,:] = normal[:,2,:,:]*-1 
    return normal

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


def depth2mesh_R(data_uint16, save_path, name_uint16, R, img_crop, image_size=512):
    # data_uint16 = render_dep*255*255
#    image_size = 256
    # data_uint16 = cv2.resize(data_uint16, (image_size,image_size),interpolation = cv2.INTER_NEAREST)
    
    
    img_crop = (img_crop*image_size/512).astype(int)
    
    # data_uint16[:,:50] = 255*255 
    min_filter =  np.where(data_uint16>255*255-10000,0,data_uint16).max()
    data_uint16 = cv2.resize(data_uint16, (image_size,image_size))
    # data_uint16 = cv2.resize(data_uint16, (image_size,image_size),interpolation = cv2.INTER_NEAREST)
    data_uint16_mask = cv2.resize(data_uint16, (image_size,image_size),interpolation = cv2.INTER_NEAREST)
    
    data_uint16_mask = np.where(data_uint16_mask>255*255-20000,0,1)
    
    data_uint16_mask[:, :img_crop[0]] = 0
    data_uint16_mask[:,img_crop[1]: ] = 0
    data_uint16_mask[:img_crop[2], :] = 0
    data_uint16_mask[img_crop[3]:, :] = 0
    
    kernel = np.ones((3, 3), dtype=np.uint8)
    data_uint16_mask_erode = cv2.erode(data_uint16_mask.astype("uint8"), kernel, iterations=1)*1.
    
    
    # plt.imshow((data_uint16_mask).astype("uint8"))
    # plt.imshow((data_uint16/255).astype("uint8"))
    
    # data_uint16 = cv2.resize(data_uint16, (image_size,image_size))
    calib_1 = np.array([[ 107.78947529,    0.  ,    0.        ,    0. ],[   0. ,  107.78947529,    0.,    0.],
                        [   0.        ,    0.  , -107.78947529,    0. ],[   0. ,   95.        ,    0.,    1.]])
    
    
    
    data_uint16 = data_uint16*data_uint16_mask_erode + (1-data_uint16_mask_erode)*255*255
    # data_uint16[:, :img_crop[0]] = 65025
    # data_uint16[:,img_crop[1]: ] = 65025
    # data_uint16[:img_crop[2], :] = 65025
    # data_uint16[img_crop[3]:, :] = 65025
    
    # data_uint16 =cv2.bilateralFilter(data_uint16.astype("float32"), 0, sigmaColor=100, sigmaSpace=15)
    # data_uint16 =cv2.bilateralFilter(data_uint16.astype("float32"), 0, sigmaColor=100, sigmaSpace=15)
    
    
    data_uint16 = np.flip(data_uint16, 0) # x
    #data_uint16 = np.flip(data_uint16, 1)  # y
    #data_uint16 = cv2.flip(data_uint16,1)
    mask = np.where(data_uint16 >= min_filter,0,1)
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
    
    uv_mat_test_2[:,:3] = np.matmul(uv_mat_test_2[:,:3], R)
    
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



def decode_sdf(decoder, latent_list, points, clamp_dist=0.1, MAX_POINTS=300000, no_grad=False):
    
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
        
        # latent_vector = latent_vector.transpose(2,1)
        
        # if latent_vector is None:
        #     inputs = points[:,start:end]
        # else:
        # latent_repeat = latent_vector.expand(end - start, -1)
#            inputs = torch.cat([latent_repeat, points[start:end]], 1)
        # inputs = latent_repeat.transpose(1,0)[None]
        
        sdf_batch = decoder(latent_vector)[0].squeeze(1)
        
        # TSDF Formulation
        # sdf_batch = (2*sdf_batch-1)/10
        sdf_batch = (2*sdf_batch-1)*25
        
        # 
        sdf_batch = sdf_batch/128
        
        # sdf_batch = sdf_batch.transpose(1,0)
        start = end
        if no_grad:
            sdf_batch = sdf_batch.detach()
        output_list.append(sdf_batch)
        if end == num_all:
            break
    sdf = torch.cat(output_list, 0)
#    print(sdf.shape)
    if clamp_dist != None:
        sdf = torch.clamp(sdf, -clamp_dist, clamp_dist)
    return sdf


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
    return torch.tensor(R).float()



def make_rotate_torch(rx, ry, rz):
    sinX = torch.sin(rx)
    sinY = torch.sin(ry)
    sinZ = torch.sin(rz)

    cosX = torch.cos(rx)
    cosY = torch.cos(ry)
    cosZ = torch.cos(rz)

    Rx = torch.zeros((3,3))
    Rx[0, 0] = 1.0
    Rx[1, 1] = cosX
    Rx[1, 2] = -sinX
    Rx[2, 1] = sinX
    Rx[2, 2] = cosX

    Ry = torch.zeros((3,3))
    Ry[0, 0] = cosY
    Ry[0, 2] = sinY
    Ry[1, 1] = 1.0
    Ry[2, 0] = -sinY
    Ry[2, 2] = cosY

    Rz = torch.zeros((3,3))
    Rz[0, 0] = cosZ
    Rz[0, 1] = -sinZ
    Rz[1, 0] = sinZ
    Rz[1, 1] = cosZ
    Rz[2, 2] = 1.0
    R = torch.matmul(torch.matmul(Rz,Ry),Rx)
    return R.float()



class SDFRenderer(object):
    def __init__(self,  cuda,
                 decoder, intrinsic, img_hw=None, transform_matrix=None, march_step=50, buffer_size=5, ray_marching_ratio=1, use_depth2normal=False, max_sample_dist=0.2, radius=1.0, threshold=5e-5, scale_list=[4, 2, 1], march_step_list=[3, 3, -1], use_gpu=True, is_eval=True):
        self.decoder = decoder
#        self.device = next(self.decoder.parameters()).get_device()
        self.device = cuda
        if is_eval:
            self.decoder.eval()
        self.march_step = march_step
        self.buffer_size = buffer_size
        self.max_sample_dist = max_sample_dist
        self.ray_marching_ratio = ray_marching_ratio
        self.use_depth2normal=use_depth2normal
        self.radius = radius
        self.threshold = threshold
        self.scale_list = scale_list
        self.march_step_list = march_step_list
        if type(intrinsic) == torch.Tensor:
            intrinsic = intrinsic.detach().cpu().numpy()
        self.intrinsic = intrinsic
        
        if img_hw is None:
            img_h, img_w = int(intrinsic[1,2] * 2), int(intrinsic[0,2] * 2)
            self.img_hw = (img_h, img_w)
        else:
            self.img_hw = img_hw

        self.homo_2d = self.init_grid_homo_2d(self.img_hw)
        self.K, self.K_inv = self.init_intrinsic(intrinsic)
        self.homo_calib = torch.matmul(self.K_inv, self.homo_2d) # (3, H*W)
        self.homo_calib.requires_grad=False
        self.imgmap_init = self.init_imgmap(self.img_hw)
        self.imgmap_init.requires_grad=False

        if transform_matrix is None:
            self.transform_matrix = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
#            self.transform_matrix_ortho = np.array([[1., 0., 0.], [0., -1, 0.], [0., 0., -1.]])
        else:
            self.transform_matrix = transform_matrix
        self.transform_matrix = torch.from_numpy(self.transform_matrix).float()
        self.use_gpu = use_gpu
        if use_gpu:
            if torch.cuda.device_count() == 0:
                raise ValueError('No GPU device found.')
#            print(self.device)
            self.homo_2d = self.homo_2d.to(self.device)
            self.homo_calib = self.homo_calib.to(self.device) # (3, H*W)
            self.imgmap_init = self.imgmap_init.to(self.device) # (H*W)
            self.transform_matrix = self.transform_matrix.to(self.device) # (3,3)
            self.K, self.K_inv = self.K.to(self.device), self.K_inv.to(self.device)

        self.calib_map = self.normalize_vectors(self.homo_calib)[2,:]
        self.clip_cube = 0.95
        
        #20211012
        self.cam_pos_dist = 2
        self.mask_threshold = 0.0008

    def get_intrinsic(self):
        return self.intrinsic

    def get_threshold(self):
        return self.threshold

    def get_img_hw(self):
        return self.img_hw

  
    def apply_3Dsim(self, points,sim3_mtrx,inv=False):
        sR,t = sim3_mtrx[:,:3],sim3_mtrx[:,3]
        points = (points-t)@sR.inverse().t() if inv else \
                 points@sR.t()+t
        return points

    def transform_points(self, points):
        '''
        transformation for point coordinates.
        Input:
        - points	type: torch.Tensor (3, H*W)
        Return:
        - points_new	type: torch.Tensor (3, H*W)
        '''
        if self.transform_matrix.shape[1] == 4:
            # sR, t = self.transform_matrix[:,:3], self.transform_matrix[:,3]
            # points_new = sR @ points + t[:, None]
            points_new = self.apply_3Dsim(points.t(), self.transform_matrix).t()
        else:
            points_new = torch.matmul(self.transform_matrix, points)
        return points_new

    def inv_transform_points(self, points):
        '''
        inverse transformation for point coordinates.
        Input:
        - points	type: torch.Tensor (3, H*W)
        Return:
        - points_new	type: torch.Tensor (3, H*W)
        '''
        if self.transform_matrix.shape[1] == 4:
            # sR, t = self.transform_matrix[:,:3], self.transform_matrix[:,3]
            # points_new = sR.inverse() @ (points-t[:, None])
            #pdb.set_trace()
            #points = np.array([0.419, 1.837, 2.495])
            #points = torch.from_numpy(points)
            #points = points[:, None]

            # pdb.set_trace()
            points_new = self.apply_3Dsim(points.t(), self.transform_matrix, inv=True).t()
        else:
            points_new = torch.matmul(self.transform_matrix.transpose(1,0), points)
        return points_new

    def get_meshgrid(self, img_hw):
        '''
        To get meshgrid:
        Input:
        - img_hw	(h, w)
        Return:
        - grid_map	type: torch.Tensor (H, W, 2)
        '''
        h, w = img_hw
        Y, X = torch.meshgrid(torch.arange(0, h), torch.arange(0, w))
        grid_map = torch.cat([X[:,:,None], Y[:,:,None]], 2) # (h, w, 2)
        grid_map = grid_map.float()
        return grid_map

    def get_homo_2d_from_xy(self, xy):
        # xy = grid_map
        '''
        get homo 2d from xy
        Input:
        - xy		type: torch.Tensor (H, W, 2)
        Return:
        - homo		type: torch.Tensor (H, W, 3)
        '''
        H, W = xy.shape[0], xy.shape[1]
        homo_ones = torch.ones(H, W, 1)
        if xy.get_device() != -1:
            homo_ones = homo_ones.to(xy.get_device())
        homo_2d = torch.cat([xy, homo_ones], 2)
        return homo_2d

    def get_homo_2d(self, img_hw):
        xy = self.get_meshgrid(img_hw)
        homo_2d = self.get_homo_2d_from_xy(xy)
        return homo_2d

    def init_grid_homo_2d(self, img_hw):
        homo_2d = self.get_homo_2d(img_hw)
        homo_2d = homo_2d.reshape(-1, 3).transpose(1,0) # (3, H*W)
        return homo_2d

    def init_intrinsic(self, intrinsic):
        K = torch.from_numpy(intrinsic).float()
        K_inv = torch.from_numpy(np.linalg.inv(intrinsic)).float()
        return K, K_inv

    def init_imgmap(self, img_hw):
        h, w = img_hw
        imgmap_init = torch.zeros(h, w)
        return imgmap_init

    def normalize_vectors(self, x):
        '''
        normalize the vector by the first dim
        '''
        norm = torch.norm(x, p=2, dim=0).expand_as(x)
        eps = 1e-12
        x = x.div(norm + eps)
        return x

    def get_camera_location(self, R, T):
        '''
        Input:
        - R	type: torch.Tensor (3,3)
        - T	type: torch.Tensor (3)
        '''
        pos = torch.matmul(-R.transpose(1,0), T[:,None]) # (3,1)
        pos = pos.squeeze(1) # (3)
        return pos
#   
    
    def get_distance_from_origin(self, cam_pos, cam_rays):
        '''
        get_distance_from_origin
        Input:
        - cam_pos	type torch.FloatTensor (3)
        - cam_rays	type torch.FloatTensor (3, H*W)
        '''
        N = cam_rays.shape[1]
        cam_pos_pad = cam_pos[:,None].expand_as(cam_rays) # (3, N)

        p, q = cam_pos_pad, cam_rays # (3, N), (3, N)
        ptq = (p * q).sum(0) # (N)
        dist = p - ptq[None,:].repeat(3,1) * q # (3, N)
        dist = torch.norm(dist, p=2, dim=0) # (N)
        return dist

    def get_maxbound_zdepth_from_dist(self, dist):
        '''
        Input:
        - dist		type torch.FloatTensor (N)
        '''
        with torch.no_grad():
            value = self.radius ** 2 - dist ** 2
            valid_mask = (value >= 0)

            maxbound_zdepth = torch.zeros_like(dist)
            maxbound_zdepth[valid_mask] = 2 * torch.sqrt(value[valid_mask])
        return maxbound_zdepth

    def get_intersections_with_unit_spheres(self, cam_pos, cam_rays):
        '''
        get_intersections_with_unit_sphere
        Input:
        - cam_pos	type torch.FloatTensor (3)
        - cam_rays	type torch.FloatTensor (3, H*W)
        '''
        with torch.no_grad():
            dist = self.get_distance_from_origin(cam_pos, cam_rays)
            valid_mask = (dist <= self.radius)
            maxbound_marching_zdepth = self.get_maxbound_zdepth_from_dist(dist) # (H*W)

            cam_pos_dist = torch.sqrt((cam_pos ** 2).sum())
            if torch.nonzero((cam_pos_dist < self.radius).unsqueeze(0)).shape[0] != 0:
                init_zdepth = torch.zeros_like(dist)
            else:
                init_zdepth_valid = torch.sqrt(cam_pos_dist ** 2 - dist[valid_mask] ** 2) - maxbound_marching_zdepth[valid_mask] / 2.0 # (N)
                init_zdepth = torch.ones_like(dist) * init_zdepth_valid.max() # (H*W)
                init_zdepth = self.copy_index(init_zdepth, valid_mask, init_zdepth_valid)
        return init_zdepth, valid_mask

    def get_maxbound_zdepth(self, cam_pos, valid_cam_rays):
        with torch.no_grad():
            init_zdepth, _ = self.get_intersections_with_unit_spheres(cam_pos, valid_cam_rays) # (N)

            dist = self.get_distance_from_origin(cam_pos, valid_cam_rays) # (N)
            maxbound_marching_zdepth = self.get_maxbound_zdepth_from_dist(dist) # (N)
            max_zdepth = init_zdepth + maxbound_marching_zdepth # (N)
        return max_zdepth

    def copy_index(self, inputs, mask, src):
        '''
        out-of-place copy index.
        Input:
        - inputs:	torch.Tensor (H*W) / (H, W) / (H, W, k)
        - mask:		torch.Tensor (H*W)
        - src:		torch.Tensor (N) / (N, k)
        '''
        inputs_shape = inputs.shape
        if len(inputs_shape) <= 2:
            inputs, mask = inputs.reshape(-1), mask.reshape(-1)
        elif len(inputs_shape) == 3:
            inputs, mask = inputs.reshape(-1, inputs_shape[-1]), mask.reshape(-1)
        else:
            raise NotImplementedError
        index = torch.nonzero(mask).reshape(-1).long()
        outputs = inputs.index_copy(0, index, src)
        outputs = outputs.reshape(inputs_shape)
        return outputs

    def get_index_from_sdf_list(self, sdf_list, index_size, index_type='min', clamp_dist=0.1):
        '''
        get index with certain method.
        Input:
        - sdf_list:		type: torch.Tensor (self.march_step, N)
        Return:
        - sdf:			type: torch.Tensor (N, index_size)
        - index:		type: torch.Tensor (N, index_size). Note: the first dimension (index[0]) is always the min index.
        '''
        if index_type == 'min':
            sdf, index = torch.topk(-sdf_list.transpose(1,0), index_size, dim=1)
            sdf = -sdf
        elif index_type == 'min_abs':
            sdf_list_new = torch.abs(sdf_list)
            _, index = torch.topk(-sdf_list_new.transpose(1,0), index_size, dim=1)
            # index = index.to(self.device)
            sdf = self.collect_data_from_index(sdf_list, index.to(self.device))
        elif index_type == 'max_neg':
            sdf_list_new = sdf_list.clone()
            sdf_list_pos = (sdf_list_new >= 0)
            sdf_list_new[sdf_list_pos] = sdf_list_new[sdf_list_pos].clone() * (-1) - 2
            sdf, index = torch.topk(sdf_list_new.transpose(1,0), index_size, dim=1) # (N, index_size)
            sdf_pos = (sdf <= -2)
            sdf[sdf_pos] = sdf[sdf_pos].clone() * (-1) - 2
        elif index_type == 'last_valid':
            march_step, N = sdf_list.shape[0], sdf_list.shape[1]
            valid = (torch.abs(sdf_list) < clamp_dist)
            idx_list = torch.arange(0, march_step)[:,None].repeat(1,N).to(sdf_list.get_device())
            idx_list = idx_list.float() * valid.float()
            _, index = torch.topk(idx_list.transpose(1,0), index_size, dim=1) # (N, index_size)
            sdf = self.collect_data_from_index(sdf_list, index)[0].transpose(1,0)
        elif index_type == 'last':
            march_step, N = sdf_list.shape[0], sdf_list.shape[1]
            sdf = sdf_list[-index_size:, :].transpose(1,0)
            index = torch.arange(march_step - index_size, march_step)[None,:].repeat(N, 1)
            index = index.to(sdf.get_device())
        else:
            raise NotImplementedError
        return sdf, index

    def collect_data_from_index(self, data, index):
        '''
        data = sdf_list
        Input:
        - data:		type: torch.Tensor (self.march_step, N) / (self.march_step, N, k)
        - index:	type: torch.Tensor (N, index_size)
        Return:
        - data_sampled:	type: torch.Tensor (index_size, N) / (index_size, N, k)
        '''
        index_size = index.shape[1]
        count_index = torch.arange(index.shape[0]).repeat(index_size).to(self.device)
#        count_index = torch.arange(index.shape[0]).repeat(index_size)
        point_index = index.transpose(1,0).reshape(-1) * data.shape[1] + count_index

        if len(data.shape) == 3:
            data_shape = data.shape
            data_sampled = data.reshape(-1, data_shape[-1])[point_index].reshape(index_size, -1, data_shape[-1]).clone() # (index_size, N, 3)
        elif len(data.shape) == 2:
            data_sampled = data.reshape(-1)[point_index].reshape(index_size, -1).clone() # (index_size, N)
        else:
            raise NotImplementedError
        return data_sampled
    
    def collect_data_from_index_ori(self, data, index):
        '''
        data = sdf_list
        Input:
        - data:		type: torch.Tensor (self.march_step, N) / (self.march_step, N, k)
        - index:	type: torch.Tensor (N, index_size)
        Return:
        - data_sampled:	type: torch.Tensor (index_size, N) / (index_size, N, k)
        '''
        index_size = index.shape[1]
        count_index = torch.arange(index.shape[0]).repeat(index_size).to(self.device)
#        count_index = torch.arange(index.shape[0]).repeat(index_size)
        point_index = index.transpose(1,0).reshape(-1) * data.shape[1] + count_index

        if len(data.shape) == 3:
            data_shape = data.shape
            data_sampled = data.reshape(-1, data_shape[-1])[point_index].reshape(index_size, -1, data_shape[-1]).clone() # (index_size, N, 3)
        elif len(data.shape) == 2:
            data_sampled = data.reshape(-1)[point_index].reshape(index_size, -1).clone() # (index_size, N)
        else:
            raise NotImplementedError
        return data_sampled

    def sample_points_uniform(self, points, cam_rays, num_samples=None):
        '''
        Input:
        points:		type: torch.Tensor (N, 3)
        cam_rays:	type: torch.Tensor (3, N)
        Return:
        points_sampled:	type: torch.Tensor (num_samples, N, 3)
        '''
        if num_samples == None:
            num_samples = self.buffer_size
        N = points.shape[0]
        points = points[None,:,:].repeat(num_samples, 1, 1) # (num_samples, N, 3)
        cam_rays = cam_rays.transpose(1, 0)[None,:,:].repeat(num_samples, 1, 1) # (num_samples, N, 3)
        delta_depth = torch.linspace(0, -self.max_sample_dist, num_samples).to(points.get_device()) # (num_samples)
        delta_depth = delta_depth[:,None,None].repeat(1, N, 3) # (num_samples, N, 3)
        points_sampled = delta_depth * cam_rays + points # (num_smaples, N, 3)
        return points_sampled

    def get_min_sdf_sample(self, sdf_list, points_list, latent, index_type='min_abs', clamp_dist=0.1, no_grad=False):
        #        profiler = Profiler(silent = not profile)
        #  sdf_list, points_list, latent
        # _, index = self.get_index_from_sdf_list(sdf_list, 1, index_type=index_type)
        sdf_list_new = torch.abs(sdf_list)
        _, index = torch.topk(-sdf_list_new, 1, dim=0)
        # _, index = sdf_renderer.get_index_from_sdf_list(sdf_list, 1, index_type=index_type)
        ''''''
        index_merge = index.reshape(index.shape[0],-1).T
        points_list_merge = points_list.reshape(points_list.shape[0],-1,3)
        points = self.collect_data_from_index(points_list_merge, index_merge)[0] # (N, 3)
        # points = sdf_renderer.collect_data_from_index(points_list_merge, index_merge)[0]# (N, 3)
        
        points = points.reshape(points_list.shape[1],-1,3).transpose(2,1)
        # points_list.transpose(2,1)
        # points = self.collect_data_from_index(points_list, index)[0] # (N, 3)
        ''''''
        
        # points = self.collect_data_from_index(points_list, index)[0] # (N, 3)
        
        min_sdf_sample = decode_sdf(self.decoder, latent, points, clamp_dist=None, no_grad=no_grad).squeeze(-1)
#        profiler.report_process('[DEPTH] [SAMPLING] sample min sdf time\t')
        if no_grad:
            min_sdf_sample = min_sdf_sample.detach()
        return min_sdf_sample

    def get_sample_on_marching_zdepth_along_ray(self, marching_zdepth_list, sdf_list, points_list, latent, index_type='min_abs',  clamp_dist=0.1,  no_grad=False):
        # initialization
        # collect points
        # if use_uniform_sample:
        #     sdf_selected, index_selected = self.get_index_from_sdf_list(sdf_list, 1, index_type=index_type, clamp_dist=clamp_dist)
        #     points = self.collect_data_from_index(points_list, index_selected)[0] # (N, 3)
        #     points_sampled = self.sample_points_uniform(points, cam_rays)
        # else:
            # sdf_selected, index_selected = self.get_index_from_sdf_list(sdf_list, self.buffer_size, index_type=index_type, clamp_dist=clamp_dist)
            # points_sampled = self.collect_data_from_index(points_list, index_selected)
            
        sdf_list_new = torch.abs(sdf_list)
        sdf_selected, index_selected = torch.topk( -sdf_list_new.transpose(1,0), self.buffer_size, dim=1)
        # sdf_selected, index_selected = torch.topk(-sdf_list_new.transpose(1,0), sdf_renderer.buffer_size, dim=1)
        sdf_selected, index_selected = sdf_selected.transpose(1,0), index_selected.transpose(1,0)
       
        index_selected_merge = index_selected.reshape(index_selected.shape[0],-1).T
        points_list_merge = points_list.reshape(points_list.shape[0],-1,3)
        points_sampled_merge = self.collect_data_from_index(points_list_merge, index_selected_merge) # (N, 3)
        # points_sampled_merge = sdf_renderer.collect_data_from_index(points_list_merge, index_selected_merge) # (N, 3)
        
        points_sampled = points_sampled_merge.reshape(self.buffer_size, points_list.shape[1],-1,3).transpose(2,3)
            # points_sampled = points_sampled_merge.reshape(sdf_renderer.buffer_size, points_list.shape[1],-1,3).transpose(2,3)
            
            # index = index.to(self.device)
            #sdf = self.collect_data_from_index(sdf_list, index.to(self.device))
        
        # mask_points = points_sampled_merge
        
        # generate new marching zdepth
        marching_zdepth_list_merge = marching_zdepth_list.reshape(marching_zdepth_list.shape[0],-1)
        marching_zdepth = self.collect_data_from_index(marching_zdepth_list_merge, index_selected_merge[:,[0]])[0] # (N)
        # marching_zdepth = sdf_renderer.collect_data_from_index(marching_zdepth_list_merge, index_selected_merge[:,[0]])[0] # (N)
        marching_zdepth = marching_zdepth.reshape(points_list.shape[1],-1)
        # index = index.to(self.device)
        
        marching_zdepth = marching_zdepth + (1 - self.ray_marching_ratio) * torch.clamp(sdf_selected[0,:], -clamp_dist, clamp_dist) # (N)
        # marching_zdepth = marching_zdepth + (1 - sdf_renderer.ray_marching_ratio) * torch.clamp(sdf_selected[0,:], -clamp_dist, clamp_dist) # (N)

        if no_grad:
            marching_zdepth_final = marching_zdepth
        else:
            marching_zdepth_new = marching_zdepth
            for i in range(self.buffer_size):
                sdf = decode_sdf(self.decoder, latent, points_sampled[i], clamp_dist=clamp_dist, no_grad=no_grad).squeeze(-1)
                # sdf = decode_sdf(sdf_renderer.decoder, latent, points_sampled[i], clamp_dist=clamp_dist, no_grad=no_grad).squeeze(-1)
                
                marching_zdepth_new = marching_zdepth_new - sdf.detach() * self.ray_marching_ratio
                marching_zdepth_new = marching_zdepth_new + sdf * self.ray_marching_ratio
#            profiler.report_process('[DEPTH] [SAMPLING] re-ray marching time')
            marching_zdepth_final = marching_zdepth_new
        return marching_zdepth_final


    def index_sample(self, basemap, indexmap):
        '''
        To use indexmap to index basemap.
        Inputs:
        - basemap		type: torch.Tensor (H', W', C)
        - indexmap		type: torch.Tensor (H, W, 2)
        Returns:
        - newmap		type: torch.Tensor (H, W, C)
        '''
        h, w, c = basemap.shape[0], basemap.shape[1], basemap.shape[2]
        h_index, w_index = indexmap.shape[0], indexmap.shape[1]

        index = indexmap.reshape(-1, 2)
        index = (index[:,0] + index[:,1] * w).type(torch.long)

        newmap = basemap.reshape(-1, c)[index]
        newmap = newmap.reshape(h_index, w_index, c)
        return newmap

        
    #20211012
    
    def render_ortho(self, cam_x, cam_y, latent, R, T, clamp_dist=0.1, sample_index_type='min_abs', profile=False, no_grad=False, no_grad_depth=False, no_grad_normal=False, no_grad_mask=False, no_grad_camera=False, normalize_normal=True, use_transform=True, ray_marching_type='pyramid_recursive', num_forward_sampling=0):
        '''
        differentiable rendering.
        Input:
        - latent	type torch.Tensor (1, latent_size)
        - R		type torch.Tensor (3,3)
        - T		type torch.Tensor (3)
        Return:
        - Zdepth		type torch.Tensor (H, W) - rendered depth
        - Znormal		type torch.Tensor (H, W, 3) - rendered normal
        - valid_mask		type torch.Tensor (H, W) - rendered silhoutte
        - min_sdf_sample	type torch.Tensor (H, W) - minimum_depth_sample
        '''
        if no_grad:
            no_grad_depth, no_grad_normal, no_grad_mask, no_grad_camera = True, True, True, True
        
        # numpy 
        # self.rotate = np.matmul(make_rotate(math.radians(cam_y), 0, 0), make_rotate(0, math.radians(cam_x), 0))
        #  rotate_np = np.matmul(make_rotate(math.radians(11), 0, 0), make_rotate(0, math.radians(0), 0))
        #torch
        # self.rotate = torch.matmul(make_rotate_torch(torch.deg2rad(torch.tensor(cam_y)), torch.tensor(0), torch.tensor(0)), make_rotate_torch( torch.tensor(0), torch.deg2rad(torch.tensor(cam_x)),  torch.tensor(0)))
        self.rotate = make_rotate_torch(torch.deg2rad(torch.tensor(cam_y)), torch.deg2rad(torch.tensor(cam_x)), torch.tensor(0)).to(self.device)
        self.rotate = self.rotate[None].repeat(latent[5].size(0),1,1)
        self.rotate_inv = make_rotate_torch(torch.deg2rad(-torch.tensor(cam_y)), -torch.deg2rad(torch.tensor(cam_x)), torch.tensor(0)).to(self.device)

#        profiler = Profiler(silent = not profile)
        h, w = self.img_hw
        # render depth
        #Zdepth, valid_mask, min_abs_query = self.render_depth(latent, R, T, clamp_dist=clamp_dist, sample_index_type=sample_index_type, profile=profile, no_grad=no_grad, no_grad_depth=no_grad_depth, no_grad_mask=no_grad_mask, no_grad_camera=no_grad_camera, ray_marching_type=ray_marching_type, use_transform=use_transform) # (H*W), (H*W), (H*W)   
        Zdepth, valid_mask, min_abs_query = self.render_depth_ortho(latent, R, T, clamp_dist=clamp_dist, sample_index_type=sample_index_type, profile=profile, no_grad=no_grad, no_grad_depth=no_grad_depth, no_grad_mask=no_grad_mask, no_grad_camera=no_grad_camera, use_transform=use_transform) # (H*W), (H*W), (H*W)
        # Zdepth, valid_mask, min_abs_query = sdf_renderer.render_depth_ortho(latent, R, T, clamp_dist=clamp_dist, sample_index_type=sample_index_type, profile=profile, no_grad=no_grad, no_grad_depth=no_grad_depth, no_grad_mask=no_grad_mask, no_grad_camera=no_grad_camera, use_transform=use_transform) # (H*W), (H*W), (H*W)
 

        depth = torch.ones_like(Zdepth) * 1e11
#        depth[valid_mask] = Zdepth[valid_mask].clone() * self.calib_map[valid_mask]
        #20211017  ortho!!!!
        depth[valid_mask] = Zdepth[valid_mask].clone() 
        depth = depth.reshape(-1, h, w)
        return depth        
   
    def get_camera_location_ortho(self, R, homo=None):
        '''
        Input:
        - R	type: torch.Tensor (3,3)
        - T	type: torch.Tensor (3)
        '''
        if homo is None:
            homo = self.homo_calib
        pos_ortho = torch.matmul(R.transpose(1,0), homo) # (3,N)
#        pos = torch.matmul(R.transpose(1,0), homo) # (3,N)
        pos_ortho[1,:] = pos_ortho[1,:]*(-1)
#        pos_ortho[1,:] = -2
        pos_ortho = self.normalize_vectors(pos_ortho)
        return pos_ortho
    
    def get_distance_from_origin_ortho(self, cam_pos_ortho, cam_rays):
        '''
        get_distance_from_origin
        Input:
        - cam_pos	type torch.FloatTensor (3)
        - cam_rays	type torch.FloatTensor (3, H*W)
        '''
        dist =  abs(cam_pos_ortho - cam_rays).sum(0)
#        dist =  torch.norm(dist, p=2, dim=0) 
        return dist
    
    def get_intersections_with_unit_spheres_ortho(self, cam_pos_ortho, cam_pos, cam_rays):
#        cam_pos_ortho, cam_pos, valid_cam_rays
        '''
        get_intersections_with_unit_sphere
        Input:
        - cam_pos	type torch.FloatTensor (3)
        - cam_rays	type torch.FloatTensor (3, H*W)
        '''
        with torch.no_grad():
            dist = self.get_distance_from_origin_ortho(cam_pos_ortho, cam_rays)
            valid_mask = (dist <= self.radius)
            maxbound_marching_zdepth = self.get_maxbound_zdepth_from_dist_ortho(dist) # (H*W)
            
#            maxbound_marching_zdepth = sdf_renderer.get_maxbound_zdepth_from_dist_ortho(dist)
            
            cam_pos_dist = torch.sqrt((cam_pos ** 2).sum())
            #20211012
#            cam_pos_dist = self.cam_pos_dist
            
            if torch.nonzero((cam_pos_dist < self.radius).unsqueeze(0)).shape[0] != 0:
                init_zdepth = torch.zeros_like(dist)
            else:
                init_zdepth_valid = torch.sqrt(cam_pos_dist ** 2 - dist[valid_mask] ** 2) - maxbound_marching_zdepth[valid_mask] / 2.0 # (N)
                init_zdepth = torch.ones_like(dist) * init_zdepth_valid.max() # (H*W)
                init_zdepth = self.copy_index(init_zdepth, valid_mask, init_zdepth_valid)
        return init_zdepth, valid_mask
    
    
    def render_depth_ortho(self, latent, R, T, clamp_dist=0.1, sample_index_type='min_abs', profile=False, no_grad=True, no_grad_depth=True, no_grad_mask=True, no_grad_camera=True, use_transform=True):
        if no_grad:
            no_grad_depth, no_grad_mask = True, True
            # no_grad_depth, no_grad_mask, no_grad_camera = True, True, True
       
        cam_pos = self.get_camera_location(R, T)
        # cam_pos = sdf_renderer.get_camera_location(R, T)
#        cam_pos_ortho, cam_rays = self.get_rays_ortho(R)
        rays_start, rays_end = self.get_rays_ortho(R)
        # rays_start, rays_end  = sdf_renderer.get_rays_ortho(R)
        
        dist = self.get_distance_from_origin_ortho(rays_start, rays_end)
        # dist = sdf_renderer.get_distance_from_origin_ortho(rays_start, rays_end)
        # bs = 
        # initialization on the unit sphere
        h, w = self.img_hw
        # h, w = sdf_renderer.img_hw
        
#        init_zdepth, valid_mask = self.get_intersections_with_unit_spheres(cam_pos, cam_rays)
#        init_zdepth, valid_mask = self.get_intersections_with_unit_spheres_ortho(cam_pos_ortho, cam_pos, cam_rays)
        # init_zdepth, valid_mask = self.get_intersections_with_unit_spheres_ortho(rays_start, cam_pos, rays_end)
        init_zdepth, valid_mask = torch.zeros_like(latent[5])*1, torch.zeros_like(latent[5])< torch.tensor(1)
        # valid_mask
        # init_zdepth = torch.zeros_like(latent_cache[5])
        # init_zdepth, valid_mask  = sdf_renderer.get_intersections_with_unit_spheres_ortho(rays_start, cam_pos, rays_end)
        
#        print(init_zdepth.shape)
        self.valid_mask_ori = valid_mask
        # ray marching
#        sdf_list, marching_zdepth_list, points_list, valid_mask_render = self.ray_marching_ortho(cam_pos_ortho, cam_pos, R, init_zdepth, valid_mask, latent, clamp_dist=clamp_dist, no_grad=no_grad_camera, ray_marching_type=ray_marching_type, use_transform=use_transform)
        # sdf_list, marching_zdepth_list, points_list, valid_mask_render = sdf_renderer.ray_marching_recursive_ortho(rays_start, cam_pos, rays_end, init_zdepth, valid_mask, latent, march_step=None, clamp_dist=clamp_dist, no_grad=no_grad, use_transform=use_transform)
        sdf_list, marching_zdepth_list, points_list, valid_mask_render = self.ray_marching_recursive_ortho(rays_start, cam_pos, rays_end, init_zdepth, valid_mask, latent, march_step=None, clamp_dist=clamp_dist, no_grad=no_grad, use_transform=use_transform)
        # sdf_list, marching_zdepth_list, points_list, valid_mask_render = sdf_renderer.ray_marching_recursive_ortho(rays_start, cam_pos, rays_end, init_zdepth, valid_mask, latent, march_step=None, clamp_dist=clamp_dist, no_grad=no_grad, use_transform=use_transform)
        
#        profiler.report_process('[DEPTH] ray marching time')
        self.sdf_list_show = sdf_list.to(self.device)
        points_list = points_list.to(self.device)

        # get differnetiable samples
        min_sdf_sample = self.get_min_sdf_sample(sdf_list, points_list, latent, index_type='min_abs', clamp_dist=clamp_dist, no_grad=no_grad_mask)
        # min_sdf_sample = sdf_renderer.get_min_sdf_sample(sdf_list, points_list, latent, index_type='min_abs', clamp_dist=clamp_dist,no_grad=no_grad_mask)
        '''
        h, w = sdf_renderer.img_hw
        no_grad, use_transform = True, False
        no_grad_depth, no_grad_mask, clamp_dist = True, True, 0.1
        cam_pos = sdf_renderer.get_camera_location(R, T)
        rays_start, rays_end  = sdf_renderer.get_rays_ortho(R)
        dist = sdf_renderer.get_distance_from_origin_ortho(rays_start, rays_end)
        init_zdepth, valid_mask = torch.zeros_like(latent[5])*1, torch.zeros_like(latent[5])< torch.tensor(1)
        sdf_list, marching_zdepth_list, points_list, valid_mask_render = sdf_renderer.ray_marching_recursive_ortho(rays_start, cam_pos, rays_end, init_zdepth, valid_mask, latent, march_step=None, clamp_dist=clamp_dist, no_grad=no_grad, use_transform=use_transform)
        min_sdf_sample = sdf_renderer.get_min_sdf_sample(sdf_list, points_list, latent, index_type='min_abs', clamp_dist=clamp_dist,no_grad=no_grad_mask)

        '''
        sdf_list_new = torch.abs(sdf_list)
        _, index_selected = torch.topk(-sdf_list_new, 1, dim=0)
        # _, index = sdf_renderer.get_index_from_sdf_list(sdf_list, 1, index_type=index_type)
        ''''''
        index_merge = index_selected.reshape(index_selected.shape[0],-1).T
        points_list_merge = points_list.reshape(points_list.shape[0],-1,3)
        points_cube = self.collect_data_from_index(points_list_merge, index_merge)[0] # (N, 3)
        # points_cube = sdf_renderer.collect_data_from_index(points_list_merge, index_merge)[0] # (N, 3)
        
        points_cube_r = torch.bmm(points_cube[None], self.rotate[0:1,:,:])[0]
        # points_cube_r = torch.bmm(points_cube[None], sdf_renderer.rotate[0:1,:,:])[0]
        
        
        points_cube_out = points_cube_r >-0.999
        points_cube_mask = (points_cube_out[:,0]* points_cube_out[:,1]* points_cube_out[:,2]).reshape(valid_mask_render.shape)
        
        mask_cache = points_cube_mask[1].reshape(128,128)
        plt.imshow(mask_cache.detach())
        
        mask_cache_ = valid_mask[1].reshape(128,128)
        plt.imshow(mask_cache_.detach())
        
        # plt.imshow((mask_cache_*mask_cache).detach())
       
        # marching_zdepth_list_merge = marching_zdepth_list.reshape(marching_zdepth_list.shape[0],-1)
        # marching_zdepth = self.collect_data_from_index(marching_zdepth_list, index_selected[:,[0]])[0]
        
        # marching_zdepth = self.collect_data_from_index(marching_zdepth_list, index_selected[:,[0]])[0]
        
#        print(min_sdf_sample.shape)
        marching_zdepth = self.get_sample_on_marching_zdepth_along_ray(marching_zdepth_list, sdf_list, points_list, latent, index_type='min_abs', clamp_dist=clamp_dist, no_grad=no_grad_depth)
                                                                      # (marching_zdepth_list, sdf_list, points_list, latent, index_type='min_abs',  clamp_dist=0.1,  no_grad=False):
                                             # initialization
#        marching_zdepth = sdf_renderer.get_sample_on_marching_zdepth_along_ray(marching_zdepth_list, sdf_list, points_list, latent, index_type='min_abs', clamp_dist=clamp_dist, no_grad=no_grad_depth)
#        profiler.report_process('[DEPTH] re-sampling time')

        # generate output
        # valid_mask = latent[5]
        # min_sdf_sample_new = torch.zeros_like(valid_mask).float() # (H, W)
        # min_sdf_sample_new.requires_grad = True
        # min_sdf_sample_new = self.copy_index(min_sdf_sample_new, valid_mask, min_sdf_sample)
        # min_sdf_sample_new = self.copy_index(min_sdf_sample_new, ~valid_mask, dist[~valid_mask] + self.threshold - self.radius) # help handle camera gradient
        
        
        #20211017
        valid_mask = min_sdf_sample < self.mask_threshold
        # valid_mask = min_sdf_sample < sdf_renderer.mask_threshold
        valid_mask = points_cube_mask *valid_mask
        ## get zdepth 
        # Zdepth = torch.ones_like(self.imgmap_init) * 1e11 # (H, W)
        # Zdepth = torch.ones([init_zdepth.shape[0], self.img_hw[0], self.img_hw[1]]) * 1e11 # (H, W)
        # Zdepth = torch.ones([init_zdepth.shape[0], sdf_renderer.img_hw[0], sdf_renderer.img_hw[1]]) * 1e11 # (H, W)
        Zdepth = torch.ones_like(init_zdepth) * 1e11 # (H, W)
        
        
        Zdepth.requires_grad = True
#        print(init_zdepth.shape)
#        print(marching_zdepth.shape)
#        print(valid_mask.shape)
#        src_zdepth = init_zdepth[valid_mask] + marching_zdepth # (N)
        #20211017
        src_zdepth = init_zdepth[valid_mask] + marching_zdepth[valid_mask]
        
        # marching_zdepth.reshape(-1)[valid_mask.reshape(-1)]
        
        # Zdepth = self.copy_index_ortho(Zdepth, valid_mask, src_zdepth)
        Zdepth = Zdepth.reshape(-1) # (H*W)
        
        # Zdepth[valid_mask] =  src_zdepth
        # index = torch.nonzero(valid_mask).reshape(-1).long()
        # Zdepth = Zdepth[torch.nonzero(valid_mask)]
        
        valid_mask_ = torch.nonzero(valid_mask.reshape(-1))[:,0].long()
        # inputs.index_copy(0, index, src)
        Zdepth_new = Zdepth.index_copy(0, valid_mask_, src_zdepth)
        
        
        # depth = valid_mask.reshape(-1, h, w)
        # plt.imshow(depth[1].detach())
        
#        profiler.report_process('[DEPTH] finalize time\t')
        if no_grad_depth:
            Zdepth = Zdepth.detach()
            
        #20211012
        self.valid_mask_render = valid_mask_render
        self.points_list = points_list
        self.sdf_list= sdf_list
        self.marching_zdepth=marching_zdepth
        self.cam_pos = cam_pos
        self.rays_end = rays_end
        self.dist = dist
#        self.init_zdepth = init_zdepth
        self.valid_mask = valid_mask
        self.min_sdf_sample = min_sdf_sample
        self.Zdepth = Zdepth
        self.rays_start = rays_start
            
        return Zdepth_new, valid_mask_, None # (H*W), (H*W), (H*W)
    
#              ray_marching_recursive(self, cam_pos,               cam_rays, init_zdepth, valid_mask, latent, march_step=None, stop_threshold=None, clamp_dist=0.1, no_grad=False, use_transform=True, use_first_query_check=True)
    
    # def ray_marching_ortho(self, rays_start, cam_pos, R, init_zdepth, valid_mask, latent, march_step=None, clamp_dist=0.1, no_grad=False, use_transform=True, ray_marching_type='recursive', split_type='raydepth'):
    #     '''
    #     ray marching function
    #     Input:
    #     - init_zdepth			type: torch.Tensor (H*W)
    #     - valid_mask			type: torch.Tensor (H*W) with N valid entries
    #     - split_type                    ['depth', 'raydepth'], which is the spliting strategy for pyramid recursive marching
    #     Return:
    #     - sdf_list			type: torch.Tensor (march_step, N)
    #     - marching_zdepth_list		type: torch.Tensor (march_step, N)
    #     - points_list			type: torch.Tensor (march_step, N, 3)
    #     - valid_mask_render		type: torch.Tensor (N)
    #     '''
    #     if not (split_type in ['depth', 'raydepth']):
    #         raise NotImplementedError
    #     elif ray_marching_type == 'recursive':
    #         rays_start, rays_end = self.get_rays_ortho(R)
    #         return self.ray_marching_recursive_ortho(rays_start, cam_pos, rays_end, init_zdepth, valid_mask, latent, march_step=None, clamp_dist=clamp_dist, no_grad=no_grad, use_transform=use_transform)
    #     # elif ray_marching_type == 'pyramid_recursive':
    #     #     return self.ray_marching_pyramid_recursive_ortho(cam_pos_ortho, R, valid_mask, latent, march_step=None, clamp_dist=clamp_dist, no_grad=no_grad, use_transform=use_transform, split_type=split_type)
    #     else:
    #         raise ValueError('Error! Invalid type of ray marching: {}.'.format(ray_marching_type))
            
    def ray_marching_recursive_ortho(self, rays_start, cam_pos, rays_end, init_zdepth, valid_mask, latent, march_step=None, stop_threshold=None, clamp_dist=0.1, no_grad=True, use_transform=True, use_first_query_check=True):
        if stop_threshold is None:
            stop_threshold = self.threshold
        # valid_rays_end = rays_end[:, valid_mask]
        # init_zdepth = init_zdepth[valid_mask]
        # valid_cam_ray_start = rays_start[:, valid_mask]
        bs = latent[5].size(0)
        valid_rays_end = rays_end[None].repeat(bs,1,1)
        init_zdepth = init_zdepth
        valid_cam_ray_start = rays_start[None].repeat(bs,1,1)

        if march_step is None:
            march_step = self.march_step
            
        maxbound_zdepth = torch.ones_like(init_zdepth)*2
        # maxbound_zdepth = self.get_maxbound_zdepth_ortho(valid_cam_ray_start, cam_pos, valid_rays_end)
        # maxbound_zdepth = sdf_renderer.get_maxbound_zdepth_ortho(rays_start, cam_pos, rays_end)
        
        
        self.maxbound_zdepth = maxbound_zdepth
        
        marching_zdepth_list, sdf_list, points_list = [], [], []
        marching_zdepth = torch.zeros_like(init_zdepth)
        self.marching_zdepth_ori = marching_zdepth

        valid_mask_max_marching_zdepth = (marching_zdepth + init_zdepth < maxbound_zdepth)
        unfinished_mask = valid_mask_max_marching_zdepth # (N)
        for i in range(march_step):
            # get unfinished
            # cam_rays_now = valid_rays_end[:, unfinished_mask] # (3, K)
            # init_zdepth_now = init_zdepth[unfinished_mask] # (K)
            # marching_zdepth_now = marching_zdepth[unfinished_mask] # (K)
            # valid_cam_pos_ortho_now = valid_cam_ray_start[:, unfinished_mask]
            cam_rays_now = valid_rays_end[:, :] # (3, K)
            init_zdepth_now = init_zdepth[:] # (K)
            marching_zdepth_now = marching_zdepth[:] # (K)
            valid_cam_pos_ortho_now = valid_cam_ray_start[:, :]

            # get corresponding sdf value
            points_now = self.generate_point_samples_ortho(valid_cam_pos_ortho_now, cam_pos, cam_rays_now, init_zdepth_now + marching_zdepth_now, inv_transform=use_transform) # (3, K)
            # points_now = sdf_renderer.generate_point_samples_ortho(valid_cam_pos_ortho_now, cam_pos, cam_rays_now, init_zdepth_now + marching_zdepth_now, inv_transform=use_transform) # (3, K)
            
            # self.points_now = points_now
            if no_grad:
                points_now = points_now.detach()
            # points_now = torch.matmul(points_now.T, self.rotate).T
            points_now = torch.bmm(points_now.transpose(2,1), self.rotate).transpose(2,1)
            #  points_now = torch.bmm(points_now.transpose(2,1), sdf_renderer.rotate).transpose(2,1)
            
            sdf_now = decode_sdf(self.decoder, latent, points_now, clamp_dist=None, no_grad=no_grad).squeeze(-1) # (K)
            # sdf_now = decode_sdf(sdf_renderer.decoder, latent, points_now, clamp_dist=None, no_grad=no_grad).squeeze(-1) # (K)
            
            # cube_index = torch.where(points_now>0.9)
            
            # show_cube_index = torch.stack([cube_index[0], cube_index[2]]).numpy()
            
            points = torch.zeros_like(marching_zdepth)[:,:,None].repeat(1,1,3)
            points[:,:,:] = points_now.transpose(2,1)
            if no_grad:
                points = points.detach()
            points_list.append(points[None,:])

            # clamp sdf from below if the flag is invalid, which means that it has not meet any sdf < 0
            sdf = marching_zdepth*0.0
            sdf[:] = sdf_now.detach()
            sdf_marching = torch.clamp(sdf, -clamp_dist, clamp_dist)

            # aggressive ray marching
            marching_zdepth = marching_zdepth + sdf_marching * self.ray_marching_ratio
            marching_zdepth_list.append(marching_zdepth[None,:])

            # update sdf list
            # sdf[~unfinished_mask] = 1.0
            sdf_list.append(sdf[None,:])

            # update unfinised mask
            valid_mask_max_marching_zdepth = (marching_zdepth + init_zdepth < maxbound_zdepth)
            # unstop_mask = torch.abs(sdf) >= stop_threshold
            # unfinished_mask = unfinished_mask & valid_mask_max_marching_zdepth & unstop_mask
            if torch.nonzero(unfinished_mask).shape[0] == 0:
                while(len(marching_zdepth_list) < self.buffer_size):
                    marching_zdepth_list.append(marching_zdepth[None,:])
                    sdf_list.append(sdf[None,:])
                    points_list.append(points[None,:])
                break
            
        # concat ray marching info
        marching_zdepth_list = torch.cat(marching_zdepth_list, 0) # (self.march_step, N)
        sdf_list = torch.cat(sdf_list, 0)
        points_list = torch.cat(points_list, 0)
        
        # get valid mask
        valid_mask_max_marching_zdepth = (marching_zdepth_list[-1] + init_zdepth < maxbound_zdepth)
        min_sdf, _ = torch.abs(sdf_list).min(0)
        valid_mask_ray_marching = (min_sdf <= self.threshold)

        # get corner case: the first query is lower than threshold.
        valid_mask_render = valid_mask_max_marching_zdepth & valid_mask_ray_marching # (N)
        if use_first_query_check:
            valid_mask_first_query = sdf_list[0] > self.threshold
            valid_mask_render = valid_mask_render & valid_mask_first_query
        
        #20211012
        self.min_sdf = min_sdf
        self.init_zdepth = init_zdepth
        
        self.marching_zdepth_list = marching_zdepth_list
        return sdf_list, marching_zdepth_list, points_list, valid_mask_render
    
    def generate_point_samples_ortho(self, cam_pos_ortho, cam_pos, cam_rays, Zdepth, inv_transform=True, has_zdepth_grad=False):
        '''
        cam_pos_ortho, cam_pos, cam_rays, Zdepth = valid_cam_pos_ortho_now, cam_pos, cam_rays_now, init_zdepth_now + marching_zdepth_now
        Input:
        - cam_pos	type torch.Tensor (3)
        - cam_ays	type torch.Tensor (3, N)
        - Zdepth	type torch.Tensor (N)
        Return:
        - points	type torch.Tensor (3, N)
        '''
        if not has_zdepth_grad:
            Zdepth = Zdepth.detach()
        # N = Zdepth.shape[0]
        N = Zdepth.shape[1]
        if N == 0:
            raise ValueError('No valid depth.')
#        cam_pos_pad = cam_pos[:,None].repeat(1,N) # (3, N)
        cam_pos_pad = cam_pos_ortho # (3, N)
#        Zdepth_pad = Zdepth[None,:].repeat(3,1) # (3, N)
#        points = cam_rays * Zdepth_pad + cam_pos_pad # (3, N)
        
        # 20211012 ortho 
        points = cam_pos_pad.clone()
        points[:,2,:] = points[:,2,:] + Zdepth
        
        if inv_transform:
            points = self.inv_transform_points(points)
        if not points.requires_grad:
            points.requires_grad=True
        return points
    
    def generate_point_samples_ortho_ori(self, cam_pos_ortho, cam_pos, cam_rays, Zdepth, inv_transform=True, has_zdepth_grad=False):
        '''
        cam_pos_ortho, cam_pos, cam_rays, Zdepth = valid_cam_pos_ortho_now, cam_pos, cam_rays_now, init_zdepth_now + marching_zdepth_now
        Input:
        - cam_pos	type torch.Tensor (3)
        - cam_ays	type torch.Tensor (3, N)
        - Zdepth	type torch.Tensor (N)
        Return:
        - points	type torch.Tensor (3, N)
        '''
        if not has_zdepth_grad:
            Zdepth = Zdepth.detach()
        # N = Zdepth.shape[0]
        N = Zdepth.shape[1]
        if N == 0:
            raise ValueError('No valid depth.')
#        cam_pos_pad = cam_pos[:,None].repeat(1,N) # (3, N)
        cam_pos_pad = cam_pos_ortho # (3, N)
#        Zdepth_pad = Zdepth[None,:].repeat(3,1) # (3, N)
#        points = cam_rays * Zdepth_pad + cam_pos_pad # (3, N)
        
        # 20211012 ortho 
        points = cam_pos_pad.clone()
        points[2,:] = points[2,:] + Zdepth
        
        if inv_transform:
            points = self.inv_transform_points(points)
        if not points.requires_grad:
            points.requires_grad=True
        return points
    
    def get_maxbound_zdepth_ortho(self,cam_pos_ortho, cam_pos, cam_rays):
        with torch.no_grad():
            init_zdepth, _ = self.get_intersections_with_unit_spheres_ortho(cam_pos_ortho, cam_pos, cam_rays) # (N)
            
#            init_zdepth, _ = sdf_renderer.get_intersections_with_unit_spheres_ortho(cam_pos_ortho, cam_pos, cam_rays) # (N)
#            print(valid_cam_rays.shape)
            dist = self.get_distance_from_origin_ortho(cam_pos_ortho, cam_rays) # (N)
            maxbound_marching_zdepth = self.get_maxbound_zdepth_from_dist_ortho(dist) # (N)

            max_zdepth = init_zdepth + maxbound_marching_zdepth # (N)
        return max_zdepth
    
    def get_maxbound_zdepth_from_dist_ortho(self, dist):
        '''
        Input:
        - dist		type torch.FloatTensor (N)
        '''
        with torch.no_grad():
            value = self.radius ** 2 - dist ** 2 +5e-12
#            value = radius ** 2 - dist ** 2 +5e-12
            valid_mask = (value >= 0)

            maxbound_zdepth = torch.zeros_like(dist)
            maxbound_zdepth[valid_mask] = 2 * torch.sqrt(value[valid_mask])
        return maxbound_zdepth
    
    def get_rays_ortho(self, R, homo=None):
        '''
        Input:
        - R	type: torch.Tensor (3,3)
        - T	type: torch.Tensor (3)
        '''
        if homo is None:
            homo = self.homo_calib
        rays = torch.matmul(R.transpose(1,0), homo) # (3, H*W)
#        rays = self.normalize_vectors(rays) # (3, H*W)
        rays_start = rays.clone()

        rays_end = rays.clone()
        rays_end[2,:] = rays_end[2,:]*(-1)
 
        return rays_start,  rays_end
    
    
    
    def get_camera_rays_ortho(self, R, homo=None):
        '''
        Input:
        - R	type: torch.Tensor (3,3)
        - T	type: torch.Tensor (3)
        '''
        if homo is None:
            homo = self.homo_calib
        rays = torch.matmul(R.transpose(1,0), homo) # (3, H*W)
        rays = self.normalize_vectors(rays) # (3, H*W)
        rays_ortho = rays
        rays_ortho[0,:] = 1e-10
        rays_ortho[2,:] = 1e-10
        rays_ortho[1,:] = 1
        
        return rays_ortho
            
    def copy_index_ortho(self, inputs, mask, src):
        '''
        inputs, mask, src =  Zdepth, valid_mask, src_zdepth
        out-of-place copy index.
        Input:
        - inputs:	torch.Tensor (H*W) / (H, W) / (H, W, k)
        - mask:		torch.Tensor (H*W)
        - src:		torch.Tensor (N) / (N, k)
        '''
        inputs_shape = inputs.shape
        if len(inputs_shape) <= 2:
            inputs, mask = inputs.reshape(-1), mask.reshape(-1)
        elif len(inputs_shape) == 3:
            inputs, mask = inputs.reshape(-1, inputs_shape[-1]), mask.reshape(-1)
        else:
            raise NotImplementedError
        index = torch.nonzero(mask).reshape(-1).long()
        outputs = inputs.index_copy(0, index, src)
        outputs = outputs.reshape(inputs_shape)
        return outputs
if __name__ == '__main__':
    pass
        

# index = torch.nonzero(mask).reshape(-1).long()
# outputs = Zdepth.index_copy(0, valid_mask, src)







