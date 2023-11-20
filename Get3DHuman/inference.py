import os
from libs.model_refinenet import refinenet
import torch
import numpy as np
import dnnlib
from opt_all import load_opt
import time
import matplotlib.pyplot as plt
from tools import   avatar_infer_shape, create_grid,  inference_avatar
from libs.DepthNormalizer import DepthNormalizer
import cv2
from libs.geometry import orthogonal, index 
import math
from scipy.spatial import cKDTree
import argparse

parser = argparse.ArgumentParser()
parser.add_argument( '--model_path', type=str, default="")
parser.add_argument( '--out_dir', type=str, default='')
parser.add_argument( '--cuda',  type=int, default=0, help='')
parser.add_argument( '--vis_cuda',  type=int, default=1, help='')
parser.add_argument( '--sample_num',  type=int, default=1, help='')
parser.add_argument( '--render_size',  type=int, default=512, help='')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.vis_cuda)


ticks = time.strftime("%Y%m%d_%H_%M", time.localtime()) 

'''pretrain model path and save path '''
save_path = args.out_dir
pretrain_st_path = os.path.join(args.model_path, "Get3DHuman_shape_texture_v2.pth")
pretrain_refine_path = os.path.join(args.model_path, "Get3DHuman_refinement_v1.pth")



'''# define  parameters'''
opt_all = load_opt()
spatial_enc = DepthNormalizer(opt_all[0])
res = args.render_size
b_min = np.array([-128,  -28, -128])
b_max = np.array([128, 228, 128])
coords, mat = create_grid(res, res, res, b_min, b_max, transform=None)
cuda = torch.device('cuda:%d' % int(0))
calib_r =  torch.Tensor(np.array([[[ 0.00927734,  0.0000,  0.0000,  0.0000], [ 0.0000, -0.00927734,  0.0000,  0.88134766],
                                   [ 0.0000,  0.0000,  0.00927734,  0.0000], [ 0.0000,  0.0000,  0.0000,  1.0000]]])).to(device=cuda)

calib_define = np.array([[ 107.78947529,    0.  ,    0.        ,    0. ],[   0. ,  107.78947529,    0.,    0.],
                         [   0.        ,    0.  , -107.78947529,    0. ],[   0. ,   95.        ,    0.,    1.]])

projection_matrix = np.identity(4)
projection_matrix[1, 1] = -1
calib = torch.Tensor(projection_matrix).float().to(device=cuda)

extrinsic_pifu = np.array([[ 1. ,  0. ,  0. , -0. ],
       [-0. , -1. , -0. , -0. ],
       [ 0. ,  0. , -1. , 0]])
R, T = extrinsic_pifu[:,:3], extrinsic_pifu[:,3]
R, T = torch.from_numpy(R).float().to(device=cuda), torch.from_numpy(T).float().to(device=cuda)
kernel = np.ones((4,4),np.uint8) 
name = "results"
args.render_size = res




'''# define  parameters'''
G_kwargs = opt_all[1].G_kwargs
G_kwargs.update({ "render_size":args.render_size, 'cuda_':cuda, "use_gpu":True, "class_name":"training.generator_tex.I3DGAN_Generator"})
common_kwargs = {'c_dim': 0, 'img_resolution': 512, 'img_channels': 3}
Get3DHuman = dnnlib.util.construct_class_by_name(**G_kwargs, **common_kwargs).cuda()

if pretrain_st_path != None:
    pretext_model = torch.load(pretrain_st_path,map_location=cuda)
    state_dict=[]
    state_dict = { k[7:]:v  for k,v in pretext_model.items() if (k[:7]=="module.")and(k[7:10]!="Dis")}
    if len(state_dict)== 0:
        state_dict = pretext_model
   
Get3DHuman_dict = Get3DHuman.state_dict()
Get3DHuman_dict.update(state_dict)
Get3DHuman.load_state_dict(Get3DHuman_dict, strict=False)
del Get3DHuman_dict, state_dict

print("Loaded shape and texture model")


model_name = os.path.split(pretrain_st_path)[-1][:-4]
model_i2i = refinenet(isTrain=False)
model_i2i = model_i2i.to(device=cuda)
if pretrain_refine_path != None:   
    model_i2i.load_state_dict(torch.load(pretrain_refine_path,map_location=cuda), strict=False)
    pretrain_refine_name = os.path.split(pretrain_refine_path)[-1][:-26]

print("Loaded refine model")




for random_i in range(args.sample_num):

    random_z_s =  torch.from_numpy(np.random.randn(1, 512)).to(device=cuda)
    random_z_t =  torch.from_numpy(np.random.randn(1, 512)).to(device=cuda)
    
    name_obj = "%05d"%random_i
    save_path_obj = os.path.join(save_path,"%s"%( name) )
    # save_root_random = save_path_obj
    

    
    with torch.no_grad():
        shape_feature, fv_shape = Get3DHuman.forward_shape_(random_z_s, 0)
        
    with torch.no_grad():
        verts_pred, faces_pred = avatar_infer_shape(spatial_enc, cuda, calib_r, mat, fv_shape, None, Get3DHuman.mlp_shape, coords)

    for tex_i in range(1):
        with torch.no_grad():
            texture_field = Get3DHuman.forward_texture(shape_feature, random_z_t, 0)
            fv_texture = torch.cat([fv_shape, texture_field], 1)

            if os.path.exists(save_path_obj)== False:
                os.makedirs(save_path_obj)

        point_rgb_all, save_imgs = inference_avatar(Get3DHuman, model_i2i, fv_shape, fv_texture, args, calib_define, calib, kernel, spatial_enc, R, T, cuda)
        

            
        point_rgb_all = point_rgb_all[1:,:]
        points_mesh = verts_pred
        point_rgb_all
        
        Tns = []
        Tns.append(point_rgb_all[:,:3])

        merge_vertices = np.concatenate(Tns)
        tree = cKDTree(merge_vertices)
        k = 1
        indices = tree.query(points_mesh, k)[1]
        points_mesh_color = point_rgb_all[indices][:,3:]

        points_mesh_recolor = np.concatenate((points_mesh, points_mesh_color),1)
        
        # Write OBJ file
        
        f = open(os.path.join(save_path_obj , '%s_texture.obj'%(name_obj)), 'w')
        for i in range(len(points_mesh_recolor)):
            f.write('v %f %f %f %f %f %f\n' % (points_mesh_recolor[i][0], points_mesh_recolor[i][1], points_mesh_recolor[i][2], 
                                      (points_mesh_recolor[i][5])/255, (points_mesh_recolor[i][4])/255, (points_mesh_recolor[i][3])/255))
        for i_f in range(len(faces_pred)):
            f.write('f %d %d %d\n' % (faces_pred[i_f][0]+1, faces_pred[i_f][2]+1, faces_pred[i_f][1]+1))
        f.close()
        
        plt.imsave(os.path.join(save_path_obj,"%s_180.png"%(name_obj)), (save_imgs[0]*255).astype("uint8"))
        plt.imsave(os.path.join(save_path_obj,"%s_000.png"%(name_obj)), (save_imgs[1]*255).astype("uint8"))



