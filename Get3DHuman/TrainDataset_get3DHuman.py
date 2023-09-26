from torch.utils.data import Dataset
import numpy as np
import os
import random
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import cv2
import torch
from PIL.ImageFilter import GaussianBlur
import argparse
import logging
import matplotlib.pyplot as plt
import platform
import glob
import natsort

class TrainDataset(Dataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def __init__(self, opt, phase='train'):
        self.opt = opt
        # Path setup
        self.root_label = self.opt.data_root_label
        self.root_nolabel = self.opt.data_root_nolabel

        # self.RENDER = os.path.join(self.root)
        
        self.subjects = self.get_subjects()
        self.is_train = (phase == 'train')
        # self.RENDER = os.path.join(self.root)
        # self.MASK = os.path.join(self.root)
        self.load_size = self.opt.load_size
        self.num_sample_inout = 10000 
        # PIL to tensor
        self.to_tensor = transforms.Compose([
            transforms.Resize(self.load_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        self.to_tensor_512 = transforms.Compose([
            transforms.Resize(512),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # augmentation
        self.aug_trans = transforms.Compose([
            transforms.ColorJitter(brightness=opt.aug_bri, contrast=opt.aug_con, saturation=opt.aug_sat,
                                   hue=opt.aug_hue)
        ])
            
#        self.mesh_dic = load_trimesh(self.OBJ)

    def get_subjects(self):
        all_subjects_label = []
        all_subjects_nolabel = []
        for root_label_i in self.root_label:
        # for root_label_i in opt_data:
            all_subjects_label = all_subjects_label + glob.glob(os.path.join(root_label_i, "*_.jpg"))
            
        for all_subjects_nolabel_i in self.root_nolabel:
            all_subjects_nolabel = all_subjects_nolabel + glob.glob(os.path.join(all_subjects_nolabel_i, "*_.jpg"))
        all_subjects_nolabel = all_subjects_nolabel + all_subjects_label
        # for all_subjects_nolabel_i in opt_all[0].nolabel_data_root:
        #         all_subjects_nolabel = all_subjects_nolabel + glob.glob(os.path.join(all_subjects_nolabel_i, "*_.jpg"))
            
        all_subjects_nolabel = natsort.natsorted(all_subjects_nolabel)
        all_subjects_label = natsort.natsorted(all_subjects_label)
        return all_subjects_label, all_subjects_nolabel

        # if self.is_train:
        #     return all_subjects
        # else:
        #     return all_subjects

    def __len__(self):
        return len(self.subjects[1]) #, len(self.subjects[1])
    
    # def len_(self):
    #     return len(self.subjects[0]), len(self.subjects[1])

    def get_render_label(self, image_path,  random_sample=False):
        
        render_path = image_path
        mask_path = image_path[:-4] + "mask.png"
        # mask_path = latent_path[:-5] + "mask.png"
        latent_path =  image_path[:-4] + "z.npy"
        # depth_f_path = image_path[:-4] + "depth.png"
        # normal_f_path = image_path[:-4] + "normal.jpg"
        path = os.path.split(image_path)
        
        depth_f_path = os.path.join(path[0]+"_nd", path[1][:-4] + "depth.png") 
        normal_f_path = os.path.join(path[0]+"_nd", path[1][:-4] + "normal.jpg") 
        
      
        
        render= Image.open(render_path)
        # if render
        latent_z = np.load(latent_path, allow_pickle=True)
 
        mask = Image.open(mask_path).convert('L')
        depth_f_img = Image.open(depth_f_path)
        normal_f_img = Image.open(normal_f_path)
        # calib = torch.Tensor(np.array([[[ 0.0093,  0.0000,  0.0000,  0.0000],[ 0.0000, -0.0093,  0.0000,  0.8813],
        #                                 [ 0.0000,  0.0000,  0.0093,  0.0000],[ 0.0000,  0.0000,  0.0000,  1.0000]]]))
        calib = torch.Tensor(np.array([[[ 0.00927734,  0.0000,  0.0000,  0.0000],
                                               [ 0.0000, -0.00927734,  0.0000,  0.88134766],
                                               [ 0.0000,  0.0000,  0.00927734,  0.0000],
                                               [ 0.0000,  0.0000,  0.0000,  1.0000]]]))

        # mask = transforms.Resize(self.load_size)(mask)
        # mask_b = ImageOps.mirror(mask)
        # mask_b = transforms.ToTensor()(mask_b).float()
        # mask_1024 = transforms.Resize(1024)(mask)
        # mask_1024 = transforms.ToTensor()(mask_1024).float()
        mask_512 = transforms.Resize(512)(mask)
        mask_512 = transforms.ToTensor()(mask_512).float()
        mask_512 = torch.where(mask_512>0.5, torch.tensor(1), torch.tensor(0))
        
        normal_f_img_512_ori = normal_f_img.resize((512,512), Image.NEAREST)
        normal_f_img_512_ori = np.array(normal_f_img_512_ori)/255*2-1
        normal_f_img_512_ori = torch.tensor(normal_f_img_512_ori).float()
        normal_f_img_512_ori = normal_f_img_512_ori.permute(2,0,1)
        # normal_f_img_512_ori = normal_f_img_512_ori * ( mask_512.repeat(3, 1 ,1))    
        # mask_ori = transforms.ToTensor()(mask).float()
    
        render_512 = render.resize((512, 512), Image.BILINEAR)
        render_512 = self.to_tensor_512(render_512)
        # render_512 = to_tensor_512(render_512)
        
        render_512 = render_512 * ( mask_512.repeat(3, 1 ,1))       
        
        depth_f_img_512_ori = depth_f_img.resize((512,512), Image.NEAREST)
        depth_f_img_512_ori = np.array(depth_f_img_512_ori)/65025*2-1
        depth_f_img_512_ori = torch.tensor(depth_f_img_512_ori).float()
        depth_f_img_512 = np.repeat(depth_f_img_512_ori[np.newaxis,:],3,0)
        # depth_f_img_512 = torch.tensor(depth_f_img_512).float()

        pc = self.dep2pc(depth_f_img_512_ori, mask_512) 
        # mask_512.repeat(3, 1 ,1)

        # normal_f_img_512_ori = normal_f_img.resize((512,512), Image.NEAREST)
        # z_random =  torch.randn([512])

        return {
            'img_latent': render_512,
            'mask_latent': mask_512,
            "latent":latent_z,
            "normal_latent":normal_f_img_512_ori,
            "depth_latent":depth_f_img_512,
            "pc_latent":pc,
            "calib_latent":calib,
            # ""
        }
    
    def get_render_labelno(self, image_path,  random_sample=False):
        render_path = image_path
        mask_path = image_path[:-4] + "mask.png"

        
        path = os.path.split(image_path)
        depth_f_path = os.path.join(path[0]+"_nd", path[1][:-4] + "depth.png") 
        normal_f_path = os.path.join(path[0]+"_nd", path[1][:-4] + "normal.jpg") 
        
       
        
        
        render= Image.open(render_path)
        # latent_z = np.load(latent_path, allow_pickle=True)
 
        mask = Image.open(mask_path).convert('L')
        depth_f_img = Image.open(depth_f_path)
        normal_f_img = Image.open(normal_f_path)
        # calib = torch.Tensor(np.array([[[ 0.0093,  0.0000,  0.0000,  0.0000],[ 0.0000, -0.0093,  0.0000,  0.8813],
        #                                 [ 0.0000,  0.0000,  0.0093,  0.0000],[ 0.0000,  0.0000,  0.0000,  1.0000]]]))
        

       
        mask_512 = transforms.Resize(512)(mask)
        mask_512 = transforms.ToTensor()(mask_512).float()
        mask_512 = torch.where(mask_512>0.5, torch.tensor(1), torch.tensor(0))
        
        normal_f_img_512_ori = normal_f_img.resize((512,512), Image.NEAREST)
        normal_f_img_512_ori = np.array(normal_f_img_512_ori)/255*2-1
        normal_f_img_512_ori = torch.tensor(normal_f_img_512_ori).float()
        normal_f_img_512_ori = normal_f_img_512_ori.permute(2,0,1)
        # normal_f_img_512_ori = normal_f_img_512_ori * ( mask_512.repeat(3, 1 ,1))    
        # mask_ori = transforms.ToTensor()(mask).float()
    
        render_512 = render.resize((512, 512), Image.BILINEAR)
        render_512 = self.to_tensor_512(render_512)
        # render_512 = to_tensor_512(render_512)
        
        render_512 = render_512 * ( mask_512.repeat(3, 1 ,1))       
        
        depth_f_img_512_ori = depth_f_img.resize((512,512), Image.NEAREST)
        depth_f_img_512_ori = np.array(depth_f_img_512_ori)/65025*2-1
        depth_f_img_512_ori = torch.tensor(depth_f_img_512_ori).float()
        depth_f_img_512 = np.repeat(depth_f_img_512_ori[np.newaxis,:],3,0)
        # depth_f_img_512 = torch.tensor(depth_f_img_512).float()

      

        return {
            'img_latentno': render_512,
            'mask_latentno': mask_512,
            # "latent":latent_z,
            "normal_latentno":normal_f_img_512_ori,
            "depth_latentno":depth_f_img_512,
            # "pc_latentno":pc,
        }
    
   

    def dep2pc(self, depth_gt_1, mask_gt, image_size = 512, sigma = 3):
       
        depth_gt = torch.flipud(depth_gt_1)
       
        
        calib_1 = torch.tensor([[ 107.78947529,    0.  ,    0.        ,    0. ],[   0. ,  107.78947529,    0.,    0.],
                            [   0.        ,    0.  , -107.26315789,    0. ],[   0. ,   95.        ,    0.,    1.]]).float()
        
        # vid, uid = torch.where(mask_gt[0] > 0.5)
        vid, uid = torch.where(depth_gt < 0.8)
        uv_mat = torch.ones((len(vid), 4), dtype=np.float)
        pc_num_surface = torch.randperm(len(vid))[:(self.num_sample_inout)]
        uv_mat = torch.ones((len(pc_num_surface), 4), dtype=np.float)
        vid_sample, uid_sample = vid[pc_num_surface], uid[pc_num_surface]
        uv_mat[:,2]  = depth_gt[vid_sample, uid_sample]
        uv_mat[:, 0] = uid_sample/image_size*2-1
        uv_mat[:, 1] = vid_sample/image_size*2-1
        
        uv_mat_surface = torch.matmul(uv_mat.float(), calib_1)
        
        '''sample 3cm'''
        # random_3cm = (torch.rand(uv_mat_surface.shape)*6-3)
        # random_3cm[:,3]=0
        # uv_mat_3cm = uv_mat_surface + random_3cm
        
        random_3cm = torch.tensor(np.random.normal(scale=sigma, size=uv_mat_surface.shape)).float()
        random_3cm[:,3]=0
        uv_mat_3cm = uv_mat_surface + random_3cm
     
        
        # self.sample_number
        return uv_mat_3cm[:,:3].T

    def get_item(self, index):
        # In case of a missing file or IO error, switch to a random sample instead
        # try:
        img_id = index 

        # name of the subject 'rp_xxxx_xxx'
        path_label = self.subjects[0][np.random.randint(len(self.subjects[0]))]
        
        # path_labelno = self.subjects[1][np.random.randint(len(self.subjects[1]))]
        path_labelno = self.subjects[1][index]
        
        # if img_id > len(self.subjects[1]):
        #     path_labelno = self.subjects[1][np.random.randint(len(self.subjects[1]))]
        # else:
        #     path_labelno = self.subjects[1][img_id]
        # z_random = torch.randn([512])
        # z_random = torch.tensor(np.random.randn(512)).float()
        res = {
            'name_label': path_label,
            'name_labelno': path_labelno,
            'img_id_label': img_id,
            'img_id_labelno': img_id,
            "index":index
            # "random_z":z_random
        }
        render_data_label = self.get_render_label(path_label)
        res.update(render_data_label)
        
        render_data_labelno = self.get_render_labelno(path_labelno)
        res.update(render_data_labelno)
        # get_render_labelno

        return res

    def __getitem__(self, index):
        return self.get_item(index)
    

    
    
    
    
    
    
    