from torch.utils.data import Dataset
import numpy as np
import os
import random
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import cv2
import torch
from PIL.ImageFilter import GaussianBlur

import logging
import matplotlib.pyplot as plt
import platform
import glob
import natsort

class TrainDataset(Dataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def __init__(self, rp_data, rp_data_pred, load_size, real_data=None, phase='train'):
        # self.opt = opt
        # self.projection_mode = 'orthogonal'
        self.loadSize = load_size
        self.load_size = load_size
        
        # Path setup
        self.rp_data_pred = rp_data_pred
        self.rp_data = rp_data
        self.real_data = real_data

        # self.RENDER = os.path.join(self.root, 'ALBEDO')


        self.is_train = (phase == 'train')
        self.random_flip =True
        self.random_scale =True
        self.random_trans =True

        self.subjects = self.get_subjects(self.rp_data_pred, self.real_data)

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

#        self.mesh_dic = load_trimesh(self.OBJ)

    def get_subjects(self, rp_pred_root, real_root=None):
        all_subjects =[]
        gt_subjects = []
        rp_pred_fn_fb = []
        real_pred_fn = []
        
        rp_pred_fn = glob.glob(os.path.join(rp_pred_root, "*", "*_00.png"))
        rp_pred_fn_ = natsort.natsorted(rp_pred_fn)
        
        for rp_i in range(int(len(rp_pred_fn_)/360)):
            rp_pred_fn_cache = rp_pred_fn_[rp_i*360:(rp_i+1)*360]
            rp_pred_fn_cache = rp_pred_fn_cache[:30] + rp_pred_fn_cache[150:210] + rp_pred_fn_cache[330:]
            rp_pred_fn_fb = rp_pred_fn_fb + rp_pred_fn_cache
            
        self.rp_num = len(rp_pred_fn_fb)
        # self.rp_pred_fn_fb = len(rp_pred_fn_fb)
        
        if real_root!=None:
            real_pred_fn = glob.glob(os.path.join(real_root,  "*_pred.png"))
            real_pred_fn = natsort.natsorted(real_pred_fn)

        all_subjects = gt_subjects + rp_pred_fn_fb + real_pred_fn
        
        return all_subjects
    

    def __len__(self):
        return len(self.subjects) #* len(self.yaw_list) * len(self.pitch_list)

    def get_render(self, subject, yid):
        '''
        Return the render data
        :param subject: subject name
        :param num_views: how many views to return
        :param view_id: the first view_id. If None, select a random one.
        :return:
            'img': [num_views, C, W, H] images
            'calib': [num_views, 4, 4] calibration matrix
            'extrinsic': [num_views, 4, 4] extrinsic matrix
            'mask': [num_views, 1, W, H] masks
        '''
        
        if yid < self.rp_num:
            # root_rp_pred = self.rp_data_pred
            render_path_pred = subject
            mask_path_pred = subject[:-4] + "_mask.png"
            
            image_id = os.path.split(render_path_pred)
            rp_name = os.path.split(image_id[0])

            mask_path = os.path.join(self.rp_data,"MASK",rp_name[1], image_id[1][:-3]+"jpg")
            render_path = os.path.join(self.rp_data,"ALBEDO",rp_name[1], image_id[1][:-3]+"jpg")

        else:
            # subject = real_pred_fn[0]
            mask_path = subject[:-8] + "gt_mask.png"
            render_path = subject[:-8] + "gt.png"
            mask_path_pred = subject[:-8] + "pred_mask.png"
            render_path_pred = subject
        
      
        mask = Image.open(mask_path).convert('L')
        mask = mask.resize((512,512), Image.NEAREST)
        render = Image.open(render_path).convert('RGB')
        render = render.resize((512,512), Image.BILINEAR)
        
        mask_pred = Image.open(mask_path_pred).convert('L')
        render_pred = Image.open(render_path_pred).convert('RGB')

        if self.is_train:
            # Pad images
            pad_size = int(0.1 * self.load_size)
            render = ImageOps.expand(render, pad_size, fill=0)
            mask = ImageOps.expand(mask, pad_size, fill=0)
            
            render_pred = ImageOps.expand(render_pred, pad_size, fill=0)
            mask_pred = ImageOps.expand(mask_pred, pad_size, fill=0)
            
            w, h = render.size
            th, tw = self.load_size, self.load_size

            # random flip
            if self.random_flip and np.random.rand() > 0.5:
                
                render = transforms.RandomHorizontalFlip(p=1.0)(render)
                mask = transforms.RandomHorizontalFlip(p=1.0)(mask)
                
                render_pred = transforms.RandomHorizontalFlip(p=1.0)(render_pred)
                mask_pred = transforms.RandomHorizontalFlip(p=1.0)(mask_pred)

            # random scale
            if self.random_scale:
                rand_scale = random.uniform(0.9, 1.1)
                w = int(rand_scale * w)
                h = int(rand_scale * h)
                
                render = render.resize((w, h), Image.BILINEAR)
                mask = mask.resize((w, h), Image.NEAREST)
                
                render_pred = render_pred.resize((w, h), Image.BILINEAR)
                mask_pred = mask_pred.resize((w, h), Image.NEAREST)
                

            # random translate in the pixel space
            if self.random_trans:
#               
                dx = random.randint(-int(round((w ) / 10.)),
                                    int(round((w ) / 10.)))
                dy = random.randint(-int(round((h ) / 20.)),
                                    int(round((h ) / 20.)))
            else:
                dx = 0
                dy = 0

            
            x1 = int(round((w - tw) / 2.)) + dx
            y1 = int(round((h - th) / 2.)) + dy

            render = render.crop((x1, y1, x1 + tw, y1 + th))
            mask = mask.crop((x1, y1, x1 + tw, y1 + th))
            
            render_pred = render_pred.crop((x1, y1, x1 + tw, y1 + th))
            mask_pred = mask_pred.crop((x1, y1, x1 + tw, y1 + th))
            # render = self.aug_trans(render)
            
        


        # mask = transforms.Resize(self.load_size)(mask)
       
        mask_512 = transforms.Resize(512)(mask)
        mask_512 = transforms.ToTensor()(mask_512).float()
        mask_512 = torch.where(mask_512>0.5,1,0)
        
        

        render = render.resize((512,512), Image.BILINEAR)
        render = self.to_tensor_512(render)
        mask_512_3 = mask_512.expand_as(render)
        render = mask_512_3 * render  + (1-mask_512_3)
        
        mask_512_pred = transforms.Resize(512)(mask_pred)
        mask_512_pred = transforms.ToTensor()(mask_512_pred).float()
        mask_512_pred = torch.where(mask_512_pred>0.5,1,0)
        

        render_pred = render_pred.resize((512,512), Image.BILINEAR)
        render_pred = self.to_tensor_512(render_pred)
        mask_512_pred_3 = mask_512_pred.expand_as(render_pred) 
        render_pred = mask_512_pred_3 * render_pred + (1-mask_512_pred_3)

        
        return {
            'img': render,
            'mask': mask_512,
            'img_pred': render_pred,
            'mask_pred': mask_512_pred
        }

 
    def get_item(self, index):

        path_img = self.subjects[index]
        # name of the subject 'rp_xxxx_xxx'
        # subject = self.subjects[sid]
        res = {'index': index}
        render_data = self.get_render(path_img,index)
        res.update(render_data)

        return res

    def __getitem__(self, index):
        return self.get_item(index)
    
    
    
    
    
'''
#opt_data.random_scale=True
#opt_data.random_trans=True

data = TrainDataset(opt_data)

train_data = data.__getitem__(0)

image_tensor_512 = train_data['img_512'].numpy()
image_tensor_1024 = train_data['img_1024'].numpy()
image_tensor = train_data['img'].numpy()
calib_tensor = train_data['calib'].numpy()
sample_tensor = train_data['samples'].numpy()
label_tensor = train_data['labels'].numpy()
normal_f_tensor = train_data['normal_f_img_list'].numpy()
normal_b_tensor = train_data['normal_b_img_list'].numpy()

img = np.transpose(np.concatenate([image_tensor_1024,normal_f_tensor,normal_b_tensor],3)[0],[1,2,0])
img = (img+1)/2

plt.imshow(img)

normal_f_tensor_1 = normal_f_tensor+0
normal_f_tensor_1[:,1,:,:] = normal_f_tensor_1[:,1,:,:]*(-1)
img = np.transpose(np.concatenate([image_tensor_1024,normal_f_tensor_1,normal_b_tensor],3)[0],[1,2,0])
img = (img+1)/2

plt.imshow(img)

'''    
    
    
    
    
    
    
    
    