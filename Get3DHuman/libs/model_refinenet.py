# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# import numpy as np
import torch
import torch.nn as nn
# import torch.nn.functional as F 
# from libs.BasePIFuNet import BasePIFuNet
# from libs.MLP import MLP
# from libs.DepthNormalizer import DepthNormalizer
# from DepthNormalizer_selfpifu import DepthNormalizer
# from libs.HGFilters_hd import HGFilter
# from libs.net_util import init_net
# from libs.networks import define_G
# import cv2
# import model_u
# from diff_render.renderer import SDFRenderer
# from diff_render.decoder_utils import load_decoder
from libs.models_social_media import Generate
# from libs.geometry import orthogonal, index 
# from surface_tracing import SDFRenderer

# from lib_hrnet.hrnet_config import load_config
# from lib_hrnet.seg_hrnet import HighResolutionNet
# from lib_refine.image_pool import ImagePool
from lib_refine.networks_loss import GANLoss, VGGLoss, define_D


class refinenet(nn.Module):
    '''
    HGPIFu uses stacked hourglass as an image encoder.
    '''

    def __init__(self, isTrain):
        super(refinenet, self).__init__()
        self.name = 'refinenet'
        self.rnet = Generate(3,3)
        
        self.isTrain = isTrain
        if self.isTrain:
            # 
            self.criterionGAN = GANLoss(use_lsgan=True, tensor=torch.cuda.FloatTensor)   
            self.criterionFeat = torch.nn.L1Loss()
            self.criterionVGG = VGGLoss()
            
            netD_input_nc= 6
            self.netD = define_D(netD_input_nc, 64, 3, "instance", "False", 2, True)
    # def init_loss_filter(self, use_gan_feat_loss, use_vgg_loss):
    #     flags = (True, use_gan_feat_loss, use_vgg_loss, True, True)
    #     def loss_filter(g_gan, g_gan_feat, g_vgg, d_real, d_fake):
    #         return [l for (l,f) in zip((g_gan,g_gan_feat,g_vgg,d_real,d_fake),flags) if f]
    #     return loss_filter
    
    def discriminate(self, input_label, test_image, use_pool=False):
        input_concat = torch.cat((input_label, test_image.detach()), dim=1)
        if use_pool:            
            fake_query = self.fake_pool.query(input_concat)
            return self.netD.forward(fake_query)
        else:
            return self.netD.forward(input_concat)

    def forward(self, image_input, image_gt):
       
        input_label, real_image = image_input, image_gt
        # input_concat = input_label
        
        fake_image = self.rnet(input_label)

        # Fake Detection and Loss
        pred_fake_pool = self.discriminate(input_label, fake_image, use_pool=False)
        loss_D_fake = self.criterionGAN(pred_fake_pool, False)        

        # Real Detection and Loss        
        pred_real = self.discriminate(input_label, real_image)
        loss_D_real = self.criterionGAN(pred_real, True)

        # GAN loss (Fake Passability Loss)        
        pred_fake = self.netD.forward(torch.cat((input_label, fake_image), dim=1))        
        loss_G_GAN = self.criterionGAN(pred_fake, True)               
        
        # GAN feature matching loss
        loss_G_GAN_Feat = 0
        if True:
            feat_weights = 4.0 / (3 + 1)
            D_weights = 1.0 / 2
            for i in range(2):
                for j in range(len(pred_fake[i])-1):
                    loss_G_GAN_Feat += D_weights * feat_weights * \
                        self.criterionFeat(pred_fake[i][j], pred_real[i][j].detach()) * 10.0
                   
        # VGG feature matching loss
        loss_G_VGG = 0
        if True:
            loss_G_VGG = self.criterionVGG(fake_image, real_image) * 10.0
        
        # Only return the fake_B image if necessary to save BW
        return [  {"G_GAN":loss_G_GAN, "G_GAN_Feat":loss_G_GAN_Feat, "G_VGG":loss_G_VGG, "D_real":loss_D_real, "D_fake":loss_D_fake} ,  fake_image ]

    
    def img2img(self,images):
#        input_img = torch.cat([images, denspose],1)
#        pred_norm = self.netG_n.forward(torch.cat([images, denspose],1))
        pred_image = self.rnet(images)
        return pred_image
    def img2img_infer(self,images):
#        input_img = torch.cat([images, denspose],1)
#        pred_norm = self.netG_n.forward(torch.cat([images, denspose],1))
        with torch.no_grad():
            pred_image = self.rnet(images)
        return pred_image
    
