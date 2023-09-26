# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F 

class DepthNormalizer(nn.Module):
    def __init__(self, z_size = 200.0, loadSize = 512):
        super(DepthNormalizer, self).__init__()
        self.z_size = z_size
        self.loadSize = loadSize
        
    def forward(self, xyz, calibs=None, index_feat=None):
        '''
        normalize depth value
        args:
            xyz: [B, 3, N] depth value
        '''
#        print(self.opt.loadSize)
        z_feat = xyz[:,2:3,:] * (self.loadSize // 2) / self.z_size

        return z_feat