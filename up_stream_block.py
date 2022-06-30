# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch import rand as rand
from torch import cat as cat
from torch import tensor
import torch.utils.model_zoo as model_zoo
from torch.nn import functional as F

class Up_Stream_Block(nn.Module):
    def __init__(self, block, inplanes, planes1, planes2, stride=1, dilation=1):
        self.block = block
        self.inplanes = inplanes
        self.planes1 = planes1
        self.planes2 = planes2
        self.stride = stride
        self.dilation = dilation
        super(Up_Stream_Block, self).__init__()
        
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners= False)
        self.cnn = self.block(self.inplanes, self.planes1, self.planes2, self.stride, self.dilation)
        
    def forward(self, x1, x_list):
        x1 = self.up(x1)
        
        if isinstance(x_list, list):
            diffY = x_list[0].size()[2] - x1.size()[2]
            diffX = x_list[0].size()[3] - x1.size()[3]
    
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
            for x2 in x_list:
                x1 = cat([x1, x2], dim=1)
        else:
            diffY = x_list.size()[2] - x1.size()[2]
            diffX = x_list.size()[3] - x1.size()[3]
    
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
            x1 = cat([x1, x_list], dim=1)   
        
        x = self.cnn(x1)
        return x
"""
Created on Wed Jun 29 17:09:23 2022

@author: Martin
"""
