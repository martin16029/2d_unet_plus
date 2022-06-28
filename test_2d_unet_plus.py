# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch import rand as rand
from torch import cat as cat
from torch import tensor
import torch.utils.model_zoo as model_zoo
from torch.nn import functional as F
import numpy as np

from test_2d_unet_final import Convblock, Down_Stream_Block


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



class _2D_Unet_Plus(nn.Module):
    def __init__(self, block, depth, num_classes, dilation=1, gray_scale = True, base=16):
        self.depth = depth
        self.num_classes = num_classes
        if gray_scale:
            self.input_channel = 1
        else:
            self.input_channel = 3
        self.base = base
        
        
        super(_2D_Unet_Plus, self).__init__()
        
        self.initial = nn.Conv2d(self.input_channel, self.base, kernel_size=3, stride=1, padding=1, bias=False)
        self.top_section = Convblock(self.base, self.base, self.base)
        
        self.down_part = nn.ModuleList()
        for i in range(1, int(self.depth)):
            self.down_part.append(Down_Stream_Block(block, self.base*2**(i-1), self.base*2**(i), self.base*2**(i)))
            
        
        self.up_part = nn.ModuleDict()
        for i in range(1, int(self.depth)):
            self.decode = nn.ModuleList()
            for j in range(i-1, -1, -1):
                self.decode.append(Up_Stream_Block(block, (self.base*2**j)*(2+i-j), self.base*2**(j), self.base*2**(j)))
                
            self.decode.append(nn.Conv2d(int(self.base*2**(j)), int(self.num_classes), kernel_size=1, stride=1, padding=0, bias=False))
            #self.decode.append(nn.Sigmoid())
            
            self.up_part["decode_"+str(i)]=self.decode
        
        
    
    def forward(self, x):
        x = self.initial(x)
        x = self.top_section(x)
        encode_products = list()
        encode_products.append(x)
        decode_products = dict()
        decode_product_list = list()
        final_output = list()
        for k in range(int(self.depth)-1):
            l = self.down_part[k]
            x = l(x)
            encode_products.append(x)       
        
        
        for i in range(1, int(self.depth)):
           encode_p1 = encode_products[i]
           up = self.up_part["decode_"+str(i)]
           out_list = list()
           for j in range(i):
               if j == 0:
                   encode_p2 = encode_products[i-1]
                   x = up[j](encode_p1, encode_p2)
               else:
                   
                   for key, value in decode_products.items():
                       try:
                           decode_product_list.append(value[j-i])
                       except IndexError:
                           pass
                   decode_product_list.append(encode_products[i-j-1])
                   #for product in decode_product_list:
                       #print(product.shape)
                   x = up[j](x, decode_product_list)
                   
                   
               out_list.append(x)
               
               if j == i-1:
                  x = up[-1](x) 
                  #x = torch.sigmoid(x)
                  final_output.append(x)
              
               decode_product_list = list() 
           
           decode_products[str(i)] = out_list
        return final_output



def _2D_unetplus(n_class=20, gray_scale = True, base=16):
    _2d_unetplus = _2D_Unet_Plus(Convblock, 4, n_class, gray_scale = gray_scale, base = base)
    return _2d_unetplus     

    

if __name__ == "__main__":
    
    from torchsummary import summary
    test_net = _2D_unetplus(gray_scale = False, base = 16)
    test_net.cuda()
    summary(test_net, (3,64,64))
    

"""
Created on Tue Apr 26 11:57:24 2022

@author: ai
"""
