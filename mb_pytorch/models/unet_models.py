#function: list of unet models
from torch.nn import functional as F
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
from .blocks import attention_block, conv_with_relu, rnn,conv_block

__all__ = ['Unet']

class Unet(nn.Module):
    """
    Basic Unet model
    """
    def __init__(self,**kwargs):
        super(Unet,self).__init__()
        self._data=kwargs
        if 'conv_depth' in self._data:
            self.conv_depth=self._data['conv_depth']
        else:
            self.conv_depth=2
        if 'bottleneck_conv_depth' in self._data:
            self.bottleneck_conv_depth=self._data['bottleneck_conv_depth']
        else:
            self.bottleneck_conv_depth=self.conv_depth
        if 'unet_depth' in self._data:
            self.unet_depth=self._data['unet_depth']
        else:
            self.unet_depth=3
        if 'n_channels' in self._data:
            self.n_channels=self._data['n_channels']
        else:
            self.n_channels=3
        if 'n_classes' in self._data:
            self.n_classes=self._data['n_classes']
        else:
            self.n_classes=1
        if 'n_filters' in self._data:
            self.n_filters=self._data['n_filters']
        else:
            self.n_filters=64
        if 'linear_layers' in self._data:
            self.linear_layers=self._data['linear_layers']
        else:
            self.linear_layers=0

        self.in_channels = [self.n_channels] + [self.n_filters*(2**i) for i in range(self.unet_depth)]
        self.out_channels = self.in_channels[::-1][:-1] + [self.n_classes]
        
        self.unet_conv = nn.Sequential()
        for i in range(self.unet_depth):
            temp_conv = conv_block.ConvBlock(in_channels=self.in_channels[i],out_channels=self.in_channels[i+1],
                                             depth=self.conv_depth,**kwargs)
            self.unet_conv.add_module(f"unet_conv_{i}",temp_conv)
        

        self.bottle_neck = nn.Sequential(conv_block.ConvBlock(in_channels=self.out_channels[0],out_channels=self.out_channels[0],
                                                              depth=self.bottleneck_conv_depth,sample_type='bottleneck',**kwargs))

        self.unet_deconv = nn.Sequential()
        for i in range(self.unet_depth-1):
            temp_deconv = conv_block.ConvBlock(in_channels=self.out_channels[i],out_channels=self.out_channels[i+1],
                                               depth=self.conv_depth,sample_type='up',**kwargs)
            self.unet_deconv.add_module(f"unet_deconv_{i}",temp_deconv)
        

        self.final = nn.Sequential()
        self.final.add_module(f'final_conv',nn.Conv2d(self.in_channels[1],self.n_classes,kernel_size=1))

        if self.linear_layers > 0:
            self.final.add_module(f"unet_linear",nn.Linear(self.n_classes,self.n_classes))


        if self.n_classes == 1:
            self.final.add_module(f'final_activation',nn.Sigmoid())
        else:
            self.final.add_module(f'final_activation',nn.Softmax(dim=1))

    def forward(self,x):

        x = self.unet_conv(x)
        x = self.bottle_neck(x)
        x = self.unet_deconv(x)
        x = self.final(x)
        return x

