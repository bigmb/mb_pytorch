#function: list of unet models
from torch.nn import functional as F
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
from blocks import attention_block, conv_with_relu, rnn,conv_block



class Unet(nn.Module):
    """
    Basic Unet model
    """
    def __init__(self,**kwargs):
        super(Unet,self).__init__()
        self._data=kwargs
        if 'depth' in self._data:
            self.depth=self._data['depth']
        else:
            self.depth=4
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
        
        out_channels = [self.n_filters*(2**i) for i in range(self.depth)]
        self.convs = conv_block.ConvBlock(in_channels=self.n_channels,out_channels=out_channels,depth=self.depth,**kwargs)
        self.deconvs = conv_block.ConvBlock(in_channels=out_channels[::-1][0],out_channels=out_channels[0],depth=self.depth,**kwargs)
        self.linears = nn.ModuleList([nn.Linear(out_channels[0],out_channels[0]) for i in range(self.linear_layers)])

    def forward(self,x):
        x = self.convs(x)
        x = self.deconvs(x)
        if self.linear_layers > 0:
            for i in range(self.linear_layers):
                x = self.linears[i](x)
        return x