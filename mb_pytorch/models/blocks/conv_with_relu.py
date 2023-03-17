import torch
from torch import nn

__all__ = ['ConvBlock']

class ConvBlock(nn.Module):
    """
    Applies 'num_layers', 'stride_conv'(3x3) convolutions each followed by ReLU then downsamples via 'max_pool_stride'(2x2) max pool.
    """

    def __init__(self, num_layers=1, in_channels=4, out_channels=128,conv_kernel_size=3,stride_conv=3,padding=1,max_pool_kernal=2,max_pool_stride=2):
       super().__init__()
       self.convs = nn.ModuleList(
           [nn.Sequential( 
           nn.Conv2d(in_channels if i==0 else out_channels, out_channels, kernel_size=conv_kernel_size, stride=stride_conv,padding=padding),
           nn.ReLU()) 
           for i in range(num_layers)]
       )
       self.downsample = nn.MaxPool2d(kernel_size=max_pool_kernal, stride=max_pool_stride)
      
    def forward(self, x):
       for conv in self.convs:
           x = conv(x)
       x = self.downsample(x)
       return x
      