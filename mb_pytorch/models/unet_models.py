#function: list of unet models
from torch.nn import functional as F
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
from .blocks import attention_block, conv_with_relu, rnn,conv_block


__all__ = ['Unet','Unet_attention']

class Unet_old(nn.Module):
    """
    Basic Unet model
    """
    def __init__(self,**kwargs):
        super(Unet_old,self).__init__()
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
        
        #self.unet_conv = nn.Sequential()
        
        for i in range(self.unet_depth):
            temp_conv = conv_block.ConvBlock(in_channels=self.in_channels[i],out_channels=self.in_channels[i+1],
                                             depth=self.conv_depth,**kwargs)
            #self.unet_conv.add_module(f"unet_conv_{i}",temp_conv)
            self.add_module(f"unet_conv_{i}",temp_conv)
        

        #self.bottle_neck = nn.Sequential(conv_block.ConvBlock(in_channels=self.out_channels[0],out_channels=self.out_channels[0],
        #                                                      depth=self.bottleneck_conv_depth,sample_type='bottleneck',**kwargs))
        self.bottle_neck = conv_block.ConvBlock(in_channels=self.out_channels[0],out_channels=self.out_channels[0],
                                                depth=self.bottleneck_conv_depth,sample_type='bottleneck',**kwargs)

        #self.unet_deconv = nn.Sequential()
        for i in range(self.unet_depth-1):
            temp_deconv = conv_block.ConvBlock(in_channels=self.out_channels[i],out_channels=self.out_channels[i+1],
                                               depth=self.conv_depth,sample_type='up',**kwargs)
            #self.unet_deconv.add_module(f"unet_deconv_{i}",temp_deconv)
            self.add_module(f"unet_deconv_{i}",temp_deconv)

        #self.final = nn.Sequential()
        #self.final.add_module(f'final_conv',nn.Conv2d(self.in_channels[1],self.n_classes,kernel_size=1))
        #self.final.add_module(f'up_sample',nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True))
        self.add_module(f'final_conv',nn.Conv2d(self.in_channels[1],self.n_classes,kernel_size=1))
        self.add_module(f'up_sample',nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True))



        if self.linear_layers > 0:
            #self.final.add_module(f"unet_linear",nn.Linear(self.n_classes,self.n_classes))
            self.add_module(f"unet_linear",nn.Linear(self.n_classes,self.n_classes))

        if self.n_classes == 1:
            #self.final.add_module(f'final_activation',nn.Sigmoid())
            self.add_module(f'final_activation',nn.Sigmoid())
        else:
            #self.final.add_module(f'final_activation',nn.Softmax(dim=1))
            self.add_module(f'final_activation',nn.Softmax(dim=1))

    def forward(self,x):

        #x = self.unet_conv(x)
        x = self.unet_conv_0(x)
        x1 = self.unet_conv_1(x)
        x2 = self.unet_conv_2(x1)
        x3 = self.bottle_neck(x2)
        x4 = self.unet_deconv_0(x3+x2)
        x5 = self.unet_deconv_1(x4+x1)
        x6 = self.final_conv(x5+x)
        final = self.final_activation(self.up_sample(x6))
        return final


class Unet(nn.Module):
    """
    Basic Unet model
    """
    def __init__(self, conv_depth=2, bottleneck_conv_depth=None, unet_depth=3, n_channels=3, n_classes=1, n_filters=64, linear_layers=0, **kwargs):
        super(Unet, self).__init__()
        if 'conv_depth' in kwargs:
            self.conv_depth = kwargs['conv_depth']
        else:
            self.conv_depth = conv_depth
        if 'bottleneck_conv_depth' in kwargs:
            self.bottleneck_conv_depth = kwargs['bottleneck_conv_depth']
        else:
            self.bottleneck_conv_depth = bottleneck_conv_depth if bottleneck_conv_depth is not None else conv_depth
        if 'unet_depth' in kwargs:
            self.unet_depth = kwargs['unet_depth']
        else:
            self.unet_depth = unet_depth
        if 'n_channels' in kwargs:
            self.n_channels = kwargs['n_channels']
        else:
            self.n_channels = n_channels
        if 'n_classes' in kwargs:
            self.n_classes = kwargs['n_classes']
        else:
            self.n_classes = n_classes
        if 'n_filters' in kwargs:
            self.n_filters = kwargs['n_filters']
        else:
            self.n_filters = n_filters
        if 'linear_layers' in kwargs:
            self.linear_layers = kwargs['linear_layers']
        else:
            self.linear_layers = linear_layers

        self.in_channels = [self.n_channels] + [self.n_filters*(2**i) for i in range(self.unet_depth)]
        self.out_channels = self.in_channels[::-1][:-1] + [self.n_classes]

        self.unet_conv = nn.ModuleList([conv_block.ConvBlock(in_channels=self.in_channels[i], out_channels=self.in_channels[i+1], **kwargs) for i in range(self.unet_depth)])

        self.bottle_neck = conv_block.ConvBlock(in_channels=self.out_channels[0],out_channels=self.out_channels[0], **kwargs)

        self.unet_deconv = nn.ModuleList([conv_block.ConvBlock(in_channels=self.out_channels[i], out_channels=self.out_channels[i+1], sample_type='up', **kwargs) for i in range(self.unet_depth-1)])

        self.final_conv = nn.Conv2d(self.in_channels[1], self.n_classes, kernel_size=1)
        self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        if self.linear_layers > 0:
            self.unet_linear = nn.Linear(self.n_classes,self.n_classes)

        if self.n_classes == 1:
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = nn.Softmax(dim=1)

    def forward(self, x):
        temp_out = []
        for i in range(self.unet_depth):
            x = self.unet_conv[i](x)
            temp_out.append(x)
            x = temp_out[-1]

        x = self.bottle_neck(x)

        for i in range(self.unet_depth-1):
            x = self.unet_deconv[i](x + temp_out[-(i+1)])

        x = self.final_conv(x + temp_out[0])
        x = self.up_sample(x)
        final = self.final_activation(x)

        if self.linear_layers > 0:
            final = final.view(final.size(0),-1)
            final = self.unet_linear(final)
            
        return final
    

class Unet_attention(nn.Module):
    """
    Unet with attnetion module
    """
    def __init__(self, conv_depth=2, bottleneck_conv_depth=None, unet_depth=3, n_channels=3, n_classes=1, n_filters=64, linear_layers=0, **kwargs):
        super(Unet_attention, self).__init__()
        if 'conv_depth' in kwargs:
            self.conv_depth = kwargs['conv_depth']
        else:
            self.conv_depth = conv_depth
        if 'bottleneck_conv_depth' in kwargs:
            self.bottleneck_conv_depth = kwargs['bottleneck_conv_depth']
        else:
            self.bottleneck_conv_depth = bottleneck_conv_depth or conv_depth
        if 'unet_depth' in kwargs:
            self.unet_depth = kwargs['unet_depth']
        else:
            self.unet_depth = unet_depth
        if 'n_channels' in kwargs:
            self.n_channels = kwargs['n_channels']
        else:
            self.n_channels = n_channels
        if 'n_classes' in kwargs:
            self.n_classes = kwargs['n_classes']
        else:
            self.n_classes = n_classes
        if 'n_filters' in kwargs:
            self.n_filters = kwargs['n_filters']
        else:
            self.n_filters = n_filters
        if 'linear_layers' in kwargs:
            self.linear_layers = kwargs['linear_layers']
        else:
            self.linear_layers = linear_layers

        self.in_channels = [self.n_channels] + [self.n_filters*(2**i) for i in range(self.unet_depth)]
        self.out_channels = self.in_channels[::-1][:-1] + [self.n_classes]

        self.unet_conv = nn.ModuleList([conv_block.ConvBlock(in_channels=self.in_channels[i], out_channels=self.in_channels[i+1], depth=self.conv_depth, **kwargs) for i in range(self.unet_depth)])

        self.bottle_neck = conv_block.ConvBlock(in_channels=self.out_channels[0],out_channels=self.out_channels[0], depth=self.bottleneck_conv_depth, sample_type='bottleneck',**kwargs)

        self.unet_deconv = nn.ModuleList([conv_block.ConvBlock(in_channels=self.out_channels[i], out_channels=self.out_channels[i+1], depth=self.conv_depth, sample_type='up', **kwargs) for i in range(self.unet_depth-1)])
    
        self.attention = nn.ModuleList([attention_block.Attention(x=self.out_channels[i],g=self.out_channels[i]) for i in range(self.unet_depth-1)])

        self.final_conv = nn.Conv2d(self.in_channels[1], self.n_classes, kernel_size=1)
        self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        if self.linear_layers > 0:
            self.unet_linear = nn.Linear(self.n_classes,self.n_classes)

        if self.n_classes == 1:
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = nn.Softmax(dim=1)

    def forward(self, x):
        temp_out = []
        for i in range(self.unet_depth):
            x = self.unet_conv[i](x)
            temp_out.append(x)
            x = temp_out[-1]

        x = self.bottle_neck(x)

        for i in range(self.unet_depth-1):
            x = self.attention[i](x,temp_out[-(i+1)])
            x = self.unet_deconv[i](x + temp_out[-(i+1)])

        x = self.final_conv(x + temp_out[0])
        x = self.up_sample(x)
        final = self.final_activation(x)

        if self.linear_layers > 0:
            final = final.view(final.size(0),-1)
            final = self.unet_linear(final)
            
        return final