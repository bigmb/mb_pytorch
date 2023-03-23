##function for convolutional blocks module

import torch
import torch.nn as nn

__all__ = ['ConvBlock']

class ConvBlock(nn.Module):

    def __init__(self,*args,**kwargs):
        """
        Function to create a convolutional block of different forms
        Inputs:
            depth (int): Number of convolutional layers
            in_channels (int or list): Number of input channels
            out_channels (int or list): Number of output channels
            pooling_kernal (int): Pooling kernal size
            pooling_stride (int): Pooling stride size
            stride (int or list): Stride size
            padding (int or list): Padding size
            kernel_size (int or list): Kernel size
            activation (str): Activation function
            kwargs (dict): Keyword arguments
                conv_type (str): Type of convolutional layer
                pooling_type (str): Type of pooling layer
                dropout (float): Dropout rate
        Returns:
            conv_block (torch.nn.Module): Convolutional block
        """
        super(ConvBlock, self).__init__()

        if 'conv_depth' in kwargs:
            depth = kwargs['conv_depth']
        else:
            depth = 2
        if 'in_channels' in kwargs:
            in_channels = kwargs['in_channels']
        else:
            in_channels = 64
        if 'out_channels' in kwargs:
            out_channels = kwargs['out_channels']
        else:
            out_channels = 64
        if 'kernel_size' in kwargs:
            kernel_size = kwargs['kernel_size']
        else:
            kernel_size = 3
        if 'stride' in kwargs:
            stride = kwargs['stride']
        else:
            stride = 1
        if 'pooling_kernal' in kwargs:
            pooling_kernal = kwargs['pooling_kernal']
        else:
            pooling_kernal = 2
        if 'pooling_stride' in kwargs:
            pooling_stride = kwargs['pooling_stride']
        else:
            pooling_stride = 2
        if 'padding' in kwargs:
            padding = kwargs['padding']
        else:
            padding = 0
        if 'pooling_type' in kwargs:
            self.pool = kwargs['pooling_type']
        else:
            self.pool = 'MaxPool2d'
        if 'dropout' in kwargs:
            self.dropout = kwargs['dropout']
        else:
            self.dropout = 0.0
        
        if 'activation' in kwargs:
            if  kwargs['activation'] == 'relu':
                self.activation = 'ReLU'
            elif kwargs['activation'] == 'leaky_relu':
                self.activation = 'LeakyReLU'
            else:
                activation = kwargs['activation']
        else:
            self.activation = 'ReLU'

        if 'conv_type' in kwargs:
            conv_type = kwargs['conv_type']
        else:
            conv_type = 'Conv2d'
            conv = getattr(nn,conv_type)

        self.convs = nn.Sequential()
        for i in range(depth):
            if isinstance(in_channels,list):
                in_channels = in_channels[i]
            else:
                if i==0:
                    in_channels = in_channels
                else:
                    in_channels = out_channels
            if isinstance(out_channels,list):
                out_channels = out_channels[i]
            else:
                out_channels = out_channels
            if isinstance(stride,list):
                stride = stride[i]
            else:
                stride = stride
            if isinstance(padding,list):
                padding = padding[i]
            else:
                padding = padding
            if isinstance(kernel_size,list):
                kernel_size = kernel_size[i]
            else:
                kernel_size = kernel_size
            #temp_str = 'conv'+str(i+1)
            temp_conv = conv(in_channels=in_channels,out_channels=out_channels,stride=stride,padding=padding,kernel_size=kernel_size)
            #temp_2 = '{}={}'.format(temp_str,temp_conv)
            #exec(temp_2)
            self.convs.add_module(f"conv_{i}",temp_conv)
            self.convs.add_module(f"activation_{i}",getattr(nn,self.activation)())
            self.convs.add_module(f"dropout_{i}",getattr(nn,'Dropout')(self.dropout))

        if 'sample_type' in kwargs:
            sample_type = kwargs['sample_type']
        else:
            sample_type = None
        if sample_type =='up':
            self.convs.add_module(f'up_sample_{i}',getattr(nn,'Upsample')(scale_factor=2,mode='bilinear',align_corners=True))
        elif sample_type =='bottleneck':
            pass
        else:
            self.convs.add_module(f"pool_conv_block_{i}",getattr(nn,self.pool)(kernel_size=pooling_kernal,stride=pooling_stride))

    def forward(self,x):
        x = self.convs(x)
        return x