##function for convolutional blocks module

import torch
import torch.nn as nn

__all__ = ['ConvBlock']

class ConvBlock(nn.Module):

    def __init__(self,depth=2,in_channels=3,out_channels=3,kernel_size=3,stride=1,pooling_kernal=1,pooling_stride=2,padding=0,activation='relu',**kwargs):
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

        if 'pooling_type' in kwargs:
            self.pool = kwargs['pooling_type']
        else:
            self.pool = 'MaxPool2d'
        if 'dropout' in kwargs:
            self.dropout = kwargs['dropout']
        else:
            self.dropout = 0.0
        
        if activation == 'relu':
            self.activation = 'ReLU'
        elif activation == 'leaky_relu':
            self.activation = 'LeakyReLU'

        if 'conv_type' in kwargs:
            conv_type = kwargs['conv_type']
        else:
            conv_type = 'Conv2d'
            conv = getattr(nn,conv_type)

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
            self.add_module(f"conv_{i}",temp_conv)
            self.add_module(f"activation_{i}",getattr(nn,self.activation)())
            self.add_module(f"dropout_{i}",getattr(nn,'Dropout')(self.dropout))
        self.add_module(f"pool_conv_block",getattr(nn,self.pool)(kernel_size=pooling_kernal,stride=pooling_stride))

    def forward(self,x):
        for i,module in enumerate(self.add_module):
            x = module(x)
        return x