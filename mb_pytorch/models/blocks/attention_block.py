##attention module

import torch
import torch.nn as nn

__all__ = ['Attention','Attention_transformers']
class Attention(nn.Module):
    
    def __init__(self, x= None,g=None) -> None:
        """
        Attention module for U-Net
        Input:
            x: (batch_size, channels, height, width)
            x: (batch_size, channels, height, width)
        Output:
            attention: (batch_size, channels, height, width)
        """

        super(Attention,self).__init__()
        self.x_conv = nn.Conv2d(in_channels=x,out_channels=x, kernel_size=1, stride=1, padding=0)
        self.g_conv = nn.Conv2d(in_channels=g, out_channels=g, kernel_size=1, stride=1, padding=0)
        self.psi_conv = nn.Conv2d(in_channels=x, out_channels=1, kernel_size=1, stride=1, padding=0)

    def forward(self,x,g):
        """
        Input:
            x: (batch_size, channels, height, width)
            g:(batch_size, channels, height, width)
        Output:
            attention: (batch_size, channels, height, width)
        """
        x_conv = self.x_conv(x)
        g_conv = self.g_conv(g)
        x_plus_g = (x_conv + g_conv)
        relu = nn.ReLU()(x_plus_g)
        psi = self.psi_conv(relu)
        bnorm = nn.BatchNorm2d(psi.shape[1])(psi)
        attention = nn.Sigmoid()(bnorm)
        return attention


class Attention_transformers(nn.Module):
    """
    Attention module used for transformers
    Input:
        x_query: (batch_size, channels, height, width)
        x_key: (batch_size, channels, height, width)
        x_value: (batch_size, channels, height, width)
    Output:
        attention: (batch_size, channels, height, width)
    """
    
    def __init__(self, x_query=None, x_key=None, x_value=None) -> None:
        super(Attention_transformers,self).__init__()
        self.x_query_conv = nn.Conv2d(in_channels=x_query, out_channels=x_query, kernel_size=1, stride=1, padding=0)
        self.x_key_conv = nn.Conv2d(in_channels=x_key, out_channels=x_key, kernel_size=1, stride=1, padding=0)
        self.x_value_conv = nn.Conv2d(in_channels=x_value, out_channels=x_value, kernel_size=1, stride=1, padding=0)
        
    def forward(self,x_query, x_key, x_value):
        """
        Input:
            x_query: (batch_size, channels, height, width)
            x_key: (batch_size, channels, height, width)
            x_value: (batch_size, channels, height, width)
        Output:
            attention: (batch_size, channels, height, width)
        """
        x_query_conv = self.x_query_conv(x_query)
        x_key_conv = self.x_key_conv(x_key)
        x_value_conv = self.x_value_conv(x_value)
        x_query_plus_key = (x_query_conv + x_key_conv)
        relu = nn.ReLU()(x_query_plus_key)
        attention = nn.Softmax(dim=-1)(relu)
        return attention
    