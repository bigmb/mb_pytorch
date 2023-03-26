##attention module

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['Attention']
class Attention(nn.Module):
    
    def __init__(self, x= None,g=None) -> None:
        """
        Attention module for U-Net
        Input:
            x_k: (batch_size, channels, height, width)
            x_v: (batch_size, channels, height, width)
        Output:
            attention: (batch_size, channels, height, width)
        """

        super(Attention,self).__init__()
        self.x = nn.Conv2d(in_channels=x,out_channels=x, kernel_size=1, stride=1, padding=1)(x)
        self.g = nn.Conv2d(in_channels=g, out_channels=g, kernel_size=1, stride=1, padding=1)(g)
        self.psi = nn.Conv2d(in_channels=x, out_channels=1, kernel_size=1, stride=1, padding=1)

    def forward(self,x,g):
        """
        Input:
            x: (batch_size, channels, height, width)
            g:(batch_size, channels, height, width)
        Output:
            attention: (batch_size, channels, height, width)
        """
        x = self.x(x)
        g = self.g(g)
        rel = nn.relu(x + g)
        psi = self.psi(rel)
        psi = nn.BatchNorm2d(psi.shape[1])(psi)
        attention = F.sigmoid(psi)
        return attention


