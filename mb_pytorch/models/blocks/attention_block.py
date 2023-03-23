##attention module

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['Attention']
class Attention(nn.Module):
    
    def __init__(self, x,g) -> None:
        """
        Attention module for U-Net
        Input:
            x_k: (batch_size, channels, height, width)
            x_v: (batch_size, channels, height, width)
        Output:
            attention: (batch_size, channels, height, width)
        """

        super(Attention,self).__init__()
        self.x = nn.Conv2d(x.shape[1], x.shape[1], kernel_size=1, stride=2, padding=0)(x)
        self.g = nn.Conv2d(g.shape[1], g.shape[1], kernel_size=1, stride=1, padding=0)(g)
        self.psi = nn.Conv2d(x.shape[1], 1, kernel_size=1, stride=1, padding=0)

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

