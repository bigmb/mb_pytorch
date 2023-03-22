##attention module

import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    
    def __init__(self, x) -> None:
        """
        Attention module
        Input:
            x: (batch_size, channels, height, width)
        Output:
            attention: (batch_size, channels, height, width)
        """

        super(Attention,self).__init__()
        
        self.x = x
        self.weight_keys = nn.Conv2d(x.shape[1], x.shape[1], kernel_size=1, stride=1, padding=0)
        self.weight_queries = nn.Conv2d(x.shape[1], x.shape[1], kernel_size=1, stride=1, padding=0)
        self.weight_values = nn.Conv2d(x.shape[1], x.shape[1], kernel_size=1, stride=1, padding=0)

    def forward(self, x, gatting):
        """
        Input:
            x: (batch_size, channels, height, width)
            gating: (batch_size, channels, height, width)
        Output:
            attention: (batch_size, channels, height, width)
        """
        keys = self.weight_keys(x)
        queries = self.weight_queries(x)
        values = self.weight_values(x)
        
        # (batch_size, channels, height, width)
        attention = torch.matmul(keys, queries)
        attention = torch.div(attention, keys.shape[1])
        attention = F.softmax(attention, dim=1)
        attention = torch.mul(attention, values)
        
        return attention
