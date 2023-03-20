##attention module

import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    
    def __init__(self, x, gatting) -> None:
        """
        Attention module
        Input:
            x: (batch_size, channels, height, width)
            gating: (batch_size, channels, height, width)
        Output:
            attention: (batch_size, channels, height, width)
        """

        super(Attention,self).__init__()
        
        self.x = x
        self.gating = gatting

