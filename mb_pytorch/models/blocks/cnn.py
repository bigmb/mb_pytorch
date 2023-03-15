import torch
from torch import nn
from .conv_with_relu import ConvBlock

class CNN(nn.Module):
   """
   Applies several ConvBlocks each doubling the number of channels, and halving the feature map size, before taking a global average and classifying.
   """

   def __init__(self, in_channels, num_blocks, num_classes):
       super().__init__()
       first_channels = 64
       self.blocks = nn.ModuleList(
           [ConvBlock(
               num_layers=3,
               in_channels=(in_channels if i == 0 else first_channels*(2**(i-1))),
               out_channels=first_channels*(2**i))
            for i in range(num_blocks)]
       )
       self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
       self.cls = nn.Linear(first_channels*(2**(num_blocks-1)), num_classes)

   def forward(self, x):
       for block in self.blocks:
           x = block(x)
       x = self.global_pool(x)
       x = x.flatten(1)
       x = self.cls(x)
       return x
