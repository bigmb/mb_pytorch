import torch
from torch import nn
from .conv_with_relu import ConvBlock


__all__ = ['CNN','CustomCNN']

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

class CustomCNN(nn.Module):
    """
    Custom CNN model.
    Inputs:
        depth (int): Number of convolutional layers
        layer_of_extraction (int): Layer of feature extraction
        skip_connection (bool): Whether to use skip connections
    
    """
    def __init__(self, depth, layer_of_extraction, skip_connection):
        super(CustomCNN, self).__init__()
        
        self.depth = depth
        self.layer_of_extraction = layer_of_extraction
        self.skip_connection = skip_connection
        
        self.conv_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        self.skip_connections = nn.ModuleList()
        
        # Define convolutional layers
        for i in range(depth):
            in_channels = 3 if i == 0 else 64
            out_channels = 64
            kernel_size = 3
            padding = 1
            conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, padding,stride=1)
            self.conv_layers.append(conv_layer)
            
            # Define skip connections
            if skip_connection and i > 0:
                skip_connection_layer = nn.Conv2d(in_channels, out_channels, 1)
                identity = nn.Identity()
                self.skip_connections_layer.append([skip_connection_layer, identity])
                
            # Define pooling layers
            if i < depth - 1:
                pool_layer = nn.MaxPool2d(2, 2)
                self.pool_layers.append(pool_layer)
        
        # Define fully connected layer
        self.fc_layer = nn.Linear(64 * 8 * 8, 10)
        
    def forward(self, x):
        layer_outputs = []
        
        for i in range(self.depth):
            x = self.conv_layers[i](x)
            
            if self.skip_connection and i > 0:
                skip_connection_layer, skip_connection_identity = self.skip_connections[i - 1]
                if skip_connection_layer is not None:
                    skip_connection_out = skip_connection_layer(x)
                x = x + skip_connection_out
            
            x = nn.functional.relu(x)
            
            if i < self.depth - 1:
                x = self.pool_layers[i](x)
            
            if i == self.layer_of_extraction:
                layer_outputs.append(x)
        
        x = torch.flatten(x, start_dim=1)
        x = self.fc_layer(x)
        
        if self.layer_of_extraction == self.depth-1 and len(layer_outputs) == 0:            
            layer_outputs.append(x)
        elif self.layer_of_extraction < self.depth-1:
            layer_outputs.append(x)
        
        return x, layer_outputs[-1]
