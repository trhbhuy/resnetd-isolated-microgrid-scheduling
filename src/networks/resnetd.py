import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, shape):
        super(ResidualBlock, self).__init__()
        self.layer = nn.Linear(shape, shape)

    def forward(self, x):
        return self.layer(x)
    
class ConnectionBlock(nn.Module):
    def __init__(self, shape):
        super(ConnectionBlock, self).__init__()
        self.layer_1 = nn.Linear(shape, shape)
        self.layer_2 = nn.Linear(shape, shape)

    def _activation(self, x):
        return F.relu(x)

    def forward(self, x):
        x = self._activation(self.layer_1(x))
        x = self.layer_2(x)
        return x

class ResNetD(nn.Module):
    def __init__(self, input_shape, num_classes, num_blocks = 4):
        super(ResNetD, self).__init__()
        layer_shape = input_shape
        
        # Init Residual Blocks
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            block = nn.ModuleDict({
                'residual': self._ResidualLayer(layer_shape),
                'connection_layer': self._ConnectionLayer(layer_shape)
            })
            self.blocks.append(block)
        
        # Output dense layer
        self.predictions = nn.Linear(layer_shape, num_classes)

    def _ResidualLayer(self, shape):
        return ResidualBlock(shape)
    
    def _ConnectionLayer(self, shape):
        return ConnectionBlock(shape)

    def _activation(self, data):
        return F.relu(data)
    
    def forward(self, x):
        previous_connection = None

        out = x
        for i, block in enumerate(self.blocks):
            #  Residual connection
            conn = block['connection_layer'](out)

            res = block['residual'](x)

            out = res + conn
            if (previous_connection is not None):
                out += previous_connection
            previous_connection = conn
            
            # last layer doesn't require activation
            out = out if (i == len(self.blocks) - 1) else self._activation(out) 

        predictions = self.predictions(out)

        return predictions