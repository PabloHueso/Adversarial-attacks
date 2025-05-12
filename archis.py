# Small architechture for MNIST 
import torch
import torch.nn as nn

class SimpleCNN(nn.Module): 
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, 10, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(1690, output_channels)

    def forward(self, x):
        # Initial shape is [B, 1, 28, 28]
        x = self.conv(x)
        # After convoluting, shape is [B, 10, 26, 26]
        x = nn.ReLU()(x)
        # After ReLU, shape is [B, 10, 26, 26] (it's an element-wise operation)
        x = self.pool(x)
        # After pooling, shape is [B, 10, 13, 13]
        x = nn.Flatten()(x)
        # After flattening shape is [B, 1690] (naturally, 1690 = 10x13x13)
        x = self.fc(x)
        # Final shape is [B, 10] (we have logits for 10 possible classes)
        return x
    
# Small architechture with proba output for MNIST 

class SimpleCNNprobas(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, 10, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(1690, output_channels)

    def forward(self, x):
        # Initial shape is [B, 1, 28, 28]
        x = self.conv(x)
        # After convoluting, shape is [B, 10, 26, 26]
        x = nn.ReLU()(x)
        # After ReLU, shape is [B, 10, 26, 26] (it's an element-wise operation)
        x = self.pool(x)
        # After pooling, shape is [B, 10, 13, 13]
        x = nn.Flatten()(x)
        # After flattening shape is [B, 1690] (naturally, 1690 = 10x13x13)
        x = self.fc(x)
        # Final shape is [B, 10] (we have logits for 10 possible classes)
        x = nn.functional.softmax(x, dim=1)   
        # Does not change shape either
        return x