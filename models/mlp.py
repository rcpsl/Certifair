import torch
import torch.nn as nn
class MLP(nn.Module):
    def __init__(self, layers_sizes):
        super().__init__()
        self.layers = []
        for idx in range(len(layers_sizes) - 1):
            self.layers.append(nn.Linear(layers_sizes[idx], layers_sizes[idx+1]))
            if(idx != len(layers_sizes) - 2):
                self.layers.append(nn.ReLU())
        self.net = nn.Sequential(*self.layers)
    
    def forward(self,x):
        return self.net(x)