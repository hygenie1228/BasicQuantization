import torch 
import numpy as np 
import torch.nn as nn 
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from .func import quantizer

class Tuning():
    def __init__(self):
        self.cor_factor = 0.5
        self.cor_interval = 1
        self.weight_ratio = 1.0

        self.full_fp = True
        self.remove_coord = False
    
    def set_tuning(self, cor_factor, cor_interval, weight_ratio=1.0):
        self.cor_factor = cor_factor
        self.cor_interval = cor_interval
        self.weight_ratio = weight_ratio

    def apply_quantization(self, model, epoch):
        if epoch % self.cor_interval is not 0:
            return

        if self.weight_ratio < 1e-4:
            self.weight_ratio = 0.0
        
        for name, module in model.named_modules():  
            if isinstance(module, (Q_Conv2d, Q_Linear)):
                self.weight_init(module)

        self.weight_ratio = self.weight_ratio * self.cor_factor

    def weight_init(self, m):
        nn.init.normal_(m.coord_weight, mean=0.0, std=0.5)

    def train_mode(self):
        self.full_fp = False
        self.remove_coord = False
    
    def test_mode(self):
        self.remove_coord = True

tuning = Tuning()

class Q_ReLU(nn.Module):
    def __init__(self):
        super(Q_ReLU, self).__init__()
        self.bound = 1.0
        
    def forward(self, x):
        if tuning.full_fp:
            return F.relu(x)
        else: 
            x = F.relu(x)
            return quantizer.activation_quantize_fn(x)

class Q_Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, *args, **kargs):
        super(Q_Conv2d, self).__init__(in_channels, out_channels, kernel_size, *args, **kargs)
        self.scale = 1.0
        self.coord_weight = nn.Parameter(data=torch.rand(self.weight.shape), requires_grad=True)
        nn.init.normal_(self.coord_weight, mean=0.0, std=0.2)

    def forward(self, x):
        if tuning.full_fp:
            weight = self.weight
        elif tuning.remove_coord:
            scale, weight = quantizer.weight_quantize_fn(self.weight)
            weight = scale * weight
        else:
            scale, weight = quantizer.weight_quantize_fn(self.weight)
            weight = tuning.weight_ratio * self.coord_weight + scale * weight
        return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)  

class Q_Linear(nn.Linear):
    def __init__(self, in_channels, out_channels, *args, **kargs):
        super(Q_Linear, self).__init__(in_channels, out_channels, *args, **kargs)
        self.scale = 1.0
        self.coord_weight = nn.Parameter(data=torch.rand(self.weight.shape), requires_grad=True)
        nn.init.normal_(self.coord_weight, mean=0.0, std=0.2)

    def forward(self, x):
        if tuning.full_fp:
            weight = self.weight
        elif tuning.remove_coord:
            scale, weight = quantizer.weight_quantize_fn(self.weight)
            weight = scale * weight
        else:
            scale, weight = quantizer.weight_quantize_fn(self.weight)
            weight = tuning.weight_ratio * self.coord_weight + scale * weight
        return F.linear(x, weight, self.bias)
