import torch
from torch import nn
from torch.nn import functional as F

class VehicleExtension(nn.Module):

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(7, 512, 1, 1))
        self.bias = nn.Parameter(torch.empty(7))
        self.vehicle_weight = nn.Parameter(torch.empty(1, 512, 1, 1))
        self.vehicle_bias = nn.Parameter(torch.empty(1))

    def forward(self, features):
        weight = torch.cat((self.weight, self.vehicle_weight))
        bias = torch.cat((self.bias, self.vehicle_bias))
        return F.conv2d(features, weight, bias)
