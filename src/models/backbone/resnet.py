import torch
import torch.nn as nn
from typing import List, Union, cast
import torchvision

class ResNet(nn.Module):
    def __init__(
        self, 
        in_channels: int = 1,
        num_classes: int = 2, 
    ) -> None:
        super().__init__()

        self.inplanes = 64
        
        self.resnet = torchvision.models.resnet18(num_classes=num_classes)
        self.resnet.conv1 = nn.Conv2d(in_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.resnet(x)
        out = self.sigmoid(out)
        return out