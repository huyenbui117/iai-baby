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
        
        self.resnet = torchvision.models.resnet50(weights="IMAGENET1K_V2")
        
        conv1_weight = self.resnet.conv1.weight.clone()
        self.resnet.conv1 = nn.Conv2d(in_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.conv1.weight = torch.nn.Parameter(conv1_weight[:, 0:1])

        self.resnet.fc = nn.Linear(in_features=self.resnet.fc.in_features, out_features=num_classes, bias=True)
        
        self.preset_bn_layers(self.resnet)

        self.sigmoid = nn.Sigmoid()

    def preset_bn_layers(self, module):
        for k in module._modules.keys():
            submodule = module._modules[k]
            if isinstance(submodule, nn.BatchNorm2d):
                submodule.momentum = 1.
            else:
                self.preset_bn_layers(submodule)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.resnet(x)
        out = self.sigmoid(out)
        return out