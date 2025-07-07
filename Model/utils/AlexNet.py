import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import random
import numpy as np


class AlexNetSlice(nn.Module):
    def __init__(self, num_classes=3, pretrained=True):
        super(AlexNetSlice, self).__init__()
        self.num_classes = num_classes

        # load alexnet from torchvision
        alexnet = torchvision.models.alexnet(pretrained=pretrained) # pretrained by default
        self.features = alexnet.features # only get the conv layers from alexnet
        self.pool = nn.AvgPool2d(kernel_size=2, stride=1, padding=0)
        self.flatten = nn.Flatten()

        # the slice attention block
        self.slice_attention = nn.Sequential(
        nn.Linear(256*6*6, 128),
            nn.Tanh(),
            nn.Linear(128, 1), # get the score
        )

        # classifier block
        self.classifier = nn.Sequential(
            nn.Linear(256*6*6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):

        # shape of x [B,C,D,H,W]
        B, C, D, H, W = x.size()
        x = x.permute(0, 2, 1, 3, 4).reshape(B*D, C, H, W) # [BxD, C, H, W]

        x = self.features(x) # [BxD, 256, 7, 7]
        x = self.pool(x) # [BxD, 256, 6, 6]
        x = self.flatten(x)  # [BxD, 256x6x6]
        x = x.view(B, D, -1)  # [B, D, 256x6x6]

        # score the slice
        score_slice = self.slice_attention(x) # [B, D, 1]
        score_slice = score_slice.squeeze(-1)
        weight_slice = torch.softmax(score_slice, dim=1) # convert socre to 0-1

        # aggregate slices into one
        x = torch.sum(weight_slice.unsqueeze(-1) * x, dim=1) # [B, 256x6x6]
        x = self.classifier(x) # [B, num_classes]

        return x


# model = AlexNetSlice(num_classes=3, pretrained=False)
# print(model)
#
# x = torch.randn(2, 3, 32, 256, 256)
# y = model(x)
# print(y)