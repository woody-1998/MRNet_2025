import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets.resnet import ResNet, get_inplanes

# --- Channel Attention Block ---
class ChannelAttention3D(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.fc = nn.Sequential(
            nn.Conv3d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return x * self.sigmoid(avg_out + max_out)

# --- Spatial Attention Block ---
class SpatialAttention3D(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv3d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        attn = self.sigmoid(self.conv(x_cat))
        return x * attn, attn  # Return both the result and the attention map

# --- CBAM Block ---
class CBAM3D(nn.Module):
    def __init__(self, in_planes):
        super().__init__()
        self.ca = ChannelAttention3D(in_planes)
        self.sa = SpatialAttention3D()

    def forward(self, x):
        x = self.ca(x)
        x, attn = self.sa(x)
        return x, attn


# --- Final Model ---
class ResNet18WithCBAM(nn.Module):
    def __init__(self, in_channels=3, num_classes=3):
        super().__init__()
        self.resnet = ResNet(
            block="basic",
            layers=(2, 2, 2, 2),
            block_inplanes=get_inplanes(),
            spatial_dims=3,
            n_input_channels=in_channels,
            num_classes=num_classes
        )
        self.resnet.fc = nn.Identity()
        self.cbam = CBAM3D(in_planes=512)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x, return_attn=False):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = F.relu(x, inplace=True)
        x = F.max_pool3d(x, kernel_size=3, stride=2, padding=1)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x, attn = self.cbam(x)
        x = F.adaptive_avg_pool3d(x, 1).flatten(1)
        output = self.classifier(x)

        return (output, attn) if return_attn else output
