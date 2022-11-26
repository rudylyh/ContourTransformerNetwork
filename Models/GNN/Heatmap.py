import torch
import torch.nn as nn
from torch.nn import functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Heatmap(nn.Module):
    def __init__(self, in_dim=256, out_dim=100, img_size=None):

        super(Heatmap, self).__init__()
        self.heatmap = nn.Sequential(
            nn.Conv2d(
                in_channels=in_dim,
                out_channels=in_dim,
                kernel_size=1,
                stride=1,
                padding=0),
            nn.BatchNorm2d(in_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=in_dim,
                out_channels=out_dim,
                kernel_size=1,
                stride=1,
                padding=0),
            # nn.Sigmoid()
        )
        self.img_size = img_size

    def forward(self, x):
        # x = F.upsample(input=x, size=(self.img_size, self.img_size), mode='bilinear', align_corners=True)
        x = self.heatmap(x)
        return x
