import pdb

import torch
# import torch.utils.model_zoo as model_zoo
from torch.nn.parameter import Parameter
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import numpy as np


class ConvBlock(nn.Module):
    def __init__(self, in_num_ch, out_num_ch):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_num_ch, out_num_ch, 3, padding=1),
            nn.BatchNorm2d(out_num_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_num_ch, out_num_ch, 3, padding=1),
            nn.BatchNorm2d(out_num_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x

class DownBlock(nn.Module):
    def __init__(self, in_num_ch, out_num_ch):
        super(DownBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_num_ch, out_num_ch)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class UpBlock(nn.Module):
    def __init__(self, in_num_ch, out_num_ch, upsample=True):
        super(UpBlock, self).__init__()
        if upsampling == True:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_num_ch//2, in_num_ch//2, 2, stride=2)

        self.conv = ConvBlock(in_num_ch, out_num_ch)

    def forward(self, x_down, x_up):
        x_up = self.up(x_up)
        x = torch.cat([x_down, x_up])
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_num_ch, out_num_ch, first_num_ch=64, input_size=256, output_activation='softplus'):
        super(UNet, self).__init__()
        self.down_1 = ConvBlock(in_num_ch, first_num_ch)
        self.down_2 = DownBlock(first_num_ch, 2*first_num_ch)
        self.down_3 = DownBlock(2*first_num_ch, 4*first_num_ch)
        self.down_4 = DownBlock(4*first_num_ch, 8*first_num_ch)
        self.down_5 = DownBlock(8*first_num_ch, 16*first_num_ch)
        self.up_4 = UpBlock(16*first_num_ch, 8*first_num_ch)
        self.up_3 = UpBlock(8*first_num_ch, 4*first_num_ch)
        self.up_2 = UpBlock(4*first_num_ch, 2*first_num_ch)
        self.up_1 = UpBlock(2*first_num_ch, first_num_ch)
        self.output = nn.Conv2d(first_num_ch, out_num_ch, 1)
