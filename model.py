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
    def __init__(self, in_num_ch, out_num_ch, filter_size=3):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_num_ch, out_num_ch, filter_size, padding=1),
            nn.BatchNorm2d(out_num_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_num_ch, out_num_ch, filter_size, padding=1),
            nn.BatchNorm2d(out_num_ch),
            nn.ReLU(inplace=True)
            )

    def forward(self, x):
        x = self.conv(x)
        return x

class DownBlock(nn.Module):
    def __init__(self, in_num_ch, out_num_ch):
        super(DownBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_num_ch, out_num_ch, 3)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class UpBlock(nn.Module):
    def __init__(self, in_num_ch, out_num_ch, upsample=False):
        super(UpBlock, self).__init__()
        if upsample == True:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_num_ch, in_num_ch//2, 2)
                )
        else:
            self.up = nn.ConvTranspose2d(in_num_ch, in_num_ch//2, 3, padding=1, stride=2) # (H-1)*stride-2*padding+kernel_size

        self.conv = ConvBlock(in_num_ch, out_num_ch, 3)

    def forward(self, x_down, x_up):
        x_up = self.up(x_up)
        # after the upsample/convtrans, the HW is smaller than the down-sampling map
        x_up = F.pad(x_up, (0,1,0,1), mode='replicate')
        x = torch.cat([x_down, x_up], 1)
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_num_ch, out_num_ch, first_num_ch=64, input_size=256, output_activation='softplus'):
        super(UNet, self).__init__()
        self.down_1 = ConvBlock(in_num_ch, first_num_ch, 3)
        self.down_2 = DownBlock(first_num_ch, 2*first_num_ch)
        self.down_3 = DownBlock(2*first_num_ch, 4*first_num_ch)
        self.down_4 = DownBlock(4*first_num_ch, 8*first_num_ch)
        self.down_5 = DownBlock(8*first_num_ch, 16*first_num_ch)
        self.up_4 = UpBlock(16*first_num_ch, 8*first_num_ch)
        self.up_3 = UpBlock(8*first_num_ch, 4*first_num_ch)
        self.up_2 = UpBlock(4*first_num_ch, 2*first_num_ch)
        self.up_1 = UpBlock(2*first_num_ch, first_num_ch)
        self.output = nn.Conv2d(first_num_ch, out_num_ch, 1)
        # choose different activation layer
        if output_activation == 'sigmoid':
            self.output_act = nn.Sigmoid()
        elif output_activation == 'tanh':
            self.output_act = nn.Tanh()
        elif output_activation == 'linear':
            self.output_act = nn.Linear()
        else:
            self.output_act = nn.Softplus()

    def forward(self, x):
        down_1 = self.down_1(x)
        down_2 = self.down_2(down_1)
        down_3 = self.down_3(down_2)
        down_4 = self.down_4(down_3)
        down_5 = self.down_5(down_4)
        up_4 = self.up_4(down_4, down_5)
        up_3 = self.up_3(down_3, up_4)
        up_2 = self.up_2(down_2, up_3)
        up_1 = self.up_1(down_1, up_2)
        output = self.output(up_1)
        output_act = self.output_act(output)
        return output_act
