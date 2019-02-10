import pdb

import torch
# import torch.utils.model_zoo as model_zoo
from torch.nn.parameter import Parameter
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import numpy as np


class ConvDoubleBlock(nn.Module):
    def __init__(self, in_num_ch, out_num_ch, filter_size=3):
        super(ConvDoubleBlock, self).__init__()
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
            ConvDoubleBlock(in_num_ch, out_num_ch, 3)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class UpBlock(nn.Module):
    def __init__(self, in_num_ch, out_num_ch, upsample=True):
        super(UpBlock, self).__init__()
        if upsample == True:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_num_ch, in_num_ch//2, 2)
                )
        else:
            self.up = nn.ConvTranspose2d(in_num_ch, in_num_ch//2, 3, padding=1, stride=2) # (H-1)*stride-2*padding+kernel_size

        self.conv = ConvDoubleBlock(in_num_ch, out_num_ch, 3)

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
        self.down_1 = ConvDoubleBlock(in_num_ch, first_num_ch, 3)
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

'''
GAN's architecture
'''
class GANStandardGenerator(nn.Module):
    def __init__(self, in_num_ch, out_num_ch, first_num_ch=64, input_size=256, output_activation='softplus'):
        super(GANStandardGenerator, self).__init__()
        self.down_1 = nn.Sequential(
            nn.Conv2d(in_num_ch, first_num_ch, 4, 2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.down_2 = ConvBlock(first_num_ch, 2*first_num_ch)
        self.down_3 = ConvBlock(2*first_num_ch, 4*first_num_ch)
        self.down_4 = ConvBlock(4*first_num_ch, 8*first_num_ch)
        self.down_5 = ConvBlock(8*first_num_ch, 8*first_num_ch)
        self.down_6 = ConvBlock(8*first_num_ch, 8*first_num_ch)
        self.down_7 = ConvBlock(8*first_num_ch, 8*first_num_ch)
        self.down_8 = ConvBlock(8*first_num_ch, 8*first_num_ch, activation='no') # 1 x 1
        self.up_7 = DeconvBlock(8*first_num_ch, 8*first_num_ch)     # 512
        self.up_6 = DeconvBlock(16*first_num_ch, 8*first_num_ch)    #
        self.up_5 = DeconvBlock(16*first_num_ch, 8*first_num_ch)
        self.up_4 = DeconvBlock(16*first_num_ch, 8*first_num_ch)
        self.up_3 = DeconvBlock(16*first_num_ch, 4*first_num_ch)
        self.up_2 = DeconvBlock(8*first_num_ch, 2*first_num_ch)
        self.up_1 = DeconvBlock(4*first_num_ch, first_num_ch)
        self.output = DeconvBlock(2*first_num_ch, out_num_ch, is_last=True)
        # choose different activation layer
        if output_activation == 'sigmoid':
            self.output_act = nn.Sigmoid()
        elif output_activation == 'tanh':
            self.output_act = nn.Tanh()
        elif output_activation == 'no':
            self.output_act = nn.Sequential()
        else:
            self.output_act = nn.Softplus()

    def forward(self, x):
        down_1 = self.down_1(x)
        down_2 = self.down_2(down_1)
        down_3 = self.down_3(down_2)
        down_4 = self.down_4(down_3)
        down_5 = self.down_5(down_4)
        down_6 = self.down_6(down_5)
        down_7 = self.down_7(down_6)
        down_8 = self.down_8(down_7)
        up_7 = self.up_7(down_7, down_8)
        up_6 = self.up_6(down_6, up_7)
        up_5 = self.up_5(down_5, up_6)
        up_4 = self.up_4(down_4, up_5)
        up_3 = self.up_3(down_3, up_4)
        up_2 = self.up_2(down_2, up_3)
        up_1 = self.up_1(down_1, up_2)
        output = self.output(None, up_1)
        output_act = self.output_act(output)
        return output_act

class ConvBlock(nn.Module):
    def __init__(self, in_num_ch, out_num_ch, filter_size=4, stride=2, padding=1, activation='lrelu'):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_num_ch, out_num_ch, filter_size, stride, padding=padding),
            nn.BatchNorm2d(out_num_ch)
            )
        if activation == 'lrelu':
            self.act = nn.LeakyReLU(0.2, inplace=True)
        if activation == 'relu':
            self.act = nn.ReLU(inplace=True)
        if activation == 'elu':
            self.act = nn.ELU(inplace=True)
        else:
            self.act = nn.Sequential()

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        return x

class DeconvBlock(nn.Module):
    def __init__(self, in_num_ch, out_num_ch, filter_size=3, stride=1, padding=1, activation='relu', upsample=True, is_last=False):
        super(DeconvBlock, self).__init__()
        if activation == 'lrelu':
            self.act = nn.LeakyReLU(0.2, inplace=True)
        if activation == 'relu':
            self.act = nn.ReLU(inplace=True)
        if activation == 'elu':
            self.act = nn.ELU(inplace=True)
        else:
            self.act = nn.Sequential()
        self.is_last = is_last

        if upsample == True:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_num_ch, out_num_ch, filter_size, stride, padding=padding)
                )
        else:
            self.up = nn.ConvTranspose2d(in_num_ch, out_num_ch, filter_size, padding=padding, stride=stride) # (H-1)*stride-2*padding+kernel_size
        self.bn = nn.BatchNorm2d(out_num_ch)

    def forward(self, x_down, x_up):
        x_up = self.act(x_up)
        x_up = self.up(x_up)
        if self.is_last == False:
            x_up = self.bn(x_up)
            x = torch.cat([x_down, x_up], 1)
        else:
            x = x_up
        return x

class Discriminator(nn.Module):
    def __init__(self, in_num_ch, first_num_ch=64):
        super(Discriminator, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_num_ch, first_num_ch, 4, 2, padding=2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv_2 = ConvBlock(first_num_ch, 2*first_num_ch)
        self.conv_3 = ConvBlock(2*first_num_ch, 4*first_num_ch)
        self.conv_4 = ConvBlock(4*first_num_ch, 8*first_num_ch, stride=1)
        self.output = nn.Conv2d(8*first_num_ch, 1, 4)
        self.output_act = nn.Sigmoid()

    def forward(self, x):
        conv_1 = self.conv_1(x)
        conv_2 = self.conv_2(conv_1)
        conv_3 = self.conv_3(conv_2)
        conv_4 = self.conv_4(conv_3)
        output = self.output(conv_4)
        output_act = self.output_act(output)
        return output_act
