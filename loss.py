import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pdb

# class testLoss(nn.Module):
#     def __init__(self):
#         super(testLoss, self).__init__()
#
#     def forward(self, prediction, target):
#         return F.l1_loss(prediction, target, reduction='mean')

class PSNRLoss(nn.Module):
    '''
    PSNR is Peek Signal to Noise Ratio, which is similar to mean squared error.

    It can be calculated as
    PSNR = 20 * log10(MAXp) - 10 * log10(MSE)

    '''
    def __init__(self, value_range=1.):
        super(PSNRLoss, self).__init__()
        self.value_range = value_range

    def forward(self, prediction, target):
        PSNR = 20.*torch.log10(self.value_range) - 10.*torch.log10(F.mse_loss(prediction, target))
        return PSNR


class DSSIMLoss(nn.Module):
    '''
    DSSIM is structural dissimilarity metric.
    DSSIM = (1 - SSIM) / 2

    Assume the input shape is NCHW
    '''
    def __init__(self, kernel_size=7, stride=3, value_range=1.):
        super(DSSIMLoss, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.value_range = value_range

    def extract_patches(self, prediction, target):
        # extract patches from NCHW to N * kernel_size * kernel_size * #patches
        batch_size = prediction.shape[0]
        patches_prediction = prediction.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        patches_prediction = patches_prediction.contiguous().view(batch_size, -1, self.kernel_size, self.kernel_size).permute(0,2,3,1)
        patches_target = prediction.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        patches_target = patches_target.contiguous().view(batch_size, -1, self.kernel_size, self.kernel_size).permute(0,2,3,1)
        return patches_prediction, patches_target

    def forward(self, prediction, target):
        k1 = 0.01
        k2 = 0.03

        # get patches
        patches_prediction, patches_target = self.extract_patches(prediction, target)

        # compute SSIM
        c1 = (k1 * self.value_range) ** 2
        c2 = (k2 * self.value_range) ** 2
        ux = patches_prediction.mean(-1)
        uy = patches_target.mean(-1)
        vx = patches_prediction.var(-1)
        vy = patches_target.var(-1)
        cov = torch.mean(patches_prediction * patches_target, dim=-1) - ux * uy

        SSIM = (2 * ux * uy + c1) * (2 * cov + c2)
        dominator = (ux * ux + uy * uy + c1) * (vx * vx + vy * vy + c2)
        SSIM /= dominator
        return torch.mean((1. - SSIM) / 2.)
