import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import models

import numpy as np
# import copy
import pdb

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
        # parameters
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
        SSIM = SSIM / dominator
        return torch.mean((1. - SSIM) / 2.)

class PerceptualLoss(nn.Module):
    '''
    content loss and style loss extracted by selected model
    '''
    def __init__(self, device, model_type='vgg19', content_layers=['conv_4'],
                style_layers=['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5'], channel_idx=0):
        super(PerceptualLoss, self).__init__()
        self.device = device
        self.model_type = model_type
        self.content_layers = content_layers
        self.style_layers = style_layers
        self.channel_idx = channel_idx   # if output and target have multiple channels
        if model_type == 'vgg19':
            self.model = models.vgg19(pretrained=True).features.to(device)
        self.model.eval()

    def normalize_image(self, image):
        # normalize to 0~1
        image = image / image.max()
        # image /= image.max()      # in-place operation, can't compute gradient

        # grayscale to rgb
        image = image[:, self.channel_idx, :, :]
        image = image.unsqueeze(1).expand(-1, 3, -1, -1)

        # normalize by mean/std
        mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1).to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1).to(self.device)
        image = (image - mean) / std

        # crop the middle region
        crop_size = (image.shape[-1] - 224) // 2
        image = image[:, :, crop_size:crop_size+224, crop_size:crop_size+224]

        return image

    def content_loss(self, feature_prediction, feature_target):
        content_loss = F.mse_loss(feature_prediction, feature_target)
        return content_loss

    def style_loss(self, feature_prediction, feature_target):
        gram_prediction = self.gram_matrix(feature_prediction)
        gram_target = self.gram_matrix(feature_target)
        style_loss = F.mse_loss(gram_prediction, gram_target)
        return style_loss

    def gram_matrix(self, feature):
        batch_size, num_ch, height, width = feature.size()  # NCHW
        feature = feature.view(batch_size * num_ch, height * width)
        gram = torch.mm(feature, feature.t())
        return gram.div(batch_size * num_ch * height * width)

    def forward(self, prediction, target):
        # noramlize image
        prediction = self.normalize_image(prediction)
        target = self.normalize_image(target)

        # get features from selected layers
        conv_block_idx = 0
        model_new = nn.Sequential().to(self.device)
        content_losses = []
        style_losses = []
        # feature_prediction = prediction
        # feature_target = target
        for i, layer in enumerate(self.model):
            # TODO: might have error when it's self defined model
            # the official code given by tutorial, a bit faster than the code below
            if isinstance(layer, nn.Conv2d):
                conv_block_idx += 1
                name = 'conv_' + str(conv_block_idx)
            else:
                name = str(i)
            model_new.add_module(name, layer)

            # get loss
            if name in self.content_layers or name in self.style_layers:
                feature_prediction = model_new(prediction)
                feature_target = model_new(target)
                if name in self.content_layers:
                    content_losses.append(self.content_loss(feature_prediction, feature_target))
                if name in self.style_layers:
                    style_losses.append(self.style_loss(feature_prediction, feature_target))

            # another version without building the new model
            # feature_prediction = layer(feature_prediction)
            # feature_target = layer(feature_target)
            # if isinstance(layer, nn.Conv2d):
            #     conv_block_idx += 1
            #     name = 'conv_' + str(conv_block_idx)
            #     if name in self.content_layers:
            #         content_losses.append(self.content_loss(feature_prediction, feature_target))

        return content_losses, style_losses

class GANLoss(nn.Module):
    def __init__(self, device, ls_gan=True):
        super(GANLoss, self).__init__()
        self.device = device
        if ls_gan == True:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def forward(self, prediction, target_label_value=1.):
        target_label = torch.tensor(target_label_value).to(self.device)
        target = target_label.expand_as(prediction)
        loss = self.loss(prediction, target)
        return loss
