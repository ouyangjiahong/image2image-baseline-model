#!/usr/bin/env python
import argparse
import os
import time
import pdb

import torch
# import torch.utils.model_zoo as model_zoo
from torch.nn.parameter import Parameter
import torchvision.models as models
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import numpy as np
import pickle
# from skimage.io import imsave
# import matplotlib.pyplot as plt

from loss import *
from util import *
from model import *  # UNet, GAN, VAE
from logger import Logger   # API for tensorboard

localtime = time.localtime(time.time())
time_label = str(localtime.tm_year) + '_' + str(localtime.tm_mon) + '_' + str(localtime.tm_mday) + \
            '_' + str(localtime.tm_hour) + '_' + str(localtime.tm_min)

parser = argparse.ArgumentParser(description='')
parser.add_argument('--gpu', dest='gpu', default='0', help='0,1,2,3')
parser.add_argument('--phase', dest='phase', default='train', help='train, test')
parser.add_argument('--task_label', dest='task_label', default=time_label, help='task specific name')
parser.add_argument('--model_name', dest='model_name', default='UNet', help='UNet or GAN or VAE')
parser.add_argument('--dataset_dir', dest='dataset_dir', default='../data/AD', help='name of the dataset')
parser.add_argument('--data_train_name', dest='data_train_name', default='train.pickle')
parser.add_argument('--data_test_name', dest='data_test_name', default='test.pickle')
parser.add_argument('--data_val_name', dest='data_val_name', default='test.pickle')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='checkpoint', help='models are saved here')
parser.add_argument('--log_dir', dest='log_dir', default='log', help='logs are saved here')
parser.add_argument('--test_dir', dest='test_dir', default='test', help='test results are saved here')
parser.add_argument('--epochs', dest='epochs', type=int, default=50, help='# of epoch')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=8, help='# images in batch')
parser.add_argument('--input_size', dest='input_size', type=int, default=256, help='resize input image size')
parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='initial learning rate for adam')
# parser.add_argument('--beta1', dest='beta1', type=float, default=0.5, help='momentum term of adam')
parser.add_argument('--print_freq', dest='print_freq', type=int, default=20, help='print the debug information every print_freq iterations')
parser.add_argument('--save_freq', dest='save_freq', type=int, default=20, help='save the checkpoint')
parser.add_argument('--continue_train', dest='continue_train', action="store_true", help='if continue training, load the latest model')
parser.set_defaults(continue_train=False)

args = parser.parse_args()

def main():
    # train
    if args.phase == 'train':
        train(args)
    else:
        test(args)

def train(args):
    # define device
    device = torch.device('cuda:' + args.gpu)

    # mkdir for log and checkpoint
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    checkpoint_dir = args.checkpoint_dir + '/' + args.task_label
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # check data exist
    dataset_dir = args.dataset_dir
    if dataset_dir[-1] == '/':
        dataset_dir = dataset_dir[:-1]
    train_data_path = dataset_dir + '/' + args.data_train_name
    val_data_path = dataset_dir + '/' + args.data_val_name
    if not os.path.isfile(train_data_path):
        raise ValueError('no training data file')
    if not os.path.isfile(val_data_path):
        raise ValueError('no validation data file')

    # build Logger
    logger = Logger(args.log_dir, args.task_label)

    # load data
    trainData = MedicalDataset(train_data_path)
    trainDataLoader = DataLoader(trainData, batch_size=args.batch_size, shuffle=True, num_workers=4)
    valData = MedicalDataset(val_data_path)
    valDataLoader = DataLoader(valData, batch_size=args.batch_size, shuffle=False, num_workers=4)
    print('built data loader')

    # define model
    # if args.model_name == 'UNet':
    net_G = UNet(in_num_ch=9, out_num_ch=1, first_num_ch=64, input_size=256,
                output_activation='softplus').to(device)
    # net_G.cuda()      # already have to(device)

    # define loss type
    criterionL1 = nn.L1Loss().to(device)
    criteriontest = DSSIMLoss(value_range=20).to(device)
    criterionMSE = nn.MSELoss().to(device)

    # optimizer
    optimizer_G = optim.Adam(net_G.parameters(), lr=args.lr)
    scheduler_G = optim.lr_scheduler.ReduceLROnPlateau(optimizer_G, 'min')   # dynamic change lr according to val_loss

    # continue training
    start_epoch = 0
    if args.continue_train:
        [optimizer_G, scheduler_G, net_G], start_epoch = load_checkpoint_by_key([optimizer_G, scheduler_G, net_G],\
                                                            checkpoint_dir, ['optimizer_G','scheduler_G','net_G'])

    # start training
    global_iter = 0
    monitor_loss_best = 100
    iter_per_epoch = len(trainDataLoader)
    start_time = time.time()
    for epoch in range(start_epoch, args.epochs):
        net_G.train()
        for iter, sample in enumerate(trainDataLoader):
            global_iter += 1

            # forward G
            # change data from NHWC to NCHW
            real_A = sample['input'].permute(0,3,1,2).to(device)
            real_B = sample['target'].permute(0,3,1,2).to(device)
            fake_B = net_G(real_A)

            # define loss and do backward
            optimizer_G.zero_grad()
            loss_L1 = criterionL1(fake_B, real_B)
            loss_test = criteriontest(fake_B, real_B)
            loss_G = loss_L1 + 10*loss_test
            # loss_G = criterionL1(fake_B, real_B)
            loss_G.backward()
            optimizer_G.step()

            # print msg
            if global_iter % args.print_freq == 0:
                print('Epoch: [%2d] [%4d/%4d] time: %4.4f, L1_loss: %.8f, PSNR_loss: %.8f' % \
                    (epoch, iter, iter_per_epoch, time.time()-start_time, loss_L1.item(), loss_test.item()))
                logger.scalar_summary('train/L1_loss', loss_G.item(), global_iter)
                # print('Epoch: [%2d] [%4d/%4d] time: %4.4f, L1_loss: %.8f' % \
                #     (epoch, iter, iter_per_epoch, time.time()-start_time, loss_G.item()))
                # logger.scalar_summary('train/L1_loss', loss_G.item(), global_iter)

        # validation
        monitor_loss = validate(net_G, valDataLoader, logger, epoch, global_iter, device)
        scheduler_G.step(monitor_loss)

        # save checkpoint
        if monitor_loss_best > monitor_loss or epoch % args.save_freq == 1 or epoch == args.epochs-1:
            is_best = (monitor_loss_best > monitor_loss)
            state = {'epoch': epoch, 'task_label': args.task_label, 'monitor_loss': monitor_loss, \
                    'optimizer_G': optimizer_G.state_dict(), 'scheduler_G': scheduler_G.state_dict(), \
                    'net_G': net_G.state_dict()}
            save_checkpoint(state, is_best, checkpoint_dir)

def validate(net, valDataLoader, logger, epoch, global_iter, device):
    net.eval()
    loss_G_all = 0
    for iter, sample in enumerate(valDataLoader):
        real_A = sample['input'].permute(0,3,1,2).to(device)
        real_B = sample['target'].permute(0,3,1,2).to(device)
        fake_B = net(real_A)
        loss_G_all += F.l1_loss(fake_B, real_B).item()  # need to accumulate using float not tensor, else the memory will be explode
    loss_G_mean = loss_G_all / (iter + 1)
    print('Validation: Epoch: [%2d], L1_loss: %.8f' % (epoch, loss_G_mean))

    logger.scalar_summary('val/L1_loss', loss_G_mean, global_iter)
    logger.image_summary('val/real_A', real_A.cpu().numpy()[:,real_A.shape[1]//2,:,:], global_iter)
    logger.image_summary('val/real_B', real_B.cpu().numpy(), global_iter)
    logger.image_summary('val/fake_B', fake_B.detach().cpu().numpy(), global_iter)  # need detach, can't transform from tensor with grad to numpy

    return loss_G_mean

def test(args):
    # define device
    device = torch.device('cuda:' + args.gpu)

    # mkdir for test and checkpoint
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    checkpoint_dir = args.checkpoint_dir + '/' + args.task_label
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    if not os.path.exists(args.test_dir):
        os.makedirs(args.test_dir)
    test_dir = args.test_dir + '/' + args.task_label
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    # check data exist
    dataset_dir = args.dataset_dir
    if dataset_dir[-1] == '/':
        dataset_dir = dataset_dir[:-1]
    test_data_path = dataset_dir + '/' + args.data_test_name
    if not os.path.isfile(test_data_path):
        raise ValueError('no testing data file')

    # load data
    testData = MedicalDataset(test_data_path)
    testDataLoader = DataLoader(testData, batch_size=args.batch_size, shuffle=True, num_workers=4)
    print('built data loader')

    # define model
    # if args.model_name == 'UNet':
    net_G = UNet(in_num_ch=9, out_num_ch=1, first_num_ch=64, input_size=256,
                output_activation='softplus').to(device)

    # load checkpoint
    [net_G], _ = load_checkpoint_by_key([net_G], checkpoint_dir, ['net_G'])

    # start testing
    net_G.eval()
    for iter, sample in enumerate(testDataLoader):
        real_A = sample['input'].permute(0,3,1,2).to(device)
        real_B = sample['target'].permute(0,3,1,2).to(device)
        fake_B = net_G(real_A)
        save_test_result({real_A, real_B, fake_B}, test_dir)

# run the main function
main()
