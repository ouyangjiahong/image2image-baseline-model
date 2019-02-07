#!/usr/bin/env python
import argparse
import os
import time
import pdb

import torch
# import torch.utils.model_zoo as model_zoo
from torch.nn.parameter import Parameter
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import numpy as np
import pickle
# from skimage.io import imsave
# import matplotlib.pyplot as plt

from util import *
# from model import UNet  # GAN, VAE
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
parser.add_argument('--epochs', dest='epochs', type=int, default=50, help='# of epoch')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=8, help='# images in batch')
parser.add_argument('--input_size', dest='input_size', type=int, default=256, help='resize input image size')
parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='initial learning rate for adam')
# parser.add_argument('--beta1', dest='beta1', type=float, default=0.5, help='momentum term of adam')
parser.add_argument('--print_freq', dest='print_freq', type=int, default=20, help='print the debug information every print_freq iterations')
parser.add_argument('--continue_train', dest='continue_train', action="store_true", help='if continue training, load the latest model')
parser.set_defaults(continue_train=False)

args = parser.parse_args()

def main():
    # train
    if args.phase == 'train':
        train(args)
    else:
        test(args)

    # check data exist
    # if not os.path.isfile(dataset_dir + '/' + data_train_name):
    #     raise ValueError('no testing data file')


def train(args):
    # define device
    device = torch.device('cuda:' + args.gpu)

    # mkdir for log and checkpoint
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    checkpoint_dir = args.checkpoint_dir + '/' + args.task_label
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    log_dir = args.log_dir + '/' + args.task_label
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

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
    logger = Logger(log_dir, args.task_label)

    # load data
    trainData = MedicalDataset(train_data_path)
    trainDataLoader = DataLoader(trainData, batch_size=args.batch_size, shuffle=True, num_workers=4)
    valData = MedicalDataset(val_data_path)
    valDataLoader = DataLoader(valData, batch_size=args.batch_size, shuffle=False, num_workers=4)
    print('built data loader')

    # define model
    # if args.model_name == 'UNet':
    net_G = UNet(input_size).to(device)
    net_G.cuda()

    # define loss type
    criterionL1 = nn.L1Loss().to(device)
    criterionMSE = nn.MSELoss().to(device)

    # optimizer
    optimizer_G = optim.Adam(net_g.parameters(), lr=args.lr)
    scheduler_G = ReduceLROnPlateau(optimizer_G, 'min')   # dynamic change lr according to val_loss

    # start training
    global_iter = 0
    iter_per_epoch = len(trainDataLoader)
    start_time = time.time()
    for epoch in range(args.epochs):
        for iter, sample in enumerate(trainDataLoader):
            global_iter += 1
            pdb.set_trace()
            # forward G
            real_A = sample['input']
            real_B = sample['target']
            fake_B = net_G(real_A)

            # define loss and do backward
            optimizer_G.zero_grad()
            loss_G = criterionL1(fake_B, real_B)
            loss_G.backward()
            optimizer.step()

            # print msg
            if global_iter % args.print_freq == 0:
                print('Epoch: [%2d] [%4d/%4d] time: %4.4f, L1_loss: %.8f' % \
                    (epoch, iter, iter_per_epoch, time.time()-start_time, loss_G.item()))

        # validation
        # monitor_loss = validate()
        # scheduler_G.step(monitor_loss)

# def validate(args, valDataLoader):
#     for iter, sample in enumerate(valDataLoader):




# run the main function
main()
