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
from model import *  # UNet, GAN
from logger import Logger   # API for tensorboard

localtime = time.localtime(time.time())
time_label = str(localtime.tm_year) + '_' + str(localtime.tm_mon) + '_' + str(localtime.tm_mday) + \
            '_' + str(localtime.tm_hour) + '_' + str(localtime.tm_min)

parser = argparse.ArgumentParser(description='')
parser.add_argument('--gpu', dest='gpu', default='0', help='0,1,2,3')
parser.add_argument('--phase', dest='phase', default='train', help='train, test')
parser.add_argument('--task_label', dest='task_label', default=time_label, help='task specific name')
parser.add_argument('--model_name', dest='model_name', default='Base', help='Base or GAN')
parser.add_argument('--generator_name', dest='generator_name', default='UNet', help='UNet or ResNet or DenseNet... Standard for GANStandardGenerator')
parser.add_argument('--dataset_dir', dest='dataset_dir', default='../data/AD', help='name of the dataset')
parser.add_argument('--data_train_name', dest='data_train_name', default='train.pickle')
parser.add_argument('--data_test_name', dest='data_test_name', default='test.pickle')
parser.add_argument('--data_val_name', dest='data_val_name', default='test.pickle')
parser.add_argument('--in_num_ch', dest='in_num_ch', type=int, default=9)
parser.add_argument('--out_num_ch', dest='out_num_ch', type=int, default=1)
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='checkpoint', help='models are saved here')
parser.add_argument('--log_dir', dest='log_dir', default='log', help='logs are saved here')
parser.add_argument('--test_dir', dest='test_dir', default='test', help='test results are saved here')
parser.add_argument('--epochs', dest='epochs', type=int, default=50, help='# of epoch')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=8, help='# images in batch')
parser.add_argument('--input_size', dest='input_size', type=int, default=256, help='resize input image size')
parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--print_freq', dest='print_freq', type=int, default=20, help='print the debug information every print_freq iterations')
parser.add_argument('--save_freq', dest='save_freq', type=int, default=20, help='save the checkpoint')
parser.add_argument('--continue_train', dest='continue_train', action="store_true", help='if continue training, load the latest model')
parser.set_defaults(continue_train=False)
parser.add_argument('--g_times', dest='g_times', type=int, default=1, help='train D once, train G several times')
# define lambdas
parser.add_argument('--L1_lambda', dest='L1_lambda', type=float, default=100., help='lambda for L1 loss')
parser.add_argument('--GAN_lambda', dest='GAN_lambda', type=float, default=1., help='lambda for GAN loss')

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

    # define generator model
    if args.generator_name == 'UNet':
        net_G = UNet(in_num_ch=args.in_num_ch, out_num_ch=args.out_num_ch, first_num_ch=64, input_size=256,
                    output_activation='softplus').to(device)
    elif args.generator_name == 'Standard':
        net_G = GANStandardGenerator(in_num_ch=args.in_num_ch, out_num_ch=args.out_num_ch, first_num_ch=64, input_size=256,
                    output_activation='softplus').to(device)
    else:
        raise ValueError('not supporting other models yet!')

    if args.model_name == 'Base':
        train_Base(args, net_G, device, trainDataLoader, valDataLoader, logger, args.in_num_ch, args.out_num_ch, checkpoint_dir)
    else: # GAN
        train_GAN(args, net_G, device, trainDataLoader, valDataLoader, logger, args.in_num_ch, args.out_num_ch, checkpoint_dir)

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

    # define network G
    if args.generator_name == 'UNet':
        net_G = UNet(in_num_ch=9, out_num_ch=1, first_num_ch=64, input_size=256,
                    output_activation='softplus').to(device)
    elif args.generator_name == 'Standard':
        net_G = GANStandardGenerator(in_num_ch=9, out_num_ch=1, first_num_ch=64, input_size=256,
                    output_activation='softplus').to(device)
    else:
        raise ValueError('not supporting other models yet!')

    # load checkpoint
    [net_G], _ = load_checkpoint_by_key([net_G], checkpoint_dir, ['net_G'])

    # start testing
    net_G.eval()
    for iter, sample in enumerate(testDataLoader):
        real_A = sample['input'].permute(0,3,1,2).to(device)
        real_B = sample['target'].permute(0,3,1,2).to(device)
        fake_B = net_G(real_A)
        save_test_result({real_A, real_B, fake_B}, test_dir)


def train_Base(args, net_G, device, trainDataLoader, valDataLoader, logger, in_num_ch, out_num_ch, checkpoint_dir):
    # define loss type
    criterionL1 = nn.L1Loss().to(device)
    # criteriontest = DSSIMLoss(device).to(device)
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
            # loss_test = criteriontest(fake_B, real_B)
            # loss_G = loss_L1 + loss_test
            loss_G = loss_L1
            loss_G.backward()
            optimizer_G.step()

            # print msg
            if global_iter % args.print_freq == 0:
                print('Epoch: [%2d] [%4d/%4d] time: %4.4f, L1_loss: %.8f' % \
                    (epoch, iter, iter_per_epoch, time.time()-start_time, loss_L1.item()))
                logger.scalar_summary('train/L1_loss', loss_G.item(), global_iter)

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


def train_GAN(args, net_G, device, trainDataLoader, valDataLoader, logger, in_num_ch, out_num_ch, checkpoint_dir):
    # define network D
    net_D = Discriminator(in_num_ch=in_num_ch+out_num_ch, first_num_ch=64).to(device)

    # define loss type
    criterionL1 = nn.L1Loss().to(device)
    criterionGAN = GANLoss(device, ls_gan=True).to(device)

    # optimizer
    optimizer_G = optim.Adam(net_G.parameters(), lr=args.lr)
    scheduler_G = optim.lr_scheduler.ReduceLROnPlateau(optimizer_G, 'min')   # dynamic change lr according to val_loss
    optimizer_D = optim.Adam(net_D.parameters(), lr=args.lr)
    scheduler_D = optim.lr_scheduler.ReduceLROnPlateau(optimizer_D, 'min')

    # continue training
    start_epoch = 0
    if args.continue_train:
        params = [optimizer_G, scheduler_G, net_G, optimizer_D, scheduler_D, net_D]
        params, start_epoch = load_checkpoint_by_key(params, checkpoint_dir,
                            ['optimizer_G','scheduler_G','net_G','optimizer_D','scheduler_D','net_D'])

    # start training
    global_iter = 0
    monitor_loss_best = 100
    iter_per_epoch = len(trainDataLoader)
    start_time = time.time()
    for epoch in range(start_epoch, args.epochs):
        net_G.train()
        # pdb.set_trace()
        for iter, sample in enumerate(trainDataLoader):
            global_iter += 1

            # forward G
            # change data from NHWC to NCHW
            real_A = sample['input'].permute(0,3,1,2).to(device)
            real_B = sample['target'].permute(0,3,1,2).to(device)
            fake_B = net_G(real_A)

            # Stage 1: update Discriminator
            optimizer_D.zero_grad()

            # train with real samples
            real_AB = torch.cat((real_A, real_B), 1)
            pred_real = net_D.forward(real_AB)
            loss_D_real = criterionGAN(pred_real, 1.)

            # train with fake samples
            fake_AB = torch.cat((real_A, fake_B), 1)
            pred_fake = net_D.forward(fake_AB)      # need detech?
            loss_D_fake = criterionGAN(pred_fake, 0.)

            # backward D
            loss_D = (loss_D_real + loss_D_fake) / 2.
            loss_D.backward(retain_graph=True)
            # each backward delete intermediate results, thus have error when calling loss_G.backward
            optimizer_D.step()

            # Stage 2: update Generator
            for i in range(args.g_times):
                retain_graph = True
                if i == args.g_times-1:
                    retain_graph = False
                optimizer_G.zero_grad()
                loss_L1 = criterionL1(fake_B, real_B)
                loss_GAN = criterionGAN(pred_fake, 1.)
                loss_G = args.L1_lambda * loss_L1 + args.GAN_lambda * loss_GAN
                loss_G.backward(retain_graph=retain_graph)
                optimizer_G.step()

            # print msg
            if global_iter % args.print_freq == 0:
                print('Epoch: [%2d] [%4d/%4d] time: %4.4f, L1_loss: %.8f, GAN_loss: %.8f, \
                        D_loss: %.8f, D_loss_real: %.8f, D_loss_fake: %.8f' % \
                    (epoch, iter, iter_per_epoch, time.time()-start_time, loss_L1.item(), loss_GAN.item(),
                    loss_D.item(), loss_D_real.item(), loss_D_fake.item()))
                logger.scalar_summary('train/L1_loss', loss_L1.item(), global_iter)
                logger.scalar_summary('train/GAN_loss', loss_GAN.item(), global_iter)
                logger.scalar_summary('train/D_loss', loss_D.item(), global_iter)
                logger.scalar_summary('train/D_loss_real', loss_D_real.item(), global_iter)
                logger.scalar_summary('train/D_loss_fake', loss_D_fake.item(), global_iter)

        # validation
        monitor_loss = validate(net_G, valDataLoader, logger, epoch, global_iter, device)
        scheduler_G.step(monitor_loss)

        # save checkpoint
        if monitor_loss_best > monitor_loss or epoch % args.save_freq == 1 or epoch == args.epochs-1:
            is_best = (monitor_loss_best > monitor_loss)
            state = {'epoch': epoch, 'task_label': args.task_label, 'monitor_loss': monitor_loss, \
                    'optimizer_G': optimizer_G.state_dict(), 'scheduler_G': scheduler_G.state_dict(), \
                    'net_G': net_G.state_dict(), 'optimizer_D': optimizer_D.state_dict(), \
                    'scheduler_D': scheduler_D.state_dict(), 'net_D': net_D.state_dict()}
            save_checkpoint(state, is_best, checkpoint_dir)


# run the main function
main()
