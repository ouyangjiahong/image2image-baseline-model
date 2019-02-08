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
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import shutil

class MedicalDataset(Dataset):
    """ define the dataset with it's features """

    def __init__(self, data_path, transform=None):
        with open(data_path, 'rb') as handle:
            data = pickle.load(handle)
        self.samples = data
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        input = sample['input']
        target = sample['target']
        return sample   # {'input':img2, 'target':img2}

def save_checkpoint(state, is_best, checkpoint_dir):
    print("save checkpoint")
    filename = checkpoint_dir+'/epoch'+str(state['epoch'])+'.pth.tar'
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, checkpoint_dir+'/model_best.pth.tar')

def load_checkpoint_by_key(values, checkpoint_dir, keys):
    '''
    the key can be state_dict for both optimizer or model,
    value is the optimizer or model that define outside
    '''
    filename = checkpoint_dir+'/model_best.pth.tar'
    if os.path.isfile(filename):
        checkpoint = torch.load(filename)
        epoch = checkpoint['epoch']
        for i, key in enumerate(keys):
            values[i].load_state_dict(checkpoint[key])
        print("loaded checkpoint from '{}' (epoch: {}, monitor loss: {})".format(filename, \
                epoch, checkpoint['monitor_loss']))
    else:
        raise ValueError('No correct checkpoint')
    return values, epoch


def save_test_result(res, test_dir):
    '''self define function to save results or visualization'''
    print('edit here')
    return
