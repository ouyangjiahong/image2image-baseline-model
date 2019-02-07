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
        return sample   # {'input':img2, 'target':img2}
