import os
import numpy as np
from tqdm import tqdm

import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

from .net import *
from .loss_functions import *
from .transforms import *
from .datasets import ImageDataset

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        # xavier(m.weight.data)
        # xavier(m.bias.data)
        nn.init.xavier_uniform(m.weight)
        nn.init.constant(m.bias, 0.1)

def get_network(pretrained):
    # net = Custom()
    net = Resnet(pretrained=pretrained)

    return net

def get_dataset(data_dir, mode):
    if mode == 'train':
        image_dir = os.path.join(data_dir, 'train')
        label_file = os.path.join(data_dir, 'train.csv')
        dataset = ImageDataset(image_dir=image_dir, label_file=label_file, mode=mode)
    else:
        image_dir = os.path.join(data_dir, 'test')
        label_file = os.path.join(data_dir, 'sample_submission.csv')
        dataset = ImageDataset(image_dir=image_dir, label_file=label_file, mode=mode)

    return dataset

def get_train_val_split(dataset):
    n_images = len(dataset)
    val_split = 0.1

    arr = np.random.choice(n_images, n_images, replace=False)
    train_idxs = arr[:int(n_images * (1 - val_split))]
    val_idxs = arr[int(n_images * (1 - val_split)):]

    trainset = []
    valset = []
    print('getting training set...')
    for i in tqdm(train_idxs):
        sample = dataset[i]
        trainset.append(sample)
    print('getting validating set...')
    for i in tqdm(val_idxs):
        sample = dataset[i]
        valset.append(sample)

    return trainset, valset

def get_loss_function():
    return sigmoid_cross_entropy_with_logits

def l1_penalty(var):
    return torch.abs(var).sum()

def l2_penalty(var):
    return torch.sqrt(torch.pow(var, 2).sum())