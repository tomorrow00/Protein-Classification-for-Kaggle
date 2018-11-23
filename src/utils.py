import numpy as np

from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from .rather_small_net import Net
from .loss_functions import f1_loss, binary_cross_entropy_with_logits
from .transforms import *
from .datasets import TrainImageDataset, TestImageDataset


def get_dataset(image_dir, train=True):
    transform = transforms.Compose([CombineColors(), ToTensor()])

    if train:
        dataset = TrainImageDataset(label_file='/home/wcc/data/kaggle_protein/train.csv',
                         image_dir=image_dir,
                         transform=transform)
    else:
        dataset = TestImageDataset(image_dir=image_dir,
                         transform=transform)
    return dataset


def get_train_val_split(dataset, trainBS, valBS, val_split, subsample, n_subsample=1000, **kwargs):
    n_images = len(dataset)
    if subsample:
        arr = np.random.choice(n_images, n_subsample, replace=False)
        train_idxs = arr[:int(n_subsample * (1 - val_split))]
        val_idxs = arr[int(n_subsample * (1 - val_split)):]
    else:
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
    trainLoader = DataLoader(dataset=trainset, batch_size=trainBS, shuffle=True, **kwargs)
    valLoader = DataLoader(dataset=valset, batch_size=valBS, shuffle=False, **kwargs)
    return trainLoader, valLoader


def get_network(pretrained=False):
    if pretrained:
        pass # can't pass pretrained net yet
    else:
        net = Net()
        return net


def get_loss_function(lf='bce'):
    if lf == 'bce':
        return binary_cross_entropy_with_logits
    elif lf == 'f1':
        return f1_loss
    else:
        raise ModuleNotFoundError('loss function not found')