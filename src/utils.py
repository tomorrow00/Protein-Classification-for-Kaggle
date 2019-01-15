import os
import numpy as np
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from iterstrat.ml_stratifiers import RepeatedMultilabelStratifiedKFold

import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

from .net import *
from .datasets import ImageDataset

cw = {0: 3.94, 1: 40.5, 2: 14.02, 3: 32.53, 4: 27.33, 5: 20.21, 6: 50.38, 7: 18.0,
      8: 958.15, 9: 1128.49, 10: 1813.64, 11: 46.46, 12: 73.81, 13: 94.57, 14: 47.64,
      15: 2418.19, 16: 95.82, 17: 241.82, 18: 56.3, 19: 34.27, 20: 295.24, 21: 13.45,
      22: 63.32, 23: 17.13, 24: 157.71, 25: 6.17, 26: 154.82, 27: 4616.55}

cw_log = {0: 1.0, 1: 3.01, 2: 1.95, 3: 2.79, 4: 2.61, 5: 2.31, 6: 3.23, 7: 2.2,
          8: 6.17, 9: 6.34, 10: 6.81, 11: 3.15, 12: 3.61, 13: 3.86, 14: 3.17,
          15: 7.1, 16: 3.87, 17: 4.8, 18: 3.34, 19: 2.84, 20: 4.99, 21: 1.91,
          22: 3.46, 23: 2.15, 24: 4.37, 25: 1.13, 26: 4.35, 27: 7.74}

def get_network(architecture, pretrained):
    if architecture == 'custom':
        net = Custom()
    elif architecture == 'resnet':
        net = Resnet(pretrained=pretrained)
    elif architecture == 'bcnn':
        net = BCNN(pretrained=pretrained)
    elif architecture == 'densenet':
        net = Densenet(pretrained=pretrained)
    elif architecture == 'inception':
        net = Inception(pretrained=pretrained)
    elif architecture == 'squeezenet':
        net = Squeezenet(pretrained=pretrained)
    elif architecture == 'bninception':
        net = get_pretrained_network()

    return net

def get_dataset(data_dir, mode, split, subsample, folds, foldnum):
    if mode == 'train':
        data_dir = os.path.join(data_dir, mode)

        internal_image_dir = os.path.join(data_dir, 'internal')
        external_image_dir = os.path.join(data_dir, 'external')

        internal_df = pd.read_csv(os.path.join(data_dir, 'train_internal.csv'))
        external_df = pd.read_csv(os.path.join(data_dir, 'train_external.csv'))
        # subsample_df = pd.read_csv(os.path.join(data_dir, 'train_subsample.csv')).sample(frac=1)
        # internal_df = internal_df.sample(200)
        # external_df = external_df.sample(200)

        # oversample
        internal_df = oversample(internal_df, "in")
        external_df = oversample(external_df, "ex")

        if subsample:
            all_df = all_df[:100]

        if split:
            # KFolds
            print(str(folds) + " folds cross validation with fold " + str(foldnum))
            train_df, val_df = kfolds(external_df, folds, foldnum - 1)

            train_dataset_internal = ImageDataset(image_dir=internal_image_dir, df=internal_df, mode=mode, augument=True)
            train_dataset_external = ImageDataset(image_dir=external_image_dir, df=train_df, mode=mode, augument=True)
            train_dataset = train_dataset_internal + train_dataset_external
            # train_dataset = ImageDataset(image_dir=internal_image_dir, df=subsample_df, mode=mode, augument=True)

            val_dataset = ImageDataset(image_dir=external_image_dir, df=val_df, mode=mode, augument=False)

            dataset = [train_dataset, val_dataset]
        else:
            dataset = ImageDataset(image_dir=image_dir, df=all_df, mode=mode, augument=True)
    else:
        image_dir = os.path.join(data_dir, 'test')
        all_df = pd.read_csv(os.path.join(data_dir, 'sample_submission.csv'))
        dataset = ImageDataset(image_dir=image_dir, df=all_df, mode=mode, augument=False)

    return dataset

def oversample(train_df, cls):
    train_df_orig = train_df.copy()
    # lows = [15, 15, 15, 8, 9, 10, 8, 9, 10, 8, 9, 10, 17, 20, 24, 26, 15, 27, 15, 20, 24, 17, 8, 15, 27, 27, 27]

    if cls == "in":
        lows = [15, 15, 15, 15, 15, 15, 8, 9, 10, 8, 9, 10, 8, 9, 10, 17, 27, 27, 27, 27, 27, 27, 20, 24, 26, 15, 27, 15,
           20, 24, 17, 8, 15, 27, 27, 27, 27, 27, 27]
    else:
        lows = [15, 15, 15, 15, 15, 15, 8, 9, 10, 8, 9, 10, 8, 9, 10, 17, 20, 24, 26, 15, 27, 15, 20, 24, 17, 8, 15, 27,
                27, 27]

    for i in lows:
        target = str(i)
        indicies = train_df_orig.loc[train_df_orig['Target'] == target].index
        train_df = pd.concat([train_df, train_df_orig.loc[indicies]], ignore_index=True)
        indicies = train_df_orig.loc[train_df_orig['Target'].str.startswith(target + " ")].index
        train_df = pd.concat([train_df, train_df_orig.loc[indicies]], ignore_index=True)
        indicies = train_df_orig.loc[train_df_orig['Target'].str.endswith(" " + target)].index
        train_df = pd.concat([train_df, train_df_orig.loc[indicies]], ignore_index=True)
        indicies = train_df_orig.loc[train_df_orig['Target'].str.contains(" " + target + " ")].index
        train_df = pd.concat([train_df, train_df_orig.loc[indicies]], ignore_index=True)

    return train_df

def kfolds(df, folds, foldnum):
    train_df = pd.DataFrame(columns=["Id", "Target"])

    for i in range(folds):
        start = int(len(df) / folds * i)
        end = int(len(df) / folds * (i + 1))
        if i == folds - 1:
            end += 1

        if i == foldnum:
            val_df = df[start:end]
        else:
            train_df = pd.concat([train_df, df[start:end]], ignore_index = True)

    return train_df, val_df

def get_loss_function():
    return sigmoid_cross_entropy_with_logits

def l1_penalty(var):
    return torch.abs(var).sum()

def l2_penalty(var):
    return torch.sqrt(torch.pow(var, 2).sum())
