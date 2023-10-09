""" Import packages """

# Numerical Operations
import random
import numpy as np


# Reading/Writing/Cleaning Data
import pandas as pd
from PIL import Image
import os
import gc

# For Progress Bar
from tqdm.auto import tqdm

# For Drawing
import matplotlib
import matplotlib.pyplot as plt

# Pytorch
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset, TensorDataset, random_split
from torchvision.datasets import DatasetFolder, VisionDataset

""" Set Device """


def register_device(gpu_no=0):
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_no)
        return torch.device("cuda")
    else:
        return torch.device("cpu")
    # TODO: add option to use multiple gpu??


""" Configuration """


def print_config(device, model_load, model_name, translate_dirct,hp):
    print(
        "============CONFIGURATIONS============",
        "Device = "+str(device)+'',
        "Model is loaded "+str(model_load),
        "Model = "+str(model_name),
        "Translation Dirct. = "+str(translate_dirct),
        "--------------------------------------",
        "Transforms Opt. = "+str(hp['trans']),
        "Img. Scale Size = "+str(hp['scale_size']),
        "Img. Crop Size = "+str(hp['crop_size']),
        "Flipped = "+str(hp['flip']),
        "--------------------------------------",
        "G's Arch. = "+str(hp['G_arch']),
        "D's Arch. = "+str(hp['D_arch']),
        "num. of D's layers = "+str(hp['D_layers']),
        "num. of Input's Channels = "+str(hp['in_chan']),
        "num. of Output's Channels = "+str(hp['out_chan']),
        "--------------------------------------",
        "Batch Size = "+str(hp['batch_size']),
        "Normalization = "+str(hp['norm_type']),
        "G's Dropout = "+str(hp['G_dropout']),
        "D's Dropout = " + str(hp['D_dropout']),
        "======================================",
        sep='\n'
    )


""" Preparing Data """


# CSV reader
def read_my_csv(path: str, fname: str):
    raw_data = pd.read_csv(os.path.join(path, fname)).values
    # (raw_train_data isn't split yet)
    return raw_data  # this is a numpy array


# Images reader
def read_my_img(path: str):
    raw_data = sorted([os.path.join(path, x) for x in os.listdir(path) if x.endswith(".jpg")])
    # (fetch a list of image data in the directory of 'path')
    return raw_data  # this is a list of images


# Dataset Spliter
def split_train_valid(data_set, train_ratio, seed):
    train_set_size = int(train_ratio * len(data_set))
    valid_set_size = len(data_set) - train_set_size
    train_set, valid_set = random_split(data_set, [train_set_size, valid_set_size],
                                        generator=torch.Generator().manual_seed(seed))
    return np.array(train_set), np.array(valid_set)


# Sample-Label Divider
def div_sample_label(train_data, valid_data, test_data):
    y_train, y_valid = train_data[:, -1], valid_data[:, -1]
    raw_x_train, raw_x_valid, raw_x_test = train_data[:, :-1], valid_data[:, :-1], test_data
    return raw_x_train, raw_x_valid, raw_x_test, y_train, y_valid


# Feature Selector (Manual Feature Engineering)
def select_features(raw_x_train, raw_x_valid, raw_x_test, idx_list: list, select_all=True):
    if select_all:
        feat_idx = list(range(raw_x_train.shape[1]))
    else:
        feat_idx = idx_list  # TODO: Select suitable feature columns.
    x_train, x_valid, x_test = raw_x_train[:, feat_idx], raw_x_valid[:, feat_idx], raw_x_test[:, feat_idx]
    return x_train, x_valid, x_test


class Imageset(Dataset):
    def __init__(self, transforms, imgs: list):
        # Initialize with path, customized transforms & a maybe-specified images list
        super().__init__()
        self.transforms = transforms
        self.imgs = imgs

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = self.imgs[idx]
        sample = Image.open(img)  # using PIL.Image.open to fetch the actual image
        sample = self.transforms(sample)  # transform when getting item
        try:
            label = int(img.split("/")[-1].split("_")[0])
            # (use the first number in the image name to represent the 'label' value)
        except:
            label = -1
            # (test has no label, no '_' in name so throws exception, set 'label' to '-1')
        return sample, label


class CSVset(Dataset):
    def __init__(self, X, X_type=torch.FloatTensor, y=None, y_type=torch.FloatTensor):
        # X, y are numpy arrays
        self.data = X_type(X)
        if y is not None:
            self.label = y_type(y)
        else:
            self.label = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.label is not None:
            return self.data[idx], self.label[idx]
        else:
            return self.data[idx], None


# Reproducibility Assurer
def assure_reproduce(seed):
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
