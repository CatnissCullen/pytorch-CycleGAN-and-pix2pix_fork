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
import torchvision.transforms.functional as trans_func
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


""" Reproducibility Assurer """


def assure_reproduce(seed):
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)


""" Configuration """


def print_config(device, model_load, model_save, model_name, translate_dirct, opt):
	print(
		"============CONFIGURATIONS============",
		"Device = " + str(device),
		"Model is loaded " + str(model_load),
		"Model is saved " + str(model_save),
		"Model = " + str(model_name),
		"Translation Dirct. = " + str(translate_dirct),
		"--------------------------------------",
		"Transforms Opt. = " + str(opt['trans']),
		"Img. Scale Size = " + str(opt['scale_size']),
		"Img. Crop Size = " + str(opt['crop_size']),
		"Flipped = " + str(opt['flip']),
		"--------------------------------------",
		"G's Arch. = " + str(opt['G_arch']),
		"D's Arch. = " + str(opt['D_arch']),
		"num. of D's layers = " + str(opt['D_layers']),
		"num. of Input's Channels = " + str(opt['in_chan']),
		"num. of Output's Channels = " + str(opt['out_chan']),
		"--------------------------------------",
		"Batch Size = " + str(opt['batch_size']),
		"Normalization = " + str(opt['norm_type']),
		"G's Dropout = " + str(opt['G_dropout']),
		"D's Dropout = " + str(opt['D_dropout']),
		"--------------------------------------",
		"Loss Mode = " + str(opt['loss_mode']),
		"L1's Lambda = " + str(opt['L1_lambda']),
		"Beta1 = " + str(opt['beta1']),
		"Initial Learning-rate = " + str(opt['lr']),
		"Decay Mode = " + str(opt['lr_dec_mode']),
		"======================================",
		sep='\n'
	)


""" Preparing Data """


# Images reader
def read_img(path: str, mode):
	mode_path = os.path.join(path, mode)
	raw_imageset = sorted([os.path.join(mode_path, x) for x in os.listdir(mode_path) if x.endswith(".jpg")])
	# (fetch a list of image data in the directory of 'path')
	# print("from " + mode_path + " fetched " + str(len(raw_imageset)) + " data")
	return raw_imageset  # this is a list of images


# Image Spliter
def split_photo_facade(img):
	w, h = img.size
	ww = int(w / 2)
	photo = img.crop((0, 0, ww, h))
	facade = img.crop((ww, 0, w, h))
	return photo, facade  # return pillow Images


# Image Transformer
def my_transforms(img, opt):
	"""
	:param img: a pillow Image sample
	:param opt: {
		'trans': 'scale & crop', # | crop | scale width | scale width & crop
		'scale_size': 286, # !!! in TEST TIME set to crop_size !!!
		'crop_size': 256,
		'flip': False, # whether to flip images in augmentation
	}
	:return: a transformed img. ready for input
	"""
	trans, scale_size, crop_size = opt['trans'], opt['scale_size'], opt['crop_size']
	if trans == 'scale & crop':
		img = trans_func.resize(img, [scale_size, scale_size])
	elif 'width' in trans:
		w, h = img.size
		if w == scale_size and h >= crop_size:
			pass
		else:  # w/h = scale_size/h' & remain large enough to crop
			ww, hh = scale_size, int(max(scale_size * h / w, crop_size))
			w, h = ww, hh
			img = trans_func.resize(img, [w, h])

	if 'crop' in trans:
		w, h = img.size
		x = torch.randint(0, w - crop_size + 1, size=(1,)).item()
		y = torch.randint(0, h - crop_size + 1, size=(1,)).item()
		img = trans_func.crop(img, y, x, crop_size, crop_size)

	if opt['flip']:
		img = trans_func.hflip(img)

	return img  # return pillow Image


class Imageset(Dataset):
	def __init__(self, trans_opt, img_files: list):
		# Initialize with customized transforms & an img. files list
		super().__init__()
		self.trans_opt = trans_opt
		self.img_files = img_files  # left: photo; right: facade

	def __len__(self):
		return len(self.img_files)

	def preproc(self, img_file):
		sample = Image.open(img_file).convert('RGB')  # create pillow Image sample from raw image file
		photo, facade = split_photo_facade(sample)  # split sample to photo & facade Images
		photo = my_transforms(photo, self.trans_opt)
		facade = my_transforms(facade, self.trans_opt)
		return photo, facade, sample  # return pillow Images

	def __getitem__(self, idx):
		img_file = self.img_files[idx]
		photo, facade, _ = self.preproc(img_file)
		return trans_func.to_tensor(photo), trans_func.to_tensor(facade)  # return torch tensors of a data point

	def check_preproc(self, idx):
		img_file = self.img_files[idx]
		photo, facades, sample = self.preproc(img_file)
		plt.figure(1)
		plt.subplot(121)
		plt.imshow(photo)
		plt.subplot(122)
		plt.imshow(facades)
		plt.show()
		plt.imshow(sample)
		plt.show()
