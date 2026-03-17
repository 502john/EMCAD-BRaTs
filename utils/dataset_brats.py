import os
import random
import h5py
import numpy as np
import torch
import cv2
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    # rotate in H,W plane not C,H plane
    image = np.rot90(image, k, axes=(1, 2))
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    # flip along spatial dims not channel dim
    image = np.flip(image, axis=axis + 1).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    # rotate each channel independently
    image = np.stack([
        ndimage.rotate(image[c], angle, order=3, reshape=False) 
        for c in range(image.shape[0])
    ], axis=0)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        _, x, y = image.shape



        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (1, self.output_size[0] / x, self.output_size[1] / y), order=3)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32))
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample


class BraTS_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, nclass=4, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir
        self.nclass = nclass

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name + '.npz')            
            data = np.load(data_path)
            image, label = data['image'], data['label']

            
            
        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = os.path.join(self.data_dir, vol_name + '.npy.h5')
            data = h5py.File(filepath)
            image, label = data['image'][:], data['label'][:]

            
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample
