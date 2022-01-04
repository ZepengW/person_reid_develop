# encoding: utf-8

import math
import random
import torchvision.transforms as T
import torch
from timm.data.random_erasing import RandomErasing

def build_transforms(size):
    normalize_transform = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transforms = T.Compose([
        T.Resize(size),
        #T.Pad(10),
        #T.RandomCrop([256, 128]),
        T.ToTensor(),
        normalize_transform,
        #RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
    ])
    return transforms

def build_transorms_shape(size, mode = 'train', padding = 10, flip_prob = 0.5):
    if mode == 'train':
        transforms = T.Compose([
            T.Resize(size),
            T.RandomHorizontalFlip(flip_prob),
            T.Pad(padding),
            T.RandomCrop(size)
        ])
    else:
        transforms = T.Compose([
            T.Resize(size)
        ])
    return transforms

def build_transorms_value(mode = 'train', pixel_mean = [0.485, 0.456, 0.406], pixel_std = [0.229, 0.224, 0.225],
                          re_prob = 0.5):
    t_l = [
        T.Normalize(mean=pixel_mean, std=pixel_std)
    ]
    if mode == 'train':
        t_l.append(RandomErasing(probability=re_prob, mode='pixel', max_count=1, device='cpu'))
    transforms = T.Compose(t_l)
    return transforms

def build_transforms_mask():
    transforms = T.Compose([
        T.Resize([256, 128]),
        #T.Pad(10),
        #T.RandomCrop([256, 128]),
        #ChangeZeroTo(0.1)
    ])
    return transforms

class ChangeZeroTo(object):
    '''
    for a tensor, change all zero elements to x
    and others do not change
    '''
    def __init__(self, x):
        self.x = x

    def __call__(self, t_input):
        t_flag = (t_input == 0).int()
        return t_input + t_flag * self.x

