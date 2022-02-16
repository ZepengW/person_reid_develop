# encoding: utf-8

import math
import random
from cv2 import transform
import torchvision.transforms as T
import torch
from timm.data.random_erasing import RandomErasing
import timm.data.random_erasing as r_e

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

def build_transforms_bgerase(mode = 'train', re_prob = 0.5):
    t_l = []
    if mode == 'train':
        t_l.append(RandomEraseBackground(probability=re_prob, mode='pixel', max_count=1, device='cpu'))
    transform = T.Compose(t_l)
    return transform

def build_transforms_mask():
    transforms = T.Compose([
        T.Resize([256, 128]),
        #T.Pad(10),
        #T.RandomCrop([256, 128]),
        #ChangeZeroTo(0.1)
    ])
    return transforms

class RandomEraseBackground(object):
    '''
    for a tensor, which dim is 4 (RGB + Mask)
    Random erase some area
    However the human body area not erased
    '''
    def __init__(
            self,
            probability=0.5, min_area=0.02, max_area=1/3, min_aspect=0.3, max_aspect=None,
            mode='const', min_count=1, max_count=None, num_splits=0, device='cuda'):
        self.probability = probability
        self.min_area = min_area
        self.max_area = max_area
        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))
        self.min_count = min_count
        self.max_count = max_count or min_count
        self.num_splits = num_splits
        mode = mode.lower()
        self.rand_color = False
        self.per_pixel = False
        if mode == 'rand':
            self.rand_color = True  # per block random normal
        elif mode == 'pixel':
            self.per_pixel = True  # per pixel random normal
        else:
            assert not mode or mode == 'const'
        self.device = device

    def _erase(self, img, chan, img_h, img_w, dtype):
        # todo : 只消除掉 mask 为背景的rgb区域，保留人物rgb区域
        img_rgb = img[0:3]
        img_mask = img[3]
        img_human = torch.einsum('chw, hw->chw',img_rgb, img_mask)
        if random.random() > self.probability:
            return
        area = img_h * img_w
        count = self.min_count if self.min_count == self.max_count else \
            random.randint(self.min_count, self.max_count)
        for _ in range(count):
            for attempt in range(10):
                target_area = random.uniform(self.min_area, self.max_area) * area / count
                aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
                h = int(round(math.sqrt(target_area * aspect_ratio)))
                w = int(round(math.sqrt(target_area / aspect_ratio)))
                if w < img_w and h < img_h:
                    top = random.randint(0, img_h - h)
                    left = random.randint(0, img_w - w)
                    img[:, top:top + h, left:left + w] = r_e._get_pixels(
                        self.per_pixel, self.rand_color, (chan, h, w),
                        dtype=dtype, device=self.device)
                    break
        return torch.where(img_human == 0, img, img_human)

    def __call__(self, input):
        if len(input.size()) == 3:
            output = self._erase(input, *input.size(), input.dtype)
        else:
            batch_size, chan, img_h, img_w = input.size()
            output = []
            # skip first slice of batch if num_splits is set (for clean portion of samples)
            batch_start = batch_size // self.num_splits if self.num_splits > 1 else 0
            for i in range(batch_start, batch_size):
                output.append(self._erase(input[i], chan, img_h, img_w, input.dtype))
            output =  torch.cat(output, dim = 0)
        return output
        

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

