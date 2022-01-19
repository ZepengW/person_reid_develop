'''
this file contains method of reading dataset
'''
from PIL import Image
import logging
import os
from dataset.transforms import build_transforms, build_transorms_shape, build_transorms_value
import torch
import numpy as np
import torchvision.transforms as T


class GetImg(object):
    """
    read img data from img path, and resize it to uniform size
    :param data:
    :param resize:
    :return:
    """

    def __init__(self, **kwargs):
        self.img_size = kwargs.get('image_size')
        self.tf = build_transforms(self.img_size)

    def __call__(self, data: dict):
        img_path = data.get('img_path')
        pid = data.get('pid')
        cid = data.get('cid')
        if not os.path.isfile(img_path):
            logging.error(f'Can Not Read Image {img_path}')
            img = None
        else:
            img = Image.open(img_path).convert('RGB')
            img = self.tf(img)
        data_dict = {'img': img, 'pid': pid, 'camera_id': cid, 'img_path': img_path}
        return data_dict


class GetImgHeatmap(object):
    """
    read img data from img path, and resize it to uniform size
    read heatmap data from .npy
    """

    def __init__(self, **kwargs):
        self.img_size = kwargs.get('image_size')
        self.tf = build_transforms(self.img_size)

    def __call__(self, data: dict):
        img_path = data.get('img_path')
        pid = data.get('pid')
        cid = data.get('cid')
        if not os.path.isfile(img_path):
            logging.error(f'Can Not Read Image {img_path}')
            img = None
        else:
            img = Image.open(img_path).convert('RGB')
            img = self.tf(img)
        hm_path = data.get('hm_path', '')
        if not os.path.isfile(hm_path):
            hm = -1
        else:
            hm = np.load(hm_path)
            hm = torch.from_numpy(hm)
        data_dict = {'img': img, 'pid': pid, 'camera_id': cid, 'heatmap': hm, 'img_path': img_path}
        return data_dict


class GetImgWithSem(object):
    """
    read img data from img path
    read img sem data from certain path
    resize them to uniform size
    """

    def __init__(self, **kwargs):
        self.img_size = kwargs.get('image_size', [256, 128])
        self.mode = kwargs.get('mode', 'train')
        padding = kwargs.get('padding', 10)
        flip_prob = kwargs.get('flip_prob', 0.5)
        self.tf_shape = build_transorms_shape(self.img_size, self.mode, padding = padding, flip_prob = flip_prob)
        re_prob = kwargs.get('re_prob', 0.5)
        pixel_mean = kwargs.get('pixel_mean', [0.485, 0.456, 0.406])
        pixel_std = kwargs.get('pixel_std', [0.229, 0.224, 0.225])
        self.tf_value = build_transorms_value(mode = self.mode, re_prob = re_prob, pixel_mean = pixel_mean, pixel_std = pixel_std)
        self.to_tensor = T.ToTensor()

    def __call__(self, data: dict):
        img_path = data.get('img_path')
        pid = data.get('pid')
        cid = data.get('cid')
        img_dir = os.path.dirname(img_path)
        ss_dir = img_dir + '_ss'
        img_name = os.path.basename(img_path)
        ss_name = img_name.split('.')[0] + '.pt'
        ss_path = os.path.join(ss_dir, ss_name)
        if not os.path.isfile(img_path):
            logging.error(f'Can Not Read Image {img_path}')
            raise Exception
        else:
            img = Image.open(img_path).convert('RGB')
            img = self.to_tensor(img)
        ss = torch.load(ss_path)
        ss = (ss == 11) | (ss == 12)  # 11 is id of person, 12 is id of rider, which are all if of human
        ss = ss.float()  # 0-1 matrix
        ss = torch.unsqueeze(ss, dim=0)
        reshape_data = self.tf_shape(torch.cat([img, ss], dim=0))
        img = reshape_data[0:3]
        img = self.tf_shape(img)
        ss = reshape_data[3]
        # pre transforms
        data_dict = {'img': img, 'pid': pid, 'camera_id': cid, 'ss': ss, 'img_path': img_path}
        return data_dict
