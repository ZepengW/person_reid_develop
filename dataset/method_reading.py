'''
this file contains method of reading dataset
'''
from PIL import Image
import logging
import os
from dataset.transforms import build_transforms
import torch
import numpy as np

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

    def __call__(self, data:dict):
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

    def __call__(self, data:dict):
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
        data_dict = {'img': img, 'pid': pid, 'camera_id': cid, 'heatmap':hm, 'img_path': img_path}
        return data_dict
