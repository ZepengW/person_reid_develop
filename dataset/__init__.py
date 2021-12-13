import random

import torch
import os.path as osp
from PIL import Image
from torch.utils.data import Dataset
from dataset.dataset_duke import Duke
from dataset.dataset_market import Market
from dataset.dataset_msmt import MSMT17
from dataset.dataset_mars import Mars
from dataset.dataset_ltcc import LTCC
import numpy as np

import logging

DATASET_MAP = {
    'market': Market,
    'duke': Duke,
    'msmt17': MSMT17,
    'mars': Mars,
    'ltcc': LTCC
}


class DatasetManager(object):
    def __init__(self, dataset_name, dataset_dir, num_mask = 6):
        self.dataset_name = dataset_name
        self.num_mask = num_mask
        if not dataset_name in DATASET_MAP.keys():
            logging.error('dataset_name no exist. support:' + ','.join(DATASET_MAP.keys()))
            return
        self.dataset = DATASET_MAP.get(dataset_name)(dataset_dir)

    def get_train_pid_num(self):
        return self.dataset.num_train_pids

    def get_dataset_list(self, mode):
        return getattr(self.dataset, mode)

    def get_dataset_image(self, mode, transform=None, transform_mask=None):
        if not hasattr(self.dataset, mode):
            logging.error('can not find mode:' + mode + ' in dataset:' + self.dataset_name)
            return None
        datalist = getattr(self.dataset, mode)
        return DatasetImage(datalist, transform=transform, transform_mask=transform_mask, num_mask=self.num_mask)


class DatasetImage(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, dataset, transform=None, transform_mask = None, num_mask=6):
        self.dataset = dataset
        self.transform = transform
        self.transform_mask =transform_mask
        # for mask
        self.num_mask = num_mask

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        img_path = item[0]
        p_id = item[1]
        cam_id = item[2]
        if len(item) > 3:
            clothes_id = item[3]
        else:
            clothes_id = 0
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        data_dict = {'img':img, 'pid':p_id, 'camera_id':cam_id, 'clothes_id':clothes_id}
        return data_dict

