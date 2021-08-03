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
from dataset.mask import read_person_mask
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


    def get_dataset_image(self, mode, transform=None, transform_mask=None):
        if not hasattr(self.dataset, mode):
            logging.error('can not find mode:' + mode + ' in dataset:' + self.dataset_name)
            return None
        datalist = getattr(self.dataset, mode)
        return DatasetImage(datalist, transform=transform, transform_mask=transform_mask, num_mask=self.num_mask)

    def get_dataset_video(self, mode, seq_len=15, transform=None, sample='constrain_random',transform_mask=None):
        if not hasattr(self.dataset, mode):
            logging.error('can not find mode:' + mode + ' in dataset:' + self.dataset_name)
            return None
        datalist = getattr(self.dataset, mode)
        return DatasetVideo(datalist, seq_len=seq_len, transform=transform, sample=sample,transform_mask=transform_mask,
                            num_mask=self.num_mask)


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
        clothes_id = item[3]
        mask_path = item[4]
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        mask = read_person_mask(mask_path)
        mask = torch.from_numpy(mask).float()
        if self.transform_mask is not None:
            mask = self.transform_mask(mask)
        return img, p_id, cam_id, clothes_id, mask


class DatasetVideo(Dataset):
    '''
    seq_num: sequence number
    datalist: [(img_paths:list,pid:int,cid:int,mask_paths:list)]
    '''

    def __init__(self, datalist, seq_len=15, sample='constrain_random', transform=None,transform_mask=None,
                 num_mask = 6):
        self.data_list = datalist
        self.transform = transform
        self.seq_len = seq_len
        self.sample = sample
        self.transform_mask = transform_mask
        # for body mask
        self.num_mask = num_mask

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        img_paths, pid, cid = self.data_list[index]
        # 适配image类型数据集，令其作为seq_len为1的video类型数据
        if type(img_paths) is str:
            img_paths = [img_paths]
        num = len(img_paths)

        if self.sample == 'constrain_random':
            """
            Evenly and randomly sample seq_len items from num items .
            """

            def random_int_list(start, stop, length):
                start, stop = (int(start), int(stop)) if start <= stop else (int(stop), int(start))
                length = int(abs(length)) if length else 0
                random_list = []
                for i in range(length):
                    random_list.append(random.randint(start, stop))
                return random_list

            if num >= self.seq_len:
                r = num % self.seq_len
                stride = num // self.seq_len
                if r != 0:
                    stride += 1
                bias_indices = random_int_list(0, stride - 1, self.seq_len)
                ach_indices = np.arange(0, self.seq_len) * stride
                indices = bias_indices + ach_indices
                indices = indices.clip(max=num - 1)
            else:
                # if num is smaller than seq_len, simply replicate the last image
                # until the seq_len requirement is satisfied
                indices = np.arange(0, num)
                num_pads = self.seq_len - num
                indices = np.concatenate([indices, np.ones(num_pads).astype(np.int32) * (num - 1)])

            imgs = []
            for index in indices:
                img_path = img_paths[int(index)]
                img = Image.open(img_path).convert('RGB')
                if self.transform is not None:
                    img = self.transform(img)
                h = img.shape[1]
                w = img.shape[2]
                img = img.unsqueeze(0)
                imgs.append(img)

            imgs = torch.cat(imgs, dim=0)
            return imgs, pid, cid, 0
        elif self.sample == 'evenly':
            """
            Evenly sample seq_len items from num items.
            """
            indices_list = []
            if num >= self.seq_len:
                r = num % self.seq_len
                stride = num // self.seq_len
                if r != 0:
                    stride += 1
                for i in range(stride):
                    indices = np.arange(i, stride * self.seq_len, stride)
                    indices = indices.clip(max=num - 1)
                    indices_list.append(indices)
            else:
                # if num is smaller than seq_len, simply replicate the last image
                # until the seq_len requirement is satisfied
                indices = np.arange(0, num)
                num_pads = self.seq_len - num
                indices = np.concatenate([indices, np.ones(num_pads).astype(np.int32) * (num - 1)])
                indices_list.append(indices)

            if len(indices_list) > 50:
                indices_list = indices_list[:50]

            imgs_list = []
            for indices in indices_list:
                imgs = []
                for index in indices:
                    img_path = img_paths[int(index)]
                    img = Image.open(img_path).convert('RGB')
                    if self.transform is not None:
                        img = self.transform(img)
                    img = img.unsqueeze(0)
                    imgs.append(img)

                imgs = torch.cat(imgs, dim=0)
                imgs_list.append(imgs)

            imgs_array = torch.stack(imgs_list)
            return imgs_array, pid, cid, 0
