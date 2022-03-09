'''
this file contains method of reading dataset
'''
from PIL import Image
import logging
import os
from dataset.transforms import build_transforms, build_transorms_shape, build_transorms_value, build_transforms_bgerase
import torch
import numpy as np
import torchvision.transforms as T
import json
from scipy.ndimage.filters import gaussian_filter


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

    def __init__(self, using_bg_erase = False, **kwargs):
        self.img_size = kwargs.get('image_size', [256, 128])
        self.mode = kwargs.get('mode', 'train')
        padding = kwargs.get('padding', 10)
        flip_prob = kwargs.get('flip_prob', 0.5)
        self.tf_shape = build_transorms_shape(self.img_size, self.mode, padding = padding, flip_prob = flip_prob)
        re_prob = kwargs.get('re_prob', 0.5)
        pixel_mean = kwargs.get('pixel_mean', [0.485, 0.456, 0.406])
        pixel_std = kwargs.get('pixel_std', [0.229, 0.224, 0.225])
        self.using_bg_erase = using_bg_erase
        if using_bg_erase:
            self.tf_value = build_transorms_value(mode = '', re_prob = re_prob, pixel_mean = pixel_mean, pixel_std = pixel_std)
            self.tf_erase = build_transforms_bgerase(mode = self.mode, re_prob = re_prob)
        else:
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
        ss = reshape_data[3]
        img = self.tf_value(img)
        if self.using_bg_erase and self.mode == 'train':
            img = self.tf_erase(torch.cat([img, ss.unsqueeze(0)], dim = 0))
        # pre transforms
        data_dict = {'img': img, 'pid': pid, 'camera_id': cid, 'ss': ss, 'img_path': img_path}
        return data_dict

class GetImgWithHm(object):
    """
    read img data from img path
    read alphapose-result from .json
    output img, heatmap, 
    """

    def __init__(self, using_bg_erase = False, **kwargs):
        self.img_size = kwargs.get('image_size', [256, 128])
        self.mode = kwargs.get('mode', 'train')
        padding = kwargs.get('padding', 10)
        flip_prob = kwargs.get('flip_prob', 0.5)
        # 尺度变换预处理
        self.tf_shape = build_transorms_shape(self.img_size, self.mode, padding = padding, flip_prob = flip_prob)
        re_prob = kwargs.get('re_prob', 0.5)
        pixel_mean = kwargs.get('pixel_mean', [0.485, 0.456, 0.406])
        pixel_std = kwargs.get('pixel_std', [0.229, 0.224, 0.225])
        self.vis_thre = kwargs.get('vis_thre', 0.2)
        # 色彩变换预处理
        self.tf_value = build_transorms_value(mode = "", re_prob = re_prob, pixel_mean = pixel_mean, pixel_std = pixel_std)

        self.to_tensor = T.ToTensor()

    def __call__(self, data: dict):
        img_path = data.get('img_path')
        pid = data.get('pid')
        cid = data.get('cid')
        json_name = os.path.basename(img_path).split('.jpg')[0]+'.json'
        json_dir = os.path.dirname(img_path)+'-alphapose'
        json_path = os.path.join(json_dir, json_name)
        
        if not os.path.isfile(img_path):
            logging.error(f'Can Not Read Image {img_path}')
            raise Exception
        else:
            img = Image.open(img_path).convert('RGB')
            img = self.to_tensor(img)
            hm, vis_score = genHeatMap(json_path, img.shape[1], img.shape[2])
            hm = torch.tensor(hm)
            vis_score = torch.tensor(vis_score)

        reshape_data = self.tf_shape(torch.cat([img, hm], dim=0))
        img = reshape_data[0:3]
        hm = reshape_data[3:]
        img = self.tf_value(img)
        # pre transforms
        img = img.float()
        hm = hm.float()
        vis_score = vis_score.float()
        data_dict = {'img': img, 'pid': pid, 'camera_id': cid, 'hm': hm, 'img_path': img_path, 'vis_score': vis_score}
        return data_dict


class GetImgWithRandomMask(object):
    """
    read img data from img path
    generate mask randomly
    """

    def __init__(self, using_bg_erase = False, **kwargs):
        self.img_size = kwargs.get('image_size', [256, 128])
        self.mode = kwargs.get('mode', 'train')
        padding = kwargs.get('padding', 10)
        flip_prob = kwargs.get('flip_prob', 0.5)
        self.tf_shape = build_transorms_shape(self.img_size, self.mode, padding = padding, flip_prob = flip_prob)
        re_prob = kwargs.get('re_prob', 0.5)
        pixel_mean = kwargs.get('pixel_mean', [0.485, 0.456, 0.406])
        pixel_std = kwargs.get('pixel_std', [0.229, 0.224, 0.225])
        self.using_bg_erase = using_bg_erase
        self.tf_value = build_transorms_value(mode = self.mode, re_prob = re_prob, pixel_mean = pixel_mean, pixel_std = pixel_std)
        self.mask_generator = MaskGenerator(
            input_size=self.img_size[0],
            mask_patch_size=32,
            model_patch_size=16,
            mask_ratio=0.6,
        )
        self.to_tensor = T.ToTensor()

    def __call__(self, data: dict):
        img_path = data.get('img_path')
        pid = data.get('pid')
        cid = data.get('cid')
        if not os.path.isfile(img_path):
            logging.error(f'Can Not Read Image {img_path}')
            raise Exception
        else:
            img = Image.open(img_path).convert('RGB')
            img = self.to_tensor(img)
        img = self.tf_value(self.tf_shape(img))
        mask = self.mask_generator()
        # pre transforms
        data_dict = {'img': img, 'pid': pid, 'camera_id': cid, 'mask': mask, 'img_path': img_path}
        return data_dict


class MaskGenerator:
    def __init__(self, input_size=224, mask_patch_size=32, model_patch_size=4, mask_ratio=0.6):
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio
        
        assert self.input_size % self.mask_patch_size == 0
        assert self.mask_patch_size % self.model_patch_size == 0
        
        self.rand_size = self.input_size // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size
        
        self.token_count = self.rand_size ** 2
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))
        
    def __call__(self):
        mask_idx = np.random.permutation(self.token_count)[:self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1
        
        mask = mask.reshape((self.rand_size, self.rand_size))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)
        
        return mask


def genHeatMap(json_path, imgh, imgw):
    if os.path.isfile(json_path):
        with open(json_path,'r') as f:
            a=json.load(f)
            p=a['people']
    else:
        hm = np.zeros([18, imgh, imgw])
        vis_score = np.zeros([18])
        return hm, vis_score
    
    vis_score_value_max = -1
    vis_score_select = np.zeros([18])
    p_points_select = np.zeros([18, 3])
    for i in range(len(p)):
        p_points=p[i]
        p_points=p_points['pose_keypoints_2d']
        p_points=np.array(p_points)
        p_points=p_points.reshape(18,3)
        vis_score =p_points[:,2]

        # get vis_score_value is max
        vis_score_value = vis_score.sum()
        if vis_score_value > vis_score_value_max:
            vis_score_value_max = vis_score_value
            p_points_select = p_points
            vis_score_select = vis_score

    hm = np.zeros([p_points_select.shape[0], imgh, imgw])
    for j in range(p_points_select.shape[0]):
        w,h = p_points_select[j][:2]
        w,h=int(w),int(h)
        hm[j][min(h,imgh-1)][min(w, imgw-1)] = vis_score_select[j]
        hm[j] = gaussian_filter(hm[j],[3,1.5],mode='wrap')
        #hm[j]=CenterGaussianHeatMap(imgh,imgw,w,h,(imgh*imgw/1000.))
    return hm, vis_score_select
    

# Compute gaussian kernel
def CenterGaussianHeatMap(img_height, img_width, c_x, c_y, variance):
    gaussian_map = np.zeros((img_height, img_width))
    for x_p in range(img_width):
        for y_p in range(img_height):
            dist_sq = (x_p - c_x) * (x_p - c_x) + \
                      (y_p - c_y) * (y_p - c_y)
            exponent = dist_sq / 2.0 / variance / variance
            gaussian_map[y_p, x_p] = np.exp(-exponent)
    return gaussian_map