'''
this file contains method of reading dataset
'''
from PIL import Image
import logging
import os
from dataset.transforms import build_transforms

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
        data_dict = {'img': img, 'pid': pid, 'camera_id': cid}
        return data_dict

