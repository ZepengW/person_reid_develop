'''
this file contains method of reading dataset
'''
from PIL import Image
from utils.logger import Logger
logging = Logger()
import os
from dataset.transforms import build_transforms, build_transorms_shape, build_transorms_value, build_transforms_bgerase


class GetImg(object):
    """
    read img data from img path, and resize it to uniform size
    :param data:
    :param resize:
    :return:
    """

    def __init__(self, **kwargs):
        self.tf = build_transforms(**kwargs['transform_params'])

    def __str__(self):
        return self.tf.__str__()

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

class get_img_wo_path(object):
    # for inference
    def __init__(self, **kwargs):
        self.tf = build_transforms(**kwargs['transform_params'])

    def __str__(self):
        return self.tf.__str__()

    def __call__(self, inputs):
        img_path, cid, img_name = inputs
        if not os.path.isfile(img_path):
            logging.error(f'Can Not Read Image {img_path}')
            img = None
        else:
            img = Image.open(img_path).convert('RGB')
            img = self.tf(img)
        return img, cid, img_name

