import os.path as osp
import os
from utils.logger import Logger
logging = Logger()

def get_img_dir(dir_path):
    data_list = []
    cid = 0
    cid_flag = True
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.endswith('.jpg') or file.endswith('.png'):
                img_path = os.path.join(root, file)
                data_list.append((img_path, cid, os.path.relpath(img_path, dir_path)))

    print(f'img number: {len(data_list)}')
    return data_list
