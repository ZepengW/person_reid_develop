import os
from dataset.mask import get_person_semantics_mask
import torch
import math
import torch.nn as nn
from dataset.transforms import build_transforms_mask
import itertools
import time

LIP_LABEL = ['Background', 'Hat', 'Hair', 'Glove', 'Sunglasses', 'Upper-clothes', 'Dress', 'Coat',
                  'Socks', 'Pants', 'Jumpsuits', 'Scarf', 'Skirt', 'Face', 'Left-arm', 'Right-arm',
                  'Left-leg', 'Right-leg', 'Left-shoe', 'Right-shoe']

LABEL_CLOTHES = [1,3,4,5,6,7,8,9,10,11,12,18,19]
LABEL_BODY = [13,14,15,16,17]

if __name__ == '__main__':
    time_b = time.time()
    # initialize Embeding Information
    embed_dim = 768
    # initialize label weight matrix
    num_label = len(LIP_LABEL)
    weight_mat = torch.eye(num_label) * 0.8 + 0.2
    for id_body in LABEL_BODY:
        for id_clothes in LABEL_CLOTHES:
            weight_mat[id_body,id_clothes] = 0.5
            weight_mat[id_clothes,id_body] = 0.5
    for p in itertools.combinations(LABEL_CLOTHES, 2):
        weight_mat[p[0],p[1]] = 0.8
        weight_mat[p[1],p[0]] = 0.8
    for p in itertools.combinations(LABEL_BODY, 2):
        weight_mat[p[0],p[1]] = 0.8
        weight_mat[p[1],p[0]] = 0.8
    print('generate weight mat:')
    print(weight_mat)
    
    # processMask
    dirs_mask = {'../dataset/DukeMTMC-reID/vit_patch_mask'}
    dir_output = '../dataset/DukeMTMC-reID/vit_patch_mask_attn'
    if not os.path.exists(dir_output):
        os.mkdir(dir_output)
    for dir_mask in dirs_mask:
        print(f'processing dir: {dir_mask}')
        files_mask = os.listdir(dir_mask)
        for i,file in enumerate(files_mask):
            cost_time = time.time() - time_b
            print(f'\rprocessing files [{i+1}/{len(files_mask)}]: {file}, left time: {int((len(files_mask) - i - 1)/(i+1) *cost_time)}s',end='')
            file_path = os.path.join(dir_mask,file)
            output_path = os.path.join(dir_output,file)

            mask_ids = torch.load(file_path)
            mask_ids = torch.flatten(mask_ids,0)
            num_patch = mask_ids.shape[0]
            mask_attn = torch.zeros([num_patch,num_patch])

            # 依据mask_ids 生成 mask_attn
            for i in range(0,num_patch):
                for j in range(0,num_patch):
                    mask_attn[i,j] = weight_mat[int(mask_ids[i]),int(mask_ids[j])]


            torch.save(mask_attn,output_path)
    

            