import os.path
import sys
# 将项目目录添加到Python路径中
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
from model import make_model
from dataset.method_reading import get_img_wo_path
from dataset.dataset_dir import get_img_dir
from dataset import DatasetImage
import torch
from torch.utils.data import DataLoader
from utils.dist_compute import compute_euclidean_distance
from utils.re_ranking import compute_dis_matrix
import numpy as np
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
import shutil
import openpyxl
def extract_feat(model, dataloader, output_dir):
    model.eval()
    feat_l = []
    img_path_l = []
    for img_batch, cid, img_name_batch in tqdm(dataloader):
        img_batch = img_batch.cuda()
        cid = cid.cuda()
        with torch.no_grad():
            feat_batch = model(img_batch)
            feat_l.append(feat_batch['feat'].cpu().clone().detach())
            img_path_l += img_name_batch
    feat_l = torch.cat(feat_l, dim=0)
    # save result
    os.makedirs(output_dir, exist_ok=True)
    torch.save(feat_l, os.path.join(output_dir, 'feat.pkl'))
    with open(os.path.join(output_dir, 'name.txt'), 'w') as f:
        for img_path in img_path_l:
            f.write(img_path + '\n')
    return feat_l, img_path_l


def search_persons(query_name_l, feats, name_l, threshold = 500, topk = 100):
    idx_query = [name_l.index(name) for name in query_name_l]
    idx_gallery = list(set(range(len(name_l))) - set(idx_query))
    feats_query = feats[idx_query]
    feats_gallery = feats[idx_gallery]
    print('compute distance matrix')
    #dist_m = compute_euclidean_distance(feats_query, feats_gallery)
    dist_m = compute_dis_matrix(feats_query, feats_gallery, 'euclidean', is_re_ranking=False)
    print(f'[Finish] distance matrix shape: {dist_m.shape} ')
    print('ranking')
    # 对数组进行排序
    indices = np.argsort(dist_m, axis=1)
    values = np.sort(dist_m, axis=1)
    result = defaultdict(list)
    if topk < 0:
        for i in range(indices.shape[0]):
            idx_gallery_rank_i = indices[i]
            for j in idx_gallery_rank_i:
                if values[i, j] > threshold:
                    break
                result[query_name_l[i]].append((name_l[idx_gallery[j]], dist_m[i, j]))
    else:
        for i in range(indices.shape[0]):
            idx_gallery_rank_i = indices[i][:topk]
            for j in idx_gallery_rank_i:
                result[query_name_l[i]].append((name_l[idx_gallery[j]], dist_m[i, j]))
    return result

def write_to_excel(result, filename):
    data = []
    for key, values in result.items():
        row1 = [key] + [name for name, _ in values]
        row2 = [''] + [value for _, value in values]
        data.append(row1)
        data.append(row2)
    df = pd.DataFrame(data)
    df.to_excel(filename, index=False, header=False)


def read_file(input_dir):
    try:
        pkl_path = os.path.join(input_dir, 'feat.pkl')
        feats = torch.load(pkl_path)
        name_l = []
        with open(os.path.join(input_dir, 'name.txt'), 'r') as f:
            for line in f.readlines():
                name_l.append(line.strip())
        return feats, name_l
    except:
        return None, None

if __name__ == '__main__':
    query_name_l = [
        'query/a1.png',
        'query/a2.png',
        'query/b1.png',
        'query/b2.png'
    ]
    output_feat_dir = '/home/ck/workspace/dataset/sjtu-search/result/feat'
    dir_path = '/home/ck/workspace/dataset/sjtu-search/sjtu-filter'
    output_search_dir = '/home/ck/workspace/dataset/sjtu-search/result'
    feats, name_l = read_file(output_feat_dir)


    if not isinstance(name_l, list):
        model_name = 'osnet'
        model_args = {
            'name': 'osnet_ain_x1_0',
            'num_classes': 2510
        }
        # model_name = 'transreid'
        # model_args = {
        #     'PRETRAIN_CHOICE': 'imagenet',
        #     'PRETRAIN_PATH': './pretrained/jx_vit_base_p16_224-80ecf9dd.pth',
        #     'METRIC_LOSS_TYPE': 'triplet',
        #     'IF_LABELSMOOTH': 'off',
        #     'IF_WITH_CENTER': 'no',
        #     'NAME': 'transformer',
        #     'NO_MARGIN': True,
        #     'TRANSFORMER_TYPE': 'vit_base_patch16_224_TransReID',
        #     'STRIDE_SIZE': [ 12, 12 ],
        #     'SIE_CAMERA': True,
        #     'SIE_COE': 3.0,
        #     'JPM': True,
        #     'RE_ARRANGE': True,
        #     'CAMERA_NUM': 15,
        #     'SIZE_INPUT': [256, 128],
        #     'num_classes': 1041,
        # }
        model = make_model(model_name, model_args)
        # load pretrained
        #model.load_state_dict(torch.load('./pretrained/vit_transreid_msmt.pth'), strict=True)
        model = model.cuda()
        state_dict = torch.load('../pretrained/osnet_ain_ms_d_c.pth.tar')['state_dict']
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=True)

        # load dataset dir
        transform_params = {
            'size': [256, 128],
            'mean': [0.5, 0.5, 0.5],
            'std': [0.5, 0.5, 0.5],
            'type': 'test'
        }
        m_reading = get_img_wo_path(transform_params = transform_params)

        dataloader = DataLoader(DatasetImage(get_img_dir(dir_path), m_reading), shuffle=False, batch_size=128, num_workers=8)


        # extract feat
        feats, name_l = extract_feat(model, dataloader, output_feat_dir)
    # person search
    result = search_persons(query_name_l, feats, name_l, topk=50)
    write_to_excel(result, '../result.xlsx')

    # save to
    print('Save Result')
    for key, values in result.items():
        output_dir = os.path.join(output_search_dir, key)
        os.system(f'rm -rf {output_dir}')
        os.makedirs(output_dir, exist_ok=True)
        for idx, (name, _) in enumerate(values):
            shutil.copy(os.path.join(dir_path, name), os.path.join(output_dir, f'{idx:04d}-{os.path.basename(name)}'))
