import os
import sys
# 将项目目录添加到Python路径中
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
import torch
from utils.re_ranking import compute_dis_matrix
from tqdm import tqdm, trange
from search import extract_feat
from model.model_factory import make_model
from dataset.method_reading import get_img_wo_path
from dataset.dataset_dir import get_img_dir
from dataset import DatasetImage
from torch.utils.data import DataLoader

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

def calculate_distance(input_dir, output_dir):
    model_name = 'osnet'
    model_args = {
        'name': 'osnet_ain_x1_0',
        'num_classes': 2510
    }
    model = make_model(model_name, model_args)
    # load pretrained
    # model.load_state_dict(torch.load('./pretrained/vit_transreid_msmt.pth'), strict=True)
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
    m_reading = get_img_wo_path(transform_params=transform_params)

    dataloader = DataLoader(DatasetImage(get_img_dir(input_dir), m_reading), shuffle=True, batch_size=128,
                            num_workers=8)

    feats, name_l = extract_feat(model, dataloader, output_dir)

    print('calculate distance matrix')
    num_sample = len(name_l)
    for i in trange(num_sample // 10000 + 1):
        dist_mat = compute_dis_matrix(feats[i * 10000: (i + 1) * 10000], feats, 'euclidean', is_re_ranking=False)
        dist_mat = torch.from_numpy(dist_mat)
        torch.save(dist_mat, os.path.join(output_dir, f'dist_mat_{i}.pkl'))
    print('save distance matrix')

def cluster(input_dir, name_l, dist_dir, output_dir, new_dir,  threshold = 200):

    print('cluster...')
    os.system(f'rm -rf {output_dir}')
    os.system(f'rm -rf {new_dir}')
    os.makedirs(new_dir, exist_ok=True)
    idx_except = []
    id_dist_map = 0
    print(f'load distance matrix {id_dist_map}')
    dist_mat = torch.load(os.path.join(dist_dir, f'dist_mat_{id_dist_map}.pkl'))
    _, index = torch.sort(dist_mat, dim=1)
    for i in trange(len(name_l)):
        name = name_l[i]
        if i // 10000 > id_dist_map:
            id_dist_map += 1
            print(f'load distance matrix {id_dist_map}')
            dist_mat = torch.load(os.path.join(dist_dir, f'dist_mat_{id_dist_map}.pkl'))
            _, index = torch.sort(dist_mat, dim=1)
        if i in idx_except:
            continue
        os.system(f'cp {os.path.join(input_dir, name)} {os.path.join(new_dir, os.path.basename(name))}')
        sample_output_dir = os.path.join(output_dir, os.path.basename(name))
        os.makedirs(sample_output_dir, exist_ok=True)
        for j in index[i%10000]:
            if dist_mat[i%10000, j] > threshold:
                break
            idx_except.append(j)
            os.system(f'cp {os.path.join(input_dir, name_l[j])} {os.path.join(sample_output_dir, os.path.basename(name_l[j]))}')
    print(f'remove sample number {len(idx_except)}')
    print(f'remain sample number {len(os.listdir(new_dir))}')





if __name__ == '__main__':
    input_dir = '../../dataset/sjtu/0304-1635-1838'
    output_dir = '../dataset/sjtu-search/sjtu-cluster/0304-1635-1838'
    new_dir = '../../dataset/sjtu-search/sjtu-filter/0304-1635-1838'
    feat_dir = '../dataset/sjtu-search/sjtu-feat/0304-1635-1838'
    calculate_distance(input_dir, feat_dir)

    name_l = []
    with open(os.path.join(feat_dir, 'name.txt'), 'r') as f:
        for line in f.readlines():
            name_l.append(line.strip())
    cluster(input_dir,name_l, feat_dir ,output_dir,new_dir)