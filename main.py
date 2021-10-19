import argparse
import torch
import os
import logging
import datetime
from dataset import transforms as tf
from dataset import DatasetManager, DatasetVideo
from model import ModelManager
from torch.utils.data import DataLoader
import yaml
from tensorboardX import SummaryWriter


# set cuda visible devices, and return the first gpu device
def set_gpus_env(gpu_ids):
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(id) for id in gpu_ids])
    if not torch.cuda.is_available():
        logging.warning('Cuda is not available using cpu')
        return torch.device('cpu')
    gpus_count = torch.cuda.device_count()
    for gpu_id in gpu_ids:
        if gpu_id >= gpus_count:
            logging.warning('gpu id:{0} exceeds the limit , which only have {1} gpus'.format(gpu_id, gpus_count))
            gpu_ids.remove(gpu_id)
        logging.info('using gpu: id is ' + str(gpu_id) + ' name is ' + torch.cuda.get_device_name(gpu_id))
    if len(gpu_ids) == 0:
        gpu_ids.append(0)
        logging.warning('all the config gpus can not be used, use gpu:0')
    return torch.device('cuda:{0}'.format(gpu_ids[0]))


def main(config, writer_tensorboardX):
    device = set_gpus_env(config.get('gpu', [0]))
    vis_bool = config.get('vis', False)     # draw feature distribution
    vis_interval = config.get('vis_interval', 20)    # interval of drawing feature distribution
    eval_interval = config.get('eval_interval', 20)

    ### pre transform methods on images
    dataset_config = config.get('dataset', dict())
    ### pre transform methods on images
    size = dataset_config.get('image_size', [256,128])
    t = tf.build_transforms(size)
    dataset_manager = DatasetManager(dataset_config.get('dataset_name', ''), dataset_config.get('dataset_path', ''),
                                     num_mask=dataset_config.get('num-mask', 6))

    model_config = config.get('model-manager', dict())
    model = ModelManager(model_config, device, class_num=dataset_manager.get_train_pid_num(),
                         writer=writer_tensorboardX)

    mode = config.get('mode', 'train')
    dataset_type = dataset_config.get('type', 'image')
    get_dataset = getattr(dataset_manager, 'get_dataset_' + dataset_type)
    if 'train' == mode:
        logging.info("loading train data")
        loader_train_source = DataLoader(
            get_dataset('train', transform=t),
            batch_size=dataset_config.get('batch_size_train', 16),
            num_workers=dataset_config.get('num_workers', 8),
            drop_last=True,
            shuffle=True
        )
        logging.info("load train data finish")
        logging.info("loading test data")
        loader_gallery_source = DataLoader(
            get_dataset('test', transform=t),
            batch_size=dataset_config.get('batch_size_test', 16),
            num_workers=dataset_config.get('num_workers', 8),
            drop_last=False,
            shuffle=True
        )
        loader_query_source = DataLoader(
            get_dataset('query', transform=t),
            batch_size=dataset_config.get('batch_size_test', 16),
            num_workers=dataset_config.get('num_workers', 8),
            drop_last=False,
            shuffle=True
        )
        logging.info("load test data finish")
        logging.info("prepare to train from epoch[{0}] to epoch[{1}]".format(model.trained_epoches,
                                                                             model_config.get('epoch', 64) - 1))
        for i in range(model.trained_epoches+1, model_config.get('epoch', 64)+1):
            is_vis = (i % vis_interval == 0 or i == model_config.get('epoch', 64) - 1) #each vis_interval or last epoch
            is_vis = is_vis and vis_bool
            model.train(loader_train_source, i, is_vis)
            if (i % eval_interval == 0) or i == model_config.get('epoch', 64) - 1:
                model.test(loader_query_source, loader_gallery_source, epoch=i, is_vis=vis_bool)
    elif 'test' == mode:
        logging.info("loading test data")
        loader_gallery_source = DataLoader(
            get_dataset('test', transform=t),
            batch_size=dataset_config.get('batch_size_test', 16),
            num_workers=dataset_config.get('num_workers', 8),
            drop_last=False,
            shuffle=True
        )
        loader_query_source = DataLoader(
            get_dataset('query', transform=t),
            batch_size=dataset_config.get('batch_size_test', 16),
            num_workers=dataset_config.get('num_workers', 8),
            drop_last=False,
            shuffle=True
        )
        logging.info("load test data finish")
        model.test(loader_query_source, loader_gallery_source, is_vis=vis_bool)
    else:
        logging.error(f'not support mode:{mode}')

    logging.info("finish!")


def init_logging(task_name=''):
    # log config
    log_dir_name = str(datetime.datetime.now().year).rjust(4, '0') \
                   + str(datetime.datetime.now().month).rjust(2, '0') \
                   + str(datetime.datetime.now().day).rjust(2, '0') \
                   + str(datetime.datetime.now().hour).rjust(2, '0') \
                   + str(datetime.datetime.now().minute).rjust(2, '0') \
                   + str(datetime.datetime.now().second).rjust(2, '0')
    if task_name != '':
        log_dir_name = f'{task_name}-{log_dir_name}'
    if not os.path.isdir(f'./output/log/{log_dir_name}'):
        os.mkdir(f'./output/log/{log_dir_name}')
    logging.basicConfig(filename=f'./output/log/{log_dir_name}/log.txt',
                        level=logging.INFO,
                        format='###%(levelname)s###[%(asctime)s]%(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter('[%(asctime)s]%(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    logging.getLogger().addHandler(console)
    print(f'writing log to ./output/log/{log_dir_name}')
    return log_dir_name

def merge_data(data_1, data_2):
    """
    merge data of two nested dict
    :param data_1:
    :param data_2: priority
    :return:
    """
    if isinstance(data_1, dict) and isinstance(data_2, dict):
        new_dict = {}
        d2_keys = list(data_2.keys())
        for d1k in data_1.keys():
            if d1k in d2_keys:
                d2_keys.remove(d1k)
                new_dict[d1k] = merge_data(data_1.get(d1k), data_2.get(d1k))
            else:
                new_dict[d1k] = data_1.get(d1k)
        for d2k in d2_keys:
            new_dict[d2k] = data_2.get(d2k)
        return new_dict
    else:
        if data_2 == None:
            return data_1
        else:
            return data_2

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_base', type=str, default='config/cfg-base.yaml', help='the config file(.yaml)')
    parser.add_argument('--cfg', type=str, default='config/train-market1501.yaml', help='the config file(.yaml)')

    config = parser.parse_args()
    # base config
    cfg_base_path = config.cfg_base
    if not os.path.exists(cfg_base_path):
        logging.error(f'can not find the base config file:{cfg_base_path}')
    with open(cfg_base_path) as f:
        cfg = f.read()
        yaml_cfg_base = yaml.safe_load(cfg)

    # detail config
    cfg_path = config.cfg
    if not os.path.exists(cfg_path):
        logging.error(f'can not find the config file:{cfg_path}')
    with open(cfg_path) as f:
        cfg = f.read()
        yaml_cfg_detail = yaml.safe_load(cfg)

    yaml_cfg = merge_data(yaml_cfg_base,yaml_cfg_detail)
    if not os.path.isdir('./output'):
        os.mkdir('./output')
    if not os.path.isdir('./output/log'):
        os.mkdir('./output/log')

    # initial logging module
    log_dir_name = init_logging(task_name=yaml_cfg.get('task-name', ''))
    # initial tensorboardX
    writer_tensorboardx = SummaryWriter(f'./output/log/{log_dir_name}')

    logging.info(str(yaml_cfg))

    main(yaml_cfg, writer_tensorboardx)

    writer_tensorboardx.close()
