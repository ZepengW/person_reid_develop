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
    ### pre transform methods on images
    t = tf.build_transforms()
    t_mask = tf.build_transforms_mask()

    dataset_config = config.get('dataset', dict())
    dataset_manager = DatasetManager(dataset_config.get('dataset_name', ''), dataset_config.get('dataset_path', ''),
                                     num_mask=dataset_config.get('num-mask', 6))

    model_config = config.get('network-params', dict())
    model = ModelManager(model_config, device, class_num=dataset_manager.get_train_pid_num(),
                         writer=writer_tensorboardX)

    mode = config.get('mode', 'train')
    dataset_type = dataset_config.get('type', 'image')
    get_dataset = getattr(dataset_manager, 'get_dataset_' + dataset_type)

    vis_bool = config.get('vis', False)
    if 'train' == mode:
        logging.info("loading train data")
        loader_train_source = DataLoader(
            get_dataset('train', transform=t, transform_mask=t_mask),
            batch_size=dataset_config.get('batch_size', 16),
            num_workers=dataset_config.get('num_workers', 8),
            drop_last=False,
            shuffle=True
        )
        logging.info("load train data finish")
        logging.info("prepare to train from epoch[{0}] to epoch[{1}]".format(model.trained_epoches,
                                                                             model_config.get('epoch', 64) - 1))
        for i in range(model.trained_epoches, model_config.get('epoch', 64)):
            model.train(loader_train_source, i, is_vis= vis_bool)
    elif 'test' == mode:
        logging.info("loading test data")
        loader_gallery_source = DataLoader(
            get_dataset('test', transform=t, transform_mask=t_mask),
            batch_size=dataset_config.get('batch_size', 16),
            num_workers=dataset_config.get('num_workers', 8),
            drop_last=False,
            shuffle=True
        )
        loader_query_source = DataLoader(
            get_dataset('query', transform=t, transform_mask=t_mask),
            batch_size=dataset_config.get('batch_size', 16),
            num_workers=dataset_config.get('num_workers', 8),
            drop_last=False,
            shuffle=True
        )
        logging.info("load test data finish")
        model.test(loader_query_source, loader_gallery_source, is_vis= vis_bool)
    elif 'train_test' == mode:
        logging.info("loading train data")
        loader_train_source = DataLoader(
            get_dataset('train', transform=t, transform_mask=t_mask),
            batch_size=dataset_config.get('batch_size', 16),
            num_workers=dataset_config.get('num_workers', 8),
            drop_last=False,
            shuffle=True
        )
        logging.info("load train data finish")
        logging.info("loading test data")
        loader_gallery_source = DataLoader(
            get_dataset('test', transform=t, transform_mask=t_mask),
            batch_size=dataset_config.get('batch_size', 16),
            num_workers=dataset_config.get('num_workers', 8),
            drop_last=False,
            shuffle=True
        )
        loader_query_source = DataLoader(
            get_dataset('query', transform=t, transform_mask=t_mask),
            batch_size=dataset_config.get('batch_size', 16),
            num_workers=dataset_config.get('num_workers', 8),
            drop_last=False,
            shuffle=True
        )
        logging.info("load test data finish")
        logging.info("prepare to train from epoch[{0}] to epoch[{1}]".format(model.trained_epoches,
                                                                             model_config.get('epoch', 64) - 1))
        for i in range(model.trained_epoches, model_config.get('epoch', 64)):
            is_vis = True if (i % 20 == 0 or i == model_config.get('epoch', 64) - 1) else False
            is_vis = is_vis and vis_bool
            model.train(loader_train_source, i, is_vis)
            if i % 10 == 0 or i == model_config.get('epoch', 64) - 1:
                model.test(loader_query_source, loader_gallery_source, epoch=i, is_vis = vis_bool)

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='config/train-mars.yaml', help='the config file(.yaml)')

    config = parser.parse_args()
    cfg_path = config.cfg
    if not os.path.exists(cfg_path):
        logging.error(f'can not find the config file:{cfg_path}')
    with open(cfg_path) as f:
        cfg = f.read()
        yaml_cfg = yaml.safe_load(cfg)


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
