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

# set cuda visible devices, and return the first gpu device
def set_gpus_env(gpu_ids):
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(id) for id in gpu_ids])
    if not torch.cuda.is_available():
        logging.warning('Cuda is not available using cpu')
        return torch.device('cpu')
    gpus_count = torch.cuda.device_count()
    for gpu_id in gpu_ids:
        if gpu_id >= gpus_count:
            logging.warning('gpu id:{0} exceeds the limit , which only have {1} gpus'.format(gpu_id,gpus_count))
            gpu_ids.remove(gpu_id)
        logging.info('using gpu: id is ' + str(gpu_id) + ' name is ' + torch.cuda.get_device_name(gpu_id))
    if len(gpu_ids)==0:
        gpu_ids.append(0)
        logging.warning('all the config gpus can not be used, use gpu:0')
    return torch.device('cuda:{0}'.format(gpu_ids[0]))


def main(config):
    device = set_gpus_env(config.get('gpu', [0]))
    ### pre transform methods on images
    t = tf.build_transforms()
    t_mask = tf.build_transforms_mask()

    dataset_config = config.get('dataset', dict())
    dataset_manager = DatasetManager(dataset_config.get('dataset_name', ''), dataset_config.get('dataset_path', ''),
                                     num_mask=dataset_config.get('num-mask', 6))

    model_config = config.get('network-params',dict())
    model = ModelManager(model_config, device, class_num=dataset_manager.get_train_pid_num())

    mode = config.get('mode', 'train')
    dataset_type = dataset_config.get('type','image')
    get_dataset = getattr(dataset_manager,'get_dataset_'+dataset_type)

    if 'train' == mode:
        logging.info("loading train data")
        loader_train_source = DataLoader(
            get_dataset('train',transform=t,transform_mask=t_mask),
            batch_size=dataset_config.get('batch_size', 16),
            num_workers=dataset_config.get('num_workers', 8),
            drop_last=False,
            shuffle=True
        )
        logging.info("load train data finish")
        logging.info("prepare to train from epoch[{0}] to epoch[{1}]".format(model.trained_epoches, model_config.get('epoch',64)-1))
        for i in range(model.trained_epoches, model_config.get('epoch',64)):
            model.train(loader_train_source, i)
    elif 'test' == mode:
        logging.info("loading test data")
        loader_gallery_source = DataLoader(
            get_dataset('test',transform=t,transform_mask=t_mask),
            batch_size=dataset_config.get('batch_size', 16),
            num_workers=dataset_config.get('num_workers', 8),
            drop_last=False,
            shuffle=True
        )
        loader_query_source = DataLoader(
            get_dataset('query',transform=t,transform_mask=t_mask),
            batch_size=dataset_config.get('batch_size', 16),
            num_workers=dataset_config.get('num_workers', 8),
            drop_last=False,
            shuffle=True
        )
        logging.info("load test data finish")
        model.test(loader_query_source, loader_gallery_source)
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
            model.train(loader_train_source, i)
            if i % 10 == 0:
                model.test(loader_query_source, loader_gallery_source)

    logging.info("finish!")

def init_logging():
    # log config
    if not os.path.exists('./output/log'):
        os.makedirs('./output/log')
    logging.basicConfig(filename='./output/log/' + str(datetime.datetime.now().year).rjust(4, '0')
                                 + str(datetime.datetime.now().month).rjust(2, '0')
                                 + str(datetime.datetime.now().day).rjust(2, '0')
                                 + str(datetime.datetime.now().hour).rjust(2, '0')
                                 + str(datetime.datetime.now().minute).rjust(2, '0')
                                 + str(datetime.datetime.now().second).rjust(2, '0') + '.txt',
                        level=logging.DEBUG,
                        format='###%(levelname)s###[%(asctime)s]%(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter('[%(asctime)s]%(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    logging.getLogger('').addHandler(console)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--cfg', type =str, default='config/train-mars.yaml',help='the config file(.yaml)')

    # initial logging module
    init_logging()

    config = parser.parse_args()
    cfg_path = config.cfg
    if not os.path.exists(cfg_path):
        logging.error('can not find the config file:'+cfg_path)
    with open(cfg_path) as f:
        cfg = f.read()
        yaml_cfg = yaml.safe_load(cfg)
    print(yaml_cfg)
    main(yaml_cfg)
