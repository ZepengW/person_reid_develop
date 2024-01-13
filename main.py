import comet_ml
from utils.logger import Logger
logging = Logger()
import argparse
import torch
import os
from lightning import seed_everything
from lightning.pytorch.loggers import CometLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import lightning.pytorch as Lp
from dataset import DatasetManager, initial_m_reading
from model import ModelManager
from torch.utils.data import DataLoader
import yaml



def set_seed(seed):
    seed_everything(seed, workers=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def main(cfg_dict: dict, logger_comet: CometLogger):
    set_seed(cfg_dict.get('seed', 1234))
    cfg_model = cfg_dict.get('model-manager', dict())

    # callback
    exp_name = cfg_dict['logger']['task_name']
    dir_weights = os.path.join('output', exp_name, 'weights')
    os.makedirs(dir_weights, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        monitor='val/mAP',
        dirpath=dir_weights,
        filename='E{epoch}-mAP{val/mAP:.2%}',
        save_last=True,
        auto_insert_metric_name=False
    )
    # Trainer for Lightning
    job = Lp.Trainer(
        callbacks=[checkpoint_callback],
        accelerator='cpu' if cfg_dict.get('gpus', 'auto') == -1 else 'gpu',
        devices=cfg_dict.get('gpus', 'auto'),
        precision=cfg_dict.get('precision', 32),
        min_epochs=cfg_model.get('epoch', 100),
        max_epochs=cfg_model.get('epoch', 100),
        logger=logger_comet,
        check_val_every_n_epoch=cfg_dict.get('eval_interval', 10),
        num_sanity_val_steps=-1
    )
    # initial dataset
    cfg_data = cfg_dict.get('dataset', dict())
    # initial model
    mode = cfg_dict.get('mode', 'train')
    model = ModelManager(cfg_model, cfg_data)
    if 'train' == mode:
        logging.info('Begin to train')
        job.fit(model)
        logging.info('End train')
    elif 'test' == mode:
        logging.info('Begin to test')
        job.test(model)
        logging.info('End test')
    else:
        logging.error(f'not support mode:{mode}')

    logging.info("finish!")


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
    parser.add_argument('--cfg', type=str, default='', help='the config file(.yaml)')
    parser.add_argument('--no_log', action='store_true', default=False, help='do not save log file')
    parser.add_argument('--params', type=str, help='Override parameters in yaml file')

    config = parser.parse_args()
    # base config
    cfg_base_path = config.cfg_base
    if os.path.exists(cfg_base_path):
        print(f'[Info] loading base config file:{cfg_base_path}')
        with open(cfg_base_path) as f:
            cfg = f.read()
            yaml_cfg_base = yaml.safe_load(cfg)
    else:
        print(f'[Warning] can not find the base config file:{cfg_base_path}')

    # detail config
    cfg_path = config.cfg
    if os.path.exists(cfg_path):
        print(f'[Info] loading the config file:{cfg_path}')
        with open(cfg_path) as f:
            cfg = f.read()
            yaml_cfg_detail = yaml.safe_load(cfg)
    else:
        print(f'[Warning] can not find the config file:{cfg_path}')
        yaml_cfg_detail = dict()

    cfg_dict = merge_data(yaml_cfg_base, yaml_cfg_detail)
    # 覆盖yaml文件中的参数
    if config.params is not None:
        overrides = config.params.split(',')
        for override in overrides:
            key, value = override.split('=')
            keys = key.split('.')
            last_key = keys.pop()
            temp = cfg_dict
            for k in keys:
                temp = temp[k]
            temp[last_key] = type(temp[last_key])(value)

    if not config.no_log:
        # initial logging module
        logger_cfg = cfg_dict['logger']
        logger_comet = CometLogger(
            api_key=logger_cfg.get('api_key', os.environ.get("COMET_API_KEY")),
            workspace=logger_cfg.get('workspace', os.environ.get("COMET_WORKSPACE")),  # Optional
            save_dir="./output",  # Optional
            project_name=logger_cfg.get('project_name', "reid"),  # Optional
            rest_api_key=os.environ.get("COMET_REST_API_KEY"),  # Optional
            experiment_key=os.environ.get("COMET_EXPERIMENT_KEY"),  # Optional
            experiment_name=logger_cfg.get('task_name', None),  # Optional
            auto_output_logging='simple'
        )
        logging.set_log_file(os.path.join('output', logger_comet.experiment.get_name(), logger_comet.experiment.get_name())+'.log')
        logger_comet.log_hyperparams(cfg_dict)
    else:
        logger_comet = None
    main(cfg_dict, logger_comet)
