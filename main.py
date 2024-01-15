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
        dirpath=dir_weights,
        filename=exp_name+'-E{epoch}',
        every_n_epochs=cfg_dict.get('eval_interval', 10),
        save_last=True,
        auto_insert_metric_name=False,
        verbose=True
    )
    # Trainer for Lightning
    cfg_engine = cfg_dict.get('Engine')
    job = Lp.Trainer(
        callbacks=[checkpoint_callback],
        accelerator='cpu' if cfg_engine.get('gpus', 'auto') == -1 else 'gpu',
        devices=cfg_engine.get('gpus', 'auto'),
        precision=cfg_engine.get('precision', 32),
        min_epochs=cfg_engine.get('epoch', 100),
        max_epochs=cfg_engine.get('epoch', 100),
        logger=logger_comet,
        check_val_every_n_epoch=cfg_engine.get('eval_interval', 10),
        # num_sanity_val_steps=-1,
        inference_mode=True,
    )
    # initial dataset
    cfg_data = cfg_dict.get('dataset', dict())
    # initial model
    model = ModelManager(cfg_model, cfg_data)
    # resume
    resume_path = cfg_engine.get('resume', None)
    if resume_path is not None:
        if not os.path.isfile(resume_path):
            logging.warning(f'Can not find resume: {resume_path}')
            resume_path = None

    mode = cfg_dict.get('mode', 'train')
    if 'train' == mode:
        logging.info('Begin to train')
        job.fit(model, ckpt_path=resume_path)
        logging.info('End train')
    elif 'test' == mode:
        logging.info('Begin to test')
        job.test(model, ckpt_path=resume_path)
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


def params_overwrite(cfg_dict, cfg_cli):
    # overwrite params from cli
    # 覆盖yaml文件中的参数
    if cfg_cli.test:
        cfg_dict['mode'] = 'test'

    if cfg_cli.resume:
        cfg_dict['Engine']['resume'] = cfg_cli.resume

    if cfg_cli.params is not None:
        overrides = cfg_cli.params.split(',')
        for override in overrides:
            key, value = override.split('=')
            keys = key.split('.')
            last_key = keys.pop()
            temp = cfg_dict
            for k in keys:
                temp = temp[k]
            temp[last_key] = type(temp[last_key])(value)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_base', type=str, default='config/cfg-base.yaml', help='the config file(.yaml)')
    parser.add_argument('--cfg', type=str, default='', help='the config file(.yaml)')
    parser.add_argument('--no_log', action='store_true', default=False, help='do not save log file')
    parser.add_argument('--test', action='store_true', default=False, help='test stage')
    parser.add_argument('--resume', type=str, help='path to load ckpt for resume or test')
    parser.add_argument('--params', type=str, help='Override parameters in yaml file')

    cfg_cli = parser.parse_args()
    # base config
    cfg_base_path = cfg_cli.cfg_base
    if os.path.exists(cfg_base_path):
        print(f'[Info] loading base config file:{cfg_base_path}')
        with open(cfg_base_path) as f:
            cfg = f.read()
            yaml_cfg_base = yaml.safe_load(cfg)
    else:
        print(f'[Warning] can not find the base config file:{cfg_base_path}')

    # detail config
    cfg_path = cfg_cli.cfg
    if os.path.exists(cfg_path):
        print(f'[Info] loading the config file:{cfg_path}')
        with open(cfg_path) as f:
            cfg = f.read()
            yaml_cfg_detail = yaml.safe_load(cfg)
    else:
        print(f'[Warning] can not find the config file:{cfg_path}')
        yaml_cfg_detail = dict()

    cfg_dict = merge_data(yaml_cfg_base, yaml_cfg_detail)
    params_overwrite(cfg_dict, cfg_cli)

    if not cfg_cli.no_log:
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
            auto_output_logging=False
        )
        logging.set_log_file(os.path.join('output', logger_comet.experiment.get_name(), logger_comet.experiment.get_name())+'.log')
        logger_comet.log_hyperparams(cfg_dict)
    else:
        logger_comet = None
    main(cfg_dict, logger_comet)
    if not cfg_cli.no_log:
        print("update logging file")
        logger_comet.experiment.log_asset(logging.log_file_path, file_name='log')


