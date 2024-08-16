from typing import Optional, Union, Callable, Any

import numpy as np
import torch
from lightning.pytorch.utilities.types import _METRIC, EVAL_DATALOADERS, TRAIN_DATALOADERS, STEP_OUTPUT
from model.model_factory import make_model
import solver
from loss import LossManager
import lightning as L
from utils.logger import Logger
logging = Logger()
from torchmetrics import Accuracy
from dataset import DatasetManager, initial_m_reading
from dataset.sampler import RandomIdentitySampler
from torch.utils.data import DataLoader
from utils.eval_reid import Evaluator

class ModelManager(L.LightningModule):
    def __init__(self, cfg_model: dict, cfg_data: dict=None):
        super().__init__()
        self.cfg_model = cfg_model
        # add your own network here
        network_params = cfg_model.get('network-params', dict())
        self.net = make_model(cfg_model.get('network_name'), network_params)
        # data keys which the network input
        self.data_input_l = cfg_model.get('network_inputs', ['img', 'pid'])
        if isinstance(self.data_input_l, list):
            self.data_input_l = {key: key for key in self.data_input_l}

        # loss function
        self.loss_f = LossManager(cfg_model.get('loss'))

        # optim
        self.cfg_solver = cfg_model.get('solver')


        # metric
        self.re_ranking = cfg_model.get('re_ranking', True)
        self.evaluator = Evaluator(
            **cfg_model.get('evaluator', {})
        )
        self.metric_train = Accuracy(task='multiclass', num_classes=network_params.get('num_classes'))

        # dataset
        logging.info('Initial dataset manager')
        self.cfg_data = cfg_data
        self.dataset_manager = DatasetManager(cfg_data.get('dataset_name', ''), cfg_data.get('dataset_path', ''))

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        # reading method for train
        logging.info('Initial train loader')
        m_reading_train_cfg = self.cfg_data.get('reading_method_train')
        params = m_reading_train_cfg.get('params')
        m_reading_train = initial_m_reading(m_reading_train_cfg.get('name'), **params)
        batch_size_train = self.cfg_data.get('batch_size_train', 16)
        num_instance = self.cfg_data.get('num_instance', 4)
        data_sampler = RandomIdentitySampler(self.dataset_manager.get_dataset_list('train'),
                                             batch_size_train,
                                             num_instance)
        logging.info("Reading method (Train)")
        logging.info(m_reading_train)
        loader_train = DataLoader(
            self.dataset_manager.get_dataset_image('train', m_reading_train),
            num_workers=self.cfg_data.get('num_workers', 8),
            batch_size=batch_size_train,
            sampler=data_sampler,
            drop_last=True
        )
        return loader_train

    def val_dataloader(self) -> EVAL_DATALOADERS:
        logging.info('Initial gallery loader')
        m_reading_test_cfg = self.cfg_data.get('reading_method_test')
        params = m_reading_test_cfg.get('params')
        m_reading_test = initial_m_reading(m_reading_test_cfg.get('name'), **params)
        logging.info("Reading method (Test)")
        logging.info(m_reading_test)
        loader_gallery = DataLoader(
            self.dataset_manager.get_dataset_image('test', m_reading_test),
            batch_size=self.cfg_data.get('batch_size_test', 16),
            num_workers=self.cfg_data.get('num_workers', 8),
            drop_last=False,
            shuffle=False
        )
        logging.info('Initial query loader')
        loader_query = DataLoader(
            self.dataset_manager.get_dataset_image('query', m_reading_test),
            batch_size=self.cfg_data.get('batch_size_test', 16),
            num_workers=self.cfg_data.get('num_workers', 8),
            drop_last=False,
            shuffle=False
        )
        return [loader_gallery, loader_query]
    
    def test_dataloader(self) -> EVAL_DATALOADERS:
        return self.val_dataloader()
    
    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return self.val_dataloader()

    def configure_optimizers(self):
        optimizer, _ = solver.make_optimizer(cfg_solver=self.cfg_solver, model=self.net)
        # scheduler for lr
        lr_scheduler = solver.create_scheduler(self.cfg_solver, optimizer)
        if lr_scheduler is None:
            logging.warning('No lr scheduler')
            return optimizer
        lr_scheduler_config = {
            "scheduler": lr_scheduler,
            "interval": "epoch",
        }
        return {
            'optimizer': optimizer,
            'lr_scheduler': lr_scheduler_config
        }

    def refactor_inputs(self, batch):
        input_data = dict()
        for l in self.data_input_l.keys():
            if not l in batch.keys():
                logging.error(f'The expected input {l} for network is accessible.')
                raise
            else:
                input_data[self.data_input_l[l]] = batch[l]
        return input_data

    def on_train_epoch_start(self):
        # logging lr
        logging.info(f"Epoch[{self.current_epoch:0>3d}] Begin to train")
        for i, optimizer in enumerate(self.trainer.optimizers):
            for j, param_group in enumerate(optimizer.param_groups):
                current_lr = param_group['lr']
                group_name = param_group['group_name']
                logging.info(f"Lr group:{group_name} | lr:{current_lr:.2e}")
                if group_name == 'default':
                    self.log('lr', current_lr)

    def training_step(self, batch, batch_idx):
        input_data = self.refactor_inputs(batch)
        output_data = self.net(**input_data)
        batch.update(output_data)  # update result
        loss_dict = self.loss_f(batch)
        # calculate acc
        self.metric_train(output_data['pred'], batch['pid'])
        self.log(f'train/acc', self.metric_train, on_epoch=True, on_step=False, prog_bar=False, logger=True)
        for key, item in loss_dict.items():
            self.log(f"train/{key}", item.item(),
                     on_step=True, on_epoch=True, prog_bar=(key == 'loss'), logger=True)
        return loss_dict['loss']

    def on_train_epoch_end(self) -> None:
        metrics = self.trainer.callback_metrics
        log_train = f'acc:{metrics["train/acc"]:.2%} loss:{metrics["train/loss"]:.4f}'
        for loss in self.cfg_model['loss']:
            log_train += f' {loss["name"]}:{metrics["train/"+loss["name"]+"_epoch"]:.4f}'
        logging.info('[Train/Metrics] ' + log_train)

    def on_validation_epoch_start(self) -> None:
        logging.info(f'Begin to validation')
        self.feat_gallery_l = []
        self.pid_gallery_l = []
        self.cid_gallery_l = []
        self.feat_query_l = []
        self.pid_query_l = []
        self.cid_query_l = []

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """
        dataloader_list: [gallery(0), query(1)]
        """
        input_data = self.refactor_inputs(batch)
        output_data = self.net(**input_data)
        if dataloader_idx == 0:
            self.feat_gallery_l.append(output_data['feat'].cpu().clone().detach())
            self.pid_gallery_l.append(batch['pid'].cpu().clone().detach().numpy())
            self.cid_gallery_l.append(batch['camera_id'].cpu().clone().detach().numpy())
        elif dataloader_idx == 1:
            self.feat_query_l.append(output_data['feat'].cpu().clone().detach())
            self.pid_query_l.append(batch['pid'].cpu().clone().detach().numpy())
            self.cid_query_l.append(batch['camera_id'].cpu().clone().detach().numpy())


    def on_validation_epoch_end(self) -> None:
        # calculate top-1 and mAP
        feat_gallery = torch.cat(self.feat_gallery_l)
        feat_query = torch.cat(self.feat_query_l)
        pid_gallery = np.concatenate(self.pid_gallery_l)
        pid_query = np.concatenate(self.pid_query_l)
        cid_gallery = np.concatenate(self.cid_gallery_l)
        cid_query = np.concatenate(self.cid_query_l)

        res = self.evaluator(feat_query, feat_gallery, pid_query, pid_gallery, cid_query, cid_gallery, re_ranking=self.re_ranking)
        cmc = res[0]
        m_ap = res[1]
        m_inp = res[2]
        logging.info(
            f'{"Eval(w/o RK):":<15} {"Rank-1:":<8} {cmc[0]:>7.2%} {"Rank-3:":<8} {cmc[2]:>7.2%} {"Rank-5:":<8} {cmc[4]:>7.2%} {"Rank-10:":<8} {cmc[9]:>7.2%}')
        logging.info(f'{" ":<15} {"mAP:":<8} {m_ap:>7.2%} {m_inp:>7.2%}')
        self.log_dict({
            'val/R1': cmc[0],
            'val/R3': cmc[2],
            'val/R5': cmc[4],
            'val/R10': cmc[9],
            'val/mAP': m_ap,
            'val/mINP': m_inp
        }, on_epoch=True)
        if self.re_ranking:
            cmc = res[3]
            m_ap = res[4]
            m_inp = res[5]
            logging.info(
                f'{"Eval(w/o RK):":<15} {"Rank-1:":<8} {cmc[0]:>7.2%} {"Rank-3:":<8} {cmc[2]:>7.2%} {"Rank-5:":<8} {cmc[4]:>7.2%} {"Rank-10:":<8} {cmc[9]:>7.2%}')
            logging.info(f'{" ":<15} {"mAP:":<8} {m_ap:>7.2%} {m_inp:>7.2%}')
            self.log_dict({
                'val/R1': cmc[0],
                'val/R3': cmc[2],
                'val/R5': cmc[4],
                'val/R10': cmc[9],
                'val/mAP': m_ap,
                'val/mINP': m_inp
            }, on_epoch=True)

        self.feat_gallery_l.clear()
        self.pid_gallery_l.clear()
        self.cid_gallery_l.clear()
        self.feat_query_l.clear()
        self.pid_query_l.clear()
        self.cid_query_l.clear()

    def on_test_epoch_start(self) -> None:
        self.on_validation_epoch_start()

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        # Call the validation_step method for testing
        return self.validation_step(batch, batch_idx, dataloader_idx)

    def on_test_epoch_end(self):
        # Call the on_validation_epoch_end method for testing
        self.on_validation_epoch_end()
    
    def on_predict_epoch_start(self) -> None:
        self.on_validation_epoch_start()
        
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # Call the validation_step method for prediction
        return self.validation_step(batch, batch_idx, dataloader_idx)
    
    def on_predict_epoch_end(self):
        # Call the on_validation_epoch_end method for prediction
        feat_gallery = torch.cat(self.feat_gallery_l)
        feat_query = torch.cat(self.feat_query_l)
        pid_gallery = np.concatenate(self.pid_gallery_l)
        pid_query = np.concatenate(self.pid_query_l)
        cid_gallery = np.concatenate(self.cid_gallery_l)
        cid_query = np.concatenate(self.cid_query_l)
        self.evaluator.infer(feat_query, feat_gallery, re_ranking=self.re_ranking)
        self.feat_gallery_l.clear()
        self.pid_gallery_l.clear()
        self.cid_gallery_l.clear()
        self.feat_query_l.clear()
        self.pid_query_l.clear()
        self.cid_query_l.clear()
