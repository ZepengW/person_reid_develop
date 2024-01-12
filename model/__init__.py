from typing import Optional, Union, Callable, Any

import torch
from lightning.pytorch.utilities.types import _METRIC, EVAL_DATALOADERS, TRAIN_DATALOADERS

from utils.eval_reid import eval_func
from utils.re_ranking import compute_dis_matrix
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

class ModelManager(L.LightningModule):
    def __init__(self, cfg_model: dict, cfg_data: dict=None):
        super().__init__()
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
        self.metrics = cfg_model.get('metric', 'euclidean')
        self.re_ranking = cfg_model.get('re_ranking', True)
        self.metric_train = Accuracy(task='multiclass', num_classes=network_params.get('num_classes'))

        # dataset
        logging.info('Initial dataset manager')
        self.cfg_data = cfg_data
        self.dataset_manager = DatasetManager(cfg_data.get('dataset_name', ''), cfg_data.get('dataset_path', ''))

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        # reading method for train
        logging.info('Initial train loader')
        m_reading_train_cfg = self.cfg_data.get('reading_method_train')
        m_reading_train = initial_m_reading(m_reading_train_cfg.get('name'), **m_reading_train_cfg)
        batch_size_train = self.cfg_data.get('batch_size_train', 16)
        num_instance = self.cfg_data.get('num_instance', 4)
        data_sampler = RandomIdentitySampler(self.dataset_manager.get_dataset_list('train'),
                                             batch_size_train,
                                             num_instance)
        loader_train = DataLoader(
            self.dataset_manager.get_dataset_image('train', m_reading_train),
            batch_size=batch_size_train,
            num_workers=self.cfg_data.get('num_workers', 8),
            sampler=data_sampler
        )
        return loader_train

    def val_dataloader(self) -> EVAL_DATALOADERS:
        logging.info('Initial gallery loader')
        m_reading_test_cfg = self.cfg_data.get('reading_method_test')
        m_reading_test = initial_m_reading(m_reading_test_cfg.get('name'), **m_reading_test_cfg)
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

    def configure_optimizers(self):
        optimizer, _ = solver.make_optimizer(cfg_solver=self.cfg_solver, model=self.net)
        # scheduler for lr
        lr_scheduler = solver.create_scheduler(self.cfg_solver, optimizer)
        if lr_scheduler is None:
            return optimizer
        lr_scheduler_config = {
            # REQUIRED: The scheduler instance
            "scheduler": lr_scheduler,
            # The unit of the scheduler's step size, could also be 'step'.
            # 'epoch' updates the scheduler on epoch end whereas 'step'
            # updates it after a optimizer update.
            "interval": "epoch",
            # How many epochs/steps should pass between calls to
            # `scheduler.step()`. 1 corresponds to updating the learning
            # rate after every epoch/step.
            "frequency": 1,
            # Metric to monitor for schedulers like `ReduceLROnPlateau`
            # "monitor": "val_loss",
            # If set to `True`, will enforce that the value specified 'monitor'
            # is available when the scheduler is updated, thus stopping
            # training if not found. If set to `False`, it will only produce a warning
            # "strict": True,
            # If using the `LearningRateMonitor` callback to monitor the
            # learning rate progress, this keyword can be used to specify
            # a custom logged name
            "name": None,
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

    def training_step(self, batch, batch_idx):
        input_data = self.refactor_inputs(batch)
        output_data = self.net(**input_data)
        batch.update(output_data)  # update result
        loss_dict = self.loss_f(batch)
        # calculate acc
        self.metric_train(output_data['pred'], batch['pid'])
        for key, item in loss_dict.items():
            self.log(f"train/{key}", item.item(),
                     on_step=True, on_epoch=True, prog_bar=(key == 'loss'), logger=True)
        return loss_dict['loss']

    def on_train_epoch_end(self) -> None:
        self.log(f"train/acc", self.metric_train, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        logging.info(f'[Train|E{self.current_epoch:0>4d}] Acc:{self.metric_train.compute():>4.2%}')
        self.metric_train.reset()

    def on_validation_epoch_start(self) -> None:
        logging.info('Begin to Validation')
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
            self.feat_gallery_l.append(output_data['feat'].cpu())
            self.pid_gallery_l.append(batch['pid'].cpu())
            self.cid_gallery_l.append(batch['camera_id'].cpu())
        elif dataloader_idx == 1:
            self.feat_query_l.append(output_data['feat'].cpu())
            self.pid_query_l.append(batch['pid'].cpu())
            self.cid_query_l.append(batch['camera_id'].cpu())

    def on_validation_epoch_end(self) -> None:
        # calculate top-1 and mAP
        feat_gallery = torch.cat(self.feat_gallery_l)
        feat_query = torch.cat(self.feat_query_l)
        pid_gallery = torch.cat(self.pid_gallery_l).numpy()
        pid_query = torch.cat(self.pid_query_l).numpy()
        cid_gallery = torch.cat(self.cid_gallery_l).numpy()
        cid_query = torch.cat(self.cid_query_l).numpy()

        logging.info("compute dist mat")
        dist_mat = compute_dis_matrix(feat_query, feat_gallery, self.metrics, is_re_ranking=False)
        logging.info("compute rank list and score")
        cmc, m_ap, m_inp, _ = eval_func(dist_mat, pid_query, pid_gallery, cid_query, cid_gallery)
        logging.info(
            f'{"Result(w/o RK):":<15} {"Rank-1:":<8} {cmc[0]:>7.2%} {"Rank-3:":<8} {cmc[2]:>7.2%} {"Rank-5:":<8} {cmc[4]:>7.2%} {"Rank-10:":<8} {cmc[9]:>7.2%}')
        logging.info(f'{" ":<15} {"mAP:":<8} {m_ap:>7.2%} {"mINP:":<8} {m_inp:>7.2%}')
        self.log_dict({
            'val/R1': cmc[0],
            'val/R3': cmc[2],
            'val/R5': cmc[4],
            'val/R10': cmc[9],
            'val/mAP': m_ap,
            'val/mINP': m_inp
        }, on_epoch=True)
        if self.re_ranking:
            logging.info("compute dist mat (with RK)")
            dist_mat = compute_dis_matrix(feat_query, feat_gallery, self.metrics, is_re_ranking=True)
            logging.info("compute rank list and score")
            cmc, m_ap, m_inp, _ = eval_func(dist_mat, pid_query, pid_gallery, cid_query, cid_gallery)
            logging.info(
                f'{"Result(with RK):":<15} {"Rank-1:":<8} {cmc[0]:>7.2%} {"Rank-3:":<8} {cmc[2]:>7.2%} {"Rank-5:":<8} {cmc[4]:>7.2%} {"Rank-10:":<8} {cmc[9]:>7.2%}')
            logging.info(f'{" ":<15} {"mAP:":<8} {m_ap:>7.2%} {"mINP:":<8} {m_inp:>7.2%}')
            self.log_dict({
                'val/R1(RK)': cmc[0],
                'val/R3(RK)': cmc[2],
                'val/R5(RK)': cmc[4],
                'val/R10(RK)': cmc[9],
                'val/mAP(RK)': m_ap,
                'val/mINP(RK)': m_inp
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
        self.validation_step(batch, batch_idx, dataloader_idx)

    def test_epoch_end(self):
        # Call the on_validation_epoch_end method for testing
        self.on_validation_epoch_end()
