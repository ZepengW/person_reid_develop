from math import inf
import torch
import os
import logging
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.eval_reid import eval_func
import numpy as np
from utils.re_ranking import re_ranking, compute_dis_matrix
from model.model_factory import make_model
from tensorboardX import SummaryWriter
import solver
from loss import make_loss




class ModelManager:
    def __init__(self, cfg: dict, device, class_num=1000, writer: SummaryWriter = None):
        self.device = device
        self.model_name = cfg.get('save_name', 'model-no-name')

        # model load
        # add your own network here
        network_params = cfg.get('network-params',dict())
        network_params['num_classes'] = class_num
        self.net = make_model(cfg.get('network_name'),network_params)
        # data keys which the network input
        self.input_keys = cfg.get('network_inputs', ['img','pid'])

        # Multi-GPU Set
        if torch.cuda.device_count() > 1:
            self.net = nn.DataParallel(self.net)
        self.net.to(self.device)

        self.trained_epoches = 0
        # load model trained before
        load_path = cfg.get('load_path', False)
        if type(load_path) is str:
            logging.info(f'loading model: {load_path}' )
            self.net.load_state_dict(torch.load(load_path))
            logging.info('load finish!')
        elif type(load_path) is bool:
            if load_path:
                self.trained_epoches = self.load_model(self.net, cfg.get('save_name','model-no-name'))

        # loss function
        self.lossesFunction = make_loss(cfg.get('loss'), num_classes = class_num)

        # optim
        self.optimizer, _ = solver.make_optimizer(cfg_solver=cfg.get('solver'), model=self.net)

        #scheduler for lr
        self.scheduler = solver.create_scheduler(cfg.get('solver'), self.optimizer)

        # metric
        self.metrics = cfg.get('metric', 'euclidean')
        self.re_ranking = cfg.get('re_ranking', True)
        #tensorboardX
        self.writer = writer


    @staticmethod
    def load_model(model, name):
        if not os.path.exists('./output'):
            logging.warning("no trained model exist, start from init")
            return 0
        pkl_l = [n for n in os.listdir('./output') if (name in n and '.pkl' in n)]
        if len(pkl_l) == 0:
            logging.warning("no trained model exist, start from init")
            return 0
        epoch_longest = max([int(n.split('_')[-1].split('.')[0]) for n in pkl_l])
        file_name = name + '_' + str(epoch_longest) + '.pkl'
        logging.info('loading model: ' + file_name)
        model.load_state_dict(torch.load(os.path.join('./output', file_name)))
        logging.info('load finish!')
        return epoch_longest

    # clear model with same name as current model
    @staticmethod
    def clear_model(name):
        if not os.path.exists('./output'):
            return
        for file in os.listdir('./output'):
            if name in file and '.pkl' in file:
                os.remove(os.path.join('./output', file))

    @staticmethod
    def save_model(model, name, epoch):
        if os.path.exists('./output/' + name + '_' + str(epoch - 1) + '.pkl'):
            os.remove('./output/' + name + '_' + str(epoch - 1) + '.pkl')
        torch.save(model.state_dict(), './output/' + name + '_' + str(epoch) + '.pkl')
        logging.info('save model success: ' + './output/' + name + '_' + str(epoch) + '.pkl')

    def train(self, dataloader: DataLoader, epoch, is_vis = False):
        self.net.train()
        batch_num = len(dataloader)
        total_loss_l = []
        loss_value_l = []
        pids_l = []
        features_vis = []
        bool_warning = False
        # learning rate adjust
        self.scheduler.step(epoch)
        logging.info(f"[Epoch:{epoch:0>4d}] Training... Base Lr:{self.scheduler._get_lr(epoch)[0]:.2e}".center(50,'='))
        for idx, data_dict in enumerate(dataloader):
            self.optimizer.zero_grad()
            # extract ids
            ids = data_dict.get('pid').to(self.device)
            # convert data inputted the network to self.device
            input_data = dict()
            for l in self.input_keys:
                if not l in data_dict.keys():
                    logging.error('data_dict does not contain the data(key) which needed to input the network')
                    return
                else:
                    input_data[l] = data_dict[l].to(self.device)
            score, feat = self.net(**input_data)
            total_loss, loss_value, loss_name = self.lossesFunction(score,feat,ids)

            total_loss_l.append(float(total_loss.cpu()))
            loss_value_l.append(loss_value)

            # output intermediate log
            if (idx + 1) % 50 == 0:
                loss_avg_batch = np.mean(np.array(loss_value_l[idx + 1 - 50:idx + 1]), axis=0)
                loss_str_batch = f' '.join([f'[{name}:{loss_avg_batch[i]:.4f}]' for i, name in enumerate(loss_name)])
                logging.info(
                    f'[E{epoch:0>4d}|Batch:{idx+1:0>4d}/{batch_num:0>4d}] '
                    f'LOSS=[total:{np.mean(np.array(total_loss_l[idx + 1 - 50:idx + 1])):.4f}]')
                logging.info(f'[E{epoch:0>4d}|Batch:{idx+1:0>4d}/{batch_num:0>4d}] LOSS Detail:' + loss_str_batch)

            # update model
            total_loss.backward()
            self.optimizer.step()

            # vis features
            pids_l += ids.tolist()  # record pid for visualization
            if is_vis:
                f = PCA_svd(feat, 3)
                features_vis = features_vis + f.tolist()

        #loging the loss
        total_loss_avg = np.mean(np.array(total_loss_l))
        logging.info('[Epoch:{:0>4d}] LOSS=[total:{:.4f}]'.format(epoch, total_loss_avg))
        loss_avg = np.mean(np.array(loss_value_l),axis=0)
        loss_str = ' '.join([f'[{name}:{loss_avg[i]:.4f}]' for i, name in enumerate(loss_name)])
        logging.info(f'[Epoch:{epoch:0>4d}] LOSS Detail : '+loss_str)
        if self.writer is not None:
            self.writer.add_scalar('train/loss', total_loss_avg, epoch)
            for i, name in enumerate(loss_name):
                self.writer.add_scalar(f'train/{name}', loss_avg[i], epoch)
            if is_vis:
                self.writer.add_embedding(features_vis, metadata=pids_l, global_step=epoch, tag='train')
        self.save_model(self.net, self.model_name, epoch)
        logging.info(f"[Epoch:{epoch:0>4d}] Train Finish".center(50,'='))

    def test(self, queryLoader: DataLoader, galleryLoader: DataLoader, epoch = 0, is_vis = False):
        logging.info("[begin to test]".center(50,'='))
        self.net.eval()
        gf = []
        gPids = np.array([], dtype=int)
        gCids = np.array([], dtype=int)
        gClothesids = np.array([], dtype=int)
        logging.info("compute features of gallery samples")
        for idx, data_dict in enumerate(galleryLoader):
            # extract ids
            pids = data_dict.get('pid')
            cids = data_dict.get('camera_id')
            clothes_ids = data_dict.get('clothes_id')
            # convert data inputted the network to self.device
            input_data = dict()
            for l in self.input_keys:
                if not l in data_dict.keys():
                    logging.error('data_dict does not contain the data(key) which needed to input the network')
                    return
                else:
                    input_data[l] = data_dict[l].to(self.device)
            with torch.no_grad():
                f_whole = self.net(**input_data)
                gf.append(f_whole)
                gPids = np.concatenate((gPids, pids.numpy()), axis=0)
                gCids = np.concatenate((gCids, cids.numpy()), axis=0)
                gClothesids = np.concatenate((gClothesids, clothes_ids.numpy()), axis=0)
        gf = torch.cat(gf, dim=0)

        logging.info("compute features of query samples")
        qf = []
        qPids = np.array([], dtype=int)
        qCids = np.array([], dtype=int)
        qClothesids = np.array([], dtype=int)
        for idx, data_dict in enumerate(queryLoader):
            # extract ids
            pids = data_dict.get('pid')
            cids = data_dict.get('camera_id')
            clothes_ids = data_dict.get('clothes_id')
            # convert data inputted the network to self.device
            input_data = dict()
            for l in self.input_keys:
                if not l in data_dict.keys():
                    logging.error('data_dict does not contain the data(key) which needed to input the network')
                    return
                else:
                    input_data[l] = data_dict[l].to(self.device)
            with torch.no_grad():
                f_whole = self.net(**input_data)
                qf.append(f_whole)
                qPids = np.concatenate((qPids, pids.numpy()), axis=0)
                qCids = np.concatenate((qCids, cids.numpy()), axis=0)
                qClothesids = np.concatenate((qClothesids, clothes_ids.numpy()), axis=0)
        qf = torch.cat(qf, dim=0)

        logging.info("compute rank list and score")
        distmat = compute_dis_matrix(qf,gf,self.metrics, self.re_ranking)
        # standard mode
        cmc_s, mAP_s, mINP_s = eval_func(distmat, qPids, gPids, qCids, gCids)
        # clothes changing mode
        # cmc_c, mAP_c, mINP_c = eval_func(distmat, qPids, gPids, qCids, gCids, q_clo_ids=qClothesids,
        #                                  g_clo_ids=gClothesids)
        logging.info(f'standard mode test result:[rank-1:{cmc_s[0]:.2%}],[rank-3:{cmc_s[2]:.2%}]'
                     f',[rank-5:{cmc_s[4]:.2%}],[rank-10:{cmc_s[9]:.2%}]')
        logging.info(f'standard mode test result:[mAP:{mAP_s:.2%}],[mINP:{mINP_s:.2%}]')
        # logging.info(f'clothes changing mode test result:[rank-1:{cmc_c[0]:.2%}],[rank-3:{cmc_c[2]:.2%}]'
        #              f',[rank-5:{cmc_c[4]:.2%}],[rank-10:{cmc_c[9]:.2%}]')
        # logging.info(f'clothes changing mode test result:[mAP:{mAP_c:.2%}],[mINP:{mINP_c:.2%}]')
        if self.writer is not None:
            self.writer.add_scalar('test-standard/rank-1', cmc_s[0], epoch)
            self.writer.add_scalar('test-standard/mAP', mAP_s, epoch)
            self.writer.add_scalar('test-standard/mINP', mINP_s, epoch)
            # for i, p in enumerate(cmc_s):
            #     self.writer.add_scalar(f'test-standard/cmc/e{epoch}', p, i)
            # self.writer.add_scalar('test-changing/rank-1', cmc_c[0], epoch)
            # self.writer.add_scalar('test-changing/mAP', mAP_c, epoch)
            # self.writer.add_scalar('test-changing/mINP', mINP_c, epoch)
            # for i, p in enumerate(cmc_c):
            #     self.writer.add_scalar(f'test-changing/cmc/e{epoch}', p, i)
            # feature visualization
            if is_vis:
                features_test = torch.cat([gf, qf])
                labels = gPids.tolist() + qPids.tolist()
                self.writer.add_embedding(features_test, metadata=labels, global_step=epoch, tag='test')
        logging.info("[test finish]".center(50,'='))

def PCA_svd(X, k, center=True):
    n = X.size()[0]
    ones = torch.ones(n).view([n, 1])
    h = ((1 / n) * torch.mm(ones, ones.t())) if center else torch.zeros(n * n).view([n, n])
    H = torch.eye(n) - h
    H = H.cuda()
    X_center = torch.mm(H.double(), X.double())
    u, s, v = torch.svd(X_center)
    components = v[:k].t()
    # explained_variance = torch.mul(s[:k], s[:k])/(n-1)
    return components
