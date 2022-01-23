from math import inf
import torch
import os
import logging
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.eval_reid import eval_func, visualize_result
import numpy as np
from utils.re_ranking import re_ranking, compute_dis_matrix
from model.model_factory import make_model
from tensorboardX import SummaryWriter
import solver
from loss import make_loss
import random




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
        self.vis_pid_train = cfg.get('vis_pid_train', 100)
        self.vis_pid_test = cfg.get('vis_pid_test', 100)


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
        logging.info(f"[Training...] Epoch:{epoch:0>3d} Base Lr:{self.scheduler._get_lr(epoch)[0]:.2e}".center(80,'='))
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
                    f'[E{epoch:0>3d}|Batch:{idx+1:0>4d}/{batch_num:0>4d}] '
                    f'LOSS=[total:{np.mean(np.array(total_loss_l[idx + 1 - 50:idx + 1])):.4f}]')
                logging.info(f'[E{epoch:0>3d}|Batch:{idx+1:0>4d}/{batch_num:0>4d}] LOSS Detail:' + loss_str_batch)

            # update model
            total_loss.backward()
            self.optimizer.step()

            # vis features
            pids_l += ids.tolist()  # record pid for visualization
            if is_vis:
                if isinstance(feat,list):
                    feat = feat[0]
                #f = PCA_svd(feat, 3)
                features_vis.append(feat.cpu())

        #loging the loss
        total_loss_avg = np.mean(np.array(total_loss_l))
        logging.info('[Epoch:{:0>3d}] LOSS=[total:{:.4f}]'.format(epoch, total_loss_avg))
        loss_avg = np.mean(np.array(loss_value_l),axis=0)
        loss_str = ' '.join([f'[{name}:{loss_avg[i]:.4f}]' for i, name in enumerate(loss_name)])
        logging.info(f'[Epoch:{epoch:0>3d}] LOSS Detail : '+loss_str)
        if self.writer is not None:
            self.writer.add_scalar('train/loss', total_loss_avg, epoch)
            for i, name in enumerate(loss_name):
                self.writer.add_scalar(f'train/{name}', loss_avg[i], epoch)
            if is_vis:
                # fliter pids
                features_vis = torch.cat(features_vis)
                features_vis, pids_arr = self.filter_feature(features_vis,pids_l,self.vis_pid_train)
                self.writer.add_embedding(features_vis, metadata=pids_arr.tolist(), global_step=epoch, tag='train')
        self.save_model(self.net, self.model_name, epoch)
        logging.info("Train Finish")

    def test(self, queryLoader: DataLoader, galleryLoader: DataLoader, epoch = 0, is_vis = False, **kwargs):
        logging.info(f"[Testing...] Epoch:{epoch:>3d}".center(80,'='))
        self.net.eval()
        gf = []
        gPids = np.array([], dtype=int)
        gCids = np.array([], dtype=int)
        g_img_paths = []
        logging.info("compute features of gallery samples")
        for idx, data_dict in enumerate(galleryLoader):
            # extract ids
            pids = data_dict.get('pid')
            cids = data_dict.get('camera_id')
            imgs_path = data_dict.get('img_path')
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
                #gClothesids = np.concatenate((gClothesids, clothes_ids.numpy()), axis=0)
                g_img_paths += imgs_path
        gf = torch.cat(gf, dim=0)

        logging.info("compute features of query samples")
        qf = []
        qPids = np.array([], dtype=int)
        qCids = np.array([], dtype=int)
        q_img_paths = []
        for idx, data_dict in enumerate(queryLoader):
            # extract ids
            pids = data_dict.get('pid')
            cids = data_dict.get('camera_id')
            imgs_path = data_dict.get('img_path')
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
                q_img_paths += imgs_path
        qf = torch.cat(qf, dim=0)

        logging.info("compute dist mat")
        distmat = compute_dis_matrix(qf, gf, self.metrics, is_re_ranking = False)
        # standard mode
        if not is_vis:
            q_img_paths = None
            g_img_paths = None
        logging.info("compute rank list and score")
        cmc_s, mAP_s, mINP_s, res_vis = eval_func(distmat, qPids, gPids, qCids, gCids,
                                                  q_img_paths=q_img_paths,g_img_paths=g_img_paths)
        logging.info(f'test result:[rank-1:{cmc_s[0]:.2%}],[rank-3:{cmc_s[2]:.2%}]'
                     f',[rank-5:{cmc_s[4]:.2%}],[rank-10:{cmc_s[9]:.2%}]')
        logging.info(f'test result:[mAP:{mAP_s:.2%}],[mINP:{mINP_s:.2%}]')
        if self.re_ranking:
            logging.info("compute dist mat (with re-ranking)")
            distmat = compute_dis_matrix(qf, gf, self.metrics)
            cmc_s_r, mAP_s_r, mINP_s_r, res_vis_r = eval_func(distmat, qPids, gPids, qCids, gCids,
                                                      q_img_paths=q_img_paths, g_img_paths=g_img_paths)
            logging.info("compute rank list and score")
            logging.info(f'test result(re-ranking):[rank-1:{cmc_s[0]:.2%}],[rank-3:{cmc_s[2]:.2%}]'
                         f',[rank-5:{cmc_s[4]:.2%}],[rank-10:{cmc_s[9]:.2%}]')
            logging.info(f'test result(re-ranking):[mAP:{mAP_s:.2%}],[mINP:{mINP_s:.2%}]')
        if self.writer is not None:
            self.writer.add_scalar('test/rank-1', cmc_s[0], epoch)
            self.writer.add_scalar('test/mAP', mAP_s, epoch)
            self.writer.add_scalar('test/mINP', mINP_s, epoch)
            # re-ranking result
            if self.re_ranking:
                self.writer.add_scalar('test/rank-1_reranking', cmc_s_r[0], epoch)
                self.writer.add_scalar('test/mAP_reranking', mAP_s_r, epoch)
                self.writer.add_scalar('test/mINP_reranking', mINP_s_r, epoch)
            # feature visualization
            if is_vis:
                logging.info(f'generate visual result (w/o re-rankig)...')
                features_test = torch.cat([gf.cpu(), qf.cpu()])
                pids_l = gPids.tolist() + qPids.tolist()
                features_test, pids_arr = self.filter_feature(features_test,pids_l,self.vis_pid_test)
                if not None == pids_arr:
                    self.writer.add_embedding(features_test, metadata=pids_arr.tolist(), global_step=epoch, tag='test')
                # vis result imgs
                visualize_result(res_vis, writer=self.writer, **kwargs)
        logging.info("[Test Finish]".center(80,'='))

    def filter_feature(self, features:torch.Tensor, pids_l:list, vis_pid):
        # fliter pids
        if isinstance(self.vis_pid_test, int):
            pids_enum = list(set(pids_l))
            if (self.vis_pid_test < 0) or (self.vis_pid_test > len(pids_enum)):
                vis_pid_list = pids_enum
            else:
                vis_pid_list = random.sample(list(set(pids_l)), self.vis_pid_test)
        elif isinstance(self.vis_pid_test, list):
            vis_pid_list = self.vis_pid_test
        else:
            logging.error('vis_pid_train Only Supports: {INT, LIST}')
            return None, None
        # get index of select pids
        pids_arr = torch.tensor(pids_l)
        vis_pid_arr = torch.tensor(vis_pid_list)
        index = torch.where(pids_arr == vis_pid_arr[:, None])[-1]
        features_vis = features[index]
        pids_arr = pids_arr[index]
        return features_vis, pids_arr

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
