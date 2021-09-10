import torch
import os
import logging
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.eval_reid import eval_func
from utils.triplet_loss import CrossEntropyLabelSmooth, WeightedRegularizedTriplet
from utils.center_loss import CenterLoss
import numpy as np
from utils.re_ranking import re_ranking, compute_dis_matrix
#from model.net.multi_branch_network import MbNetwork
from model.net.person_transformer import PersonTransformer
from tensorboardX import SummaryWriter


class ModelManager:
    def __init__(self, cfg: dict, device, class_num=1000, writer: SummaryWriter = None):
        self.device = device
        self.model_name = cfg.get('name', 'default-network')
        self.graph_nodes_num = 6

        # model load
        # add your own network here
        self.net = PersonTransformer(num_classes=class_num,camera_num=cfg.get('camera-num'),
                                     vit_pretrained_path=cfg.get('vit-pretrained-path',None))

        # Multi-GPU Set
        if torch.cuda.device_count() > 1:
            self.net = nn.DataParallel(self.net)
        self.net.to(self.device)

        # load model trained before
        if cfg.get('continue_train', True):
            self.trained_epoches = self.load_model(self.net, self.model_name) + 1
        else:
            self.clear_model(self.model_name)
            self.trained_epoches = 0

        # loss function
        self.lossesFunction = {}
        self.lossesFunction['xent'] = nn.CrossEntropyLoss()
        self.lossesFunction['trip_weight'] = WeightedRegularizedTriplet()

        # optim
        self.lr = cfg.get('lr', 0.0001)
        self.lr_adjust = cfg.get('lr_adjust', [])
        self.weight_decay = cfg.get('weight_decay', 0.0005)
        params = []

        for key, value in self.net.named_parameters():
            if not value.requires_grad:
                continue
            params += [{"params": [value], "lr": self.lr, "weight_decay": self.weight_decay}]
        self.optimizer = getattr(torch.optim, cfg.get('optimzer', 'Adam'))(params)

        # metric 
        self.metrics = cfg.get('metric','euclidean')
        self.re_ranking = cfg.get('re_ranking', True)
        # tensorboardX
        self.writer = writer


    @staticmethod
    def load_model(model, name):
        if not os.path.exists('./output'):
            logging.warning("load trained model failed")
            return -1
        pkl_l = [n for n in os.listdir('./output') if (name in n and '.pkl' in n)]
        if len(pkl_l) == 0:
            logging.warning("no trained model exist, start from init")
            return -1
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

    def train(self, dataloader: DataLoader, epoch, is_vis= False):
        logging.info("training epoch : " + str(epoch))
        self.adjust_lr(epoch)
        self.net.train()
        batch_num = len(dataloader)
        total_loss_array = np.zeros([3, batch_num])
        pids_l = []
        features_vis = []
        for idx, (imgs, ids, cam_id, _, masks) in enumerate(dataloader):
            self.optimizer.zero_grad()

            # extract body part features
            imgs = imgs.to(self.device)
            masks = masks.to(self.device)
            cam_id = cam_id.to(self.device)
            scores, feats = self.net(imgs, masks,cam_id)
            ids = ids.to(self.device)

            # compute loss
            loss_trip = 0
            loss_xent = 0
            for i in range(0,len(scores)):
                loss_xent += self.lossesFunction['xent'](scores[i],ids)
            for i in range(0,len(feats)):
                loss_trip += self.lossesFunction['trip_weight'](feats[i], ids)[0]

            total_loss = loss_trip + loss_xent

            total_loss_array[0][idx] = (total_loss.cpu())
            total_loss_array[1][idx] = (loss_xent.cpu())
            total_loss_array[2][idx] = (loss_trip.cpu())
            # update model
            total_loss.backward()
            self.optimizer.step()

            # vis features
            pids_l += ids.tolist()  # record pid for visualization
            if is_vis:
                f = PCA_svd(feats[0], 3)
                features_vis = features_vis + f.tolist()

        logging.info('[Epoch:{:0>4d}] LOSS=[total:{:.4f}]'
                     .format(epoch, np.mean(total_loss_array[0])))
        if self.writer is not None:
            self.writer.add_scalar('train/loss', np.mean(total_loss_array[0]), epoch)
            self.writer.add_scalar('train/loss-xent', np.mean(total_loss_array[1]), epoch)
            self.writer.add_scalar('train/loss-trip', np.mean(total_loss_array[2]), epoch)
            if is_vis:
                self.writer.add_embedding(features_vis, metadata=pids_l, global_step=epoch, tag='train')
        self.save_model(self.net, self.model_name, epoch)

    def test(self, queryLoader: DataLoader, galleryLoader: DataLoader, epoch=0, is_vis= True):
        logging.info("begin to test")
        self.net.eval()
        gf = []
        gPids = np.array([], dtype=int)
        gCids = np.array([], dtype=int)
        gClothesids = np.array([], dtype=int)
        logging.info("compute features of gallery samples")
        for idx, (imgs, pids, cids, clothes_ids, masks) in enumerate(galleryLoader):
            imgs = imgs.to(self.device)
            masks = masks.to(self.device)
            with torch.no_grad():
                f_whole = self.net(imgs, masks,cids)
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
        for idx, (imgs, pids, cids, clothes_ids, masks) in enumerate(queryLoader):
            imgs = imgs.to(self.device)
            masks = masks.to(self.device)
            with torch.no_grad():
                f_whole = self.net(imgs, masks,cids)
                qf.append(f_whole)
                qPids = np.concatenate((qPids, pids.numpy()), axis=0)
                qCids = np.concatenate((qCids, cids.numpy()), axis=0)
                qClothesids = np.concatenate((qClothesids, clothes_ids.numpy()), axis=0)
        qf = torch.cat(qf, dim=0)

        logging.info("compute rank list and score")
        qf = qf.cpu()
        gf = gf.cpu()
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
                labels = [str(s) + '_g' for s in gPids.tolist()] + [str(s) + '_q' for s in qPids.tolist()]
                self.writer.add_embedding(features_test, metadata=labels, global_step=epoch, tag='test')

    def adjust_lr(self, epoch):
        lr = self.lr * (
                0.1 ** np.sum(epoch >= np.array(self.lr_adjust)))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


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
