import torch
import os
import logging
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.eval_reid import eval_func
from utils.triplet_loss import CrossEntropyLabelSmooth, WeightedRegularizedTriplet
from utils.center_loss import CenterLoss
import numpy as np
from utils.re_ranking import re_ranking
from model.net.resnet import resnet50
from tensorboardX import SummaryWriter

class ModelManager:
    def __init__(self, cfg: dict, device, class_num=1000, writer: SummaryWriter = None):
        self.device = device
        self.model_name = cfg.get('name', 'default-network')
        self.graph_nodes_num = 6

        # model load
        # add your own network here
        self.net = resnet50(pretrained=False,num_classes = class_num)

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
        self.lossesFunction['xent_global'] = nn.CrossEntropyLoss()

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

        #tensorboardX
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

    def train(self, dataloader: DataLoader, epoch):
        logging.info("training epoch : " + str(epoch))
        self.adjust_lr(epoch)
        self.net.train()
        batch_num = len(dataloader)
        total_loss_array = np.zeros([1, batch_num])
        pids_l = []
        features_vis = []
        for idx, (imgs, ids, _, _) in enumerate(dataloader):
            self.optimizer.zero_grad()

            # extract body part features
            imgs = imgs.to(self.device)
            c_global, f = self.net(imgs)
            ids = ids.to(self.device)
            loss_xent_global = self.lossesFunction['xent_global'](c_global, ids)
            total_loss = loss_xent_global

            total_loss_array[0][idx] = (total_loss.cpu())
            # update model
            total_loss.backward()
            self.optimizer.step()

            #vis features
            pids_l += ids.tolist()  # record pid for visualization
            f = PCA_svd(f,3)
            features_vis = features_vis + f.tolist()

        logging.info('[Epoch:{:0>4d}] LOSS=[total:{:.4f}]'
                     .format(epoch, np.mean(total_loss_array[0])))
        if self.writer is not None:
            self.writer.add_scalar('train/loss', np.mean(total_loss_array[0]), epoch)
            self.writer.add_embedding(features_vis, metadata=pids_l, global_step=epoch, tag='train')
        self.save_model(self.net, self.model_name, epoch)

    def test(self, queryLoader: DataLoader, galleryLoader: DataLoader, epoch = 0):
        logging.info("begin to test")
        self.net.eval()
        gf = []
        gPids = np.array([], dtype=int)
        gCids = np.array([], dtype=int)
        logging.info("compute features of gallery samples")
        for idx, (imgs, pids, cids, clothes_ids) in enumerate(galleryLoader):
            imgs = imgs.to(self.device)
            with torch.no_grad():
                c, f_whole = self.net(imgs)
                gf.append(f_whole)
                gPids = np.concatenate((gPids, pids.numpy()), axis=0)
                gCids = np.concatenate((gCids, cids.numpy()), axis=0)
        gf = torch.cat(gf, dim=0)

        logging.info("compute features of query samples")
        qf = []
        qPids = np.array([], dtype=int)
        qCids = np.array([], dtype=int)
        for idx, (imgs, pids, cids, clothes_ids) in enumerate(queryLoader):
            imgs = imgs.to(self.device)
            with torch.no_grad():
                c,f_whole = self.net(imgs)
                qf.append(f_whole)
                qPids = np.concatenate((qPids, pids.numpy()), axis=0)
                qCids = np.concatenate((qCids, cids.numpy()), axis=0)
        qf = torch.cat(qf, dim=0)

        logging.info("compute rank list and score")
        # m, n = qf.shape[0], gf.shape[0]
        # distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
        #           torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        # distmat.addmm_(qf, gf.t(),beta = 1, alpha = -2)
        # distmat = distmat.cpu().numpy()
        distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
        cmc, mAP, mINP = eval_func(distmat, qPids, gPids, qCids, gCids)
        logging.info("test result:[rank-1:{:.2%}],[rank-3:{:.2%}],[rank-5:{:.2%}],[rank-10:{:.2%}]"
                     .format(cmc[0], cmc[2], cmc[4], cmc[9]))
        logging.info("test result:[mAP:{:.2%}],[mINP:{:.2%}]".format(mAP, mINP))
        if self.writer is not None:
            self.writer.add_scalar('test/rank-1', cmc[0], epoch)
            self.writer.add_scalar('test/mAP', mAP, epoch)
            self.writer.add_scalar('test/mINP', mINP, epoch)
            for i, p in enumerate(cmc):
                self.writer.add_scalar(f'test-cmc/e{epoch}',p,i)


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